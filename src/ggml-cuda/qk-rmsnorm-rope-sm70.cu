#include "qk-rmsnorm-rope-sm70.cuh"

#include "common.cuh"

#include <climits>

namespace {

constexpr int HEAD_DIM = 128;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS = WARPS_PER_BLOCK * 32;

template <bool WITH_V>
__global__ void qk_rmsnorm_rope_sm70_kernel(ggml_cuda_qk_rmsnorm_rope_sm70_params params) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t row = int64_t(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const int64_t rows = params.batches * params.heads * params.tokens;
    if (row >= rows) {
        return;
    }

    const int64_t token = row % params.tokens;
    const int64_t head = (row / params.tokens) % params.heads;
    const int64_t batch = row / (params.tokens * params.heads);
    const int64_t q_base = batch * params.q_strides.batch +
                           token * params.q_strides.token +
                           head * params.q_strides.head;
    const int64_t k_base = batch * params.k_strides.batch +
                           token * params.k_strides.token +
                           head * params.k_strides.head;
    const int64_t v_base = WITH_V ? batch * params.v_strides.batch +
                                    token * params.v_strides.token +
                                    head * params.v_strides.head : 0;

    float q_values[4];
    float k_values[4];
    float q_sum = 0.0f;
    float k_sum = 0.0f;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
        const int dim = lane + part * 32;
        q_values[part] = params.q[q_base + dim];
        k_values[part] = params.k[k_base + dim];
        q_sum += q_values[part] * q_values[part];
        k_sum += k_values[part] * k_values[part];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += __shfl_down_sync(0xffffffffu, q_sum, offset);
        k_sum += __shfl_down_sync(0xffffffffu, k_sum, offset);
    }
    q_sum = __shfl_sync(0xffffffffu, q_sum, 0);
    k_sum = __shfl_sync(0xffffffffu, k_sum, 0);
    const float q_inv = rsqrtf(q_sum * (1.0f / HEAD_DIM) + params.epsilon);
    const float k_inv = rsqrtf(k_sum * (1.0f / HEAD_DIM) + params.epsilon);

    // 1Cat 的数值合同要求 RMSNorm 输出先舍入到 FP16，再执行 RoPE。
    __shared__ half normalized[2][WARPS_PER_BLOCK][HEAD_DIM];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
        const int dim = lane + part * 32;
        normalized[0][warp][dim] = __float2half_rn((q_values[part] * q_inv) * params.q_weight[dim]);
        normalized[1][warp][dim] = __float2half_rn((k_values[part] * k_inv) * params.k_weight[dim]);
    }
    __syncwarp();

    const int half_rot = params.rot_dim / 2;
    const int64_t rope_base = token * int64_t(half_rot) * 4;
    const int64_t out_base = row * HEAD_DIM;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
        const int dim = lane + part * 32;
        half q_result = normalized[0][warp][dim];
        half k_result = normalized[1][warp][dim];
        if (dim < params.rot_dim) {
            const bool second = dim >= half_rot;
            const int pair = second ? dim - half_rot : dim;
            const half q_first = normalized[0][warp][pair];
            const half q_second = normalized[0][warp][pair + half_rot];
            const half k_first = normalized[1][warp][pair];
            const half k_second = normalized[1][warp][pair + half_rot];
            const int matrix_row = second ? 1 : 0;
            const int64_t matrix = rope_base + int64_t(pair) * 4 + matrix_row * 2;
            const half a = __float2half_rn(params.rope[matrix]);
            const half b = __float2half_rn(params.rope[matrix + 1]);

            // 每个乘积先舍入到 FP16，再以 FP32 加法合并并再次舍入；禁止 FMA 改变边界。
            const half q_product_a = __float2half_rn(__half2float(q_first) * __half2float(a));
            const half q_product_b = __float2half_rn(__half2float(q_second) * __half2float(b));
            const half k_product_a = __float2half_rn(__half2float(k_first) * __half2float(a));
            const half k_product_b = __float2half_rn(__half2float(k_second) * __half2float(b));
            q_result = __float2half_rn(__half2float(q_product_a) + __half2float(q_product_b));
            k_result = __float2half_rn(__half2float(k_product_a) + __half2float(k_product_b));
        }

        params.q_out[out_base + dim] = q_result;
        params.k_out[out_base + dim] = __float2half_rn(__half2float(k_result) * params.kv_scale);
        if constexpr (WITH_V) {
            params.v_out[out_base + dim] = __float2half_rn(params.v[v_base + dim] * params.kv_scale);
        }
    }
}

__global__ void restore_output_kernel(
        const half * src, float * dst, int64_t count, float inverse_kv_scale) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        dst[index] = __half2float(src[index]) * inverse_kv_scale;
    }
}

} // namespace

void ggml_cuda_qk_rmsnorm_rope_sm70(
        const ggml_cuda_qk_rmsnorm_rope_sm70_params & params,
        cudaStream_t stream) {
    GGML_ASSERT(params.q != nullptr && params.k != nullptr);
    GGML_ASSERT(params.q_weight != nullptr && params.k_weight != nullptr && params.rope != nullptr);
    GGML_ASSERT(params.q_out != nullptr && params.k_out != nullptr);
    GGML_ASSERT(params.heads > 0 && params.tokens > 0 && params.batches > 0);
    GGML_ASSERT(params.rot_dim == 96 || params.rot_dim == 128);
    GGML_ASSERT(params.epsilon > 0.0f && params.kv_scale > 0.0f);
    const int64_t rows = params.batches * params.heads * params.tokens;
    GGML_ASSERT(rows <= int64_t(INT_MAX) * WARPS_PER_BLOCK);
    const int blocks = int((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    if (params.v != nullptr) {
        GGML_ASSERT(params.v_out != nullptr);
        qk_rmsnorm_rope_sm70_kernel<true><<<blocks, THREADS, 0, stream>>>(params);
    } else {
        qk_rmsnorm_rope_sm70_kernel<false><<<blocks, THREADS, 0, stream>>>(params);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_qk_rmsnorm_rope_sm70_restore_output(
        const half * src,
        float * dst,
        int64_t count,
        float inverse_kv_scale,
        cudaStream_t stream) {
    GGML_ASSERT(src != nullptr && dst != nullptr && count > 0 && inverse_kv_scale > 0.0f);
    constexpr int threads = 256;
    const int blocks = int((count + threads - 1) / threads);
    restore_output_kernel<<<blocks, threads, 0, stream>>>(src, dst, count, inverse_kv_scale);
    CUDA_CHECK(cudaGetLastError());
}
