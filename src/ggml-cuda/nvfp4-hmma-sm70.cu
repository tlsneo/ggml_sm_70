#include "nvfp4-hmma-sm70.cuh"

#include "mma.cuh"

#include <algorithm>
#include <climits>

using namespace ggml_cuda_mma;

static constexpr int HMMA_ROWS = 32;
static constexpr int HMMA_K_PAD = 36;

template <int cols_per_block, int nwarps>
__launch_bounds__(WARP_SIZE*nwarps, 1)
static __global__ void nvfp4_hmma_sm70_kernel(
        const block_nvfp4 * __restrict__ weights,
        const float * __restrict__ activation,
        float * __restrict__ output,
        int k,
        int n,
        int m) {
#if defined(VOLTA_MMA_AVAILABLE)
    using tile_A = tile<32, 4, half2, DATA_LAYOUT_I_MAJOR>;
    using tile_B = tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<32, 8, float, DATA_LAYOUT_I_MAJOR>;

    constexpr int ntB = (cols_per_block + tile_B::I - 1) / tile_B::I;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row0 = blockIdx.x * HMMA_ROWS;
    const int col0 = blockIdx.y * cols_per_block;
    const int blocks_per_row = k / QK_NVFP4;
    const int pairs_per_row = k / 2;

    extern __shared__ char shared_raw[];
    half2 * tile_xy = reinterpret_cast<half2 *>(shared_raw) + warp * HMMA_ROWS * HMMA_K_PAD;
    tile_C C[ntB];

    for (int pair = warp * WARP_SIZE + lane; pair < pairs_per_row; pair += nwarps * WARP_SIZE) {
        tile_A A[WARP_SIZE / tile_A::J];

#pragma unroll
        for (int i = 0; i < HMMA_ROWS; ++i) {
            const block_nvfp4 & block = weights[(row0 + i) * blocks_per_row + pair / (QK_NVFP4 / 2)];
            const int value0 = 2 * lane;
            const int sub = value0 / QK_NVFP4_SUB;
            const int pos0 = value0 % QK_NVFP4_SUB;
            const int pos1 = pos0 + 1;
            const uint8_t packed0 = block.qs[sub * (QK_NVFP4_SUB / 2) + pos0 % (QK_NVFP4_SUB / 2)];
            const uint8_t packed1 = block.qs[sub * (QK_NVFP4_SUB / 2) + pos1 % (QK_NVFP4_SUB / 2)];
            const int q0 = pos0 < QK_NVFP4_SUB / 2 ? packed0 & 0x0F : packed0 >> 4;
            const int q1 = pos1 < QK_NVFP4_SUB / 2 ? packed1 & 0x0F : packed1 >> 4;
            const float d = ggml_cuda_ue4m3_to_fp32(block.d[sub]);
            tile_xy[i * HMMA_K_PAD + lane] = __floats2half2_rn(
                d * kvalues_mxfp4[q0], d * kvalues_mxfp4[q1]);
        }

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += tile_A::J) {
            load_ldmatrix(A[k0 / tile_A::J], tile_xy + k0, HMMA_K_PAD);
        }

        const float2 * activation2 = reinterpret_cast<const float2 *>(activation);
#pragma unroll
        for (int itB = 0; itB < ntB; ++itB) {
#pragma unroll
            for (int j0 = 0; j0 < tile_B::I; ++j0) {
                const int j = itB * tile_B::I + j0;
                const bool valid = col0 + j < m;
                const float2 value = valid
                    ? activation2[(col0 + j) * pairs_per_row + pair]
                    : make_float2(0.0f, 0.0f);
                tile_xy[j0 * HMMA_K_PAD + lane] = __floats2half2_rn(value.x, value.y);
            }
#pragma unroll
            for (int k0 = 0; k0 < WARP_SIZE; k0 += tile_B::J) {
                tile_B B;
                load_ldmatrix(B, tile_xy + k0, HMMA_K_PAD);
                mma(C[itB], A[k0 / tile_B::J], B);
            }
        }
    }

    float * partials = reinterpret_cast<float *>(shared_raw);
    constexpr int partial_stride = nwarps * HMMA_ROWS + 4;
    __syncthreads();

#pragma unroll
    for (int itB = 0; itB < ntB; ++itB) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int i = warp * HMMA_ROWS + tile_C::get_i(l);
            const int j = itB * tile_C::J + tile_C::get_j(l);
            partials[j * partial_stride + i] = C[itB].x[l];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j0 = 0; j0 < cols_per_block; j0 += nwarps) {
        const int j = j0 + warp;
        if (j >= cols_per_block || col0 + j >= m) {
            continue;
        }
        float sum = 0.0f;
#pragma unroll
        for (int i0 = 0; i0 < nwarps * HMMA_ROWS; i0 += HMMA_ROWS) {
            sum += partials[j * partial_stride + i0 + lane];
        }
        output[(col0 + j) * n + row0 + lane] = sum;
    }
#else
    GGML_UNUSED_VARS(weights, activation, output, k, n, m);
    NO_DEVICE_CODE;
#endif
}

template <int cols_per_block, int nwarps>
static void launch_nvfp4_hmma_sm70(
        const block_nvfp4 * weights,
        const float * activation,
        float * output,
        int k,
        int n,
        int m,
        cudaStream_t stream) {
    const size_t iter_bytes = size_t(nwarps) * HMMA_ROWS * HMMA_K_PAD * sizeof(half2);
    const size_t combine_bytes = size_t(cols_per_block) * (nwarps * HMMA_ROWS + 4) * sizeof(float);
    const size_t shared_bytes = std::max(iter_bytes, combine_bytes);
    const dim3 blocks(n / HMMA_ROWS, (m + cols_per_block - 1) / cols_per_block, 1);
    const dim3 threads(WARP_SIZE, nwarps, 1);
    nvfp4_hmma_sm70_kernel<cols_per_block, nwarps><<<blocks, threads, shared_bytes, stream>>>(
        weights, activation, output, k, n, m);
}

template <int cols_per_block>
static void launch_nvfp4_hmma_sm70_warps(
        const block_nvfp4 * weights,
        const float * activation,
        float * output,
        int k,
        int n,
        int m,
        cudaStream_t stream) {
    const int pairs = k / 2;
    if (pairs <= 64) {
        launch_nvfp4_hmma_sm70<cols_per_block, 1>(weights, activation, output, k, n, m, stream);
    } else if (pairs <= 128) {
        launch_nvfp4_hmma_sm70<cols_per_block, 2>(weights, activation, output, k, n, m, stream);
    } else {
        launch_nvfp4_hmma_sm70<cols_per_block, 4>(weights, activation, output, k, n, m, stream);
    }
}

void ggml_cuda_nvfp4_hmma_sm70(
        const block_nvfp4 * weights,
        const float * activation,
        float * output,
        int64_t k,
        int64_t n,
        int64_t m,
        cudaStream_t stream) {
    GGML_ASSERT(k > 0 && k % QK_NVFP4 == 0);
    GGML_ASSERT(n > 0 && n % HMMA_ROWS == 0);
    GGML_ASSERT(m > 0);
    GGML_ASSERT(k <= INT_MAX && n <= INT_MAX && m <= INT_MAX);

    if (m <= 8) {
        launch_nvfp4_hmma_sm70_warps<8>(weights, activation, output, k, n, m, stream);
    } else if (m <= 16) {
        launch_nvfp4_hmma_sm70_warps<16>(weights, activation, output, k, n, m, stream);
    } else {
        launch_nvfp4_hmma_sm70_warps<32>(weights, activation, output, k, n, m, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
