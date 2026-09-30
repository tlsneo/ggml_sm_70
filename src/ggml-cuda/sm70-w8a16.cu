// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from the 1Cat/vLLM SM70 H3 W8A16 implementation.
#include "sm70-w8a16.cuh"

#include <algorithm>
#include <climits>

namespace {

template <typename T>
__device__ __forceinline__ float sm70_input_to_float(T value);

template <>
__device__ __forceinline__ float sm70_input_to_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float sm70_input_to_float<half>(half value) {
    return __half2float(value);
}

template <typename T>
__global__ void prepare_fp16_rows_sm70(
        const T * input, half * output, float * scales, int64_t rows, int width) {
    __shared__ float warp_maxima[8];
    __shared__ float row_scale;

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;

    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        float maximum = 0.0f;
        for (int col = tid; col < width; col += blockDim.x) {
            const float value = fabsf(sm70_input_to_float(input[row * width + col]));
            maximum = fmaxf(maximum, isnan(value) ? INFINITY : value);
        }
#pragma unroll
        for (int delta = 16; delta > 0; delta /= 2) {
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
        }
        if (lane == 0) {
            warp_maxima[warp] = maximum;
        }
        __syncthreads();

        if (warp == 0) {
            maximum = lane < 8 ? warp_maxima[lane] : 0.0f;
#pragma unroll
            for (int delta = 16; delta > 0; delta /= 2) {
                maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
            }
            if (lane == 0) {
                int exponent = 0;
                if (isfinite(maximum)) {
                    frexpf(maximum, &exponent);
                }
                row_scale  = ldexpf(1.0f, max(exponent - 11, 0));
                scales[row] = row_scale;
            }
        }
        __syncthreads();

        for (int col = tid; col < width; col += blockDim.x) {
            output[row * width + col] = __float2half_rn(
                sm70_input_to_float(input[row * width + col]) / row_scale);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ float convrot_butterfly_sm70(
        float a, float b, float c, float d, int digit) {
    return digit == 0   ? a + b + c - d
         : digit == 1   ? a + b - c + d
         : digit == 2   ? a - b + c + d
                        : -a + b + c + d;
}

__global__ void convrot256_inplace_sm70(half * data, int64_t groups) {
    const int lane = threadIdx.x % 32;

    for (int64_t group = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
         group < groups;
         group += int64_t(gridDim.x) * (blockDim.x / 32)) {
        float values[8];
        float next[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            values[i] = __half2float(data[group * 256 + lane + 32 * i]);
        }
#pragma unroll
        for (int stride = 1; stride <= 4; stride *= 4) {
            const int digit = (lane / stride) % 4;
            const int base  = lane - digit * stride;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float a = __shfl_sync(0xffffffff, values[i], base);
                const float b = __shfl_sync(0xffffffff, values[i], base + stride);
                const float c = __shfl_sync(0xffffffff, values[i], base + 2 * stride);
                const float d = __shfl_sync(0xffffffff, values[i], base + 3 * stride);
                values[i] = convrot_butterfly_sm70(a, b, c, d, digit);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int even  = i & ~1;
            const int base  = lane % 16;
            const int digit = lane / 16 + 2 * (i % 2);
            const float a = __shfl_sync(0xffffffff, values[even], base);
            const float b = __shfl_sync(0xffffffff, values[even], base + 16);
            const float c = __shfl_sync(0xffffffff, values[even + 1], base);
            const float d = __shfl_sync(0xffffffff, values[even + 1], base + 16);
            next[i] = convrot_butterfly_sm70(a, b, c, d, digit);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int base = i % 2;
            const float value = convrot_butterfly_sm70(
                next[base], next[base + 2], next[base + 4], next[base + 6], i / 2);
            data[group * 256 + lane + 32 * i] = __float2half_rn(value * (1.0f / 16.0f));
        }
    }
}

__global__ void dequantize_i8_weights_sm70(
        const int8_t * weight, const float * scales, half * output, int64_t count, int64_t width) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        const float scale = scales == nullptr ? 1.0f : scales[i / width];
        output[i] = __float2half_rn(float(weight[i]) * scale);
    }
}

__global__ void restore_rows_sm70(
        float * output, const float * scales, const float * bias, int64_t n, int64_t count) {
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += int64_t(gridDim.x) * blockDim.x) {
        const int64_t row = index / n;
        const int64_t col = index - row * n;
        float value = output[index] * scales[row];
        if (bias != nullptr) {
            value += bias[col];
        }
        output[index] = value;
    }
}

} // namespace

void ggml_cuda_mul_mat_i8_sm70_w8a16(
        const int8_t * weight,
        const void * input,
        bool input_f16,
        const float * weight_scales,
        const float * bias,
        float * output,
        half * activation_f16,
        half * weight_f16,
        float * activation_scales,
        int64_t k,
        int64_t n,
        int64_t rows,
        cudaStream_t stream,
        cublasHandle_t handle) {
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(input != nullptr);
    GGML_ASSERT(output != nullptr);
    GGML_ASSERT(activation_f16 != nullptr);
    GGML_ASSERT(weight_f16 != nullptr);
    GGML_ASSERT(activation_scales != nullptr);
    GGML_ASSERT(k > 0 && k <= INT_MAX && k % 256 == 0);
    GGML_ASSERT(n > 0 && n <= INT_MAX);
    GGML_ASSERT(rows > 0 && rows <= INT_MAX);

    if (input_f16) {
        prepare_fp16_rows_sm70<<<std::min<int64_t>(rows, 65535), 256, 0, stream>>>(
            static_cast<const half *>(input), activation_f16, activation_scales, rows, (int) k);
    } else {
        prepare_fp16_rows_sm70<<<std::min<int64_t>(rows, 65535), 256, 0, stream>>>(
            static_cast<const float *>(input), activation_f16, activation_scales, rows, (int) k);
    }
    CUDA_CHECK(cudaGetLastError());

    const int64_t groups = rows * (k / 256);
    convrot256_inplace_sm70<<<std::min<int64_t>((groups + 3) / 4, 65535), 128, 0, stream>>>(
        activation_f16, groups);
    CUDA_CHECK(cudaGetLastError());

    const int64_t weight_count = n * k;
    dequantize_i8_weights_sm70<<<std::min<int64_t>((weight_count + 255) / 256, 65535), 256, 0, stream>>>(
        weight, weight_scales, weight_f16, weight_count, k);
    CUDA_CHECK(cudaGetLastError());

    CUBLAS_CHECK(cublasSetStream(handle, stream));
    cublasMath_t saved_math;
    CUBLAS_CHECK(cublasGetMathMode(handle, &saved_math));
    CUBLAS_CHECK(cublasSetMathMode(
        handle,
        static_cast<cublasMath_t>(
            CUBLAS_TENSOR_OP_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const cublasStatus_t gemm_status = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        (int) n, (int) rows, (int) k,
        &alpha,
        weight_f16, CUDA_R_16F, (int) k,
        activation_f16, CUDA_R_16F, (int) k,
        &beta,
        output, CUDA_R_32F, (int) n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    const cublasStatus_t restore_status = cublasSetMathMode(handle, saved_math);
    CUBLAS_CHECK(gemm_status);
    CUBLAS_CHECK(restore_status);

    const int64_t output_count = n * rows;
    const int grid = std::min<int64_t>((output_count + 255) / 256, 65535);
    restore_rows_sm70<<<grid, 256, 0, stream>>>(output, activation_scales, bias, n, output_count);
    CUDA_CHECK(cudaGetLastError());
}
