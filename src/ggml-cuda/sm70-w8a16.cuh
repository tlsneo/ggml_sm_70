// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from the 1Cat/vLLM SM70 H3 W8A16 implementation.
#pragma once

#include "common.cuh"

void ggml_cuda_mul_mat_i8_sm70_w8a16(
        const int8_t * weight,
        const float * input,
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
        cublasHandle_t handle);
