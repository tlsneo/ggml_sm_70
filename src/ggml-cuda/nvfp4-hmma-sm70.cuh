#pragma once

#include "common.cuh"

void ggml_cuda_nvfp4_hmma_sm70(
        const block_nvfp4 * weights,
        const float * activation,
        float * output,
        int64_t k,
        int64_t n,
        int64_t m,
        cudaStream_t stream);
