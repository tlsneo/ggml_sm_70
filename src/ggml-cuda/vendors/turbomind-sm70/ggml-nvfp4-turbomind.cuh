#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

struct ggml_cuda_nvfp4_turbomind_prepared {
    void * weight = nullptr;
    void * scales = nullptr;
    cudaEvent_t ready = nullptr;
    int device = -1;
    int64_t k = 0;
    int64_t n = 0;
    int k_ld = 0;
    int q_ld = 0;
    size_t weight_bytes = 0;
    size_t scale_bytes = 0;
};

size_t ggml_cuda_nvfp4_turbomind_barriers_size();
size_t ggml_cuda_nvfp4_turbomind_partials_size();
size_t ggml_cuda_nvfp4_turbomind_tensormaps_size();

bool ggml_cuda_nvfp4_turbomind_prepare(
        const void * src,
        int64_t k,
        int64_t n,
        void * temp_codes,
        void * temp_scales,
        ggml_cuda_nvfp4_turbomind_prepared * prepared,
        cudaStream_t stream);

void ggml_cuda_nvfp4_turbomind_release(ggml_cuda_nvfp4_turbomind_prepared * prepared);

bool ggml_cuda_nvfp4_turbomind_mul_mat(
        int device,
        const ggml_cuda_nvfp4_turbomind_prepared & prepared,
        const void * activation_fp16,
        int64_t m,
        float * dst,
        void * barriers,
        size_t barriers_size,
        void * partials,
        size_t partials_size,
        void * tensormaps,
        size_t tensormaps_size,
        int * flags,
        cudaStream_t stream);
