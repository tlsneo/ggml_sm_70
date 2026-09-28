#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

struct ggml_cuda_qk_rmsnorm_rope_sm70_strides {
    int64_t head;
    int64_t token;
    int64_t batch;
};

struct ggml_cuda_qk_rmsnorm_rope_sm70_params {
    const float * q;
    const float * k;
    const float * v;
    const float * q_weight;
    const float * k_weight;
    const float * rope;
    half * q_out;
    half * k_out;
    half * v_out;
    ggml_cuda_qk_rmsnorm_rope_sm70_strides q_strides;
    ggml_cuda_qk_rmsnorm_rope_sm70_strides k_strides;
    ggml_cuda_qk_rmsnorm_rope_sm70_strides v_strides;
    int64_t heads;
    int64_t tokens;
    int64_t batches;
    int32_t rot_dim;
    float epsilon;
    float kv_scale;
};

void ggml_cuda_qk_rmsnorm_rope_sm70(
    const ggml_cuda_qk_rmsnorm_rope_sm70_params & params,
    cudaStream_t stream);

void ggml_cuda_qk_rmsnorm_rope_sm70_restore_output(
    const half * src,
    float * dst,
    int64_t count,
    float inverse_kv_scale,
    cudaStream_t stream);
