#pragma once

#include "../../common.cuh"

struct ggml_cuda_flashinfer_sm70_attention_params {
    const half * q;
    const half * k;
    const half * v;
    half * output;
    int length;
    int heads;
    int batches;
    float scale;
    int64_t q_stride_row;
    int64_t q_stride_head;
    int64_t q_stride_batch;
    int64_t k_stride_row;
    int64_t k_stride_head;
    int64_t k_stride_batch;
    int64_t v_stride_row;
    int64_t v_stride_head;
    int64_t v_stride_batch;
    int64_t o_stride_row;
    int64_t o_stride_head;
    int64_t o_stride_batch;
};

void ggml_cuda_flashinfer_sm70_attention(
        const ggml_cuda_flashinfer_sm70_attention_params & params,
        cudaStream_t stream);
