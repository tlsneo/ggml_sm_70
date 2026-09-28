#include "ggml-cutlass-sm70-attention.cuh"

#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/kernel/default_gemm.h>

#include "default_fmha.h"

#include <climits>

namespace {

using CutlassHalf = cutlass::half_t;
constexpr int HEAD_DIM = 128;

template <int Queries, int Keys>
using KernelFor = typename cutlass::gemm::kernel::H3FMHA<
    CutlassHalf, cutlass::arch::Sm70, true, Queries, Keys, HEAD_DIM>::FMHAKernel;

template <int Queries, int Keys>
__global__ __launch_bounds__(Queries * 2, 1) void cutlass_sm70_attention_kernel(
        typename KernelFor<Queries, Keys>::DirectParams params) {
    extern __shared__ __align__(16) unsigned char storage[];
    KernelFor<Queries, Keys> kernel;
    kernel(params, *reinterpret_cast<typename KernelFor<Queries, Keys>::SharedStorage *>(storage));
}

template <int Queries, int Keys>
void launch_attention(const ggml_cuda_cutlass_sm70_attention_params & source, cudaStream_t stream) {
    using Kernel = KernelFor<Queries, Keys>;
    typename Kernel::DirectParams params{};
    params.q = reinterpret_cast<CutlassHalf *>(const_cast<half *>(source.q));
    params.k = reinterpret_cast<CutlassHalf *>(const_cast<half *>(source.k));
    params.v = reinterpret_cast<CutlassHalf *>(const_cast<half *>(source.v));
    params.output = reinterpret_cast<CutlassHalf *>(source.output);
    params.queries = source.queries;
    params.keys = source.keys;
    params.heads = source.heads;
    params.scale = source.scale;
    params.q_stride_row = source.q_stride_row;
    params.q_stride_head = source.q_stride_head;
    params.q_stride_batch = source.q_stride_batch;
    params.k_stride_row = source.k_stride_row;
    params.k_stride_head = source.k_stride_head;
    params.k_stride_batch = source.k_stride_batch;
    params.v_stride_row = source.v_stride_row;
    params.v_stride_head = source.v_stride_head;
    params.v_stride_batch = source.v_stride_batch;
    params.o_stride_row = source.o_stride_row;
    params.o_stride_head = source.o_stride_head;
    params.o_stride_batch = source.o_stride_batch;

    const size_t shared_bytes = sizeof(typename Kernel::SharedStorage);
    if constexpr (Keys == 128) {
        CUDA_CHECK(cudaFuncSetAttribute(
            cutlass_sm70_attention_kernel<Queries, Keys>,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    if (shared_bytes > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            cutlass_sm70_attention_kernel<Queries, Keys>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }

    cutlass_sm70_attention_kernel<Queries, Keys>
        <<<dim3((source.queries + Queries - 1) / Queries, source.batches * source.heads),
           Kernel::kThreadCount, shared_bytes, stream>>>(params);
    CUDA_CHECK(cudaGetLastError());
}

template <int Queries>
void launch_key_tile(const ggml_cuda_cutlass_sm70_attention_params & params, cudaStream_t stream) {
    if (params.key_tile == 128) {
        launch_attention<Queries, 128>(params, stream);
    } else {
        launch_attention<Queries, 64>(params, stream);
    }
}

} // namespace

void ggml_cuda_cutlass_sm70_attention(
        const ggml_cuda_cutlass_sm70_attention_params & params,
        cudaStream_t stream) {
    GGML_ASSERT(params.q != nullptr && params.k != nullptr && params.v != nullptr && params.output != nullptr);
    GGML_ASSERT(params.queries > 0 && params.keys > 0 && params.heads > 0 && params.batches > 0);
    GGML_ASSERT(params.queries <= INT_MAX && params.keys <= INT_MAX);
    GGML_ASSERT(params.batches * params.heads <= 65535);
    GGML_ASSERT(params.query_tile == 64 || params.query_tile == 128);
    GGML_ASSERT(params.key_tile == 64 || params.key_tile == 128);

    if (params.query_tile == 128) {
        launch_key_tile<128>(params, stream);
    } else {
        launch_key_tile<64>(params, stream);
    }
}
