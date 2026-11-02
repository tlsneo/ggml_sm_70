#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "qk-rmsnorm-rope-sm70.cuh"
#ifdef GGML_CUDA_CUTLASS_SM70_ATTN
#include "vendors/cutlass-sm70-attention/ggml-cutlass-sm70-attention.cuh"
#endif
#ifdef GGML_CUDA_FLASHINFER_SM70_ATTN
#include "vendors/flashinfer-sm70-attention/ggml-flashinfer-sm70-attention.cuh"
#endif

#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// one list per group of ncols1 queries: a column is selected if any query of the group can see it
template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            bool selected = false;
            if (i < ne30) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
                }
            }
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    // the last group of queries is partial only if ncols1 does not divide n_queries
    GGML_ASSERT(ncols1 == 1 || ncols1 == 8);
    const auto kernel = ncols1 == 1       ? flash_attn_mask_to_sparse_indices<1, false> :
                        n_queries % 8 != 0 ? flash_attn_mask_to_sparse_indices<8, true>  :
                                             flash_attn_mask_to_sparse_indices<8, false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
        (const half *) mask->data, indices, counts, int(mask->ne[0]), n_queries, n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    // the dense kernel handles up to 64/ncols2 queries per K/V pass, the single-query gather has to beat that
    const int64_t n_gather = (ncols1 == 1 ? std::min<int64_t>(Q->ne[1], 64/ncols2) : ncols1) * (int64_t) n_kv_max;

    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2*n_gather);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        // a sparse variant at the full tile width gathers the union of its queries once, prefer it for large batches
        constexpr bool has_wide_sparse = ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 64/ncols2, ncols2);
        if (!(has_wide_sparse && Q->ne[1] > 32/ncols2) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, ncols2)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20 && GGML_CUDA_CC_IS_NVIDIA(cc)) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        if (head_size == (D) && type_K == GGML_TYPE_##type_K_case && type_V == GGML_TYPE_##type_V_case) {                          \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K_in, const ggml_type type_V_in) {
    const ggml_type type_K = type_K_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_K_in;
    const ggml_type type_V = type_V_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_V_in;
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE          =   0,
    BEST_FATTN_KERNEL_TILE          = 200,
    BEST_FATTN_KERNEL_VEC           = 100,
    BEST_FATTN_KERNEL_MMA_F16                   = 400,
    BEST_FATTN_KERNEL_CUTLASS_SM70              = 500,
    BEST_FATTN_KERNEL_FLASHINFER_SM70           = 600,
    BEST_FATTN_KERNEL_QK_RMSNORM_ROPE_SM70      = 700,
};

enum class ggml_cuda_sm70_attention_mode {
    AUTO,
    MMA,
    CUTLASS,
    FLASHINFER,
};

static ggml_cuda_sm70_attention_mode ggml_cuda_get_sm70_attention_mode() {
    static const ggml_cuda_sm70_attention_mode mode = [] {
        const char * env = getenv("GGML_CUDA_SM70_ATTN");
        if (env == nullptr) {
            return ggml_cuda_sm70_attention_mode::AUTO;
        }
        std::string value = env;
        for (char & c : value) {
            c = std::tolower(c);
        }
        if (value == "auto") {
            return ggml_cuda_sm70_attention_mode::AUTO;
        }
        if (value == "mma") {
            return ggml_cuda_sm70_attention_mode::MMA;
        }
        if (value == "cutlass") {
            return ggml_cuda_sm70_attention_mode::CUTLASS;
        }
        if (value == "flashinfer") {
            return ggml_cuda_sm70_attention_mode::FLASHINFER;
        }
        GGML_ABORT("GGML_CUDA_SM70_ATTN must be auto, mma, cutlass or flashinfer");
    }();
    return mode;
}

#ifdef GGML_CUDA_CUTLASS_SM70_ATTN
static int ggml_cuda_get_sm70_attention_query_tile() {
    static const int tile = [] {
        const char * env = getenv("GGML_CUDA_SM70_ATTN_QUERY_TILE");
        if (env == nullptr) {
            return 64;
        }
        char * end = nullptr;
        const long value = strtol(env, &end, 10);
        if (end == env || *end != '\0' || (value != 64 && value != 128)) {
            GGML_ABORT("GGML_CUDA_SM70_ATTN_QUERY_TILE must be 64 or 128");
        }
        return int(value);
    }();
    return tile;
}

static int ggml_cuda_get_sm70_attention_key_tile() {
    static const int tile = [] {
        const char * env = getenv("GGML_CUDA_SM70_ATTN_KEY_TILE");
        if (env == nullptr) {
            return 0;
        }
        char * end = nullptr;
        const long value = strtol(env, &end, 10);
        if (end == env || *end != '\0' || (value != 0 && value != 64 && value != 128)) {
            GGML_ABORT("GGML_CUDA_SM70_ATTN_KEY_TILE must be 0, 64 or 128");
        }
        return int(value);
    }();
    return tile != 0 ? tile : 128;
}
#endif

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static bool ggml_cuda_is_qk_rmsnorm_rope_attention(const ggml_tensor * dst) {
    return dst->src[5] != nullptr || dst->src[6] != nullptr || dst->src[7] != nullptr;
}

static bool ggml_cuda_use_qk_rmsnorm_rope_sm70(const int device, const ggml_tensor * dst) {
#ifdef GGML_CUDA_CUTLASS_SM70_ATTN
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * q_weight = dst->src[5];
    const ggml_tensor * k_weight = dst->src[6];
    const ggml_tensor * rope = dst->src[7];
    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    const float eps = ggml_get_op_params_f32(dst, 6);
    const float kv_scale = ggml_get_op_params_f32(dst, 5);
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(scale));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(max_bias));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(logit_softcap));

    return ggml_cuda_info().devices[device].cc == GGML_CUDA_CC_VOLTA
        && Q != nullptr && K != nullptr && V != nullptr
        && q_weight != nullptr && k_weight != nullptr && rope != nullptr
        && dst->src[3] == nullptr && dst->src[4] == nullptr
        && Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F32 && V->type == GGML_TYPE_F32
        && q_weight->type == GGML_TYPE_F32 && k_weight->type == GGML_TYPE_F32
        && rope->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && ggml_flash_attn_ext_get_prec(dst) == GGML_PREC_F32
        && Q->ne[0] == 128 && ggml_are_same_shape(Q, K) && ggml_are_same_shape(Q, V)
        && Q->ne[1] > 0 && Q->ne[2] > 0 && Q->ne[3] > 0
        && ggml_is_contiguous_rows(Q) && ggml_is_contiguous_rows(K) && ggml_is_contiguous_rows(V)
        && Q->nb[0] == sizeof(float) && K->nb[0] == sizeof(float) && V->nb[0] == sizeof(float)
        && Q->nb[1] % sizeof(float) == 0 && Q->nb[2] % sizeof(float) == 0 && Q->nb[3] % sizeof(float) == 0
        && K->nb[1] % sizeof(float) == 0 && K->nb[2] % sizeof(float) == 0 && K->nb[3] % sizeof(float) == 0
        && V->nb[1] % sizeof(float) == 0 && V->nb[2] % sizeof(float) == 0 && V->nb[3] % sizeof(float) == 0
        && ggml_nelements(q_weight) == 128 && ggml_nelements(k_weight) == 128
        && ggml_is_contiguous(q_weight) && ggml_is_contiguous(k_weight)
        && ggml_is_contiguous(rope) && rope->ne[0] == 2 && rope->ne[1] == 2
        && (rope->ne[2] == 48 || rope->ne[2] == 64) && rope->ne[3] == Q->ne[2]
        && dst->ne[0] == 128 && dst->ne[1] == Q->ne[1]
        && dst->ne[2] == Q->ne[2] && dst->ne[3] == Q->ne[3]
        && ggml_is_contiguous(dst)
        && Q->ne[2] <= INT_MAX && Q->ne[1] * Q->ne[3] <= 65535
        && Q->ne[1] * 128 <= INT_MAX
        && std::isfinite(scale) && scale > 0.0f
        && std::isfinite(kv_scale) && kv_scale > 0.0f
        && std::isfinite(scale / kv_scale) && scale / kv_scale > 0.0f
        && std::isfinite(eps) && eps > 0.0f
        && max_bias == 0.0f && logit_softcap == 0.0f;
#else
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#endif
}

#if defined(GGML_CUDA_CUTLASS_SM70_ATTN) || defined(GGML_CUDA_FLASHINFER_SM70_ATTN)
static bool ggml_cuda_use_sm70_d128_attention_common(const int device, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const int cc = ggml_cuda_info().devices[device].cc;

    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(scale));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(max_bias));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(logit_softcap));

    return cc == GGML_CUDA_CC_VOLTA
        && Q != nullptr && K != nullptr && V != nullptr
        && mask == nullptr && sinks == nullptr
        && Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16
        && dst->type == GGML_TYPE_F32 && ggml_flash_attn_ext_get_prec(dst) == GGML_PREC_F32
        && Q->ne[0] == 128 && K->ne[0] == 128 && V->ne[0] == 128
        && Q->ne[1] > 0 && K->ne[1] > 0 && K->ne[1] == V->ne[1]
        && Q->ne[2] > 0 && Q->ne[2] == K->ne[2] && Q->ne[2] == V->ne[2]
        && Q->ne[3] > 0 && Q->ne[3] == K->ne[3] && Q->ne[3] == V->ne[3]
        && dst->ne[0] == 128 && dst->ne[1] == Q->ne[2]
        && dst->ne[2] == Q->ne[1] && dst->ne[3] == Q->ne[3]
        && ggml_is_contiguous(Q) && ggml_is_contiguous(dst)
        && Q->nb[0] == sizeof(float) && K->nb[0] == sizeof(half) && V->nb[0] == sizeof(half)
        && K->nb[1] == 128 * sizeof(half) && V->nb[1] == 128 * sizeof(half)
        && K->nb[2] % sizeof(half) == 0 && K->nb[3] % sizeof(half) == 0
        && V->nb[2] % sizeof(half) == 0 && V->nb[3] % sizeof(half) == 0
        && reinterpret_cast<uintptr_t>(K->data) % 16 == 0
        && reinterpret_cast<uintptr_t>(V->data) % 16 == 0
        && Q->ne[1] <= INT_MAX && K->ne[1] <= INT_MAX
        && Q->ne[2] * Q->ne[3] <= 65535
        && Q->ne[2] * 128 <= INT_MAX
        && std::isfinite(scale) && scale > 0.0f
        && max_bias == 0.0f && logit_softcap == 0.0f;
}
#endif

static bool ggml_cuda_use_cutlass_sm70_attention(const int device, const ggml_tensor * dst) {
#ifdef GGML_CUDA_CUTLASS_SM70_ATTN
    return ggml_cuda_use_sm70_d128_attention_common(device, dst);
#else
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#endif
}

static bool ggml_cuda_use_flashinfer_sm70_attention(const int device, const ggml_tensor * dst) {
#ifdef GGML_CUDA_FLASHINFER_SM70_ATTN
    return ggml_cuda_use_sm70_d128_attention_common(device, dst)
        && dst->src[0]->ne[1] == dst->src[1]->ne[1];
#else
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#endif
}

#ifdef GGML_CUDA_CUTLASS_SM70_ATTN
static void ggml_cuda_flash_attn_ext_qk_rmsnorm_rope_sm70(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_use_qk_rmsnorm_rope_sm70(ctx.device, dst));
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * q_weight = dst->src[5];
    const ggml_tensor * k_weight = dst->src[6];
    const ggml_tensor * rope = dst->src[7];
    const int64_t heads = Q->ne[1];
    const int64_t tokens = Q->ne[2];
    const int64_t batches = Q->ne[3];
    const int64_t elements = 128 * heads * tokens * batches;
    const cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<half> q_f16(ctx.pool(), elements);
    ggml_cuda_pool_alloc<half> k_f16(ctx.pool(), elements);
    ggml_cuda_pool_alloc<half> v_f16(ctx.pool(), elements);
    ggml_cuda_pool_alloc<half> output_f16(ctx.pool(), elements);

    const float eps = ggml_get_op_params_f32(dst, 6);
    const float kv_scale = ggml_get_op_params_f32(dst, 5);
    const ggml_cuda_qk_rmsnorm_rope_sm70_params prep{
        static_cast<const float *>(Q->data),
        static_cast<const float *>(K->data),
        static_cast<const float *>(V->data),
        static_cast<const float *>(q_weight->data),
        static_cast<const float *>(k_weight->data),
        static_cast<const float *>(rope->data),
        q_f16.get(),
        k_f16.get(),
        v_f16.get(),
        {static_cast<int64_t>(Q->nb[1] / sizeof(float)),
         static_cast<int64_t>(Q->nb[2] / sizeof(float)),
         static_cast<int64_t>(Q->nb[3] / sizeof(float))},
        {static_cast<int64_t>(K->nb[1] / sizeof(float)),
         static_cast<int64_t>(K->nb[2] / sizeof(float)),
         static_cast<int64_t>(K->nb[3] / sizeof(float))},
        {static_cast<int64_t>(V->nb[1] / sizeof(float)),
         static_cast<int64_t>(V->nb[2] / sizeof(float)),
         static_cast<int64_t>(V->nb[3] / sizeof(float))},
        heads,
        tokens,
        batches,
        static_cast<int32_t>(rope->ne[2] * 2),
        eps,
        kv_scale,
    };
    ggml_cuda_qk_rmsnorm_rope_sm70(prep, stream);

    float scale = 0.0f;
    memcpy(&scale, (const float *) dst->op_params, sizeof(scale));
    const ggml_cuda_cutlass_sm70_attention_params params{
        q_f16.get(),
        k_f16.get(),
        v_f16.get(),
        output_f16.get(),
        static_cast<int>(tokens),
        static_cast<int>(tokens),
        static_cast<int>(heads),
        static_cast<int>(batches),
        scale / kv_scale,
        ggml_cuda_get_sm70_attention_query_tile(),
        ggml_cuda_get_sm70_attention_key_tile(),
        128,
        tokens * 128,
        heads * tokens * 128,
        128,
        tokens * 128,
        heads * tokens * 128,
        128,
        tokens * 128,
        heads * tokens * 128,
        heads * 128,
        128,
        tokens * heads * 128,
    };
    ggml_cuda_cutlass_sm70_attention(params, stream);
    ggml_cuda_qk_rmsnorm_rope_sm70_restore_output(
        output_f16.get(), static_cast<float *>(dst->data), elements, 1.0f / kv_scale, stream);
}

static void ggml_cuda_flash_attn_ext_cutlass_sm70(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_use_cutlass_sm70_attention(ctx.device, dst));
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const int64_t queries = Q->ne[1];
    const int64_t keys = K->ne[1];
    const int64_t heads = Q->ne[2];
    const int64_t batches = Q->ne[3];
    const cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<half> q_f16(ctx.pool(), ggml_nelements(Q));
    ggml_cuda_pool_alloc<half> output_f16(ctx.pool(), ggml_nelements(dst));
    ggml_get_to_fp16_cuda(GGML_TYPE_F32)(Q->data, q_f16.get(), ggml_nelements(Q), stream);

    float scale = 0.0f;
    memcpy(&scale, (const float *) dst->op_params, sizeof(scale));
    const ggml_cuda_cutlass_sm70_attention_params params{
        q_f16.get(),
        static_cast<const half *>(K->data),
        static_cast<const half *>(V->data),
        output_f16.get(),
        static_cast<int>(queries),
        static_cast<int>(keys),
        static_cast<int>(heads),
        static_cast<int>(batches),
        scale,
        ggml_cuda_get_sm70_attention_query_tile(),
        ggml_cuda_get_sm70_attention_key_tile(),
        128,
        queries * 128,
        heads * queries * 128,
        static_cast<int64_t>(K->nb[1] / sizeof(half)),
        static_cast<int64_t>(K->nb[2] / sizeof(half)),
        static_cast<int64_t>(K->nb[3] / sizeof(half)),
        static_cast<int64_t>(V->nb[1] / sizeof(half)),
        static_cast<int64_t>(V->nb[2] / sizeof(half)),
        static_cast<int64_t>(V->nb[3] / sizeof(half)),
        heads * 128,
        128,
        queries * heads * 128,
    };
    ggml_cuda_cutlass_sm70_attention(params, stream);
    ggml_get_to_fp32_cuda(GGML_TYPE_F16)(output_f16.get(), static_cast<float *>(dst->data), ggml_nelements(dst), stream);
}
#endif

#ifdef GGML_CUDA_FLASHINFER_SM70_ATTN
static void ggml_cuda_flash_attn_ext_flashinfer_sm70(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_use_flashinfer_sm70_attention(ctx.device, dst));
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const int64_t length = Q->ne[1];
    const int64_t heads = Q->ne[2];
    const int64_t batches = Q->ne[3];
    const cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<half> q_f16(ctx.pool(), ggml_nelements(Q));
    ggml_cuda_pool_alloc<half> output_f16(ctx.pool(), ggml_nelements(dst));
    ggml_get_to_fp16_cuda(GGML_TYPE_F32)(Q->data, q_f16.get(), ggml_nelements(Q), stream);

    float scale = 0.0f;
    memcpy(&scale, (const float *) dst->op_params, sizeof(scale));
    const ggml_cuda_flashinfer_sm70_attention_params params{
        q_f16.get(),
        static_cast<const half *>(K->data),
        static_cast<const half *>(V->data),
        output_f16.get(),
        static_cast<int>(length),
        static_cast<int>(heads),
        static_cast<int>(batches),
        scale,
        128,
        length * 128,
        heads * length * 128,
        static_cast<int64_t>(K->nb[1] / sizeof(half)),
        static_cast<int64_t>(K->nb[2] / sizeof(half)),
        static_cast<int64_t>(K->nb[3] / sizeof(half)),
        static_cast<int64_t>(V->nb[1] / sizeof(half)),
        static_cast<int64_t>(V->nb[2] / sizeof(half)),
        static_cast<int64_t>(V->nb[3] / sizeof(half)),
        heads * 128,
        128,
        length * heads * 128,
    };
    ggml_cuda_flashinfer_sm70_attention(params, stream);
    ggml_get_to_fp32_cuda(GGML_TYPE_F16)(output_f16.get(), static_cast<float *>(dst->data), ggml_nelements(dst), stream);
}
#endif

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    if (ggml_cuda_is_qk_rmsnorm_rope_attention(dst)) {
        const auto mode = ggml_cuda_get_sm70_attention_mode();
        return (mode == ggml_cuda_sm70_attention_mode::AUTO ||
                mode == ggml_cuda_sm70_attention_mode::CUTLASS) &&
               ggml_cuda_use_qk_rmsnorm_rope_sm70(device, dst)
            ? BEST_FATTN_KERNEL_QK_RMSNORM_ROPE_SM70
            : BEST_FATTN_KERNEL_NONE;
    }

    if (ggml_cuda_get_sm70_attention_mode() == ggml_cuda_sm70_attention_mode::CUTLASS) {
        return ggml_cuda_use_cutlass_sm70_attention(device, dst)
            ? BEST_FATTN_KERNEL_CUTLASS_SM70
            : BEST_FATTN_KERNEL_NONE;
    }

    if (ggml_cuda_get_sm70_attention_mode() == ggml_cuda_sm70_attention_mode::FLASHINFER) {
        return ggml_cuda_use_flashinfer_sm70_attention(device, dst)
            ? BEST_FATTN_KERNEL_FLASHINFER_SM70
            : BEST_FATTN_KERNEL_NONE;
    }

    if (ggml_cuda_get_sm70_attention_mode() == ggml_cuda_sm70_attention_mode::AUTO
            && ggml_cuda_use_cutlass_sm70_attention(device, dst)) {
        return BEST_FATTN_KERNEL_CUTLASS_SM70;
    }

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (max_bias == 0.0f && gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if (Q->ne[0] > 256 && gqa_opt_applies && Q->ne[1] * gqa_ratio_eff > 128) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

const char * ggml_cuda_flash_attn_ext_get_route(int device, const ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[device].cc;
    switch (ggml_cuda_get_best_fattn_kernel(device, dst)) {
        case BEST_FATTN_KERNEL_NONE:
            return "UNSUPPORTED";
        case BEST_FATTN_KERNEL_TILE:
            return "FATTN_TILE";
        case BEST_FATTN_KERNEL_VEC:
            return "FATTN_VEC";
        case BEST_FATTN_KERNEL_MMA_F16:
            return volta_mma_available(cc) ? "FATTN_MMA_VOLTA" : "FATTN_MMA_F16";
        case BEST_FATTN_KERNEL_CUTLASS_SM70:
            return "FATTN_CUTLASS_SM70";
        case BEST_FATTN_KERNEL_FLASHINFER_SM70:
            return "FATTN_FLASHINFER_SM70";
        case BEST_FATTN_KERNEL_QK_RMSNORM_ROPE_SM70:
            return "QK_RMSNORM_ROPE_SM70";
    }
    return "UNKNOWN";
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_CUTLASS_SM70:
        case BEST_FATTN_KERNEL_FLASHINFER_SM70:
        case BEST_FATTN_KERNEL_QK_RMSNORM_ROPE_SM70:
            break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_CUTLASS_SM70:
#ifdef GGML_CUDA_CUTLASS_SM70_ATTN
            ggml_cuda_flash_attn_ext_cutlass_sm70(ctx, dst);
            break;
#else
            GGML_ABORT("SM70 CUTLASS attention backend was not built");
#endif
        case BEST_FATTN_KERNEL_FLASHINFER_SM70:
#ifdef GGML_CUDA_FLASHINFER_SM70_ATTN
            ggml_cuda_flash_attn_ext_flashinfer_sm70(ctx, dst);
            break;
#else
            GGML_ABORT("SM70 FlashInfer attention backend was not built");
#endif
        case BEST_FATTN_KERNEL_QK_RMSNORM_ROPE_SM70:
#ifdef GGML_CUDA_CUTLASS_SM70_ATTN
            ggml_cuda_flash_attn_ext_qk_rmsnorm_rope_sm70(ctx, dst);
            break;
#else
            GGML_ABORT("SM70 Q/K RMSNorm+RoPE backend was not built");
#endif
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
