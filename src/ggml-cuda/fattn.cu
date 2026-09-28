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

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

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
            if (gqa_ratio == 20) { // GLM 4.7 Flash
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
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
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

static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return true;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
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
    const float eps = ggml_get_op_params_f32(dst, 4);
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

    const float eps = ggml_get_op_params_f32(dst, 4);
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

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

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
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
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
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
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
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
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
        case BEST_FATTN_KERNEL_VEC:
            need_f16_K = K->type == GGML_TYPE_F32;
            need_f16_V = V->type == GGML_TYPE_F32;
            break;
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
