#ifndef GGML_SM70_H
#define GGML_SM70_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API struct ggml_tensor * ggml_mul_mat_i8_tensorwise(
        struct ggml_context * ctx,
        struct ggml_tensor  * weight,
        struct ggml_tensor  * input,
        struct ggml_tensor  * weight_scale,
        struct ggml_tensor  * bias,
        int                   convrot_group_size);

// Packs row-wise I8 activations followed by one F32 scale per row.
GGML_API struct ggml_tensor * ggml_quantize_i8_convrot(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   group_size);

enum ggml_sage_attn_mode {
    GGML_SAGE_ATTN_AUTO = 0,
    GGML_SAGE_ATTN_2,
    GGML_SAGE_ATTN_2_PLUS_PLUS,
};

// Q/K: contiguous F32 [D, tokens, heads, batch], V: contiguous F16 in the same layout.
// Output: F32 [D, query heads, query tokens, batch]. CUDA-only, unmasked inference.
GGML_API struct ggml_tensor * ggml_sage_attn(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        float                 scale,
        enum ggml_sage_attn_mode mode);

// Noncausal Sol-Attn: contiguous F32 Q/K/V [D=128, T, H, B], output [D, H, T, B].
GGML_API struct ggml_tensor * ggml_sol_attn(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        float                 scale,
        float                 tau);

#ifdef __cplusplus
}
#endif
#endif
