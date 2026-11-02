#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sm70.h"

#include <string.h>

int main(void) {
    const struct ggml_init_params params = {
        .mem_size   = 1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != NULL);

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 2, 3, 1);
    struct ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    struct ggml_tensor * rope = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 64, 3);
    const float kv_scale = 0.5f;
    const float eps = 1e-5f;
    struct ggml_tensor * attn = ggml_flash_attn_ext_qk_rms_norm_rope(
            ctx, q, q, q, w, w, rope, 1.0f, kv_scale, eps);
    ggml_flash_attn_ext_set_n_kv_max(attn, 3);

    float actual_kv_scale, actual_eps;
    memcpy(&actual_kv_scale, &attn->op_params[5], sizeof(actual_kv_scale));
    memcpy(&actual_eps, &attn->op_params[6], sizeof(actual_eps));
    GGML_ASSERT(attn->op_params[4] == 3);
    GGML_ASSERT(actual_kv_scale == kv_scale && actual_eps == eps);
    GGML_ASSERT(ggml_prec_get_acc(attn) == GGML_PREC_F32);

    struct ggml_tensor * v_f16 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, 2, 3, 1);
    GGML_ASSERT(ggml_sage_attn(ctx, q, q, v_f16, 1.0f, GGML_SAGE_ATTN_AUTO)->op == GGML_OP_SAGE_ATTN);
    GGML_ASSERT(ggml_sol_attn(ctx, q, q, q, 1.0f, 0.0f)->op == GGML_OP_SOL_ATTN);

    struct ggml_tensor * glu = ggml_swiglu_split_f16(ctx, q, q);
    struct ggml_tensor * oai = ggml_swiglu_oai(ctx, q, q, 1.702f, 7.0f);
    float alpha;
    memcpy(&alpha, &oai->op_params[2], sizeof(alpha));
    GGML_ASSERT(glu->type == GGML_TYPE_F16 && glu->op_params[2] == 0);
    GGML_ASSERT(oai->type == GGML_TYPE_F32 && alpha == 1.702f);

    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 128, 4);
    struct ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * packed = ggml_quantize_i8_convrot(ctx, q, 1);
    struct ggml_tensor * matmul = ggml_mul_mat_i8_tensorwise(ctx, weight, packed, scales, NULL, 0);
    GGML_ASSERT(matmul->src[1] == packed && matmul->src[4] == q);
    GGML_ASSERT(ggml_prec_set_src(matmul, GGML_PREC_F32, 1));
    GGML_ASSERT(ggml_prec_get_src(matmul, 1) == GGML_PREC_F32 && matmul->op_params[2] == 0);

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu != NULL) {
        matmul->src[1] = ggml_dup_tensor(ctx, packed);
        GGML_ASSERT(ggml_backend_dev_supports_op(cpu, matmul));
        matmul->src[1] = q;
        GGML_ASSERT(!ggml_backend_dev_supports_op(cpu, matmul));
        GGML_ASSERT(ggml_backend_dev_supports_op(cpu, ggml_swiglu_split(ctx, q, q)));

        struct ggml_tensor * fp8 = ggml_new_tensor_1d(ctx, GGML_TYPE_F8_E4M3, 8);
        struct ggml_tensor * fp8_dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F8_E4M3, 8);
        struct ggml_tensor * f32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        GGML_ASSERT(ggml_backend_dev_supports_op(cpu, ggml_cpy(ctx, fp8, fp8_dst)));
        GGML_ASSERT(!ggml_backend_dev_supports_op(cpu, ggml_cpy(ctx, f32, fp8_dst)));
        struct ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        GGML_ASSERT(ggml_backend_dev_supports_op(cpu, ggml_get_rows(ctx, fp8, indices)));
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg == NULL) {
            continue;
        }
        const char * name = ggml_backend_reg_name(reg);
        if (strcmp(name, "CUDA") == 0 || strncmp(name, "RPC", 3) == 0 || strcmp(name, "ggml-virtgpu") == 0) {
            continue;
        }
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, attn));
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, glu));
    }

    GGML_ASSERT(ggml_clamp(ctx, q, -1.0f, 1.0f)->view_src == NULL);
    GGML_ASSERT(ggml_clamp_inplace(ctx, q, -1.0f, 1.0f)->view_src == q);

    ggml_free(ctx);
    return 0;
}
