#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-ext.h"
#include "ggml-graph.h"

#include <cstdio>
#include <cstring>

// These application-owned names must coexist with the fork's native C symbols.
enum ggml_sage_attn_mode { GGML_SAGE_ATTN_AUTO, GGML_SAGE_ATTN_2, GGML_SAGE_ATTN_2_PLUS_PLUS };
ggml_tensor * ggml_quantize_i8_convrot(ggml_context *, ggml_tensor * input, int) { return input; }
ggml_tensor * ggml_mul_mat_i8_tensorwise(ggml_context *, ggml_tensor *, ggml_tensor * input,
                                         ggml_tensor *, ggml_tensor *, int, ggml_tensor * = nullptr) { return input; }
ggml_tensor * ggml_sage_attn(ggml_context *, ggml_tensor * q, ggml_tensor *, ggml_tensor *, float, ggml_sage_attn_mode) { return q; }
ggml_tensor * ggml_sol_attn(ggml_context *, ggml_tensor * q, ggml_tensor *, ggml_tensor *, float, float) { return q; }

static bool infer(ggml_tensor * const * inputs, int count, const void *, ggml_ext_tensor_desc * output) {
    if (count != 1) {
        return false;
    }
    output->type = inputs[0]->type;
    std::memcpy(output->ne, inputs[0]->ne, sizeof(output->ne));
    return true;
}

static int supports_calls;
static bool supports(ggml_backend_dev_t, const ggml_tensor *, void *) {
    ++supports_calls;
    return true;
}

static ggml_status execute(const ggml_ext_execution_context *, ggml_tensor * const * nodes, int count, void *, void *) {
    GGML_ASSERT(count == 1);
    const auto * input = static_cast<const float *>(nodes[0]->src[0]->data);
    auto * output = static_cast<float *>(nodes[0]->data);
    for (int64_t i = 0; i < ggml_nelements(nodes[0]); ++i) {
        output[i] = input[i] * 2.0f;
    }
    return GGML_STATUS_SUCCESS;
}

int main() {
    ggml_ext_op_desc op_desc = { GGML_EXT_ABI_VERSION - 1, "test.double", 0, infer };
    GGML_ASSERT(ggml_ext_register_op(&op_desc) == nullptr);
    op_desc.abi_version = GGML_EXT_ABI_VERSION;
    const auto op = ggml_ext_register_op(&op_desc);
    GGML_ASSERT(op != nullptr);

    const ggml_init_params params = { 1024*1024, nullptr, true };
    auto * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * result = ggml_ext_op_create(ctx, op, &input, 1, nullptr);
    GGML_ASSERT(result && ggml_ext_op_get(result) == op);
    GGML_ASSERT(ggml_quantize_i8_convrot(ctx, input, 1) == input);
    GGML_ASSERT(ggml_mul_mat_i8_tensorwise(ctx, input, input, nullptr, nullptr, 0) == input);
    GGML_ASSERT(ggml_sage_attn(ctx, input, input, input, 1, GGML_SAGE_ATTN_AUTO) == input);
    GGML_ASSERT(ggml_sol_attn(ctx, input, input, input, 1, 0) == input);

    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, result);
    GGML_ASSERT(ggml_graph_n_leafs(graph) == 1);
    GGML_ASSERT(ggml_graph_leaf(graph, 0) == input && ggml_graph_leaf(graph, -1) == input);
    GGML_ASSERT(ggml_graph_use_count(graph, input) == 1);

    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        auto reg = ggml_backend_reg_get(i);
        const char * name = ggml_backend_reg_name(reg);
        const bool cuda = std::strcmp(name, "CUDA") == 0;
        const bool other_gpu = std::strcmp(name, "ROCm") == 0 || std::strcmp(name, "MUSA") == 0;
        if (cuda || other_gpu) {
            GGML_ASSERT(ggml_backend_reg_get_proc_address(reg, "ggml_ext_gpu_context_v1"));
            GGML_ASSERT(ggml_backend_reg_get_proc_address(reg, "ggml_ext_gpu_device_v1"));
            GGML_ASSERT((ggml_backend_reg_get_proc_address(reg, "ggml_ext_cuda_context_v1") != nullptr) == cuda);
            GGML_ASSERT((ggml_backend_reg_get_proc_address(reg, "ggml_ext_cuda_device_v1") != nullptr) == cuda);
        }
    }

    auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu != nullptr) {
        GGML_ASSERT(!ggml_backend_dev_supports_op(cpu, result));
        ggml_ext_kernel_desc kernel = {};
        kernel.abi_version = GGML_EXT_ABI_VERSION - 1;
        kernel.name = "test.double.cpu";
        kernel.op = GGML_OP_CUSTOM;
        kernel.custom_op = op;
        kernel.supports = supports;
        kernel.execute = execute;
        GGML_ASSERT(!ggml_ext_register_kernel(cpu, &kernel));
        kernel.abi_version = GGML_EXT_ABI_VERSION;
        GGML_ASSERT(ggml_ext_register_kernel(cpu, &kernel));
        GGML_ASSERT(ggml_backend_dev_supports_op(cpu, result));

        auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 2, 3, 1);
        auto * attention = ggml_flash_attn_ext(ctx, q, q, q, nullptr, 1, 0, 0);
        kernel.name = "test.attention.cpu";
        kernel.op = GGML_OP_FLASH_ATTN_EXT;
        kernel.custom_op = nullptr;
        GGML_ASSERT(ggml_ext_register_kernel(cpu, &kernel));
        supports_calls = 0;
        GGML_ASSERT(ggml_backend_dev_supports_op(cpu, attention) && supports_calls == 1);
        attention->src[5] = q;
        GGML_ASSERT(!ggml_backend_dev_supports_op(cpu, attention) && supports_calls == 1);

        auto * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 128, 4);
        auto * matmul = ggml_mul_mat(ctx, weight, q);
        kernel.name = "test.matmul.cpu";
        kernel.op = GGML_OP_MUL_MAT;
        GGML_ASSERT(ggml_ext_register_kernel(cpu, &kernel));
        supports_calls = 0;
        GGML_ASSERT(ggml_backend_dev_supports_op(cpu, matmul) && supports_calls == 1);
        matmul->src[2] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        supports_calls = 0;
        GGML_ASSERT(ggml_backend_dev_supports_op(cpu, matmul) && supports_calls == 0);
        auto * float_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 4);
        auto * float_matmul = ggml_mul_mat(ctx, float_weight, q);
        GGML_ASSERT(ggml_backend_dev_supports_op(cpu, float_matmul) && supports_calls == 1);

        auto backend = ggml_backend_dev_init(cpu, nullptr);
        GGML_ASSERT(backend != nullptr);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer != nullptr);
        const float values[4] = { 1, 2, 3, 4 };
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        float output[4];
        ggml_backend_tensor_get(result, output, 0, sizeof(output));
        for (int i = 0; i < 4; ++i) {
            GGML_ASSERT(output[i] == 2 * values[i]);
        }
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
    } else {
        std::fprintf(stderr, "CPU extension checks skipped: no CPU backend registered\n");
    }
    ggml_free(ctx);
    return 0;
}
