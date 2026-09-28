#include "ggml-nvfp4-turbomind.cuh"

#include "ggml-cuda/common.cuh"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {

static turbomind::gemm::MatrixLayout make_layout(
        turbomind::DataType type, turbomind::gemm::Order order, int rows, int cols, int ld) {
    turbomind::gemm::MatrixLayout layout{};
    layout.type = type;
    layout.order = order;
    layout.rows = rows;
    layout.cols = cols;
    layout.ld = ld;
    return layout;
}

__global__ void ggml_cuda_nvfp4_turbomind_unpack(
        const block_nvfp4 * __restrict__ src,
        uint16_t * __restrict__ codes,
        half * __restrict__ scales,
        int64_t k,
        int64_t n,
        bool transpose_codes) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t weight_count = k * n;
    if (idx < weight_count) {
        const int64_t row = idx / k;
        const int64_t col = idx - row * k;
        const int64_t block = row * (k / QK_NVFP4) + col / QK_NVFP4;
        const int local = col % QK_NVFP4;
        const int sub = local / QK_NVFP4_SUB;
        const int elem = local % QK_NVFP4_SUB;
        const uint8_t packed = src[block].qs[sub * (QK_NVFP4_SUB / 2) + elem % (QK_NVFP4_SUB / 2)];
        const uint16_t code = elem < QK_NVFP4_SUB / 2 ? packed & 0x0f : packed >> 4;
        codes[transpose_codes ? row * k + col : col * n + row] = code;
    }

    const int64_t scale_count = (k / QK_NVFP4_SUB) * n;
    if (idx < scale_count) {
        const int64_t group = idx / n;
        const int64_t row = idx - group * n;
        const int64_t block = row * (k / QK_NVFP4) + group / (QK_NVFP4 / QK_NVFP4_SUB);
        const int sub = group % (QK_NVFP4 / QK_NVFP4_SUB);
        // GGML kvalues_mxfp4 are twice the TurboMind E2M1 primitive convention.
        scales[idx] = __float2half(2.0f * ggml_cuda_ue4m3_to_fp32(src[block].d[sub]));
    }
}

std::mutex g_gemm_mutex;
std::unordered_map<int, std::unique_ptr<turbomind::gemm::Gemm>> g_gemms;

static turbomind::gemm::Gemm & get_gemm(int device) {
    std::lock_guard<std::mutex> lock(g_gemm_mutex);
    auto & gemm = g_gemms[device];
    if (!gemm) {
        gemm = std::make_unique<turbomind::gemm::Gemm>();
    }
    return *gemm;
}

static bool allocate_prepared(ggml_cuda_nvfp4_turbomind_prepared * prepared, int64_t k, int64_t n) {
    prepared->weight_bytes = (size_t) k * n / 2;
    prepared->scale_bytes = (size_t) k * n / QK_NVFP4_SUB * sizeof(half);
    if (cudaMalloc(&prepared->weight, prepared->weight_bytes) != cudaSuccess) {
        prepared->weight = nullptr;
        return false;
    }
    if (cudaMalloc(&prepared->scales, prepared->scale_bytes) != cudaSuccess) {
        cudaFree(prepared->weight);
        prepared->weight = nullptr;
        prepared->scales = nullptr;
        return false;
    }
    if (cudaEventCreateWithFlags(&prepared->ready, cudaEventDisableTiming) != cudaSuccess) {
        cudaFree(prepared->scales);
        cudaFree(prepared->weight);
        prepared->weight = nullptr;
        prepared->scales = nullptr;
        prepared->ready = nullptr;
        return false;
    }
    return true;
}

} // namespace

size_t ggml_cuda_nvfp4_turbomind_barriers_size() {
    return turbomind::gemm::Gemm::kBarriersSize;
}

size_t ggml_cuda_nvfp4_turbomind_partials_size() {
    return turbomind::gemm::Gemm::kPartialsSize;
}

size_t ggml_cuda_nvfp4_turbomind_tensormaps_size() {
    return 8192u * 128u;
}

bool ggml_cuda_nvfp4_turbomind_prepare(
        const void * src,
        int64_t k,
        int64_t n,
        void * temp_codes,
        void * temp_scales,
        ggml_cuda_nvfp4_turbomind_prepared * prepared,
        cudaStream_t stream) {
    if (src == nullptr || temp_codes == nullptr || temp_scales == nullptr || prepared == nullptr ||
            prepared->weight != nullptr || prepared->scales != nullptr || k <= 0 || n <= 0 ||
            k % QK_NVFP4 != 0 || n % 8 != 0) {
        return false;
    }

    const auto fp4_converters = turbomind::gemm::GetConverters(
        turbomind::kHalf, turbomind::kFloat4_e2m1, turbomind::kHalf, true, 70);
    const auto fp8_converters = turbomind::gemm::GetConverters(
        turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
    const auto * conv_w = fp4_converters[0];
    const auto * conv_s = fp8_converters[1];
    if (conv_w == nullptr || conv_s == nullptr) {
        return false;
    }

    if (cudaGetDevice(&prepared->device) != cudaSuccess || !allocate_prepared(prepared, k, n)) {
        return false;
    }
    prepared->k = k;
    prepared->n = n;

    const auto order_w = conv_w->order;
    const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) == turbomind::gemm::OPERAND_A;
    const bool is_B_w = !is_A_w;
    const bool transpose_codes = order_w == turbomind::gemm::kRowMajor;

    const int64_t thread_count = std::max(k * n, (k / QK_NVFP4_SUB) * n);
    constexpr int threads = 256;
    ggml_cuda_nvfp4_turbomind_unpack<<<(thread_count + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const block_nvfp4 *>(src),
        static_cast<uint16_t *>(temp_codes),
        static_cast<half *>(temp_scales),
        k,
        n,
        transpose_codes);
    if (cudaGetLastError() != cudaSuccess) {
        ggml_cuda_nvfp4_turbomind_release(prepared);
        return false;
    }

    auto w_desc = make_layout(
        turbomind::kHalf, order_w, static_cast<int>(n), static_cast<int>(k),
        order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k) : static_cast<int>(n));
    if (is_B_w) {
        std::swap(w_desc.rows, w_desc.cols);
        w_desc.order = ~w_desc.order;
    }

    auto packed_w_desc = w_desc;
    packed_w_desc.type = turbomind::kFloat4_e2m1;
    packed_w_desc.pack = conv_w->pack;
    if (is_A_w) {
        packed_w_desc = turbomind::gemm::transpose(packed_w_desc);
    }
    if (conv_w->Convert(temp_codes, w_desc, prepared->weight, packed_w_desc, stream) != 0) {
        ggml_cuda_nvfp4_turbomind_release(prepared);
        return false;
    }

    const int64_t groups = k / QK_NVFP4_SUB;
    const auto order_s = conv_s->order;
    const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) == turbomind::gemm::OPERAND_U;
    const bool is_B_s = !is_A_s;
    auto s_desc = make_layout(
        turbomind::kUint16, order_s, static_cast<int>(n), static_cast<int>(groups), static_cast<int>(n));
    if (is_B_s) {
        std::swap(s_desc.rows, s_desc.cols);
        s_desc.order = ~s_desc.order;
    }

    auto packed_s_desc = s_desc;
    packed_s_desc.pack = conv_s->pack;
    if (is_A_s) {
        packed_s_desc = turbomind::gemm::transpose(packed_s_desc);
    }
    if (conv_s->Convert(temp_scales, s_desc, prepared->scales, packed_s_desc, stream) != 0) {
        ggml_cuda_nvfp4_turbomind_release(prepared);
        return false;
    }

    prepared->k_ld = packed_w_desc.ld;
    prepared->q_ld = packed_s_desc.ld;
    if (cudaEventRecord(prepared->ready, stream) != cudaSuccess) {
        ggml_cuda_nvfp4_turbomind_release(prepared);
        return false;
    }

    return true;
}

void ggml_cuda_nvfp4_turbomind_release(ggml_cuda_nvfp4_turbomind_prepared * prepared) {
    if (prepared == nullptr) {
        return;
    }
    int previous = -1;
    cudaGetDevice(&previous);
    if (prepared->device >= 0 && prepared->device != previous) {
        cudaSetDevice(prepared->device);
    }
    if (prepared->ready != nullptr) {
        cudaEventSynchronize(prepared->ready);
        cudaEventDestroy(prepared->ready);
    }
    if (prepared->scales != nullptr) {
        cudaFree(prepared->scales);
    }
    if (prepared->weight != nullptr) {
        cudaFree(prepared->weight);
    }
    if (previous >= 0 && prepared->device >= 0 && previous != prepared->device) {
        cudaSetDevice(previous);
    }
    *prepared = {};
}

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
        cudaStream_t stream) {
    if (prepared.weight == nullptr || prepared.scales == nullptr || prepared.ready == nullptr ||
            activation_fp16 == nullptr || dst == nullptr || m <= 0 || prepared.k <= 0 || prepared.n <= 0) {
        return false;
    }
    if (cudaStreamWaitEvent(stream, prepared.ready, 0) != cudaSuccess) {
        return false;
    }

    const auto fp4_converters = turbomind::gemm::GetConverters(
        turbomind::kHalf, turbomind::kFloat4_e2m1, turbomind::kHalf, true, 70);
    const auto fp8_converters = turbomind::gemm::GetConverters(
        turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
    const auto * conv_w = fp4_converters[0];
    const auto * conv_s = fp8_converters[1];
    if (conv_w == nullptr || conv_s == nullptr) {
        return false;
    }

    auto desc_A = make_layout(
        turbomind::kHalf, turbomind::gemm::kRowMajor, static_cast<int>(m),
        static_cast<int>(prepared.k), static_cast<int>(prepared.k));
    turbomind::gemm::MatrixLayout desc_U{};

    const auto order_w = conv_w->order;
    const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) == turbomind::gemm::OPERAND_A;
    const bool is_B_w = !is_A_w;
    auto weight_desc = make_layout(
        turbomind::kHalf, order_w, static_cast<int>(prepared.n), static_cast<int>(prepared.k),
        order_w == turbomind::gemm::kRowMajor ? static_cast<int>(prepared.k) : static_cast<int>(prepared.n));
    if (is_B_w) {
        std::swap(weight_desc.rows, weight_desc.cols);
        weight_desc.order = ~weight_desc.order;
    }
    weight_desc.type = turbomind::kFloat4_e2m1;
    weight_desc.pack = conv_w->pack;
    if (is_A_w) {
        weight_desc = turbomind::gemm::transpose(weight_desc);
    }
    weight_desc.ld = prepared.k_ld;

    const auto order_s = conv_s->order;
    const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) == turbomind::gemm::OPERAND_U;
    const bool is_B_s = !is_A_s;
    auto scale_desc = make_layout(
        turbomind::kUint16, order_s, static_cast<int>(prepared.n),
        static_cast<int>(prepared.k / QK_NVFP4_SUB), static_cast<int>(prepared.n));
    if (is_B_s) {
        std::swap(scale_desc.rows, scale_desc.cols);
        scale_desc.order = ~scale_desc.order;
    }
    scale_desc.pack = conv_s->pack;
    if (is_A_s) {
        scale_desc = turbomind::gemm::transpose(scale_desc);
    }
    scale_desc.ld = prepared.q_ld;

    auto desc_D = make_layout(
        turbomind::kFloat, turbomind::gemm::kRowMajor, static_cast<int>(m),
        static_cast<int>(prepared.n), static_cast<int>(prepared.n));

    turbomind::gemm::Operation op{};
    op.dispatch = turbomind::gemm::DispatchPolicy::kDefault;
    op.epilogue = turbomind::gemm::Epilogue::kNone;
    op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
    op.quant_b = {turbomind::gemm::QuantType::kK, QK_NVFP4_SUB};
    op.batch_dim = 0;

    turbomind::gemm::Workspace workspace{};
    workspace.barriers = barriers;
    workspace.barriers_size = barriers_size;
    workspace.partials = partials;
    workspace.partials_size = partials_size;
    workspace.tensormaps = tensormaps;
    workspace.tensormaps_size = tensormaps_size;
    workspace.flags = flags;

    auto & gemm = get_gemm(device);
    const int ec = gemm.Run(
        op,
        1.0f,
        activation_fp16,
        desc_A,
        nullptr,
        desc_U,
        prepared.weight,
        weight_desc,
        prepared.scales,
        scale_desc,
        0.0f,
        dst,
        desc_D,
        dst,
        desc_D,
        workspace,
        stream);
    return ec == 0 && cudaEventRecord(prepared.ready, stream) == cudaSuccess;
}
