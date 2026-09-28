// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "ggml-flashinfer-sm70-attention.cuh"

#include "volta_mma.cuh"

#include <climits>

namespace fi = flashinfer::attention::sm70;
namespace {
constexpr int D = 128;
constexpr int BQ = 192;
constexpr int BK = 64;
// One warp owns Q16 and both logical K32 halves, retaining FP32 arithmetic.
constexpr int KEY_WARPS = 1;
constexpr int KEY_FRAGMENTS = BK / (KEY_WARPS * 16);
constexpr int OUTPUT_FRAGMENTS = D / (KEY_WARPS * 16);
constexpr int VLD = BK + 8;
constexpr int THREADS = (BQ / 16) * KEY_WARPS * 32;
constexpr int PREFETCH_VECTORS = (BK * D + THREADS * 8 - 1) / (THREADS * 8);
static_assert(THREADS * PREFETCH_VECTORS * 8 >= BK * D);
static_assert(KEY_WARPS == 1 && BK == 64);
constexpr int shared_bytes() {
  constexpr int QLD = D + 8;
  return (BQ * D + BK * QLD + D * VLD) * 2;
}

__device__ __forceinline__ int q_swizzle(int row) {
  return ((row & 3) << 3) | ((row & 8) << 2);
}
__device__ __forceinline__ void load_q_fragment(fi::AFragment& fragment,
                                                const half* source, int row,
                                                int col) {
  const int lane = threadIdx.x & 31;
  const int physical_row =
      row + (lane & 3) + ((lane & 16) >> 2) + ((lane & 4) << 1);
  auto* values = reinterpret_cast<uint4*>(fragment.x);
  const half* base = source + physical_row * D;
  const int mask = q_swizzle(physical_row);
  values[0] = *reinterpret_cast<const uint4*>(base + (col ^ mask));
  values[1] = *reinterpret_cast<const uint4*>(base + ((col + 8) ^ mask));
}

// Convert FP16-rounded probability accumulator pairs to the existing Volta
// A fragment layout. Exchange row ownership across lane bit 1, then concatenate
// the two eight-column halves across lane bit 3. No arithmetic in the exchange.
__device__ __forceinline__ void load_probability_fragment(
    fi::AFragment& fragment, const unsigned* pairs) {
  const int lane = threadIdx.x & 31;
  const int row_bit = (lane >> 1) & 1;
  const unsigned own0 = row_bit ? pairs[1] : pairs[0];
  const unsigned own1 = row_bit ? pairs[3] : pairs[2];
  const unsigned other0 =
      __shfl_xor_sync(0xffffffff, row_bit ? pairs[0] : pairs[1], 2);
  const unsigned other1 =
      __shfl_xor_sync(0xffffffff, row_bit ? pairs[2] : pairs[3], 2);
  const unsigned local[4] = {row_bit ? other0 : own0, row_bit ? own0 : other0,
                             row_bit ? other1 : own1, row_bit ? own1 : other1};
  auto* output = reinterpret_cast<unsigned*>(fragment.x);
#pragma unroll
  for (int part = 0; part < 4; ++part) {
    const unsigned opposite = __shfl_xor_sync(0xffffffff, local[part], 8);
    output[part] = (lane & 8) ? opposite : local[part];
    output[part + 4] = (lane & 8) ? local[part] : opposite;
  }
}

__global__ __launch_bounds__(THREADS, 1) void h3_noncausal(
    ggml_cuda_flashinfer_sm70_attention_params params) {
  const half* q = params.q;
  const half* k = params.k;
  const half* v = params.v;
  half* output = params.output;
  const int length = params.length;
  const int heads = params.heads;
  const float scale = params.scale;
  constexpr int QLD = D + 8;
  extern __shared__ __align__(32) unsigned char raw[];
  half* qs = reinterpret_cast<half*>(raw);
  half* ks = qs + BQ * D;
  half* vs = ks + BK * QLD;
  float running_max[2] = {-INFINITY, -INFINITY};
  float running_sum[2] = {0.f, 0.f};
  const int tid = threadIdx.x, warp = tid / 32;
  const int warp_q = warp / KEY_WARPS, warp_k = warp % KEY_WARPS;
  const int lane = tid % 32;
  // Volta m16n16 accumulator element coordinates, matching the repository's
  // SM70 WMMA masking convention. SM70 is checked before dispatch.
  const int fragment_row =
      (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  const int fragment_col = ((lane >> 1) & 1) * 2 + ((lane >> 3) & 1) * 8;
  fi::AccumulatorFragment accumulators[OUTPUT_FRAGMENTS];
  float register_alpha[2];
#pragma unroll
  for (int part = 0; part < OUTPUT_FRAGMENTS; ++part)
    fi::init_accumulator_fragment(accumulators[part]);
  const int q_start = blockIdx.x * BQ;
  const int head = blockIdx.y % heads;
  const int batch = blockIdx.y / heads;
  const int64_t q_base = int64_t(batch) * params.q_stride_batch +
                         int64_t(head) * params.q_stride_head;
  const int64_t k_base = int64_t(batch) * params.k_stride_batch +
                         int64_t(head) * params.k_stride_head;
  const int64_t v_base = int64_t(batch) * params.v_stride_batch +
                         int64_t(head) * params.v_stride_head;
  const int64_t o_base = int64_t(batch) * params.o_stride_batch +
                         int64_t(head) * params.o_stride_head;
  for (int i = tid; i < BQ * D; i += blockDim.x) {
    const int row = q_start + i / D;
    qs[(i / D) * D + ((i % D) ^ q_swizzle(i / D))] =
        row < length ? q[q_base + int64_t(row) * params.q_stride_row + i % D]
                     : __float2half(0.f);
  }
  __syncthreads();
  for (int start = 0; start < length; start += BK) {
    if (start == 0) {
      for (int i = tid; i < BK * D; i += blockDim.x) {
        const int row = start + i / D;
        const int64_t k_position =
            k_base + int64_t(row) * params.k_stride_row + i % D;
        const int64_t v_position =
            v_base + int64_t(row) * params.v_stride_row + i % D;
        ks[(i / D) * QLD + i % D] =
            row < length ? k[k_position] : __float2half(0.f);
        vs[(i % D) * VLD + i / D] =
            row < length ? v[v_position] : __float2half(0.f);
      }
    }
    // Join both the initial loads and the previous iteration's prefetch.
    __syncthreads();
    fi::AccumulatorFragment qk[KEY_FRAGMENTS];
#pragma unroll
    for (int n = 0; n < KEY_FRAGMENTS; ++n)
      fi::init_accumulator_fragment(qk[n]);
#pragma unroll 2
    for (int dim = 0; dim < D; dim += 16) {
      fi::AFragment qa;
      load_q_fragment(qa, qs, warp_q * 16, dim);
#pragma unroll
      for (int n = 0; n < KEY_FRAGMENTS; ++n) {
        fi::QKBFragment kb;
        fi::load_qk_b_fragment(
            kb, ks + (warp_k * (BK / KEY_WARPS) + n * 16) * QLD + dim, QLD);
        fi::mma_sync_m16n16k16_row_col_f16f16f32(qk[n], qa, kb);
      }
    }
    unsigned probability_pairs[KEY_FRAGMENTS][4];
    {
      // Preserve the original two logical K32 partials and their FP32 sum
      // order even though one warp now owns both halves of this K64 tile.
      float row_max[2][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}};
#pragma unroll
      for (int n = 0; n < KEY_FRAGMENTS; ++n) {
#pragma unroll
        for (int i = 0; i < qk[n].num_elements; ++i) {
          const int col = n * 16 + fragment_col + (i & 1) + ((i >> 2) & 1) * 4;
          const int row = (i >> 1) & 1;
          qk[n].x[i] = start + col < length ? qk[n].x[i] * scale : -INFINITY;
          row_max[n / 2][row] = fmaxf(row_max[n / 2][row], qk[n].x[i]);
        }
      }
#pragma unroll
      for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int row = 0; row < 2; ++row) {
          auto& maximum = row_max[half][row];
          maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2));
          maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 8));
        }
      }
      float new_max[2], row_sum[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
      for (int row = 0; row < 2; ++row) {
        new_max[row] =
            fmaxf(fmaxf(running_max[row], row_max[0][row]), row_max[1][row]);
        register_alpha[row] = __expf(running_max[row] - new_max[row]);
      }
#pragma unroll
      for (int n = 0; n < KEY_FRAGMENTS; ++n) {
#pragma unroll
        for (int i = 0; i < qk[n].num_elements; ++i) {
          const int row = (i >> 1) & 1;
          const float p = __expf(qk[n].x[i] - new_max[row]);
          qk[n].x[i] = p;
          row_sum[n / 2][row] += p;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          union PackedPair {
            half2 value;
            unsigned bits;
          } pair;
          pair.value = __floats2half2_rn(qk[n].x[2 * i], qk[n].x[2 * i + 1]);
          probability_pairs[n][i] = pair.bits;
        }
      }
#pragma unroll
      for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int row = 0; row < 2; ++row) {
          auto& sum = row_sum[half][row];
          sum += __shfl_xor_sync(0xffffffff, sum, 2);
          sum += __shfl_xor_sync(0xffffffff, sum, 8);
        }
      }
#pragma unroll
      for (int row = 0; row < 2; ++row) {
        float sum = 0.f;
        sum += row_sum[0][row];
        sum += row_sum[1][row];
        running_sum[row] = running_sum[row] * register_alpha[row] + sum;
        running_max[row] = new_max[row];
      }
    }
    // Issue the next K/V global loads while the current V tile is consumed.
    union StagedVector {
      uint4 packed;
      half values[8];
    } next_k[PREFETCH_VECTORS], next_v[PREFETCH_VECTORS];
#pragma unroll
    for (int n = 0; n < PREFETCH_VECTORS; ++n) {
      // Each warp stages an 8-row by 32-column tile. Adjacent K/V rows
      // occupy lanes differing in bit 2, enabling the V transpose below.
      const int tile = (tid + n * THREADS) / 32;
      const int tile_row = (tile / (D / 32)) * 8 + (lane >> 2);
      const int next_row = start + BK + tile_row;
      const int next_col = (tile % (D / 32)) * 32 + (lane & 3) * 8;
      if (tile_row < BK && next_row < length) {
        const int64_t k_position =
            k_base + int64_t(next_row) * params.k_stride_row + next_col;
        const int64_t v_position =
            v_base + int64_t(next_row) * params.v_stride_row + next_col;
        if ((reinterpret_cast<uintptr_t>(k + k_position) % 16 == 0) &&
            (reinterpret_cast<uintptr_t>(v + v_position) % 16 == 0)) {
          asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(next_k[n].packed.x), "=r"(next_k[n].packed.y),
                         "=r"(next_k[n].packed.z), "=r"(next_k[n].packed.w)
                       : "l"(k + k_position));
          asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(next_v[n].packed.x), "=r"(next_v[n].packed.y),
                         "=r"(next_v[n].packed.z), "=r"(next_v[n].packed.w)
                       : "l"(v + v_position));
        } else {
          // Contiguous storage-offset views need scalar global loads.
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            next_k[n].values[j] = k[k_position + j];
            next_v[n].values[j] = v[v_position + j];
          }
        }
      } else {
        next_k[n].packed = make_uint4(0, 0, 0, 0);
        next_v[n].packed = make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int part = 0; part < OUTPUT_FRAGMENTS; ++part) {
#pragma unroll
      for (int i = 0; i < accumulators[part].num_elements; ++i)
        accumulators[part].x[i] *= register_alpha[(i >> 1) & 1];
    }
#pragma unroll
    for (int kv = 0; kv < KEY_FRAGMENTS; ++kv) {
      fi::AFragment pa;
      load_probability_fragment(pa, probability_pairs[kv]);
#pragma unroll
      for (int part = 0; part < OUTPUT_FRAGMENTS; ++part) {
        fi::QKBFragment vb;
        fi::load_qk_b_fragment(vb, vs + part * 16 * VLD + kv * 16, VLD);
        fi::mma_sync_m16n16k16_row_col_f16f16f32(accumulators[part], pa, vb);
      }
    }
    __syncthreads();
    if (start + BK < length) {
#pragma unroll
      for (int n = 0; n < PREFETCH_VECTORS; ++n) {
        const int tile = (tid + n * THREADS) / 32;
        const int tile_row = (tile / (D / 32)) * 8 + (lane >> 2);
        const int next_col = (tile % (D / 32)) * 32 + (lane & 3) * 8;
        if (tile_row >= BK) continue;
        *reinterpret_cast<uint4*>(ks + tile_row * QLD + next_col) =
            next_k[n].packed;
        // Transpose four rows with exact 32-bit lane exchanges. Each lane
        // writes two aligned 64-bit vectors instead of four half pairs,
        // reducing shared store instructions and bank conflicts. No values
        // pass through arithmetic or a shared transpose scratch buffer.
        const auto* pairs =
            reinterpret_cast<const unsigned*>(&next_v[n].packed);
        unsigned transposed[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const unsigned local = pairs[j];
          const unsigned adjacent = __shfl_xor_sync(0xffffffff, local, 4);
          transposed[j] = (lane & 4)
                              ? ((adjacent >> 16) | (local & 0xffff0000u))
                              : ((local & 0xffffu) | (adjacent << 16));
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const unsigned other0 =
              __shfl_xor_sync(0xffffffff, transposed[2 * j], 8);
          const unsigned other1 =
              __shfl_xor_sync(0xffffffff, transposed[2 * j + 1], 8);
          const unsigned local =
              ((lane & 8) ? transposed[2 * j + 1] : transposed[2 * j]);
          const unsigned other = (lane & 8) ? other1 : other0;
          const uint2 vector =
              (lane & 8) ? make_uint2(other, local) : make_uint2(local, other);
          const int d = next_col + 4 * j + ((lane >> 2) & 3);
          *reinterpret_cast<uint2*>(vs + d * VLD + (tile_row & ~3)) = vector;
        }
      }
    }
  }
  {
#pragma unroll
    for (int part = 0; part < OUTPUT_FRAGMENTS; ++part) {
      const auto& pv = accumulators[part];
#pragma unroll
      for (int i = 0; i < pv.num_elements; ++i) {
        const int row = warp_q * 16 + fragment_row + ((i >> 1) & 1) * 2;
        const int col = warp_k * (D / KEY_WARPS) + part * 16 + fragment_col +
                        (i & 1) + ((i >> 2) & 1) * 4;
        if (q_start + row < length)
          output[o_base + int64_t(q_start + row) * params.o_stride_row + col] =
              __float2half_rn(pv.x[i] / running_sum[(i >> 1) & 1]);
      }
    }
  }
}
}  // namespace

void ggml_cuda_flashinfer_sm70_attention(
        const ggml_cuda_flashinfer_sm70_attention_params & params,
        cudaStream_t stream) {
  GGML_ASSERT(params.q != nullptr && params.k != nullptr && params.v != nullptr && params.output != nullptr);
  GGML_ASSERT(params.length > 0 && params.length <= INT_MAX);
  GGML_ASSERT(params.heads > 0 && params.batches > 0 && params.heads * params.batches <= 65535);
  CUDA_CHECK(cudaFuncSetAttribute(
      h3_noncausal, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes()));
  const dim3 grid((params.length + BQ - 1) / BQ, params.batches * params.heads);
  h3_noncausal<<<grid, THREADS, shared_bytes(), stream>>>(params);
  CUDA_CHECK(cudaGetLastError());
}
