// Copyright (c) OpenMMLab. All rights reserved.
// GGML adaptation: register only generic dynamic-shape SM70 NVFP4 kernels.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm70_s884;
using S = cache_policy::Stream;
using D = cache_policy::Default;

template <Order raster_order, int group_axis = -1>
using Config_NVF4_F32 = Sm70_s884<Operand_A<half>,
                                  Transform_Default,
                                  VoidOperand,
                                  Operand_B_Pack<fp4_e2m1_t>,
                                  Transform_HMMA_SIMT_B,
                                  Operand_V_Pack<uint16_t>,
                                  kRowMajor,
                                  float,
                                  raster_order,
                                  group_axis>;

void Registry::sm70_884_4() {
  using C = Config_NVF4_F32<kColMajor, 0>;
  Add<C::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 16, 64, 128>>();
  Add<C::Type< 64, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16, 32, 128>>();
  Add<C::Type< 32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>();
  Add<C::Type< 16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>();
  Add<C::Type<  8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 16>>();
}

}  // namespace turbomind::gemm
