# ggml

## NVIDIA V100 / SM70 优化版

本仓库是面向 NVIDIA Tesla V100（Volta，Compute Capability 7.0）的 ggml 优化分支，新增专用 CUDA 路由、Tensor Core 内核和算子融合。

**算子级最高加速 5.86 倍；NVFP4 矩阵乘加速 2.64 倍；D128 Attention 加速 1.46 倍；分块 NVFP4 路径的临时权重显存减少约 43%。**

| 优化路径 | 实测变化 | 默认策略 |
|---|---:|---|
| NVFP4 矩阵乘：TurboMind | 快 **2.64 倍**，耗时下降 **62.2%** | **默认自动启用** |
| NVFP4 分块解码 | 速度基本不变，临时权重显存减少约 **43%** | 显存紧张时可手动选择 |
| D128 Attention（H56，S512～S1537） | 快 **1.46 倍**，耗时下降 **31.5%～31.7%** | 编译启用 CUTLASS 后，SM70 在支持形状内自动选择 |
| Q/K RMSNorm＋RoPE（H8/H56，S512/S1537） | 快 **3.02～5.86 倍**，耗时下降 **66.9%～82.9%** | 默认启用 |
| SiLU Gate＋F16 准备 | 快 **4.56～5.41 倍**，耗时下降 **78.1%～81.5%** | 默认启用 |
| 两路 LoRA（M512/M1537） | 快 **1.21～1.26 倍**，耗时下降 **17.7%～20.4%** | 默认启用 |
| CUDA Graph | 小图快 **1.08 倍**、耗时下降 **7.1%**；大图耗时增加 **0.07%** | 默认关闭，小图按需开启 |
| Direct HMMA NVFP4 | 慢 **2.18 倍**，耗时增加 **117.7%** | 不进入自动路由 |

> 测试平台为 Tesla V100-SXM2-16GB、SM70、CUDA 12.4。以上是固定输入下的单算子 A/B 测试结果，不代表完整模型会获得相同比例的端到端加速。
>
> TurboMind NVFP4 后端在本分支中默认编译并自动路由，无需设置环境变量。

### 编译选项与路由条件

TurboMind NVFP4 的默认启用，不等于两个 SM70 Attention 后端也默认编译。当前默认值与已使用的组合分别为：

| CMake 选项 | 源码默认值 | Krea2／Qwen 图片构建 | H3 INT8 构建 |
| --- | --- | --- | --- |
| `GGML_CUDA_TURBOMIND_SM70` | `ON` | `ON` | `OFF` |
| `GGML_CUDA_CUTLASS_SM70_ATTN` | `OFF` | `ON` | `ON` |
| `GGML_CUDA_FLASHINFER_SM70_ATTN` | `OFF` | `ON` | `ON` |
| `GGML_CUDA_GRAPHS` | `OFF`，可由上层默认值覆盖 | `OFF` | `OFF` |

需要复现图片构建的后端集合时，在下方 CUDA 构建命令中显式增加：

```sh
-DGGML_CUDA_CUTLASS_SM70_ATTN=ON \\
-DGGML_CUDA_FLASHINFER_SM70_ATTN=ON \\
-DGGML_CUDA_TURBOMIND_SM70=ON \\
-DGGML_CUDA_GRAPHS=OFF
```

H3 既有构建显式使用 `GGML_CUDA_TURBOMIND_SM70=OFF`；其 INT8 ConvRot 成功记录不等于 NVFP4 整模验收。`GGML_CUDA_SM70_ATTN=auto` 只会选择本次编译且满足形状／dtype 条件的路由，不会自动启用未编译的后端。

默认 CUTLASS 来源是仓库内固定子模块；先递归初始化，不复制其他项目的绝对 `.deps` 路径。切换组合使用新的构建目录，保留原可用二进制。上述性能来自既有算子对照，不表示迁移后的新二进制已经重新完成 V100 整模验证。

### 上游合并兼容性

- 上游的 `ggml_clamp` 现在返回新的张量；需要原地修改时改用 `ggml_clamp_inplace`。
- 本分支的 RPC 协议版本为 9，客户端和服务端均须使用兼容本分支的版本。
- Vulkan INT8 ConvRot 暂不支持 UMA 设备，需由其他后端执行。

---

[Manifesto](https://github.com/ggerganov/llama.cpp/discussions/205)

<div align="center">

<img src="https://raw.githubusercontent.com/ggml-org/media/refs/heads/master/logo/ggml-logo.jpg" width="256" height="256" alt="ggml logo" />

<b>Tensor library for machine learning</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/ggml?filter=v*)](https://github.com/ggml-org/ggml/releases)
[![CI](https://github.com/ggml-org/ggml/actions/workflows/build-cpu.yml/badge.svg)](https://github.com/ggml-org/ggml/actions/workflows/build-cpu.yml)

</div>

## Quick start

Build from source:

```bash
git clone --recurse-submodules https://github.com/tlsneo/ggml_sm_70.git
cd ggml_sm_70

# install python dependencies in a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# build the SM70 CUDA examples
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build . --config Release -j 8
```

For a minimal, fully commented example (matrix multiplication), see [examples/simple](examples/simple).

## Description

The main goal of `ggml` is to be a simple, portable, and efficient tensor library for machine learning with minimal setup.

- Plain C/C++ core; SM70 CUDA backends use the pinned CUTLASS submodule
- Cross-platform - x86, ARM, RISC-V, LoongArch, PowerPC, s390x, and WebAssembly
- SIMD-optimized kernels for x86, ARM, and RISC-V
- Broad backend support - CPU, GPU, NPU, and browser, with optimized NVIDIA Volta SM70 kernels
- Automatic differentiation and ADAM and L-BFGS optimizers
- 2- to 8-bit integer quantization, plus MXFP4 and NVFP4 microscaling formats
- Zero memory allocations during runtime

## Documentation

- [The GGUF file format](docs/gguf.md)
- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

## Contributing

- For changes to the core `ggml` library (including to the CMake build system), please open a PR in [llama.cpp](https://github.com/ggml-org/llama.cpp) - doing so will make your PR more visible, better tested, and more likely to be reviewed
