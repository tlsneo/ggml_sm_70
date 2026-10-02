# ggml

## NVIDIA V100 / SM70 优化版

本仓库是面向 NVIDIA Tesla V100（Volta，Compute Capability 7.0）的 ggml 优化分支，新增专用 CUDA 路由、Tensor Core 内核和算子融合。

**算子级最高加速 5.86 倍；NVFP4 矩阵乘加速 2.64 倍；D128 Attention 加速 1.46 倍；分块 NVFP4 路径的临时权重显存减少约 43%。**

| 优化路径 | 实测变化 | 默认策略 |
|---|---:|---|
| NVFP4 矩阵乘：TurboMind | 快 **2.64 倍**，耗时下降 **62.2%** | **默认自动启用** |
| NVFP4 分块解码 | 速度基本不变，临时权重显存减少约 **43%** | 显存紧张时可手动选择 |
| D128 Attention（H56，S512～S1537） | 快 **1.46 倍**，耗时下降 **31.5%～31.7%** | SM70 自动选择 CUTLASS |
| Q/K RMSNorm＋RoPE（H8/H56，S512/S1537） | 快 **3.02～5.86 倍**，耗时下降 **66.9%～82.9%** | 默认启用 |
| SiLU Gate＋F16 准备 | 快 **4.56～5.41 倍**，耗时下降 **78.1%～81.5%** | 默认启用 |
| 两路 LoRA（M512/M1537） | 快 **1.21～1.26 倍**，耗时下降 **17.7%～20.4%** | 默认启用 |
| CUDA Graph | 小图快 **1.08 倍**、耗时下降 **7.1%**；大图耗时增加 **0.07%** | 默认关闭，小图按需开启 |
| Direct HMMA NVFP4 | 慢 **2.18 倍**，耗时增加 **117.7%** | 不进入自动路由 |

> 测试平台为 Tesla V100-SXM2-16GB、SM70、CUDA 12.4。以上是固定输入下的单算子 A/B 测试结果，不代表完整模型会获得相同比例的端到端加速。
>
> TurboMind NVFP4 后端在本分支中默认编译并自动路由，无需设置环境变量。

---

[Manifesto](https://github.com/ggerganov/llama.cpp/discussions/205)

Tensor library for machine learning

***Note that this project is under active development. \
Some of the development is currently happening in the [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) repos***

## Features

- Low-level cross-platform implementation
- Integer quantization support
- NVIDIA Volta SM70 optimized CUDA kernels
- Automatic differentiation
- ADAM and L-BFGS optimizers
- Pinned CUTLASS v4.4.2 dependency
- Zero memory allocations during runtime

## Build

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

## GPT inference (example)

```bash
# run the GPT-2 small 117M model
../examples/gpt-2/download-ggml-model.sh 117M
./bin/gpt-2-backend -m models/gpt-2-117M/ggml-model.bin -p "This is an example"
```

For more information, checkout the corresponding programs in the [examples](examples) folder.

## Resources

- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [The GGUF file format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
