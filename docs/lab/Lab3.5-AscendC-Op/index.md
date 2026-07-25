# 实验 3.5：Ascend C 算子迁移与优化

!!! info "实验信息"

    负责助教：胡笠桁, 胡哲文, 刘天洋, 徐晨, 周冠一

    !!! tip "Bonus 实验"

        这是一个 **Bonus 实验**，不强制要求同学们完成。学有余力、对国产计算硬件感兴趣的同学欢迎挑战。

## 实验目的

在其他实验中，我们了解了 x86-64、RISC-V CPU 和 NVIDIA GPU 的硬件架构，并尝试写出更充分利用硬件特性的程序，以优化程序的运行效率。而在本次实验中，同学们可以接触到国产计算生态的代表硬件**昇腾（Ascend）NPU**：你需要在昇腾 910B4 NPU 上实现 `fused_add_rmsnorm` 算子，在保证功能正确的前提下，尽可能地提高性能。

完成本实验后，你应当能够：

- 了解昇腾 NPU 的基础架构；
- 理解算子迁移的本质：在不改变计算语义的前提下，把一个面向某种硬件写成的实现，重新落到另一种硬件的指令与存储抽象上；
- 掌握 NPU 上常见的算子优化手段，并理解这些手段背后对应着硬件的哪些特性；
- 使用 `msprof` 等工具对算子进行性能分析。

!!! note "前置课程"

    本实验的硬件与编程模型背景对应 7 月 14 日「华为 CANN」课程。如果你对昇腾 NPU 的架构、CANN 工具链或 Ascend C 编程模型不熟悉，建议先回顾该课程内容，或参考文末的参考资料。

## 背景知识

### 算子：FusedAddRmsNorm

`FusedAddRmsNorm`，顾名思义，是把「残差加法（Add）」和「RMSNorm」融合在一起的一个算子。

先来看 [RMSNorm](https://arxiv.org/abs/1910.07467)，这是一种对向量做均方根归一化的方法。对一个长度为 $H$ 的向量 $x$，它先用均方根做缩放，再乘以可学习权重 $w$：

$$
\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\dfrac{1}{H}\sum_{i=1}^{H}x_i^2 + \varepsilon}} \odot w,
$$

其中 $w$ 是与 $x$ 逐元素相乘的可学习缩放权重，$\varepsilon$ 是为数值稳定加入的小常数。RMSNorm 直接用均方根做缩放，计算较轻，在 Transformer 等模型中被广泛用作归一化层。

而在 Transformer 前向中，RMSNorm 之前往往紧跟一个残差加法（residual add），把上一路的输入累加到残差流上。这其实就是一个矢量加法。`fused_add_rmsnorm` 就是把这两步**融合成一个算子**，避免残差结果在全局显存中多写一次再读回。融合后的计算为：

$$
\begin{aligned}
R &= x + \mathit{residual}, \\
\mathit{rms} &= \sqrt{\dfrac{1}{H}\sum_{i=1}^{H}R_i^2 + \varepsilon}, \\
y &= \dfrac{R}{\mathit{rms}} \odot w.
\end{aligned}
$$

即该算子同时输出两部分：

- `residual_out` $= R = x + \mathit{residual}$，供后续残差流继续使用；
- `y` $= \operatorname{RMSNorm}(R)$，作为本层的归一化输出。

### 昇腾 NPU 与昇腾架构上的算子开发路径简介

昇腾 910B4 的核心计算单元 **AI Core** 内部有 **AI Cube（AIC）** 与 **AI Vector（AIV）** 两套独立的核：AIC 承担矩阵运算（Cube 单元，类似于 Tensor Core），AIV 承担向量与元素级运算（Vector 单元）；二者之间/AI Core 组之间通过全局显存（Global Memory）通信协作。每个核内还有多级片上存储——AIV 侧的 **UB（Unified Buffer，向量计算的快速访存区）**、AIC 侧的 L1、L0A/L0B/L0C 等。多个 AI Core 之间相互独立，通过 tiling 把数据切分到各核上并行执行。无论你选择哪种实现路径，理解这套硬件结构都是优化的前提。

在 910B4 上，AIC 与 AIV 并非一一对应，而是按 **1 : 2** 的固定比例配置——每 1 个 AIC 配 2 个 AIV，组成一个物理核（对应 Ascend C 中的 `KERNEL_TYPE_MIX_AIC_1_2` kernel 类型）。根据本机 CANN 9.1.0-beta.3 配置，一颗 910B4 共有 **20 个 AI Core group**，即 **20 个 AIC + 40 个 AIV**，合计 60 个计算子核。

<figure markdown="span">
  <div markdown="span" style="background: #ffffff; padding: 8px;">
    ![AI Core 分离架构](image/ai-core-separated.png)
  </div>
  <figcaption><a href="https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0008.html">AI Core 分离架构示意</a></figcaption>
</figure>

我们的实验框架主要支持三种开发路径，它们对硬件的暴露程度依次降低、抽象层级依次升高：

- **Ascend C** 是昇腾提供的原生算子开发语言，本质是一套 C++ 模板库与 intrinsic。它屏蔽了部分底层细节，让你能用较高层的 API（如 `DataCopy`、`Cast`、`Add`、`Mul`、`BlockReduceSum` 等）编写 kernel，同时保留了足够细粒度的控制力去压榨性能。一个完整的 Ascend C 自定义算子通常由三部分组成：

    - **算子原型（Op 定义）**：声明算子的输入、输出与属性，注册推理 shape 与数据类型的推导函数，以及 tiling 函数。其中 OpDef 类与注册宏等骨架可由 CANN 的 `msopgen` 工具从算子信息 JSON 自动生成，但 tiling 函数中的切分逻辑仍需开发者手写；
    - **Host 侧 tiling**：在 Host 上根据输入 shape、数据类型和平台信息，计算切分参数，决定每个核处理多少数据、UB 如何分配，并把 tiling 数据传给 kernel（类比于 CUDA 中设置 grid/block 维度并准备 kernel 参数的 host 侧代码，但额外承担了数据分区与 UB 规划）；
    - **Kernel 实现**：在 AI Core 上执行真正的计算，负责 GM↔UB 搬运、UB 内计算、结果写回（类比于 CUDA 中的 kernel 函数）。

- **TileLang** 在 Lab 3 中已登场，是一种面向高性能计算的 DSL。它把 tiling、shared memory、pipeline 等概念直接暴露为语言原语，同样可编译到昇腾 NPU 后端（见 [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend)）。相较 Ascend C，它省去了手写 host tiling 与 GM↔UB 搬运的样板代码，但在某些细节上对硬件的控制力弱于原生路径。

- **Triton** 是基于 block 的算子编程模型，通过 [triton-ascend](https://github.com/triton-ascend/triton-ascend) 后端可编译到昇腾 NPU。它以 `tl.programs` 切分任务、以 block 为单位组织计算，写法接近你在 GPU 上的经验，便于把已有的 GPU kernel 快速迁移过来。

三种路径最终编译出的都是跑在 Ascend NPU 上的机器码，性能上限由硬件决定；路径之间的差异在于「你能在多大程度上接近硬件」。事实上，为迎合主流计算生态、降低已有代码的迁移门槛，昇腾还提供了不少别的上层编程范式（如对齐 PyTorch 生态的 `torch_npu`、以及基于图构建的 MindSpore 等），它们同样能编译到 910B4 的机器码。理论上同学们也可以在本实验中尝试使用这些范式，但它们往往离底层硬件更远、可控性有限；本实验的框架与评测仅以上述三条路径为准——若你需要尝试其它范式，请自行解决环境与调试问题。下文的具体指令名、UB 规划等内容均以 Ascend C 路径为例，选择 TileLang / Triton 的同学请将其映射到所用 DSL 的对应抽象。

## 实验任务

### 计算语义

设输入张量形状记号如下，其中 $B$ 为行数（可理解为 batch 或 token 数），$H$ 为隐藏层宽度：

| 张量 | 形状 | 数据类型 | 含义 |
| --- | --- | --- | --- |
| `x` | $[B, H]$ | FP16 | 输入 |
| `residual` | $[B, H]$ | FP16 | 残差 |
| `weight` | $[H,]$ | FP16 | RMSNorm 的可学习缩放权重 |
| 输出 `y` | $[B, H]$ | FP16 | $\operatorname{RMSNorm}(R)$ |
| 输出 `residual_out` | $[B, H]$ | FP16 | $R = x + \mathit{residual}$ |

对每一行 $b\in[0,B)$，算子计算：

$$
\begin{aligned}
R_b &= x_b + \mathit{residual}_b, \\
\mathit{rms}_b &= \sqrt{\dfrac{1}{H}\sum_{i=0}^{H-1}R_{b,i}^{\,2} + \varepsilon}, \\
y_b &= \dfrac{R_b}{\mathit{rms}_b} \odot w.
\end{aligned}
$$

属性 $\varepsilon$（`eps`）默认 $10^{-6}$，为可选浮点属性。各行之间相互独立，可任意并行。

!!! note "关于接口约定"

    FlashInfer 原生的 `fused_add_rmsnorm` 是**原地**修改 `input` 与 `residual` 的；而本实验的算子采用**非原地**约定：`x` 与 `residual` 作为只读输入，算子输出独立的 `y` 与 `residual_out`。这一约定与昇腾 ACL 的算子调用方式（`aclnnFusedAddRmsNorm`，显式传入输出 tensor）一致，也更便于正确性与性能评测。请在实现时遵守该约定。

    算子同时接受 `enable_pdl` 布尔属性以保持与 FlashInfer 签名的兼容性；该属性源自 NVIDIA GPU 上的 programmatic dependent launch 优化，在 NPU 上无对应物，**实现中可忽略**，仅作占位。

### 任务要求

你的任务是在昇腾 NPU 上完成 `FusedAddRmsNorm` 算子的 kernel 实现并对其进行优化。你可以从 Ascend C / TileLang / Triton 三种路径中任选其一。三种路径都能达成目标，且报告中不会因为使用了某条路径或尝试了多条路径而额外加分——路径只是达成更好性能、加深对硬件理解的手段。

- 实验框架已给出算子原型、Host 侧 tiling 的骨架与一个可运行的参考实现，你可以修改 Host tiling 和 kernel 函数；
- 你需要保证算子优化后结果能通过精度校验；
- 你需要在实验报告中说明你的优化思路和依据，展示 profiling 过程，并尝试分析优化后的性能提升及其原因。

## 优化方向

下面列举一些常见的优化方向，仅供参考，并不意味着你必须使用这些方法。你可以根据自己的理解和实验结果，尝试任何手段。

### Kernel Fusion 与 UB 数据复用

「融合」本就是 `fused_add_rmsnorm` 这个算子名字的由来，也是本算子最直接的优化收益来源。如果在 NPU 上把残差加法、平方、规约、除法、缩放拆成多个独立 kernel 依次执行，那么中间的 $R$、$R^2$、部分和等都要在全局显存（GM）里反复读写，还要承担多次 kernel launch 的开销。把这些步骤融合进单个 kernel，让中间结果只停留在靠近 AIV 的 **UB（Unified Buffer）** 中，就同时免去了 GM 往返与 launch 开销——这也就是 UB 数据复用：把一行（或一组行）的 `x`、`residual`、`weight` 一次性搬入 UB，让后续的加法、平方、规约、除法、缩放全部在 UB 内完成，最后再把 `y` 和 `residual_out` 写回 GM。

在 910B4 上，单个 AIV 的 UB 可用容量只有 **192 KiB**，你需要仔细规划如何使用。TileLang / Triton 中对应的快速访存区是 shared memory，大体思路一致。

- 尝试把上述全部步骤融合进一个 kernel，让中间结果只停留在 UB 中；
- 合理规划 UB 空间，权衡输入队列、输出队列、FP32 计算与规约工作区、广播 buffer 等的占用。

### 多核切分与 Tiling

各行之间相互独立，最自然的并行方式是按行把任务切分到多个 AI Core 上。tiling 函数需要根据平台核数和输入 shape 决定每个核负责的行数，并尽量保证负载均衡。

- 注意尾核（最后一个核）可能分到的行数较少，思考这对利用率的影响；
- 当 $B$ 远大于核数时按行切分通常足够；当 $B$ 较小而 $H$ 较大时，你也可以考虑在 $H$ 维上做进一步切分，但要小心规约跨核带来的通信开销。

### 高性能规约

沿 $H$ 维的平方和规约是性能关键。无论选择哪种路径，都请认真研究所用 DSL 提供的规约原语及其 mask / 尾块机制：

- 尝试用「部分规约逐级折叠 + 整体规约收敛」的方式把长度 $H$ 的向量压成标量（对应 Ascend C 的 `BlockReduceSum` / `WholeReduceSum`、TileLang / Triton 的 `reduce`）；
- 注意规约原语对数据对齐、mask 设置和同步的要求；
- 规约得到标量 $\mathit{rms}$ 后，需要把它广播回长度 $H$ 再做逐元素除法，思考如何高效完成这一广播。

### 数据类型与精度控制

输入输出均为 FP16，但直接在 FP16 下累加平方和会损失精度。一个常见做法是：在 UB 中把数据 cast 到 FP32 完成加法与规约，最后再 cast 回 FP16 输出。这样既能在满足精度要求的前提下，又避免全程 FP32 带来的额外带宽与寄存器压力。

- 在报告中说明你选择的计算精度策略，并解释其如何同时满足精度与性能；
- 注意 cast 操作本身也有开销，思考在哪里进行 cast 最划算。

### 访存对齐与尾块处理

NPU 上的向量与搬运指令通常要求 **32B 对齐**（这是 UB 的最小常用访问粒度，FP16 下即 16 个元素对齐）。Vector 单元一次可处理的向量宽度为 256B，对应 FP16 下 **128 个元素**（FP32 则为 64 个元素）。当 $H$ 不是 16 的整数倍时，需要正确处理尾部不对齐的元素，避免越界访问或读到脏数据。各路径都提供了对应接口（如 Ascend C 的 `DataCopyPad`、TileLang / Triton 的 mask 机制）配合零填充处理这种情况。

- 确保你的实现对任意 $H$ 都正确，而不仅是评测 shape；
- 思考零填充对规约结果的影响（被填充的零是否参与求和？是否需要屏蔽？）。

### Profiling

利用性能测试工具 `msprof op` 与 `msprof op simulator` 对实现做 profiling：前者给出各计算单元与搬运通道的占用率（utilization），后者给出完整的指令流水线图与 Roofline 图。对于 simulator 模式，请注意其本质是在 CPU 上模拟 NPU 的执行，因此速度会比在 NPU 上执行慢很多，如果你发现需要很久才能完成采样，你可以适当修改数据规模，以快速地看到关键路径的流水图。

我们看到 Roofline 后就可以简要分析当前实现的性能问题是**延迟主导**（指令间存在空泡、依赖未重叠、过度插入的同步），还是**某条数据搬运路径带宽受限**（GM↔UB 吞吐被打满等），还是**某个计算单元算力受限**（Vector 单元已跑满等）。

定位到瓶颈后，我们就可以有针对性地自行思考如何修改程序以优化性能，而非依赖某种经验。例如，当我们看到向量单元因为等待搬运数据而利用率低的时候，就可以想办法实现 Double Buffer 让搬运与计算重叠，提高向量单元利用率。

!!! tip "关于调试器"

    同学们在华为官方文档中可能会看到 `msdebug` 这个调试工具。但由于该工具底层驱动的权限限制，我们无法为同学们开放使用权限；而且实测下来，`msdebug` 在同步 / 对齐类问题上反而容易给出误导性信息，对排查帮助有限。

!!! tip "关于同步与对齐"

    在 Ascend 上，你会遇到不少「意外」的同步与对齐问题：

    - **同步**：Vector 指令**不会自动识别 UB 上的 RAW（写后读）依赖**，需要在相关指令之间手动插入同步屏障；而规约、内存读写等指令往往还带有自己的同步语义，需要格外留意指令间是否存在隐式依赖。
    - **对齐**：规约指令、内存读写指令通常要求操作数按 32B 对齐，尾部不对齐的元素必须用 `DataCopyPad` / mask 等机制正确处理。

    另外需要提醒的是，**910 这一代 NPU 的报错代码高度同质化**——多种不同错误都返回同一个错误码，对定位问题的价值很低。此时最好**仔细（或者让 AI 仔细）对照 Ascend C 文档逐条核查**指令的同步与对齐要求。

## 评测方式

评测包含**正确性（精度）**与**性能**两部分。只有通过正确性检查的实现，其性能才会被计入排名。

### 精度评测

精度评测以 FP32 计算的 golden 结果为基准：参考实现先在 FP32 下完成 $R$、$\mathit{rms}$、$y$ 的计算，再 cast 到 FP16 作为 golden；你的实现输出与之逐元素比较。判定准则为**绝对误差与相对误差取或**：

$$
\text{pass}_i = \big(|y_i - y_i^{\mathit{golden}}| \le \mathit{atol}\big)\ \lor\ \big(\mathit{rtol}_i \le \mathit{rtol}\big),
\quad
\mathit{rtol}_i = \frac{|y_i - y_i^{\mathit{golden}}|}{\max(|y_i^{\mathit{golden}}|,\,\varepsilon_0)}.
$$

FP16 下取 $\mathit{atol} = \mathit{rtol} = 10^{-3}$，且要求**全部元素**均通过判定（错误比例 $=0$），`y` 与 `residual_out` 两个输出都需通过。精度评测的基准 shape 为 $[256, 1024]$、DType 为 FP16。

### 性能评测

性能评测使用昇腾的 `msprof op` 工具对单个算子进行计时，包含 warm-up 后取稳定耗时。排名以如下配置为准：

- **Shape**：$[256, 1024]$（即 $B=256,\ H=1024$）；
- **DType**：FP16；
- **eps**：默认 $10^{-6}$。

性能结果以 `msprof op` 输出的 `OpBasicInfo` 中的耗时为准。建议你在报告中给出多次运行的稳定数据，并注明统计方式。

## 实验框架

实验框架代码位于 [HPC101 课程仓库](https://github.com/ZJUSCT/HPC101) 的 `src/lab3.5` 路径下。框架按三种实现路径分别组织，共享同一套 checker：

```text
src/lab3.5
├── FusedAddRMSNorm                 # Ascend C 实现
│   ├── op_host/                    # Host 侧：算子原型与 tiling
│   │   ├── fused_add_rms_norm.cpp  #   算子定义、InferShape、TilingFunc
│   │   └── ...
│   ├── op_kernel/                  # Kernel 侧：你需要重点优化的部分
│   │   ├── fused_add_rms_norm.cpp  #   ★ kernel 实现
│   │   └── fused_add_rms_norm_tiling.h
│   ├── build.sh                    # 编译并打包算子 .run 安装包
│   └── CMakeLists.txt
├── tilelang/                       # TileLang 实现（可选）
│   ├── fused_add_rms_norm.py       #   ★ kernel 实现
│   └── run.py                      #   编译、运行与评测入口
├── triton/                         # Triton 实现（可选）
│   ├── fused_add_rms_norm.py       #   ★ kernel 实现
│   └── run.py                      #   编译、运行与评测入口
└── checker                         # 三种路径共享的正确性与性能测试
    ├── scripts/
    │   ├── gen_data.py             #   生成输入与 FP32 golden
    │   └── verify_result.py        #   精度校验
    ├── config.txt                  #   评测 shape / dtype / range 配置
    ├── run.sh                      #   一键：编译算子 → 生成数据 → 运行 → 校验
    └── profile.sh                  #   msprof op 性能测试
```

- 选择 **Ascend C** 的同学，你需要完成的代码位于 `FusedAddRMSNorm/op_kernel/`，你能且仅能修改 kernel 相关文件；算子原型与 tiling 可以根据你的优化需要调整；
- 选择 **TileLang / Triton** 的同学，你需要完成的代码位于对应目录的 kernel 文件中；
- 三种路径共用 `checker/` 下的 golden 生成与精度校验，保证评测口径一致；你可以直接运行 `run.sh` 走完「编译算子 → 生成数据 → 运行 → 精度校验」全流程，运行 `profile.sh` 进行性能测试；
- `checker/config.txt` 中默认即评测配置 $B=256,\ H=1024$、FP16，你可以在本地调试时调整，但最终排名以该配置为准。

```bash
# 一键功能测试（编译算子 + 生成数据 + 运行 + 精度校验）
cd src/lab3.5/checker && bash run.sh

# 性能测试
cd src/lab3.5/checker && bash profile.sh
```

## 如何获取计算资源

本实验的昇腾 910B4 NPU 通过课程[实验平台](https://platform.s.zjusct.io)的 DevPod 提供：

1. 登陆[实验平台](https://platform.s.zjusct.io)；
2. 创建预设为 `Ascend-910B` 的 DevPod，该 DevPod 内可直连昇腾 NPU 设备；
3. 在 DevPod 内进行算子开发、编译与运行，CANN 工具链已在环境中安装好。

算子编译与运行所需的 CANN 环境变量（`ASCEND_HOME_PATH` / `DDK_PATH` 等）已在 DevPod 中配置；`build.sh`、`run.sh` 与 `profile.sh` 会自动 `source` 对应的 `setenv.bash`，一般情况下你无需手动设置。

```bash
# 在 arm64-910B DevPod 中
git clone https://github.com/zjusct/hpc101
cd hpc101/src/lab3.5

# 功能测试
cd checker && bash run.sh

# 性能测试
cd checker && bash profile.sh
```

更详细的实验平台使用教程请参考文档 [集群使用](https://hpc101.zjusct.io/guide/)。

## 实现要求

- 在昇腾 NPU 上完成题目指定的 `FusedAddRmsNorm` 计算，实现路径可在 Ascend C / TileLang / Triton 中任选其一；无论选哪条路径，都禁止调用其他已有的 RMSNorm 实现代替被测计算；
- 保持给定的算子原型、输入输出与计算语义（非原地、输出 `y` 与 `residual_out`）；
- 可以自行决定 kernel 的拆分与融合方式、tiling 参数与片上存储规划，也可以为不同 shape 选择不同配置；
- 不得修改 checker 的测试程序、golden 生成与计时逻辑，也不得硬编码测试数据或输出结果，或利用评测程序的漏洞绕过计算；
- 实验框架所需的依赖库和工具链已经在课程环境中安装好（含 CANN、TileLang、triton-ascend），禁止使用未说明的软件包或自建工具链；
- 你可以阅读 FlashInfer 等开源实现，了解已有的算法和优化方法。实验报告中应注明参考过的实现或资料，并说明自己的修改与优化。

## 提交要求

你需要提交以下内容：

1. **实现代码**：你所选路径下的 kernel 实现及其运行所需的辅助代码（请在报告中注明使用的是 Ascend C / TileLang / Triton 哪条路径）；
2. **实验报告**：PDF 格式，说明你的优化思路、依据和结果，具体要求见下一节；
3. **优化方法介绍**：在报告中专门一节，介绍你采用的关键优化方法及其原理；
4. **精度测试结果**与**性能测试结果**：以评测 shape $[256,1024]$ / FP16 为准。

评测时只会收取指定目录中的文件。新增的辅助文件也需要放在该目录中，并保证程序可以在课程提供的环境中从干净目录直接运行。请勿提交测试数据、编译产物或profiling 生成的大文件。

## 实验报告

实验报告不需要重复大段背景知识，重点说明你做了哪些优化、为什么这样做，以及结果如何。报告至少应包含：

- 算子计算与数据依赖分析；
- baseline 的性能结果与瓶颈分析（用 `msprof` 定位）；
- 每项主要优化的设计、实现和收益；
- 最终性能数据、测试环境和运行命令；
- 精度测试结果；
- [思考题](#思考题)作答；
- 尝试过但未采用的方案及其原因。

性能数据需要注明对应的输入 shape、统计方式和单位。建议使用表格或图表展示各阶段优化前后的变化；较长的代码、命令输出和profiler 结果可以作为附件，不需要全部放在正文中。

!!! note "关于 AI 使用的要求"

    同其他实验一样：本实验允许使用 AI Agent 辅助完成；但最终递交的实验报告中不应出现大段由 AI 直接生成的文字。勿谓言之不预也！

## 思考题

!!! tip "这里可能没有标答！"

    部分思考题可能没有一个标准的正确答案。我们更希望看到你在实验过程中遇到的实际情况以及你个人的思考和理解，即使可能存在错误。

1. 以评测 shape $[256,1024]$（FP16）为例，估算算子一次执行的总访存量（每个张量各被读 / 写几遍）与总乘加次数，计算算术强度（MACs/byte）。据此判断该算子在本配置下是访存瓶颈还是计算瓶颈，并结合 `msprof` 的实测数据验证你的判断。

2. （开放题）谈谈你在昇腾 NPU 上开发算子的体验与感受。可以从硬件架构给你的印象、与 GPU / CPU 开发体验的异同、开发过程中遇到的印象深刻的坑或顿悟、以及对昇腾生态与工具链的观察等角度展开（请畅所欲言）。

## OJ 自动评测

!!! info "OJ 评分指标"

    OJ 采用分段曲线进行评分，与 Lab 2 / Lab 3 类似。对于评测 shape $[256, 1024]$ / FP16，横轴定义为你提交实现相对于 100 分 checkpoint 的加速比：

    $$
    p=\frac{t_{100}}{t},
    $$

    其中 $t$ 为提交实现的耗时，$t_{100}$ 为该 case 的 100 分 checkpoint 对应耗时。性能超过 checkpoint 时 $p>1$，并进入奖励区间。具体的 60 分与 100 分 turning point 将在 OJ 开放后公布。

!!! warning "施工中"

    目前 OJ 自动评测系统仍在建设中😭，我们会尽快发布。性能排名以 $[256,1024]$ / FP16 配置下的 `msprof op` 耗时为准。

## 参考资料

- [FlashInfer: Kernel Library for LLM Serving](https://github.com/flashinfer-ai/flashinfer)
- [FlashInfer `fused_add_rmsnorm` 文档](https://docs.flashinfer.ai/generated/flashinfer.norm.fused_add_rmsnorm.html)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [Ascend C 开发文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC31alpha001/devguide/ascendc)
- [昇腾社区 — AscendC 最佳实践](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0031.html)
- [CANN 软件下载与文档](https://www.hiascend.com/software/cann)
- [TileLang](https://tilelang.com/)
- [Triton](https://triton-lang.org/) · [triton-ascend 后端](https://github.com/triton-ascend/triton-ascend)
