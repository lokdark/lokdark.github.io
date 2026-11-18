---
title: "GPU 基础架构：SM、存储层次与多卡互联"
date: 2026-09-18T09:50:00+08:00
draft: false
author:
  name: "Luke"
description: "从 A100 的芯片布局出发，理清 GPU 的存储层次、SM 与 CUDA 执行模型（Thread/Warp/Block/Grid）、卡间与节点间互联，以及这些硬件特性如何反过来约束 TP/PP/DP 的选型。"
summary: "GPU 性能不只取决于核心数量：SM 的寄存器与共享内存、Warp 的调度粒度、HBM 与 NVLink/IB 的带宽量级，共同决定了能跑多快、能跑多大的模型。"
tags: ["GPU", "CUDA", "NVIDIA", "并行计算", "大模型基础设施"]
categories: ["learning"]
toc: true
slug: "gpu-architecture"
---

> GPU 的性能不只看「核心多不多」，而是由 **SM 的资源、存储层次、线程调度粒度、卡间互联** 四件事共同决定

先看一张整体图（A100 公开架构），后面讲的内容都能在图上找到对应位置：

![A100 公开架构](/images/posts/gpu-architecture/a100-architecture.png)

## 一、基础术语

- **FP32 / FP16 / BF16**：浮点数格式，数字代表用多少 bit 存一个数。
- **TFLOPS**：每秒万亿次浮点运算，算力单位。
- **计算密集**：运算多、数据读写少，容易打满算力。
- **访存密集**：频繁读写数据、计算量小，性能被带宽卡死（AI 里绝大多数场景属于这类）。
- **内存墙**：计算单元越来越快，但数据从显存搬到计算单元的速度跟不上，两者之间形成一堵「墙」。
- **CPU**：复杂控制逻辑、大缓存，擅长串行分支与复杂逻辑，少量强核心。
- **GPU**：大量简单计算核心，弱化分支预测、转而堆大量简单计算核心，靠**海量并行**取胜；优化的核心思路是**让计算区域的有效面积更大**（控制/缓存占比越小越好）。

## 二、存储层次

从快到慢、容量从大到小（反过来则是带宽递减）：

`寄存器 Registers` > `Shared Memory 共享内存` > `L1 Cache` > `L2 Cache` > `HBM 显存（GPU 主存）` > `Host CPU 内存`

- **寄存器**：每个线程 Thread 私有。关键问题是 **Register Spilling（寄存器溢出）**——寄存器不够用时数据被写到显存或 Cache，性能明显下降。
- **Shared Memory**：属于单个 SM，同 SM 内的线程块都可访问，**可编程**。FlashAttention、高性能矩阵乘的核心思路，就是把 HBM 上的大块数据切成小块加载到 Shared Memory 里算。
- **L1 / L2 Cache**：硬件缓存，硬件自动管理。L1 依附于每个 SM，L2 由整卡全局共享（多个 SRAM 块）。
- **HBM**：GPU 的显存（多个 DRAM 块）。
- **Host-Device 拷贝**：一般指 CPU 内存与 GPU 显存之间的数据流动，靠 PCIe 搬运，远慢于 HBM 内部带宽，因此工程上尽量避免两边的数据拷贝（让大数据常驻显存）。

| 层级 | 单 SM 容量 | 整卡总容量 | 带宽量级 |
| --- | --- | --- | --- |
| Registers 寄存器 | 256 KB | A100：108 SM × 256 KB ≈ 27 MB；H100：144 SM × 256 KB ≈ 36 MB | 百 TB/s 级（每 SM 1.5~2 TB/s） |
| Shared Memory 共享内存 | A100：最大 164 KB；H100：最大 228 KB（与 L1 共用统一缓存池） | - | 约 19~30 TB/s（每 SM 约 200 GB/s） |
| L1 Cache | A100：统一池 192 KB/SM；H100：统一池 256 KB/SM | - | 同上，与 SMEM 共享统一池 |
| L2 Cache | - | A100：40 MB；H100 SXM：50 MB | 约 5~10 TB/s（全卡口径） |
| HBM（GPU 主存） | - | A100：40/80 GB；H100 SXM：80 GB | A100 80GB：约 2.0 TB/s；H100 SXM：约 3.35 TB/s |
| Host CPU Memory | - | TB 级 | PCIe 5.0 x16：约 128 GB/s；NVLink 4（GPU-CPU）：900 GB/s |

一句话读这张表：**越往下容量越大、带宽越低**。AI 推理和训练的大部分优化，本质都是在「把数据尽量留在上层」。

## 三、SM 与 CUDA 执行模型

SM（Streaming Multiprocessor，流多处理器）是 GPU 最基础的硬件执行单元。一个 Kernel 下发之后会被拆成大量 Block（Block 内由若干线程组成），由硬件调度器把 Block 分配到空闲的 SM 上执行。

每个 SM 拥有一套独立资源：

- 一批 CUDA Core
- Tensor Core 张量核心
- 寄存器堆（Register）
- Shared Memory
- L1 Cache

### 3.1 两类计算单元

**CUDA Core**：普通算术运算单元，负责浮点与整数运算，可做通用加减乘除、逐元素算子。

**Tensor Core（张量核心）**：专门做矩阵乘-累加 **MMA（Matrix Multiply-Accumulate）**，Transformer 里绝大部分计算都来自矩阵乘，因此高度依赖 Tensor Core。

- 支持精度：FP16 / BF16 / FP8（FP8 是新一代 GPU 才有的，主要用来减少约 50% 的内存占用）；
- **不是任意 shape 都能触发 Tensor Core 加速**，输入矩阵维度需要对齐硬件要求；算子维度不对齐时矩阵乘会退回 CUDA Core 执行，性能大幅下降；
- Tensor Core 的输出依然写回寄存器，之后再写回 Shared Memory / HBM。

### 3.2 线程执行模型

**Thread（线程）**：最小执行单元，执行一小份计算，拥有私有寄存器（分配后独占，不是某个寄存器永久绑定某个线程）。

**Warp（线程束）**：GPU 硬件调度的最小单位，32 个线程组成一个 Warp，同一 Warp 内 32 个线程同时执行同一条指令。（这些是 NVIDIA 的概念，AMD 的对等概念 Wavefront 是 64 线程。）

> **Warp Divergence（分支发散）**：程序里出现 `if...else` 时，同一 Warp 的 32 个线程可能走不同分支；硬件无法真正并行执行不同指令，只能把各分支串行执行，不参与的线程就闲置了——有效工作线程变少、算力利用率下降。并不是所有 `if` 都会这样，像「判断线程号奇偶」这种必然发散。

**Block（线程块）**：由若干 Warp 组成（线程总数一般是 32 的倍数），整个 Block 被调度到**单个 SM** 上，不能跨 SM。

**Grid（网格）**：一个 Kernel 对应一个 Grid，Grid 由大量 Block 组成；硬件调度器把 Block 分发到各个空闲 SM。

```
Grid（整个任务） → Block（分配到 SM） → Warp（硬件调度单元，32 线程） → Thread（单线程）
```

一个容易混淆的点：**Block 是开发者可控的**（你决定放多少线程），**Warp 是硬件调度单元**（硬件干活就是以固定线程数为单位进行的）。

```cuda
my_kernel<<<1024, 256>>>(args);
//          ↑ grid ↑ block
//    1024 个 block，每个 256 线程
//    → 这次 kernel 调用 = 一个 Grid，包含 1024 个 Block

// 同一个 Kernel 的两次不同执行，就是两个 Grid：
my_kernel<<<1024, 256>>>(args);
my_kernel<<<65536, 256>>>(args);
```

### 3.3 SM 资源限制对并行度的影响

每个 SM 的寄存器总量、Shared Memory 总量都是固定的，而每个 Block 都要吃掉一部分。**单 Block 占用资源越多，单个 SM 能同时容纳的 Block 就越少，活跃 Warp 数量也随之下降。**

活跃 Warp 不足会怎样？当一部分 Warp 在等待访存（读写 HBM）时，如果没有其他 Warp 可以切换上来顶替执行，SM 就会空闲——这就是**算力浪费**。所以 Block 大小和资源占用要匹配 SM 的硬件规格，不是越大越好。

## 四、多卡与多节点互联

### 4.1 基础概念

1. **Node**：一台物理服务器，一台机器内可以插多张 GPU。
2. **单节点**：所有 GPU 在同一台服务器内部。
3. **多节点**：多台独立服务器通过外部网络连接，组成 GPU 集群。
4. **集合通信**：多张 GPU 之间互相交换张量数据，DP / TP / PP 全部依赖集合通信，速度取决于底层互联硬件。
5. **带宽量级排序**：`HBM 显存带宽 >> NVLink >> InfiniBand(IB) >> RoCE >> PCIe >> 普通以太网`
6. **TP（张量并行）**：把一层的权重矩阵拆到多个 GPU 上，大家一起算一层的输出。
7. **PP（流水线并行）**：把模型的层按阶段拆到多个 GPU 上接力计算。
8. **DP（数据并行）**：每个 GPU 放一份完整模型，数据拆开各算各的，最后同步梯度。

### 4.2 单节点内部：NVLink 与 PCIe

**NVLink**：NVIDIA 专用硬件链路，GPU 与 GPU 之间直接点对点连接，**不经过 CPU**，带宽远高于 PCIe。张量并行 TP 十分依赖 NVLink——没有这个等级的互联，TP 性能会严重退化。

**PCIe**：CPU 主板上的通用总线，GPU 插在主板的 PCIe 插槽。如果两块 GPU 之间走 PCIe 通信，路径是 `GPU → CPU 内存 → 另一块 GPU`，数据要绕经 CPU，开销大、带宽低。

### 4.3 跨节点：InfiniBand 与 RoCE

单节点能插的 GPU 数量有限，训练超大模型必须上多台服务器；而服务器之间用不了 NVLink，只能靠外部网络。

**InfiniBand（IB）**：高性能集群专用硬件网络（专用网卡 + IB 交换机），面向 HPC 与 AI 大模型集群。数据路径：`GPU → 本机网卡 → IB 交换机 → 对端网卡 → 远端 GPU`。

**RoCE（RDMA over Converged Ethernet）**：在普通以太网硬件上实现 RDMA 远程直接内存访问。IB 是专用硬件，RoCE 跑在以太网交换机上，成本更低。

## 五、互联硬件对分布式算法的约束

| 并行方式 | 解决什么 | 通信特征 | 硬件要求 |
| --- | --- | --- | --- |
| **TP** 张量并行 | 单层过大 | 层内算子切分，交换频繁、消息粒度小，通信压力大 | 最好同节点 NVLink；跨节点做 TP 性能很差，几乎不用 |
| **PP** 流水线并行 | 模型过大 | 按层切分，只传层的输入输出激活，消息粒度大、交互频次低 | 容忍度高，NVLink / 跨节点 IB 都能跑 |
| **DP** 数据并行 | 吞吐量 | All-Reduce 同步梯度，消息是完整梯度张量，粒度大 | 容忍度最高，NVLink / PCIe / IB / RoCE 都可 |

换句话说：**通信越频繁的并行方式，越依赖高带宽低延迟的链路**。这也是为什么 TP 通常被限制在单机 8 卡之内。

## 六、SXM 形态、HGX 与 NVL72

### 6.1 SXM 与 PCIe 两种形态

![SXM 与 PCIe 版本对比（左：SXM，右：PCIe）](/images/posts/gpu-architecture/sxm-vs-pcie.jpg)

SXM 是 GPU 的一种形态（上图左侧），右侧是常见的 PCIe 版本。英伟达的八卡互联架构是把 SXM 接入 HGX 主板，而不是像 x86 那样用金手指插在 PCIe 插槽上。

### 6.2 两种卡间拓扑：NVLink 桥 与 NVSwitch

**不走 HGX 主板、只用卡间 NVLink 桥**：只支持点对点连接，多卡只能首尾成环，没有 Switch，跨组通信必须绕 CPU。

```
   +---------- NVLink 桥（点对点）----------+
   |                                       |
  GPU0                                   GPU1
   |                                       |
   +----------------- PCIe -----------------+----> CPU (Root Complex)

   共 4 组：(0,1) (2,3) (4,5) (6,7)；组与组之间只能经 CPU 绕行
```

**走 SXM 主板 + NVSwitch**：8 卡全 Mesh 任意互通。

```
        +---------------------------------------------+
        |   NVSwitch x2~4（NVLink 交换，8 卡全 Mesh）   |
        +--+--+--+--+--+--+--+--+--+--+--+--+--+-----+
           |  |  |  |  |  |  |  |  |  |  |  |  |
        +--v--v--v--v--v--v--v--v--v--v--v--v--v--+
        | GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7 |
        +----------------------+----------------------+
                               |
                    PCIe 5.0（HGX 基板引出到 CPU 主板）
                               v
                              CPU
```

> 注：HGX 是英伟达私有的 GPU 承载基板；华为的全卡互联有自研的 Atlas 900T A3 计算板，UBSwitch 芯片在其上。

### 6.3 板间与机架级：从 HGX 到 NVL72

**单个 HGX 基板**：集成 8 颗 SXM 形态 GPU，板内通过 2~4 颗 NVSwitch 芯片实现 8 GPU 全互联（NVLink）；**板间**扩展则依赖外部高速网络（InfiniBand 或 RoCEv2）。这是 2020 年 DGX A100 确立的经典范式，至今仍是 HGX B200/B300 单节点服务器的标准架构——节点内 NVLink 全互联，跨节点 IB 扩展。

**NVL72 则是机架级系统**：72 颗 GPU 在机架尺度上构成一个统一的 NVLink 域。计算托盘上虽然也有多颗 GPU 焊在一块板上的形态，但**板上没有 NVSwitch**——那个板叫 compute tray，主要为了 CPU 紧耦合、液冷、IO 配套等功能；GPU 的链路全部通过背板铜缆直连到机架背部独立的 **NVLink Switch Tray**（共 9 个托盘、18 颗 NVSwitch 芯片）。

## 小结

把这篇的线索收一下：

1. **存储层次决定上限**：能在 Shared Memory / 寄存器里解决的问题，绝不下放到 HBM；能留在 HBM 的，绝不跨 PCIe。
2. **SM 资源决定并发**：寄存器与 SMEM 是硬约束，Block 太大反而压低活跃 Warp，靠「更多常驻线程」掩盖访存延迟才是主要手段。
3. **Warp 是调度粒度**：32 线程同指令，分支发散和维度不对齐（Tensor Core 用不上）都会直接吃掉算力。
4. **互联决定并行策略**：TP 吃带宽所以要 NVLink、限制在机内；PP 消息大但频次低，可以跨机；DP 容忍度最高，什么链路都能跑。
