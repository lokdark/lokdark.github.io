---
title: "KV Cache 与 PagedAttention：把大模型推理的显存管起来"
date: 2026-08-15T15:15:00+08:00
draft: false
author:
  name: "Luke"
description: "从 Decode 阶段的重复计算讲起，用 KV Cache 换掉冗余算力，再看 vLLM 的 PagedAttention 如何像操作系统分页一样管理显存，以及 Offload、Token Hash Cache 这些延伸方案。"
summary: "推理的两个阶段、KV Cache 为什么只缓存 K 和 V、显存碎片怎么来的、PagedAttention 的 block/block table/写时复制、抢占与恢复策略，最后到 KV Cache Offload 和前缀缓存。"
tags: ["KV Cache", "PagedAttention", "vLLM", "推理优化", "LLM"]
categories: ["learning"]
toc: true
slug: "kv-cache-pagedattention"
---

## 一、KV Cache 到底是什么

大模型的推理不是一步算完的，它分两个阶段：

- **Prefill（预填充）**：把用户输入的整段 prompt 一次性喂进去，所有 token 并行计算，得到每个 token 的 Query/Key/Value，并写出 KV Cache。这一步矩阵很大，是**算力瓶颈（compute-bound）**。
- **Decode（解码）**：从第一个生成 token 开始，每一步只处理 **1 个新 token**，自回归地一个词一个词往外吐。这一步每次只算一个向量，却要把整个模型的权重和全部 KV Cache 读一遍，是**显存带宽瓶颈（memory-bound）**。

![Prefill 与 Decode](/images/posts/kv-cache-pagedattention/01-prefill-decode.png)

KV Cache 就作用在 **Decode** 阶段：把历史 token 的 K/V 存下来，后面每一步直接复用，不再重算。

## 二、没有 KV Cache：每一步都在重复劳动

假设句子已经有 `t1 t2`，现在要生成 `t3`。没有缓存时，模型会把 `t1 t2 t3` 当成一个完整序列重新算一遍 Q/K/V——其中 `t1`、`t2` 的 K/V 明明上次已经算过了。

![无 KV Cache 的 Decode](/images/posts/kv-cache-pagedattention/02-decode-no-cache.png)

序列越长，重复计算越多：第 `n` 步要把前 `n` 个 token 全部重算，累计计算量是 **O(n²)**，而且大量算力花在完全相同的中间结果上，速度被严重拖慢。

## 三、有 KV Cache：只算新 token

有了缓存之后，每一步只需要为新 token 计算 `q/k/v`。

![有 KV Cache 的 Decode](/images/posts/kv-cache-pagedattention/03-decode-with-cache.png)

注意力计算里，新 token 的 **Q** 用来和历史所有 token 的 **K** 做点积，再对 **V** 加权求和：

```
Q_new = x_new · Wq
K_all = concat(K_cache, k_new)
V_all = concat(V_cache, v_new)
out   = softmax(Q_new · K_allᵀ / √d_k) · V_all
```

算完把 `k_new`、`v_new` 追加进缓存即可。于是每一步的计算量从 O(n²) 降到 O(n)，整个生成过程从 O(n²) 变成 O(n)（相对逐 token 的增量而言）。

### 为什么只缓存 K 和 V，不缓存 Q？

因为 **Q 是一次性的**：第 `t` 步的 `Q_t` 只在这一步和历史的 K 做点积，之后再也不会被用到；而 `K`、`V` 会被之后所有步骤反复读取。所以缓存 K/V 是"只存会被复用的东西"。

### 代价：显存换算力

KV Cache 的显存占用可以粗略写成：

```
KV 显存 = 2 × 层数 × KV头数 × head_dim × 序列长度 × batch × 每元素字节数
```

以 7B 模型为例（32 层、32 个 head、head_dim=128、fp16）：每个 token 约 `2×32×32×128×2 ≈ 512KB`，一条 4K 的序列就要 **约 2GB** 显存。批量一大，KV Cache 往往比模型权重还占地方——这也正是后面所有优化的动机。

## 四、KV Cache 的显存浪费

早期实现要求 KV Cache 是一块**连续**的显存张量，并且按 `max_seq_len` 预留，于是出现两类浪费：

- **内部碎片**：按最大长度预留，实际请求往往短得多，预留的部分全空着（vLLM 论文里指出，浪费可达 60%~80%）；
- **外部碎片**：多条序列在连续显存上排队分配，释放后留下大小不一的空洞，放不下新序列，却也没法被别处利用。

![KV Cache 的内外部碎片](/images/posts/kv-cache-pagedattention/04-kv-cache-fragmentation.png)

## 五、PagedAttention：借操作系统的分页思路

vLLM 借鉴了操作系统的**虚拟内存分页**：把 KV Cache 切成固定大小的 **block**，逻辑上连续、物理上可以任意分散。

### 5.1 三个关键设计

1. **固定大小的 block**：每块存若干个 token 的 K/V（vLLM 默认 16 个），统一了分配粒度，内部碎片最多只剩"块内未填满"的一小部分；
2. **逻辑块 → 物理块映射**：由 **block table** 维护，物理块不需要连续，按需从显存池里取；
3. **按需分配 + 共享**：用多少分配多少，支持多个序列共享同一个块，并配合 **copy-on-write（写时复制）** 做隔离。

![逻辑块到物理块的映射](/images/posts/kv-cache-pagedattention/05-paged-block-table.png)

block table 里通常还带一个 `filled` 计数，记录这个物理块里有多少个**有效 token**（写满 16 个才是 16/16）。对调用方来说序列依然是连续的，只是物理落点被打散了。

### 5.2 省在哪

- 不用为 `max_seq_len` 预留，按实际增长分配 → 内部碎片几乎消失；
- 物理块可以来自显存池的任意位置 → 外部碎片被消除；
- 相同的 prompt、相同的块可以被多条序列**引用**而不是复制 → 显存利用率大幅提升。

## 六、并行采样与写时复制

假设用一条 prompt 同时采样 4 条回答（best-of-n、beam search 等）：

- **传统做法**：4 条序列各存一份完整 KV Cache，prompt 部分被复制 4 次；
- **PagedAttention**：4 条序列**共享**同一批 prompt 物理块，各自只有生成部分独立。

![并行采样与 CoW](/images/posts/kv-cache-pagedattention/06-parallel-sampling-cow.png)

每个块有一个**引用计数（ref count）**。某条序列真正要写入新 token 时，才把共享块复制一份（copy-on-write），引用计数随之减少；**计数降到 0 时才回收**。这样既保证了数据隔离，又省下了大量显存——4 条采样在 prompt 部分只占 1 份。

## 七、调度：抢占、回收与恢复

显存不够时，vLLM 需要**抢占**某些请求，把它们的块腾出来。

- **回收策略：All-or-Nothing**。一个序列的所有 KV 块总是一起被访问，所以要么整份回收，要么一个都不动，不会只回收一半。
- **恢复策略**分两种：

| 策略 | 做法 | 代价 |
| --- | --- | --- |
| Swapping | 把被抢占请求的 KV 块搬到 CPU 内存，等有空闲 GPU 块再搬回来 | 占用 PCIe 带宽，受 CPU 内存容量限制（总量不超过 GPU 物理块数） |
| Recomputation | 直接丢掉已生成的 KV，恢复时把"原始 prompt + 已生成 token"拼起来重算 | 浪费算力，但不用搬数据 |

选哪个取决于瓶颈在带宽还是在算力：换出换入便宜就 Swapping，重算便宜就 Recomputation。

## 八、GPU kernel 优化

按块、非连续地访问 KV Cache，和通用 kernel 的假设不匹配，所以 vLLM 还改了三处 kernel：

1. **切分 + 块写入融合**：把"切分 KV"和"写入 block"合成一个 kernel，减少 kernel 启动次数；
2. **融合块读取与注意力计算**：在 FasterTransformer 的注意力 kernel 上改造，按 block table 实时读取 KV 并计算。**每个 block 交给一个 warp 去读**，保证合并访存（coalesced access），warp 读完就换下一个块；同时支持同一 batch 内不同长度的序列，**不需要 padding 到相同长度**（省掉一大块无效计算和显存）；
3. **块复制优化**：CoW 触发的复制涉及多个非连续块，`cudaMemcpyAsync()` 只能一次拷一段，调用频繁；vLLM 实现了一个融合 kernel，把多次复制合并成一次 kernel 启动。

## 九、把 KV Cache 挪出去：Offload

既然 KV Cache 吃显存，一个自然的想法是把它放到外部存储（CPU 内存、本地 SSD、远端存储）里，需要时再读回来。

**关键前提**：只有当**从外部加载 KV Cache 的速度快过 GPU 重新计算的速度**时，卸载才有意义。换句话说，瓶颈是外部存储的读写带宽，而不是容量。

常见的数据通路：

- **GPUDirect Storage（GDS，NVIDIA Magnum IO）**：在 GPU 和存储之间建一条直连路径，**绕过 CPU**；
- 本地 **NVMe SSD**：走 PCIe 点对点传输；
- 远程存储：**NVMe-over-Fabrics**、**NFS-over-RDMA**，或支持 RDMA 的分布式文件系统（如 WEKA、VAST Data、DDN Exascaler）。

一句话：Offload 是用"搬运"换"重算"，能不能赚取决于搬运通道有多快。

## 十、Token Hash Cache：别重复算相同的 prompt

很多请求共享同一段前缀（系统提示词、few-shot 示例、多轮对话的历史）。**Token Hash Cache**（也就是 prefix caching / 前缀缓存）的思路是：把 token 序列按块算哈希，命中了就直接复用已有的 KV 块。

![Token Hash Cache 流程](/images/posts/kv-cache-pagedattention/07-token-hash-cache.png)

流程要点：

- 按 block 大小（默认 16 个 token）切分，**逐块**计算哈希；
- 哈希需要**带上父块哈希**，否则不同前缀下的相同块会被误判为同一份；
- 命中的块直接引用（零计算），未命中的部分才真正跑 Prefill，并把新块写入哈希表；
- 通常只有**写满的整块**才会被缓存，避免"半块"污染。

配合 PagedAttention 的 CoW 和引用计数，这套机制可以直接落在地上，不需要额外的数据拷贝。

## 十一、小结

| 问题 | 解法 | 换来什么 |
| --- | --- | --- |
| Decode 每步重复算历史 token | KV Cache | 每步 O(n)，省大量算力 |
| KV Cache 显存占用大 | —（这是必然代价） | 用显存换算力 |
| 连续显存导致内外碎片 | PagedAttention（block + block table） | 碎片从 60%+ 降到个位数 |
| 多序列重复存同一 prompt | 共享块 + copy-on-write + ref count | 并行采样省数倍显存 |
| 显存不够 | 抢占 + Swapping / Recomputation | 保证服务不崩 |
| 显存实在放不下 | Offload 到 CPU / SSD / 远端 | 用带宽换容量 |
| 相同前缀反复 Prefill | Token Hash Cache（前缀缓存） | 命中部分零计算 |

一句话串起来：**KV Cache 用显存换算力，PagedAttention 把这份显存管好，Offload 和前缀缓存再把"容量"和"重复"两个问题补上。**
