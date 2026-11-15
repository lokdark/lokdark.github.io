---
title: "从 Transformer 到 LLM：大模型基础学习"
date: 2026-11-15T14:20:00+08:00
draft: false
author:
  name: "Luke"
description: "从 Transformer 底层架构出发，串起 BERT（仅编码器）、GPT（仅解码器）与 LLM 的预训练—微调范式，附统一风格的自注意力、多头注意力与训练流程图解。"
summary: "把 Transformer、BERT、GPT、LLM 串成一条主线：先看自注意力与多头注意力的计算过程，再对比编码器/解码器两种演化方向，最后落到大模型的预训练与微调范式。"
tags: ["AI", "Transformer", "BERT", "GPT", "LLM", "注意力机制"]
categories: ["learning"]
toc: true
slug: "transformer-bert-gpt-llm"
---

## 一、Transformer：一切的起点

Transformer 是 2017 年《Attention Is All You Need》提出的架构，它彻底摆脱了 RNN 的顺序计算，用**自注意力（Self-Attention）**并行建模序列中任意两个位置之间的关系。

### 1.1 整体架构

架构分为**编码器（Encoder）**和**解码器（Decoder）**两大部分，中间靠「编码器-解码器注意力」连接。

![Transformer 整体架构](/images/posts/transformer-bert-gpt-llm/01-transformer-architecture.png)

### 1.2 术语：Add & Norm

图中每个子层后面都跟着 **Add & Norm**：

- **Add**：残差连接（Residual Connection），把子层的输入直接加到输出上，缓解深层网络的梯度消失问题；
- **Norm**：层归一化（Layer Normalization），对每个样本的特征维度做归一化，让训练更稳定。

### 1.3 编码器与解码器

**编码器（Encoder）** 由 N 层结构完全相同的模块堆叠而成，每层包含两个子层：

- 多头自注意力（Multi-Head Self-Attention）：计算输入序列中每个词与其他所有词的相关性；
- 前馈神经网络（Feed-Forward Network，FFN）：对每个位置独立做非线性变换。

**解码器（Decoder）** 同样由 N 层模块堆叠而成，每层比编码器多一个子层，共三个：

- 掩码多头自注意力（Masked Multi-Head Self-Attention）：只允许每个词看到它之前的词，防止「偷看」未来信息；
- 编码器-解码器注意力（Encoder-Decoder / Cross Attention）：让输出序列去关注输入序列；
- 前馈神经网络（FFN）。

每个子层后面同样接 Add & Norm。

### 1.4 工作流程（三步）

**第一步：得到输入表示 X。** 词嵌入（Token Embedding）与位置嵌入（Positional Embedding）相加，得到序列表示矩阵 X。

![输入表示](/images/posts/transformer-bert-gpt-llm/02-input-embedding.png)

**第二步：Encoder 编码。** X 经过 6 个 Encoder block，得到整句的编码信息矩阵 C。

![Encoder 输出编码矩阵 C](/images/posts/transformer-bert-gpt-llm/03-encoder-output.png)

**第三步：Decoder 解码。** Decoder 接收 C，并以上一时刻的输出为输入，逐个预测下一个词（生成式，从起始符 Begin 开始）。

![Decoder 逐词预测](/images/posts/transformer-bert-gpt-llm/04-decoder-predict.png)

### 1.5 输入：词嵌入 + 位置嵌入

- **词嵌入（Word Embedding）**：把 token 映射成稠密向量，可以由 Word2Vec、GloVe 等预训练得到，也可以在 Transformer 里一起训练。
- **位置嵌入（Positional Embedding）**：因为注意力本身不区分顺序，需要显式注入位置信息。位置嵌入的维度与词嵌入相同，可以训练得到，也可以用下面的公式直接算出来：

![位置编码公式](/images/posts/transformer-bert-gpt-llm/05-positional-encoding-formula.png)

### 1.6 Self-Attention：自注意力

自注意力的核心，是让每个词都能「看到」其他词，并按相关性加权聚合。

**Q、K、V 从哪来？** 用输入 X 分别乘以三个权重矩阵 Wq、Wk、Wv（维度为 d×d，d 为模型维度）得到：

![Q K V 的生成](/images/posts/transformer-bert-gpt-llm/06-self-attention-qkv.png)

多头注意力中，权重矩阵会被拆成多份（因为多个头的结果最后要拼接）。

![Q、K、V 计算公式](/images/posts/transformer-bert-gpt-llm/07-qkv-formula.png)

**注意力计算公式**（除以 √dₖ 是为了防止向量内积过大，导致 softmax 进入梯度消失区）：

![Self-Attention 公式](/images/posts/transformer-bert-gpt-llm/08-attention-formula.png)

**计算过程：**

1. 用 Q 与 K 的转置相乘，得到「词与词」的原始相关性分数矩阵（1234 表示句子里的单词位置）：

![Q·Kᵀ 矩阵](/images/posts/transformer-bert-gpt-llm/09-qk-transpose-matrix.png)

2. 对每一行做 softmax（每一行的和变为 1），得到注意力权重。如果是解码器的掩码注意力，Mask 矩阵就在 softmax 之前作用：

![Softmax 公式](/images/posts/transformer-bert-gpt-llm/10-softmax-formula.png)

![注意力掩码](/images/posts/transformer-bert-gpt-llm/11-attention-mask.png)

3. 把注意力权重乘以 V，得到最终输出 Z：

![注意力输出](/images/posts/transformer-bert-gpt-llm/12-attention-output-formula.png)

其中每个位置的输出 Zᵢ 是对所有 Vⱼ 的加权求和：

![Zᵢ 的计算](/images/posts/transformer-bert-gpt-llm/13-zi-formula.png)

### 1.7 Multi-Head Attention：多头注意力

把 Q、K、V 线性映射到 h 个不同的子空间，分别做 self-attention（共 h 层），再把结果拼接起来。

![多头注意力结构](/images/posts/transformer-bert-gpt-llm/14-multihead-structure.png)

其中 h 就是「头」的个数；所有头的结果拼接后过一个 Linear 层，得到 Multi-Head Attention 的最终输出 Z（维度与输入 X 一致）。

![多头注意力公式](/images/posts/transformer-bert-gpt-llm/15-multihead-formula.png)

### 1.8 Add & Norm

子层输出先做残差连接，再做层归一化：

![Add & Norm 公式](/images/posts/transformer-bert-gpt-llm/16-add-norm-formula.png)

### 1.9 Feed Forward：前馈网络

一个两层的全连接网络：第一层用 ReLU 激活，第二层不用激活函数。

![FFN 公式](/images/posts/transformer-bert-gpt-llm/17-ffn-formula.png)

- **X**：上一层（多头注意力）传过来的数据；
- **第一层（X·W₁ + b₁）**：W₁、b₁ 是需要学习的权重和偏置；
- **max(0, ·)**：ReLU，输入大于 0 原样保留、小于 0 置 0，给模型引入非线性；
- **第二层（·W₂ + b₂）**：W₂、b₂ 是第二层的权重和偏置。

## 二、BERT：只留编码器

### 2.1 架构

BERT 只使用 Transformer 的**编码器**部分，完全移除了解码器，因此是一个能「双向理解」上下文的表示模型。

![BERT 架构](/images/posts/transformer-bert-gpt-llm/18-bert-architecture.png)

### 2.2 掩码语言建模（MLM）

堆叠起来的编码器很难训练，所以 BERT 用**掩码语言建模（Masked Language Modeling）**来训练：随机盖住一部分词，让模型根据上下文预测被盖住的词——这逼着模型学会「双向」理解上下文。

![掩码语言建模](/images/posts/transformer-bert-gpt-llm/19-bert-mlm.png)

## 三、GPT：只留解码器

### 3.1 架构

GPT 与 BERT 相反，只使用**解码器**部分，是自回归的生成模型。简单说：BERT 是「表示模型」（负责理解），GPT 是「生成模型」（负责续写）。

![GPT 架构](/images/posts/transformer-bert-gpt-llm/20-gpt-architecture.png)

### 3.2 从补全到对话

生成模型本质上是「补全模型」（给定前缀，续写下文）。对它微调后，可以进一步得到：

- **指令模型（instruct model）**：能遵循人类指令；
- **对话模型（chat model）**：能进行多轮对话。

![从补全到指令/对话](/images/posts/transformer-bert-gpt-llm/21-gpt-completion.png)

## 四、LLM：训练范式

### 4.1 语言建模（预训练）

第一步是**预训练**：LLM 在海量互联网文本上训练，只做一件事——预测下一个词，从而学会语法、上下文和语言模式。这一步得到的模型叫**基础模型 / 基座模型（base model）**。

### 4.2 微调（后训练）

第二步是**微调**（也叫后训练）：在已经训练好的基座模型上，用更具体的任务（指令、对话、偏好对齐等）继续训练，让它能更好地遵循人类意图。

![预训练与微调](/images/posts/transformer-bert-gpt-llm/22-llm-pretrain-finetune.png)

## 小结

1. **Transformer** 提供了「自注意力 + 前馈 + 残差归一化」的通用骨架；
2. **BERT** 取编码器 → 双向理解，适合判别式任务；
3. **GPT** 取解码器 → 自回归生成，是今天主流大模型的雏形；
4. **LLM** 把它们放大到海量数据上，走「预训练 → 微调」的范式，得到能对话、能遵循指令的模型。
