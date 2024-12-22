# README - 中文翻译

## 模型并行性

## 并行性概述

在现代机器学习中，各种并行化方法被用于：

1. 克服GPU内存限制。例如：
   - 训练非常大的模型 - 例如，t5-11b仅模型参数就达到45GB
   - 训练非常长的序列 - 例如，
2. 显著加快训练速度 - 将原本需要一年才能完成的训练缩短到几小时

我们首先深入讨论各种一维并行化技术及其优缺点，然后看看如何将它们组合成二维和三维并行化以实现更快的训练，并支持更大的模型。还将介绍其他强大的替代方法。

虽然主要概念可能适用于任何其他框架，但本文侧重于基于PyTorch的实现。

主要有两种方法可以实现比加速器内存更大的模型训练：
1. 三维并行化 - 非常网络高效，但可能对建模代码有较大侵入性，并且需要大量工作才能正确实现
2. ZeRO并行化 - 不是很网络高效，但对建模代码几乎不需要更改且非常容易实现。

## 可扩展性概念

以下是本文件后面将详细描述的主要概念的简要说明。

1. [数据并行性](#数据并行性)（DP） - 多次复制相同的设置，每个设置处理数据的一部分。处理在并行进行，并且在每个训练步骤结束时所有设置同步。
2. [张量并行性](#张量并行性)（TP） - 每个张量拆分成多个块，因此不是整个张量都驻留在单个GPU上，而是每个张量片段驻留在指定的GPU上。处理过程中每个片段在不同的GPU上独立并行处理，最终步骤同步结果。这可以称为水平并行性，因为分割发生在水平层面。
3. [流水线并行性](#流水线并行性)（PP） - 模型垂直（层级别）分布在多个GPU上，因此只有模型的一两个层次放置在一个GPU上。每个GPU并行处理流水线的不同阶段，并处理小批量数据的一部分。
4. [零冗余优化器](#零数据并行性)（ZeRO） - 类似于TP，也会对张量进行分片，除了整个张量会在前向或后向计算时重新构建，因此不需要修改模型。它还支持多种卸载技术以补偿有限的GPU内存。分片DDP是ZeRO的基本概念，被各种其他ZeRO实现使用。
5. [序列并行性](#序列并行性) - 对于长输入序列的训练需要大量的GPU内存。该技术将单个序列的处理拆分到多个GPU上。
6. [专家并行性](#专家并行性) - 可以将混合专家（MoE）分区，使每个专家都有一个专用的GPU（或几个）。

这篇论文的引言部分可能是我找到的关于最常见的并行化技术的最佳解释之一：[广度优先流水线并行性](https://arxiv.org/abs/2211.05953)。

## 数据并行性

### DDP

大多数用户只需使用两个GPU就可以通过`DataParallel`（DP）和`DistributedDataParallel`（DDP）享受到增加的训练速度，这两个功能在Pytorch中几乎是不费吹灰之力就能使用的。

详情请参阅[DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

### ZeRO数据并行性

ZeRO驱动的数据并行性（ZeRO-DP）如下面来自这篇[博客文章](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)的图表所示
![DeepSpeed-Image-1](images/parallelism-zero.png)

虽然很难理解，但实际上这个概念很简单。这只是一个普通的`DataParallel`（DP），只是每个GPU存储的不是完整的模型参数、梯度和优化器状态，而是只存储一部分。运行时当需要特定层的完整层参数时，所有GPU会同步，互相提供缺失的部分——就是这样。

考虑一个简单的包含3层的模型，每层有3个参数：
```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```

La层有参数a0, a1 和 a2。

如果有3个GPU，分片DDP（= Zero-DP）将模型这样分配到3个GPU上：

``` 
GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2
```

从某种意义上说，这与张量并行性中的水平切片相同，如果你想象典型的深度神经网络图。垂直切片是将整个层组放在不同的GPU上。但这只是起点。

现在每个GPU都会像在DP中一样获得一个通常的小批量数据：
``` 
x0 => GPU0
x1 => GPU1
x2 => GPU2
```

输入没有被修改——它们认为自己会被正常模型处理。

首先，输入到达La层。

让我们专注于GPU0：x0需要a0, a1, a2参数来进行前向路径，但GPU0只有a0——它从GPU1获取a1，从GPU2获取a2，从而将模型的所有部分组合在一起。

同时，GPU1获取小批量数据x1，它只有a1，但需要a0和a2参数，所以它从GPU0和GPU2获取这些参数。

同样的事情也发生在GPU2上，它获取输入x2。它从GPU0和GPU1获取a0和a1，再用它的a2重构完整的张量。

所有3个GPU都重构了完整的张量，并进行了前向操作。

一旦计算完成，不再需要的数据就会被丢弃——它只在计算过程中使用。重构是通过预取有效地完成的。

整个过程会重复进行，先从前向Lb开始，再到Lc，然后反向从Lc到Lb再到La。

对我来说，这听起来像是一种高效的背包分配策略：

1. 人A携带帐篷
2. 人B携带炉子
3. 人C携带斧头

现在每天晚上他们分享各自拥有的东西，并从别人那里获取缺少的东西，第二天早上他们整理好各自的装备继续前进。这就是分片DDP/Zero DP。

将其与每个人必须携带自己的帐篷、炉子和斧头的简单策略相比，后者将更加低效。这是Pytorch中的DataParallel（DP和DDP）。

在阅读有关这一主题的文献时，你可能会遇到以下同义词：分片、分区。

如果你仔细观察ZeRO划分模型权重的方式——它看起来非常类似于稍后讨论的张量并行性。这是因为它分片/分区每个层的权重，而不是接下来讨论的垂直模型并行性。

ZeRO-DP阶段1+2+3的实现：
- [DeepSpeed](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch](https://pytorch.org/docs/stable/fsdp.html)（最初在[FairScale](https://github.com/facebookresearch/fairscale/)中实现，后来被上游集成到PyTorch核心）
- [torchtitan](https://github.com/pytorch/torchtitan)

Deepspeed ZeRO集成：
- [HF Trainer集成](https://huggingface.co/docs/transformers/main_classes/deepspeed)
- [Accelerate](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html)
- [Determined.AI](https://docs.determined.ai/latest/model-dev-guide/api-guides/apis-howto/deepspeed/_index.html)

FSDP集成：
- [HF Trainer集成](https://huggingface.co/docs/transformers/main/en/fsdp)
- [Accelerate](https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html)
- [torchtitan](https://github.com/pytorch/torchtitan)

重要论文：

Deepspeed和ZeRO总体：
- [ZeRO：优化内存以训练万亿参数模型](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload：使数十亿规模模型训练民主化](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity：突破GPU内存墙以实现极端规模深度学习](https://arxiv.org/abs/2104.07857)
- [ZeRO++：为巨型模型训练提供极其高效的集体通信](https://arxiv.org/abs/2306.10209)
- [DeepSpeed Ulysses：系统优化以实现极端长序列转换器模型训练](https://arxiv.org/abs/2309.14509)
- [AMSP：减少ZeRO的通信开销以实现高效的LLM训练](https://arxiv.org/abs/2311.00257)

PyTorch：
- [PyTorch FSDP：全分片数据并行化的经验](https://arxiv.org/abs/2304.11277)

主要DeepSpeed ZeRO资源：
- [项目的github](https://github.com/microsoft/deepspeed)
- [使用文档](https://www.deepspeed.ai/getting-started/)
- [API文档](https://deepspeed.readthedocs.io/en/latest/index.html)
- [博客文章](https://www.microsoft.com/en-us/research/search/?q=deepspeed)

#### 克服巨大的全局批次大小问题

如果你使用1024个加速器，每个加速器上的碎片会很小，会有大量的空闲内存用于微批次大小（MBS）。假设你可以容纳MBS=32，那么你最终会有GBS=32k——这很可能不是你想要的。

因此，你可能需要部署[张量并行性](#张量并行性)，这不容易实现，或者通常更简单的方法是部署[序列并行性](#序列并行性)。我还没有实际尝试过，但到目前为止我了解到的是对于：

- Deepspeed ZeRO 使用 [Deepspeed-Ulysses](#deepspeed-ulysses-sp)
- FSDP 使用 [Paged Ring Attention](https://github.com/lucidrains/ring-attention-pytorch)（[论文](https://arxiv.org/abs/2402.08268)）

请注意，这可能不会像[张量并行性](#张量并行性)那样高效——但目前我还未了解实际的额外开销。

#### ZeRO与多个副本

默认情况下，ZeRO使用所有GPU创建一个单一模型副本——这意味着模型分布在所有GPU上。这导致了各种限制，比如：

1. 全局批次大小不灵活——总是总GPU数乘以微批次大小的函数——在大型集群中可能会导致巨大的全局批次大小，这可能对有效的收敛有害。当然，可以使用极小的微批次大小来控制全局批次大小，但这会导致每个GPU上的矩阵较小，从而效率降低。
2. 由于较慢的节点间网络定义了整体通信速度，因此无法利用更快的节点内网络。

[ZeRO++]通过引入ZeRO的层次权重分区（hpZ）解决了第二个限制。在这种方法中，而不是将整个模型权重分布到所有GPU上，每个模型副本被限制在一个节点内。这增加了总节点数的内存使用，但现在两次`all_gather`调用是在更快的节点内连接上进行的。只有`reduce_scatter`用于聚合和重新分发梯度是在较慢的节点间网络上执行的。

第一个限制并没有真正解决，因为总的全局批次大小保持不变，但由于每个副本更有效，并且额外的内存压力可能会限制每个GPU上的可能微批次大小，这应该会提高系统的吞吐量。

PyTorch FSDP在[shardingStrategy.HYBRID_SHARD](https://pytorch.org/docs/stable/fsdp.html)中实现了这个功能。

论文：

- [ZeRO++：为巨型模型训练提供极其高效的集体通信](https://arxiv.org/abs/2306.10209)
- [PyTorch FSDP：全分片数据并行化的经验](https://arxiv.org/abs/2304.11277)


#### ZeRO的变体

提出ZeRO协议修改的已发表论文：

- [MiCS：在公共云上训练巨大模型的近乎线性扩展](https://arxiv.org/abs/2205.00119)（2022年）
- [AMSP：通过高级模型状态分区超级扩展LLM训练](https://arxiv.org/abs/2311.00257)（2023年）