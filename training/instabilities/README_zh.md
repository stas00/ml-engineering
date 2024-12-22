# README - 中文翻译

# 避免、恢复和理解不稳定性

子章节：

* [理解训练损失模式](training-loss-patterns.md) - 峰值类型、发散、顿悟时刻、恢复等。

## 从训练日志中学习

最佳的学习方式是阅读[公开的大型语言模型/视觉语言模型训练日志](../../resources#publicly-available-training-llmvlm-logbooks)，因为在这些日志中你可以看到具体发生了什么以及问题是如何被解决的。

## 标准初始化

正确初始化张量的初始分布对训练的稳定性有着巨大的影响。`std` 值不是固定的，而是取决于隐藏维度的大小。

这在我们BLOOM 104B实验之前被证明是一个非常关键的设置，直到我们发现Megatron-LM中的默认 `--init-method-std` 值为0.02对我们模型来说太大了，我们才突破了最初的几千次迭代。

我们参考了以下两个来源：

1. 论文《Transformers without Tears》https://arxiv.org/abs/1910.05895 规定：`sqrt(2/(NHIDDEN*5))`

2. 530B训练论文 https://arxiv.org/abs/2201.11990 使用了更小的初始化公式：`sqrt(1/(NHIDDEN*3))`

我们决定采用530B的那个公式，因为它导致了一个更小的初始化值。

为了便于比较这两个公式，它们可以重写为：
1. `sqrt(0.4000/NHIDDEN)`
2. `sqrt(0.3333/NHIDDEN)`

因此，对于 `NHIDDEN=14336`，计算结果为 `sqrt(1/(14336*3)) = 0.00482`，这就是我们所使用的值。这当然不是我们在BLOOM-176B训练过程中没有遇到稳定性问题的唯一原因，但我认为这是其中的关键原因之一。

## 数值不稳定性

处理低精度数字时，某些数学运算可能会不稳定。

例如，请参阅这个非常有趣的[PyTorch数值稳定性指南](https://pytorch.org/docs/stable/notes/numerical_accuracy.html)。

现在让我们来看一个这种概念的实际应用示例。

在使用fp16混合精度进行104B训练实验期间，Corby Rosset 提出了以下改进以使自注意力更加稳定。

具体来说，这条[链接](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/c839a8aa30731f71b3738d56009be9668508e366/megatron/model/transformer.py#L303)显示 `norm_factor` 可能在Query * Key矩阵乘法之后被相乘。如果Q和K的维度非常大，输出可能会爆炸，并且 `norm_factor` 将无法挽救这种情况。

建议：将 `norm_factor` 向内移动，以便在矩阵乘法之前缩放Q和K：
``` 
matmul_result = torch.baddbmm(
    matmul_result,
    1.0/math.sqrt(self.norm_factor) * query_layer.transpose(0, 1),   # [b * np, sq, hn]
    1.0/math.sqrt(self.norm_factor) * key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
    beta=0.0 if alibi is None else 1.0, alpha=1.0)

# 改变视图到 [b, np, sq, sk]
attention_scores = matmul_result.view(*output_size)
```

为了使操作在数学上等价，向内移动 `norm_factor` 需要再次取平方根
如果n是一个标量，A和B是矩阵：
``` 
n * (A dot B) === (sqrt(n) * A) dot (sqrt(n) * B)
```

现在A和B的维度可以显著增大。

对于CUDA内核编写者，在撰写本文时[CuBlas](https://docs.nvidia.com/cuda/cublas/index.html)的 `GemmStridedBatchedEx` 存在一个类似的问题。它定义为：

``` 
C+i*strideC=αop(A+i*strideA)op(B+i*strideB)+β(C+i*strideC)，对于 i ∈[0,batchCount−1]
```

问题是 `alpha` 在矩阵-矩阵乘法完成后才相乘，因此可能导致不稳定性。

## 数据批次与模型参数状态的“不良”组合

PaLM团队在训练较大的模型时观察到数十次“高度不规则间隔”的损失峰值。尽管他们未能追踪到根本原因，但他们通过从较早的检查点重新开始并跳过潜在有问题的数据批次来缓解了这个问题。[第5.1节 训练不稳定性](https://arxiv.org/pdf/2204.02311.pdf)

## Adam的时间域相关性发散

论文《Adam在大规模机器学习中的不稳定性理论》对训练高达546B参数的大规模语言模型时的发散峰值进行了严格研究，并提出时间域相关性会导致Adam发散。这可能是由于epsilon值不够小，导致梯度估计组件接近epsilon。

在第7.1节中，他们提出了实际建议，其中最有趣的一条是将epsilon设置为0，并可能处理除零条件。