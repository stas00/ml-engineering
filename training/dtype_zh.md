# dtype - 中文翻译

# 张量精度/数据类型

以下是在撰写本文时在机器学习中常用的常见数据类型（通常称为 `dtype`）：

浮点格式：
- fp32 - 32位
- tf32 - 19位（NVIDIA Ampere+）
- fp16 - 16位
- bf16 - 16位
- fp8 - 8位（E4M3和E5M2格式）

为了进行视觉比较，请参考以下表示：

![fp32-tf32-fp16-bf16](images/fp32-tf32-fp16-bf16.png)

（来源：[NVIDIA开发者博客](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)）

![fp16-bf16-fp8](images/fp16-bf16-fp8.png)

（来源：[NVIDIA深度学习用户指南](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)）

量化中使用的整数格式：

- int8 - 8位
- int4 - 4位
- int1 - 1位

## 机器学习数据类型演进

最初，机器学习使用的是fp32，但其速度非常慢。

接下来，发明了混合精度技术，结合了fp16和fp32的使用，大大加快了训练速度。

![fp32/fp16 混合精度](images/mixed-precision-fp16.png)

（来源：[NVIDIA开发者博客](https://developer.nvidia.com/blog/video-mixed-precision-techniques-tensor-cores-deep-learning/)）

但是，fp16的稳定性较差，并且训练大型语言模型非常困难。

幸运的是，bf16出现了，它使用相同的混合精度协议取代了fp16。这使得大型语言模型的训练更加稳定。

随后，fp8出现并切换到混合精度，这使得训练速度更快。详见论文：[深度学习的FP8格式](https://arxiv.org/abs/2209.05433)。

为了了解不同格式之间的加速效果，请查看NVIDIA A100 TFLOPS规格表（无稀疏性）：

| 数据类型             | TFLOPS |
| :---                 |    --: |
| FP32                 |   19.5 |
| Tensor Float 32 (TF32) |    156 |
| BFLOAT16 Tensor Core |    312 |
| FP16 Tensor Core     |    312 |
| FP8 Tensor Core      |    624 |
| INT8 Tensor Core     |    624 |

每种后续的数据类型比前一种快约2倍（除了fp32，它比其余的慢很多）。

在混合训练机制的同时，机器学习社区开始提出各种量化方法。可能最好的例子是Tim Dettmers的[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)，它提供了许多4位和8位量化解决方案。Deepspeed团队也有一些[有趣的量化解决方案](https://www.deepspeed.ai/tutorials/model-compression/)。

## TF32

TF32是一种魔法般的数据类型，自Ampere以来在NVIDIA GPU上可用，它允许在比普通fp32 `matmul` 快得多的速度下执行fp32 `matmul`，并且精度损失很小。

以下是A100 TFLOPS的例子（无稀疏性）：

| 数据类型             | TFLOPS |
| :---                 |    --: |
| FP32                 |   19.5 |
| Tensor Float 32 (TF32) |    156 |

如您所见，TF32比FP32快8倍！

默认情况下它是禁用的。要在程序开头启用它，请添加以下代码：

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

有关实际精度损失的更多信息，请参阅[this](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)。

## 使用fp32累加器

每当使用低精度数据类型时，必须小心不要在该数据类型中累积中间结果。

像 `LayerNorm` 这样的操作不能在半精度中进行，否则可能会丢失大量数据。因此，当这些操作正确实现时，它们会在输入的数据类型中高效地进行内部工作，但使用fp32累加寄存器，然后将输出转换为输入的精度。

通常只是累加在fp32中进行，因为如果累加很多低精度数字，结果会损失很大。

以下是一些示例：

1. 减少收集

* fp16: 如果有损失缩放，则可以在fp16中进行

* bf16: 只能在fp32中进行

2. 梯度累积

* 对于fp16和bf16，最好在fp32中进行，但对于bf16来说，这是必需的

3. 优化器步骤/消失梯度

* 当向一个大数添加一个小梯度时，这种加法经常被抵消，因此通常使用fp32主权重和fp32优化状态。

* 当使用[Kahan求和算法](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)或[随机舍入](https://en.wikipedia.org/wiki/Rounding)（在[重新审视BFloat16训练](https://arxiv.org/abs/2010.06192)中引入）时，可以使用f16主权重和优化状态。

例如，参见：[AnyPrecision优化器](https://github.com/pytorch/torchdistx/pull/52)，最新版本可在[这里](https://github.com/facebookresearch/multimodal/blob/6bf3779a064dc72cde48793521a5be151695fc62/torchmultimodal/modules/optimizers/anyprecision.py#L17)找到。

## 训练后更改精度

有时在模型训练后更改精度是可以接受的。

- 使用bf16预训练模型在fp16模式下通常会失败 - 因为fp16能表示的最大数字是64k。对于深入讨论和可能的解决方法，请参阅这个[PR](https://github.com/huggingface/transformers/pull/10956)。

- 使用fp16预训练模型在bf16模式下通常可以工作 - 它在转换时会失去一些性能，但应该可以工作 - 最好在使用之前微调一下。