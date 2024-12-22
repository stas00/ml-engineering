# make-tiny-models-tokenizers-datasets - 中文翻译

使用小模型、分词器和数据集进行更快的调试和开发

如果你在调试问题并使用全尺寸模型和分词器进行开发，那么你的工作可能效率不高。不仅解决问题更加困难，程序重启和到达所需点所需的等待时间也会非常长，这可能会大大影响你的积极性和生产力。如果问题得以解决，那也将花费更长时间。

解决方案很简单：

**除非你在测试模型的质量，否则始终使用一个随机的小型模型和可能的小型分词器。**

此外，大型模型通常需要大量的资源，这些资源通常是昂贵的，并且可以使调试过程变得极其复杂。例如，任何调试器都可以处理单个进程，但如果你的模型无法适应并且需要某种形式的并行化（需要多个进程），大多数调试器要么会中断，要么无法提供你需要的信息。理想的开发环境是一个进程，而小型模型可以保证即使是最便宜的消费者GPU也能运行。你甚至可以在没有GPU的情况下使用免费的Google Colab来紧急进行开发。

因此，更新后的机器学习开发口诀变为：

- 模型越大，最终生成的产品越好。
- 模型越小，最终产品的训练开始得越快。

注：最新的研究表明，大并不总是更好，但这足以传达我沟通的重要性。

一旦你的代码运行正常，记得切换到真正的模型来测试生成的质量。但在这种情况下，仍然尝试首先使用产生高质量结果的最小模型。只有当你看到生成的内容大部分正确时，才使用最大的模型来验证你的工作是否完美。

## 创建一个小模型

重要：鉴于其流行程度和设计良好的简单API，我将讨论HF的[`transformers`](https://github.com/huggingface/transformers/)模型。但同样的原则也可以应用于任何其他模型。

简而言之：创建一个HF `transformers`模型的小版本很简单：

1. 获取全尺寸模型的配置对象
2. 缩减隐藏层大小和其他一些参数
3. 使用缩减后的配置创建模型
4. 保存此模型。完成！

注：必须记住，这将生成一个随机模型，所以不要期望其输出有任何质量。

注：这些笔记是基于HF Transformers模型编写的。如果你使用的是不同的建模库，你可能需要调整其中的一些内容。

现在让我们通过实际代码将["google/mt5-small"](https://huggingface.co/google/mt5-small/tree/main)转换为其随机的小型版本。

```python
from transformers import MT5Config, MT5ForConditionalGeneration

mname_from = "google/mt5-small"
mname_very_small = "mt5-tiny-random"

config = MT5Config.from_pretrained(mname_from)

config.update(dict(
    d_model=64,
    d_ff=256,
))
print("新的配置", config)

very_small_model = MT5ForConditionalGeneration(config)
print(f"参数数量 {very_small_model.num_parameters()}")

very_small_model.save_pretrained(mname_very_small)
```

如你所见，这很简单。如果你不需要隐藏层大小至少为64，你可以使其更小。例如尝试8——只需确保注意力头的数量不超过隐藏层大小即可。

请注意，你不需要任何GPU来执行此操作，甚至可以在像[BLOOM-176B](https://huggingface.co/bigscience/bloom)这样的拥有1760亿参数的巨大模型上执行。因为你从未加载原始模型，除了其配置对象之外。

在修改配置之前，你可以转储原始参数并选择缩减更多维度。例如，减少层数会使模型更小且更容易调试。所以这里你可以这样做：

```python
config.update(dict(
    d_model=64,
    d_ff=256,
    d_kv=8,
    num_layers=8,
    num_decoder_layers=8,
    num_heads=4,
    relative_attention_num_buckets=32,
))
```

原始["google/mt5-small"](https://huggingface.co/google/mt5-small/tree/main)模型文件大小为1.2GB。通过上述更改（以及以下部分中解释的词汇缩减）我们将其缩小到126MB。

如果你正在处理多层嵌套配置，你将需要单独更新每个子级的配置对象。例如，在[IDEFICS](https://huggingface.co/HuggingFaceM4/idefics-9b/blob/main/config.json)中，我们有一个主对象和两个嵌套对象：
```python
config
config.perceiver_config
config.vision_config
```
如果你想缩小这个模型，你想更新`config`和`config.vision_config`以使用更小的值：
```python
config.update(dict(
    hidden_size=64,
    intermediate_size=37,
    num_hidden_layers=5,
    num_attention_heads=4,
    max_position_embeddings=64,
    max_sequence_length=64,

))
# 子对象需要直接更新
config.vision_config.update(dict(embed_dim=64))
```
参见[idefics-make-tiny-model.py](tiny-scripts/idefics-make-tiny-model.py)获取一个完整的脚本（我在这里没有添加词汇缩减，因为我只是演示如何更新嵌套配置对象）。