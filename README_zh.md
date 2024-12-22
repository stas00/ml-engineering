# README - 中文翻译

# 机器学习工程开放书籍

这是一份开放的集合，包含方法论、工具以及逐步指导，以帮助成功训练大型语言模型和多模态模型及其推理。

这是适合LLM/VLM训练工程师和技术人员的技术资料。这里包含大量的脚本和可复制粘贴的命令，以便您能快速应对需求。

这个仓库是我对训练大型语言模型（LLM）（以及VLMs）的经验总结；其中很多知识是我在2022年训练开源的[BLOOM-176B](https://huggingface.co/bigscience/bloom)模型，2023年训练[IDEFICS-80B](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct)多模态模型，以及在[Contextual.AI](https://contextual.ai/)于2024年训练RAG模型的过程中积累的。

我一直将这些信息整理起来主要是为了自己，这样我就可以快速找到过去已经研究过并行且有效的解决方案，但正如往常一样，我很乐意与更广泛的机器学习社区分享这些笔记。

## 目录


**第一部分 洞察**

1. **[人工智能战场工程](./insights/ai-battlefield.md)** - 成功所需了解的内容

**第二部分 硬件**

1. **[计算](compute)** - 加速器、CPU、CPU内存。

1. **[存储](storage)** - 本地、分布式和共享文件系统。

1. **[网络](network)** - 节点间和节点内的网络连接。


**第三部分 编排**

1. **[SLURM](orchestration/slurm)** - 主要的编排环境


**第四部分 训练**

1. **[训练](training)** - 与模型训练相关的指南


**第五部分 推理**

1. **[推理](inference)** - 模型推理洞察


**第六部分 开发**

1. **[调试与故障排除](debug)** - 如何调试简单和复杂问题

1. **[更多调试](https://github.com/stas00/the-art-of-debugging)**

1. **[测试](testing)** - 多种技巧和工具，使测试编写变得愉快


**第七部分 杂项**

1. **[资源](resources)** - LLM/VLM 漫游记


## 更新

我会在我的推特频道[https://twitter.com/StasBekman](https://twitter.com/StasBekman)上发布任何重大更新。

## PDF版本

下载书籍的[PDF](https://huggingface.co/stas/ml-engineering-book/resolve/main/Stas%20Bekman%20-%20Machine%20Learning%20Engineering.pdf?download=true)版本。

我会尝试每周更新一次，但如果您需要最新的版本，请参阅[这里](build)的构建说明。

感谢HuggingFace允许我在[HuggingFace hub](https://huggingface.co/)上托管我的书籍PDF。

## 讨论

如果您想讨论与机器学习工程相关的内容，此仓库提供了[社区讨论](https://github.com/stas00/ml-engineering/discussions)，所以请随时分享您的经验或开始讨论您感兴趣的话题。

## 关键比较表

高端加速器：

- [理论加速器TFLOPS](compute/accelerator#tflops-comparison-table)
- [加速器内存大小和速度](compute/accelerator#accelerator-memory-size-and-speed)

网络：

- [理论节点间速度](network#intra-node-networking)
- [理论节点内速度](network#inter-node-networking)

## 快捷方式

您可能需要快速频繁查找的内容。

工具：

- [all_reduce_bench.py](network/benchmarks/all_reduce_bench.py) - 一种比nccl-tests更容易的网络吞吐量基准测试方法。
- [torch-distributed-gpu-test.py](debug/torch-distributed-gpu-test.py) - 一个快速测试节点间连接的工具

指南：

- [调试PyTorch应用程序](debug/pytorch.md) - 快速复制粘贴解决挂起或中断的PyTorch应用程序的方法
- [SLURM用户指南](orchestration/slurm/users.md) - SLURM速查表和技巧
- [创建小型模型/数据集/分词器](debug/make-tiny-models-tokenizers-datasets.md)
- [LLM/VLM 漫游记集合](resources#publicly-available-training-llmvlm-logbooks)


## 感激

如果没有被委托进行特定的LLM/VLM培训，我就不可能获得最初的专门知识。由于租用巨大的机器学习计算集群的成本高得令人望而却步，因此只有少数人能享受到这种特权。希望机器学习社区的其他人能够从这些笔记中获得间接的学习体验。

特别感谢[Thom Wolf](https://github.com/thomwolf)，他在我对大规模训练一无所知时提议让我领导BLOOM-176B的训练。这个项目使我进入了高强度的学习过程。当然，还要感谢HuggingFace给了我全职从事BLOOM-176B和后来IDEFICS-80B训练的机会。

最近，我在[Contextual.AI](https://contextual.ai/)训练模型和构建可扩展的训练/推理系统的过程中继续扩展我的知识和经验，对此机会我表示感激，感谢Aman和Douwe。

我还想感谢众多[贡献者](contributors.md)，他们让这本书变得出色且无误。

## 贡献

如果您发现了错误、拼写错误或有任何改进建议，请毫不犹豫地打开[问题](https://github.com/stas00/ml-engineering/issues)或提交PR。

## 许可证

本内容按照[署名-相同方式共享 4.0 国际](LICENSE-CC-BY-SA)许可协议分发。

## 引用

```bibtex
@misc{bekman2024mlengineering,
  author = {Bekman, Stas},
  title = {机器学习工程开放书籍},
  year = {2023-2024},
  publisher = {Stasosphere Online Inc.},
  journal = {GitHub仓库},
  url = {https://github.com/stas00/ml-engineering}
}
```