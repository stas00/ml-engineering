# README - 中文翻译

## 资源

### 有用的汇总

- @StellaAthena 创建了 [常见LLM设置电子表格](https://docs.google.com/spreadsheets/d/14vbBbuRMEHoqeuMHkTfw3uiZVmyXNuoSp8s-aHvfvZk/edit#gid=0)，在开始新的LLM训练之前，这可能是一个非常有用的资源，因为它告诉你已知的LLM训练有多少。

- 几年前，我开始收集关于模型训练所使用的数据类型的资料：[模型预训练精度数据库（FP16、FP32、BF16）](https://discuss.huggingface.co/t/model-pre-training-precision-database-fp16-fp32-bf16/5671)。该列表只包含少量模型，但如果要进行有关数据类型的调研，它仍然有用。我曾利用这些信息尝试编写 [模型预训练数据类型自动检测](https://github.com/stas00/ml-ways/blob/master/numbers/detect-model-pretrained-in-bf16-fp16-fp32.ipynb)，并在此分享一个相关的 [浮点16与bfloat16数值特性比较](https://github.com/stas00/ml-ways/blob/master/numbers/bfloat16-vs-float16-study.ipynb)。

### 公开可用的LLM/VLM训练日志

LLM/VLM训练的日志和编年史是学习如何处理训练不稳定性以及选择良好超参数的最佳来源之一。

如果您知道任何未列于此处的公开的LLM/VLM训练日志，请务必告知我或通过PR添加。谢谢！

这些记录按年份分组，没有特定顺序。

#### 2021年

- BigScience 在2021年的BigScience BLOOM-108B训练实验：
[编年史](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide/chronicles.md) |
[完整的规范和讨论](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide)
（备份：
[1](https://github.com/stas00/bigscience-backup/blob/master/train/tr8-104B-wide/chronicles.md) |
[2](https://github.com/stas00/bigscience-backup/blob/master/train/tr8-104B-wide))

#### 2022年

- BigScience 的 BLOOM-176B（2022年）：
[前传编年史](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles-prequel.md) |
[编年史](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md) |
[完整的规范和讨论](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/)
（备份：
[1](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles-prequel.md) |
[2](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles.md) |
[3](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/))

- Meta 的 OPT-175B（2022年）：
[训练日志](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles) | [视频](https://www.youtube.com/watch?v=p9IxoSkvZ-M) （备份：[1](https://github.com/stas00/metaseq-backup/tree/main/projects/OPT/chronicles)）

- THUDM 的 GLM-130B（2022年）：[英文版训练日志](https://github.com/THUDM/GLM-130B/blob/main/logs/main-log-en.md) | [中文版](https://github.com/THUDM/GLM-130B/blob/main/logs/main-log.md) （备份：[1](https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log-en.md) | [2](https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log.md)）

#### 2023年

- HuggingFace 的 IDEFICS-80B 多模态（Flamingo 重现）（2023年）：[学习日志](https://github.com/huggingface/m4-logs/blob/master/memos/README.md) | [训练编年史](https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md) （备份：[1](https://github.com/stas00/m4-logs-backup/blob/master/memos/README.md) | [2](https://github.com/stas00/m4-logs-backup/blob/master/tr-190-80b/chronicles.md)）

- BloombergGPT 50B LLM – 见[BloombergGPT：用于金融的大型语言模型](https://arxiv.org/abs/2303.17564)中的C部分

#### 2024年

- [MegaScale：将大规模语言模型训练扩展到超过10,000个GPU](https://arxiv.org/abs/2402.15627) – 论文涵盖了各种训练问题及其解决方案，尽管这些模型是专有的，但同样具有教学和实用价值。

- Imbue 的 [从零开始构建70B模型：基础设施设置和脚本](https://imbue.com/research/70b-infrastructure/) 是一篇非常详细的技术文章，涵盖了他们在训练专有70B参数模型时遇到的各种训练相关问题。

### 硬件设置日志

- Imbue 发布了他们如何搭建一个512节点IB胖树集群并使其运行的详细日志：[从零开始构建70B模型：基础设施设置和脚本](https://imbue.com/research/70b-infrastructure/)，同时开源了他们在过程中创建的 [集群工具](https://github.com/imbue-ai/cluster-health)。