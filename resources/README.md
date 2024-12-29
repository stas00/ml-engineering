# Resources

## Useful compilations

- [@StellaAthena](https://github.com/StellaAthena) created the [Common LLM Settings spreadsheet](https://docs.google.com/spreadsheets/d/14vbBbuRMEHoqeuMHkTfw3uiZVmyXNuoSp8s-aHvfvZk/edit#gid=0) which can be a super-useful resource when you're about to embark on a new LLM training - as it tells you how many known LLM trainings were created.

- A few years back I started compiling information on [which dtype the models were trained in](https://discuss.huggingface.co/t/model-pre-training-precision-database-fp16-fp32-bf16/5671) - it only contains a handful of models but if you're doing a research on dtypes it can still be useful. I was using this information to try and write [a model pretraining dtype auto-detection](https://github.com/stas00/ml-ways/blob/master/numbers/detect-model-pretrained-in-bf16-fp16-fp32.ipynb) and here is a related [float16 vs bfloat16 numerical properties comparison](https://github.com/stas00/ml-ways/blob/master/numbers/bfloat16-vs-float16-study.ipynb).

## Publicly available training LLM/VLM logbooks

Logbooks and chronicles of training LLM/VLM are one of the best sources to learn from about dealing with training instabilities and choosing good hyper parameters.

If you know of a public LLM/VLM training logbook that is not on this list please kindly let me know or add it via a PR. Thank you!

The listing is in no particular order other than being grouped by the year.

### 2021

- BigScience pre-BLOOM 108B training experiments (2021):
[chronicles](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide/chronicles.md) |
[the full spec and discussions](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide)
(backup:
[1](https://github.com/stas00/bigscience-backup/blob/master/train/tr8-104B-wide/chronicles.md) |
[2](https://github.com/stas00/bigscience-backup/blob/master/train/tr8-104B-wide))


### 2022

- BigScience BLOOM-176B (2022):
[chronicles-prequel](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles-prequel.md) |
[chronicles](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md) |
[the full spec and discussions](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/)
(backup:
[1](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles-prequel.md) |
[2](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles.md) |
[3](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/))

- Meta OPT-175B (2022):
 [logbook](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles) | [Video](https://www.youtube.com/watch?v=p9IxoSkvZ-M) (backup: [1](https://github.com/stas00/metaseq-backup/tree/main/projects/OPT/chronicles))

- THUDM GLM-130B (2022): [en logbook](https://github.com/THUDM/GLM-130B/blob/main/logs/main-log-en.md) | [Mandarin version](https://github.com/THUDM/GLM-130B/blob/main/logs/main-log.md) (backup:  [1](https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log-en.md) | [2](https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log.md))


### 2023

- HuggingFace IDEFICS-80B multimodal (Flamingo repro) (2023): [Learning log](https://github.com/huggingface/m4-logs/blob/master/memos/README.md) | [Training Chronicles](https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md) (backup: [1](https://github.com/stas00/m4-logs-backup/blob/master/memos/README.md) | [2](https://github.com/stas00/m4-logs-backup/blob/master/tr-190-80b/chronicles.md))

- BloombergGPT 50B LLM - section C in [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)


### 2024

- [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://arxiv.org/abs/2402.15627) - the paper covers various training issues and their resolution - albeit on models that are proprietary yet just as instructional/useful.

- Imbue's [From bare metal to a 70B model: infrastructure set-up and scripts](https://imbue.com/research/70b-infrastructure/) very detailed technical post covers many training-related issues that they had to overcome while training a proprietary 70B-param model.




## Hardware setup logbooks

- Imbue published a detailed log of how they have set up a 512-node IB-fat-tree cluster and made it to work: [From bare metal to a 70B model: infrastructure set-up and scripts](https://imbue.com/research/70b-infrastructure/), they also open-sourced the [cluster tooling](https://github.com/imbue-ai/cluster-health) they created in the process.

- SemiAnalysis published a great detailed writeup about [what it takes to set up a Neocloud cluster](https://semianalysis.com/2024/10/03/ai-neocloud-playbook-and-anatomy/).
