# Machine Learning Engineering Online Book

An open collection of methodologies to help with successful training of large language models and multi-modal models.

This is a technical material suitable for LLM/VLM training engineers and operators. That is the content here contains lots of scripts and copy-n-paste commands to enable you to quickly address your needs.

This repo is an ongoing brain dump of my experiences training Large Language Models (LLM) (and VLMs); a lot of the know-how I acquired while training the open-source [BLOOM-176B](https://huggingface.co/bigscience/bloom) model in 2022 and
[IDEFICS-80B](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct) multi-modal model in 2023. Currently, I'm working on developing/training open-source Retrieval Augmented models at [Contextual.AI](https://contextual.ai/).

I've been compiling this information mostly for myself so that I could quickly find solutions I have already researched in the past and which have worked, but as usual I'm happy to share these with the wider ML community.


## Table of Contents

My apologies if the layout is a bit unstable while I'm writing new chapters and gradually re-organizing the content to be more intuitive.

**Part 1. Insights**

1. **[The AI Battlefield Engineering - What You Need To Know](./insights/ai-battlefield.md)**

**Part 2. Key Hardware Components**

1. **[Accelerator](./accelerator/)** - the work horses of ML - GPUs, TPUs, IPUs, FPGAs, HPUs, QPUs, RDUs (WIP)

1. **[Network](./network/)** - intra-node and inter-node connectivity, calculating bandwidth requirements

1. **[IO](./io/)** - local and distributed disks and filesystems

1. **[CPU](./cpu/)** - cpus, affinities (WIP)

1. **[CPU Memory](./cpu-memory/)** - how much CPU memory is enough - the shortest chapter ever.


**Part 3. Performance**

1. **[Fault Tolerance](./fault-tolerance/)**

1. **[Performance](./performance/)**

1. **[Multi-Node networking](./multi-node)**

1. **[Model parallelism](./model-parallelism/)**


**Part 4. Operating**

1. **[SLURM](./slurm/)**

1. **[Training hyper-parameters and model initializations](./hparams/)**

1. **[Instabilities](./instabilities/)**


**Part 5. Development**

1. **[Debugging software and hardware failures](./debug/)**

1. **[And more debugging](https://github.com/stas00/the-art-of-debugging)**

1. **[Reproducibility](./reproducibility/)**

1. **[Tensor precision / Data types](./dtype/)**

1. **[HF Transformers notes](./transformers/)** - making small models, tokenizers, datasets, and other tips


**Part 6. Miscellaneous**

1. **[Resources](./resources/)** - LLM/VLM chronicles




## Shortcuts

Things that you are likely to need to find quickly and often.

Tools:

- [all_reduce_bench.py](./multi-node/all_reduce_bench.py) - a much easier way to benchmark network throughput than nccl-tests.
- [torch-distributed-gpu-test.py](./debug/torch-distributed-gpu-test.py) - a tool to quickly test your inter-node connectivity

Guides:

- [debugging pytorch applications](./debug/pytorch.md) - quick copy-n-paste solutions to resolve hanging or breaking pytorch applications
- [slurm for users](./slurm/users.md) - a slurm cheatsheet and tricks
- [make tiny models/datasets/tokenizers](./transformers/make-tiny-models.md)
- [LLM/VLM chronicles collection](https://github.com/stas00/ml-engineering/tree/master/resources#publicly-available-training-llmvlm-logbooks)


## Gratitude

None of this would have been possible without me being entrusted with doing the specific LLM/VLM trainings I have learned this know-how from. This is a privilege that only a few enjoy due to the prohibitively expensive cost of renting huge ML compute clusters. So hopefully the rest of the ML community will vicariously learn from these notes.

Special thanks go to [Thom Wolf](https://github.com/thomwolf) who proposed that I lead the BLOOM-176B training back when I didn't know anything about large scale training. This was the project that catapulted me into the intense learning process. And, of course, HuggingFace for giving me the opportunity to work full time on BLOOM-176B and later on IDEFICS-80B trainings.

## Contributing

If you found a bug, typo or would like to propose an improvement please don't hesitate to open an [Issue](https://github.com/stas00/ml-engineering/issues) or contribute a PR.


## License

The content of this site is distributed under [Attribution-ShareAlike 4.0 International](./LICENSE-CC-BY-SA).


## My repositories map

✔ **Machine Learning:**
 [ML Engineering](https://github.com/stas00/ml-engineering) |
 [ML ways](https://github.com/stas00/ml-ways) |
 [Porting](https://github.com/stas00/porting)

✔ **Guides:**
 [The Art of Debugging](https://github.com/stas00/the-art-of-debugging)

✔ **Applications:**
 [ipyexperiments](https://github.com/stas00/ipyexperiments)

✔ **Tools and Cheatsheets:**
 [bash](https://github.com/stas00/bash-tools) |
 [conda](https://github.com/stas00/conda-tools) |
 [git](https://github.com/stas00/git-tools) |
 [jupyter-notebook](https://github.com/stas00/jupyter-notebook-tools) |
 [make](https://github.com/stas00/make-tools) |
 [python](https://github.com/stas00/python-tools) |
 [tensorboard](https://github.com/stas00/tensorboard-tools) |
 [unix](https://github.com/stas00/unix-tools)
