# Machine Learning Engineering Open Book

This is an open collection of methodologies, tools and step by step instructions to help with successful training and fine-tuning of large language models and multi-modal models and their inference.

This is a technical material suitable for LLM/VLM training engineers and operators. That is the content here contains lots of scripts and copy-n-paste commands to enable you to quickly address your needs.

This repo is an ongoing brain dump of my experiences training Large Language Models (LLM) (and VLMs); a lot of the know-how I acquired while training the open-source [BLOOM-176B](https://huggingface.co/bigscience/bloom) model in 2022 and [IDEFICS-80B](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct) multi-modal model in 2023, and RAG models at [Contextual.AI](https://contextual.ai/) in 2024.

I've been compiling this information mostly for myself so that I could quickly find solutions I have already researched in the past and which have worked, but as usual I'm happy to share these notes with the wider ML community.


## Table of Contents


**Part 1. Insights**

1. **[The AI Battlefield Engineering](./insights/ai-battlefield.md)** - what you need to know in order to succeed.

1. **[How to Choose a Cloud Provider](./insights/how-to-choose-cloud-provider.md)** - these questions will empower you to have a successful compute cloud experience.

**Part 2. Hardware**

1. **[Compute](compute)** - accelerators, CPUs, CPU memory.

1. **[Storage](storage)** - local, distributed and shared file systems.

1. **[Network](network)** - intra- and inter-node networking.


**Part 3. Orchestration**

1. **[Orchestration Systems](orchestration)** - managing containers and resources
1. **[SLURM](orchestration/slurm)** - Simple Linux Utility for Resource Management


**Part 4. Training**

1. **[Training](training)** - model training-related guides


**Part 5. Inference**

1. **[Inference](inference)** - model inference insights


**Part 6. Development**

1. **[Debugging and Troubleshooting](debug)** - how to debug easy and difficult issues

1. **[And more debugging](https://github.com/stas00/the-art-of-debugging)**

1. **[Testing](testing)** - numerous tips and tools to make test writing enjoyable


**Part 7. Miscellaneous**

1. **[Resources](resources)** - LLM/VLM chronicles


## Updates

I announce any significant updates on my twitter channel [https://twitter.com/StasBekman](https://twitter.com/StasBekman).

## PDF version

Download the [PDF](https://huggingface.co/stas/ml-engineering-book/resolve/main/Stas%20Bekman%20-%20Machine%20Learning%20Engineering.pdf?download=true) version of the book.

I will try to rebuild it once a week or so, but if you want the latest, the instructions for building are [here](build).

Thanks to HuggingFace for giving me permission to host my book's PDF at the [HF hub](https://huggingface.co/).

## Discussions

If you want to discuss something related to ML engineering this repo has the [community discussions](https://github.com/stas00/ml-engineering/discussions) available - so please don't hesitate to share your experience or start a new discussion about something you're passionate about.

## Key comparison tables

High end accelerators:

- [Theoretical accelerator TFLOPS](compute/accelerator#tflops-comparison-table)
- [Accelerator memory size and speed](compute/accelerator#accelerator-memory-size-and-speed)

Networks:

- [Theoretical inter-node speed](network#inter-node-networking)
- [Theoretical intra-node speed](network#intra-node-networking)

## Shortcuts

Things that you are likely to need to find quickly and often.

Tools:

- [all_reduce_bench.py](network/benchmarks/all_reduce_bench.py) - a much easier way to benchmark network throughput than nccl-tests.
- [torch-distributed-gpu-test.py](debug/torch-distributed-gpu-test.py) - a tool to quickly test your inter-node connectivity
- [mamf-finder.py](compute/accelerator/benchmarks/mamf-finder.py) - what is the actual TFLOPS measurement you can get from your accelerator.

Guides:

- [debugging pytorch applications](debug/pytorch.md) - quick copy-n-paste solutions to resolve hanging or breaking pytorch applications
- [slurm for users](orchestration/slurm/users.md) - a slurm cheatsheet and tricks
- [make tiny models/datasets/tokenizers](debug/make-tiny-models-tokenizers-datasets.md)
- [LLM/VLM chronicles collection](resources#publicly-available-training-llmvlm-logbooks)


## Gratitude

None of this would have been possible without me being entrusted with doing the specific LLM/VLM trainings I have learned the initial know-how from. This is a privilege that only a few enjoy due to the prohibitively expensive cost of renting huge ML compute clusters. So hopefully the rest of the ML community will vicariously learn from these notes.

Special thanks go to [Thom Wolf](https://github.com/thomwolf) who proposed that I lead the BLOOM-176B training back when I didn't know anything about large scale training. This was the project that catapulted me into the intense learning process. And, of course, HuggingFace for giving me the opportunity to work full time on BLOOM-176B and later on IDEFICS-80B trainings.

Recently, I continued expanding my knowledge and experience while training models and building scalable training/inference systems at [Contextual.AI](https://contextual.ai/) and I'm grateful for that opportunity to Aman and Douwe.

I'd also like to thank the numerous [contributors](contributors.md) who have been making this text awesome and error-free.

## Contributing

If you found a bug, typo or would like to propose an improvement please don't hesitate to open an [Issue](https://github.com/stas00/ml-engineering/issues) or contribute a PR.


## License

The content of this site is distributed under [Attribution-ShareAlike 4.0 International](LICENSE-CC-BY-SA).


## Citation

```bibtex
@misc{bekman2024mlengineering,
  author = {Bekman, Stas},
  title = {Machine Learning Engineering Open Book},
  year = {2023-2024},
  publisher = {Stasosphere Online Inc.},
  journal = {GitHub repository},
  url = {https://github.com/stas00/ml-engineering}
}
```

## My repositories map

✔ **Machine Learning:**
 [ML Engineering Open Book](https://github.com/stas00/ml-engineering) |
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
