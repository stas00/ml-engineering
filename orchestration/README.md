# Orchestration

There are many container/accelerator orchestration solutions - many of which are open source.

So far I have been working with SLURM:

- [SLURM](slurm/) - Simple Linux Utility for Resource Management, which you're guaranteed to find on most HPC environments and typically it's supported by most cloud solutions.

The other popular orchestrator is Kubernetes:

- [Kubernetes](https://kubernetes.io/) - also known as K8s, is an open source system for automating deployment, scaling, and management of containerized applications. Here is a good [comparison between SLURM and K8s](https://www.fluidstack.io/post/is-kubernetes-or-slurm-the-best-orchestrator-for-512-gpu-jobs).

Here are various emerging new orchestration solutions:

- [dstack](https://github.com/dstackai/dstack) is a lightweight, open-source alternative to Kubernetes & Slurm, simplifying AI container orchestration with multi-cloud & on-prem support. It natively supports NVIDIA, AMD, & TPU.
- [run.ai](https://github.com/run-ai/genv) - GPU environment and cluster management with LLM support - and more of it is planned to be open source.
- [OpenHPC](https://github.com/openhpc/ohpc) provides a variety of common, pre-built ingredients required to deploy and manage an HPC Linux cluster including provisioning tools, resource management, I/O clients, runtimes, development tools, containers, and a variety of scientific libraries.
