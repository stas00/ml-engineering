# Orchestration

There are many container/accelerator orchestration solutions - many of which are open source.

So far I have been working with SLURM and Kubernetes.

## SLURM

- [SLURM](slurm/) - Simple Linux Utility for Resource Management, which you're guaranteed to find on most HPC environments and typically it's supported by most cloud providers.  It has been around for more than 2 decades

### SLURM on Kubernetes:

These are frameworks allowing you to run SLURM on top of Kubernetes:

- [Slinky](https://github.com/slinkyproject) - a framework from the creators of SLURM.
- [Soperator](https://github.com/nebius/soperator) - from Nebius
- CoreWeave never released their [SUNK](https://www.coreweave.com/products/sunk) open source, so it's only available to its customers.

## Kubernetes

The other most popular orchestrator is Kubernetes:

- [Kubernetes](https://kubernetes.io/) - also known as K8s, is an open source system for automating deployment, scaling, and management of containerized applications. Here is a good [comparison between SLURM and K8s](https://web.archive.org/web/20250324222116/https://www.fluidstack.io/post/is-kubernetes-or-slurm-the-best-orchestrator-for-512-gpu-jobs).

## Other orchestration frameworks

Here are various other less popular than SLURM and K8s, but still very mighty orchestration solutions:

- [dstack](https://github.com/dstackai/dstack) is a lightweight, open-source alternative to Kubernetes & Slurm, simplifying AI container orchestration with multi-cloud & on-prem support. It natively supports NVIDIA, AMD, & TPU.
- [SkyPilot](https://github.com/skypilot-org/skypilot) is a framework for running AI and batch workloads on any infra, offering unified execution, high cost savings, and high GPU availability.
- [OpenHPC](https://github.com/openhpc/ohpc) provides a variety of common, pre-built ingredients required to deploy and manage an HPC Linux cluster including provisioning tools, resource management, I/O clients, runtimes, development tools, containers, and a variety of scientific libraries.
- [run.ai](https://www.run.ai/) - got acquired by NVIDIA and is planned to be open sourced soon.
- [Docker Swarm](https://docs.docker.com/engine/swarm/) is a container orchestration tool.
- [IBM Platform Load Sharing Facility (LSF)](https://www.ibm.com/products/hpc-workload-management) Suites is a workload management platform and job scheduler for distributed high performance computing (HPC).
