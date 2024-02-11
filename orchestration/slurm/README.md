# Working in SLURM Environment

Unless you're lucky and you have a dedicated cluster that is completely under your control chances are that you will have to use SLURM to timeshare the GPUs with others. But, often, if you train at HPC, and you're given a dedicated partition you still will have to use SLURM.

The SLURM abbreviation stands for: **Simple Linux Utility for Resource Management** - though now it's called
The Slurm Workload Manager. It is a free and open-source job scheduler for Linux and Unix-like kernels, used by many of the world's supercomputers and computer clusters.

These chapters will not try to exhaustively teach you SLURM as there are many manuals out there, but will cover some specific nuances that are useful to help in the training process.

- [SLURM For Users](./users.md) - everything you need to know to do your training in the SLURM environment.
- [SLURM Administration](./admin.md) - if you're unlucky to need to also manage the SLURM cluster besides using it, there is a growing list of recipes in this document to get things done faster for you.
- [Performance](./performance.md) - SLURM performance nuances.
- [Launcher scripts](./launchers) - how to launch with `torchrun`, `accelerate`, pytorch-lightning, etc. in the SLURM environment
