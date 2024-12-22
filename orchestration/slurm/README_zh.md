# README - 中文翻译

在SLURM环境中工作

除非你很幸运，拥有完全由你控制的专用集群，否则很有可能你需要与其他人共享GPU资源，使用SLURM进行调度。但即便如此，如果你在高性能计算（HPC）环境中训练，并且被分配了一个专用分区，你仍然需要使用SLURM。

SLURM是**Simple Linux Utility for Resource Management**的缩写，现在称为《Slurm 工作负载管理器》。它是用于Linux和类Unix内核的免费开源作业调度程序，被世界上许多超级计算机和计算机集群所使用。

这些章节不会试图详尽地教你SLURM，因为有很多手册可供参考，但会涵盖一些在训练过程中特别有用的细节。

- [SLURM 用户指南](./users.md) - 在SLURM环境中进行训练所需了解的一切。
- [SLURM 管理](./admin.md) - 如果你不幸还需要管理SLURM集群，而不是仅仅使用它，这份文档中有一系列不断增长的解决方案可以帮助你更快地完成任务。
- [性能](./performance.md) - SLURM的性能细节。
- [启动脚本](./launchers) - 如何在SLURM环境中使用`torchrun`、`accelerate`、pytorch-lightning等工具启动训练。