# users - 中文翻译

## SLURM 用户指南

## 快速开始

只需复制此 [example.slurm](./example.slurm)，并根据需要进行调整。

## SLURM 分区

在这个文档中，我们将使用以下两个集群名称的例子：

- `dev`
- `prod`

要了解节点的主机名及其可用性，请使用：

```bash
sinfo -p dev
sinfo -p prod
```

Slurm 配置位于 `/opt/slurm/etc/slurm.conf`。

要查看所有分区的配置：

```bash
scontrol show partition
```

## 资源分配等待时间

```bash
squeue -u `whoami` --start
```

将显示任何待处理作业预计何时开始运行。

如果其他人在这段预留时间结束前取消了他们的预定，这些作业可能会提前开始。

## 通过依赖关系请求分配

要在当前正在运行的任务结束后（无论任务是否仍在运行或尚未开始）安排新任务，请使用依赖机制。通过告诉 `sbatch` 在当前运行的任务成功完成后启动新任务来实现：

```bash
sbatch --dependency=CURRENTLY_RUNNING_JOB_ID tr1-13B-round1.slurm
```

使用 `--dependency` 可能会比使用 `--begin` 导致更短的等待时间。因为如果 `--begin` 指定的时间允许即使有几分钟的延迟，调度器也可能会在指定时间之前开始一些优先级较低的其他任务。

## 在预定时间进行资源分配

要推迟分配的时间，请使用：
```bash
salloc --begin HH:MM MM/DD/YY
```

对于 `sbatch` 也是同样的操作。

这将简单地将作业放入队列中，就像您在该时间执行此命令一样。如果该时间有资源可用，分配将立即给出。否则它将被排队。

有时相对开始时间很有用。可以使用其他格式。例如：

```bash
--begin now+2hours
--begin=16:00
--begin=now+1hour
--begin=now+60  # 默认以秒为单位
--begin=2010-01-20T12:34:00
```

时间单位可以是 `seconds`（默认）、`minutes`、`hours`、`days` 或 `weeks`：

## 无时间限制的预分配节点

这对于运行重复的交互式实验非常有用——这样就不需要等待分配来继续。因此策略是在较长的时间内预先分配资源，然后使用这个分配多次运行交互式 `srun` 任务。

设置 `--time` 为您所需的窗口（例如 6 小时）：
```bash
salloc --partition=dev --nodes=1 --ntasks-per-node=1 --cpus-per-task=96 --gres=gpu:8 --time=6:00:00 bash
salloc: Pending job allocation 1732778
salloc: job 1732778 queued and waiting for resources
salloc: job 1732778 has been allocated resources
salloc: Granted job allocation 1732778
```
现在使用此保留的节点多次运行任务，通过传递 `salloc` 的作业 ID：
```bash
srun --jobid $SLURM_JOBID --pty bash
```
如果从通过 `salloc` 启动的 `bash` 中运行，则可以从另一个 shell 开始，但需要明确设置 `--jobid`。

如果此 `srun` 任务超时或手动退出，您可以在此相同的保留节点上重新启动它。

`srun` 当然可以直接调用实际的训练命令，而不仅仅是 `bash`。

重要：当分配一个节点时，分配的 shell 不在该节点上（它永远不在）。您需要找到节点的主机名（在分配时报告或通过 `squeue` 和 `ssh` 查找）。

完成时，通过退出 `salloc` 启动的 shell 或使用 `scancel JOBID` 释放资源。

这个保留的节点在整个分配期间都会计入小时使用量，所以用完后尽快释放。

实际上，如果这只是单个节点，那么甚至不需要使用 `salloc`，而是直接使用 `srun`，这将同时分配和提供您要使用的 shell：
```bash
srun --pty --partition=dev --nodes=1 --ntasks=1 --cpus-per-task=96 --gres=gpu:8 --time=60 bash
```

## 超线程

默认情况下，如果 CPU 已启用 [超线程](https://zh.wikipedia.org/wiki/%E8%B6%85%E7一线程)（HT），SLURM 将使用它。如果您不想使用 HT，必须指定 `--hint=nomultithread`。

注脚：HT 是英特尔特定的命名方式，通用概念是同时多线程（SMT）

例如，在具有每个节点 2 个 CPU，每个 CPU 24 核心且每个核心 2 个超线程的集群中，总共有 96 个超线程或 48 个 CPU 核心可用。因此要充分利用该节点，您需要配置：

```bash
#SBATCH --cpus-per-task=96
```
或者如果您不想使用 HT：
```bash
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
```

这种方法将为每个核心分配一个线程，在这种模式下只有 48 个 CPU 核心可供使用。

请注意，根据您的应用程序，这两种模式之间的性能差异可能相当大。因此建议尝试两种方法，看看哪种效果更好。

在某些设置（如 AWS）中，当使用 `--hint=nomultithread` 时，全减少吞吐量会显著下降！而在其他一些设置中，相反的情况是真实的——没有 HT 时吞吐量更差！

要检查实例是否启用了 HT，请运行：

```bash
$ lscpu | grep Thread
Thread(s) per core: 2
```

如果是 `2` 则表示启用了 HT，如果是 `1` 则表示未启用。