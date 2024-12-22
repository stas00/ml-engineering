# performance - 中文翻译

# SLURM 性能

在这里，您会发现一些影响性能的SLURM特定设置的讨论。

## `srun` 的 `--cpus-per-task` 可能需要显式指定

您需要确保通过 `srun` 启动的程序能够接收到预期数量的CPU核心。例如，在典型的机器学习训练程序中，每个GPU至少需要一个用于驱动它的CPU核心以及更多的核心用于 `DataLoader`。您需要多个核心以便每个任务可以并行执行。如果您有8个GPU，每个GPU有2个工作进程，则每个节点至少需要 `3*8=24` 个CPU核心。

`每个任务的CPU核心数` 由 `--cpus-per-task` 定义，该参数传递给 `sbatch` 或 `salloc`，而原来的 `srun` 会继承这个设置。然而，最近这种行为发生了变化：

`sbatch` 手册页中的一个引用：

> 注意：从22.05版本开始，`srun` 不再继承 `salloc` 或 `sbatch` 请求的 `--cpus-per-task` 值。如果希望为任务请求此值，必须在调用 `srun` 时再次请求，或者使用环境变量 `SRUN_CPUS_PER_TASK` 设置。

这意味着在过去，您的SLURM脚本可能是这样的：

``` 
#SBATCH --cpus-per-task=48
[...]
srun myprogram
```

并且由 `srun` 启动的程序将接收48个CPU核心，因为 `srun` 继承了 `sbatch` 或 `salloc` 中的 `--cpus-per-task=48` 设置。根据上述文档，自SLURM 22.05起，这种行为不再成立。

注释：我在SLURM@22.05.09上进行了测试，旧的行为仍然存在，但肯定是在23.x系列中发生了变化。因此，更改可能发生在22.05系列的后期版本中。

所以，如果您保持现状，现在程序将只接收一个CPU核心（除非 `srun` 的默认值已修改）。

您可以轻松地测试您的SLURM配置是否受到影响，使用 `os.sched_getaffinity(0)`，因为它显示当前进程可以使用的CPU核心。因此，使用 `len(os.sched_getaffinity(0))` 来计数应该是简单的。

以下是如何测试您是否受到影响：
``` 
$ cat test.slurm
#!/bin/bash
#SBATCH --job-name=test-cpu-cores-per-task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48   # 根据您的环境调整为少于48个CPU核心
#SBATCH --time=0:10:00
#SBATCH --partition=x        # 根据您的环境调整到正确的分区名称
#SBATCH --output=%x-%j.out

srun python -c 'import os; print(f"可见的CPU核心数: {len(os.sched_getaffinity(0))}")'
```

如果您得到
```
可见的CPU核心数: 48
```
那么您不需要做任何事情，但如果得到：
```
可见的CPU核心数: 1
```
或其他小于48的值，则您受到影响。

要解决这个问题，您需要更改您的SLURM脚本以：

``` 
#SBATCH --cpus-per-task=48
[...]
srun --cpus-per-task=48 myprogram
```
或：
``` 
#SBATCH --cpus-per-task=48
[...]
SRUN_CPUS_PER_TASK=48
srun myprogram
```

或者自动化处理，一次性完成：
``` 
#SBATCH --cpus-per-task=48
[...]
SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun myprogram
```