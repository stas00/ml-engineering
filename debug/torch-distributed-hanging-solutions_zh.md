# torch-distributed-hanging-solutions - 中文翻译

## 诊断多节点多GPU Python程序中的挂起和死锁

虽然本文中的方法是在处理基于PyTorch的多节点多GPU训练时开发的，但它们当然也可以帮助任何多进程多节点的Python程序。

## 辅助工具

尝试使用以下脚本 [torch-distributed-gpu-test.py](torch-distributed-gpu-test.py) 来诊断情况。

这将主要帮助发现网络相关的问题。同时也可以快速了解多GPU通信是如何工作的。

对于代码相关的问题，请阅读文档的其余部分。

## 多GPU挂起/死锁的诊断方法

### py-spy

首先执行 `pip install py-spy`。

现在你可以附加到每个进程中：

``` 
py-spy dump -n -p PID
```
它会告诉你进程在何处挂起（通常是nccl集体函数或`barrier`）。

- `PID` 是挂起的Python进程的进程ID。
- `-n` 如果你想查看由C、C++等编写的Python扩展的堆栈跟踪，这很有用，因为程序可能会在这些扩展中挂起。
- 你可能需要在命令前添加 `sudo`，详情见 [此注释](https://github.com/benfred/py-spy/blob/master/README.md#when-do-you-need-to-run-as-sudo)。

如果你没有 `sudo` 访问权限，你的系统管理员可以为你执行：
``` 
sudo echo 0 > /proc/sys/kernel/yama/ptrace_scope
```
这将允许你在不需要 `sudo` 的情况下运行 `py-spy` 和 `strace`。请注意可能的 [安全影响](https://wiki.ubuntu.com/SecurityTeam/Roadmap/KernelHardening#ptrace_Protection)，但如果您的计算节点无法从互联网访问，则风险较小。

要使此更改永久生效，请编辑 `/etc/sysctl.d/10-ptrace.conf` 并设置：
``` 
kernel.yama.ptrace_scope = 0
```

这里有一个 `py-spy dump` 的Python堆栈跟踪示例：
``` 
Thread 835995 (active): "MainThread"
    broadcast (torch/distributed/distributed_c10d.py:1191)
    _aggregate_total_loss (deepspeed/runtime/pipe/engine.py:540)
    train_batch (deepspeed/runtime/pipe/engine.py:330)
    train_step (megatron/training.py:436)
    train (megatron/training.py:851)
    pretrain (megatron/training.py:187)
    <module> (pretrain_gpt.py:239)
```
第一行是程序卡住的地方。

如果挂起发生在CPP扩展中，添加 `--native` 到 `py-spy` 中，它会显示任何非Python代码。

#### 多进程 py-spy

现在，如何为多个进程执行？逐个执行太慢了。让我们一次搞定。

如果启动命令是 `python`，你应该这样做：

``` 
pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}
```

如果是 `deepspeed`：

``` 
pgrep -P $(pgrep -o deepspeed) | xargs -I {} py-spy dump --pid {}
```

对于 `accelerate`：

``` 
pgrep -P $(pgrep -o accelerate) | xargs -I {} py-spy dump --pid {}
```

你懂的。

这种方法只会分析主进程，而不是这些进程生成的各种子进程/线程。所以如果你有8个GPU和8个进程，上述方法将生成8个堆栈跟踪。

如果你想获取所有进程及其子进程，只需运行：

``` 
pgrep -f python | xargs -I {} py-spy dump --pid {}
```
（如前所述，如果启动程序不是 `python`，请相应地调整）


#### 多节点 py-spy 通过 srun

如果你有多节点怎么办？

你可以通过 `ssh` 交互式地连接到每个节点并转储堆栈跟踪。

如果你使用的是SLURM环境，你可以使用 `srun` 为你在所有节点上完成这项工作。

现在，在另一个控制台获取 `SLURM_JOBID`（或从 `salloc` 日志中获取）：
``` 
squeue -u `whoami` -o "%.16i %9P %26j %.8T %.10M %.8l %.6D %.20S %R"
```

现在使用以下 `srun` 命令，调整 `SLURM_JOBID` 以匹配上面命令的结果：
``` 
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'ps aux | grep python | egrep -v "grep|srun" | grep `whoami` | awk "{print \$2}" | xargs -I {} py-spy dump --native --pid {}' || echo "failed"
```

注意：
- 必须使用 `--gres=gpu:0` 作为 `srun` 监控器，否则它将一直阻塞直到主 `srun`（运行训练的那个）退出。
- 每个节点都会生成一个唯一的日志文件，名为 `trace-nodename.out`，因此这有助于识别哪些节点有问题。如果你希望所有内容都转储到标准输出，可以删除 `--output=trace-%N.out`。
- 在某些SLURM版本中，你可能还需要添加 `--overlap`。
- 在某些SLURM版本中，作业ID可能与 `squeue` 报告的不同，因此你必须从你试图“附加”到的作业的日志中获取正确的 `SLURM_JOB_ID` - 即运行 `srun` 分配GPU的那个作业。
- 有时 `bash` 不工作，但 `sh` 可以。我认为这与加载的点文件有关。
- 你可能还需要激活自定义Python环境，可以这样做：
``` 
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'conda activate myenvname; ps auxc | ... ' || echo "failed"
```
或者你可以在 `~/.bashrc` 或你决定使用的任何shell的rc文件中进行。

如前所述，如果你只想获取主进程，可以使用这个命令：
``` 
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}' || echo "failed"
```
根据多GPU部分所述调整 `python`。

前一个较长的命令将为所有Python进程提供跟踪。

如果你什么都得不到，从基本调试开始：

``` 
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'date'
```
一旦你知道你正在与所有节点通信，那么你可以逐步解开调用深度，如下所示：

``` 
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'date'
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -o python'
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -P $(pgrep -o python) '
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}'
```
在每个阶段检查输出是否合理 - 例如，第2和第3个调用应该得到进程的PID。