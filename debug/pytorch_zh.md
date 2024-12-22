# pytorch - 中文翻译

## 调试 PyTorch 程序

### 让节点互相通信

一旦你需要使用多个节点来扩展训练，比如想要使用 DDP 来更快地训练，就必须让这些节点互相通信，以便通信集合能够彼此发送数据。这通常通过一个通信库（如 [NCCL](https://github.com/nVIDIA/nccl)）来实现。在我们的 DDP 示例中，在每个训练步骤结束时，所有 GPU 都需要执行一个 `all_reduce` 调用来跨所有 rank 同步梯度。

在本节中，我们将讨论一个非常简单的案例，即两个节点（每个节点有 8 块 GPU）互相通信，然后可以轻松扩展到任意数量的节点。假设这些节点的 IP 地址分别为 10.0.0.1 和 10.0.0.2。

有了 IP 地址之后，我们还需要选择一个用于通信的端口。

Unix 中有 64k 个端口。前 1k 个端口是为常见服务保留的，这样任何互联网上的计算机都可以提前知道连接到哪个端口。例如，端口 22 是为 SSH 保留的。因此，当你输入 `ssh example.com` 时，实际上程序会打开一个到 `example.com:22` 的连接。

由于有数千种服务，预留的 1k 个端口是不够的，所以各种服务可以使用几乎任何端口。但不必担心，当你在云或 HPC 上获得 Linux 机器时，不太可能有许多预安装的服务使用高号端口，因此大多数端口应该是可用的。

因此，我们可以选择端口 6000。

现在我们有两个要相互通信的地址：`10.0.0.1:6000` 和 `10.0.0.2:6000`。

首先要做的是在两个节点上都打开端口 `6000` 以允许进出连接。可能它已经打开了，或者你可能需要查阅特定设置的说明来了解如何打开给定的端口。

这里有一些你可以使用的测试方法来检查端口 6000 是否已经打开。

```bash
telnet localhost:6000
nmap -p 6000 localhost
nc -zv localhost 6000
curl -v telnet://localhost:6000
```

大多数这些命令都可以通过 `apt install` 或你的包管理器来安装。

让我们在这个示例中使用 `nmap`。如果我运行：

```bash
$ nmap -p 22 localhost
[...]
PORT   STATE SERVICE
22/tcp open  ssh
```

我们可以看到端口是打开的，并且还告诉我们分配了哪种协议和服务。

现在让我们运行：

```bash
$ nmap -p 6000 localhost
[...]
PORT     STATE  SERVICE
6000/tcp closed X11
```

在这里可以看到端口 6000 是关闭的。

现在你已经了解了如何测试，你可以继续测试 `10.0.0.1:6000` 和 `10.0.0.2:6000`。

首先在终端 A 中 ssh 登录到第一个节点并测试第二个节点上的端口 6000 是否已打开：

```bash
ssh 10.0.0.1
nmap -p 6000 10.0.0.2
```

如果一切正常，则在终端 B 中 ssh 登录到第二个节点并反向进行相同的检查：

```bash
ssh 10.0.0.2
nmap -p 6000 10.0.0.1
```

如果两个端口都已打开，那么现在你可以使用这个端口。如果任何一个或两个端口被关闭，你必须打开这些端口。由于大多数云提供商使用专有的解决方案，只需在网上搜索“打开端口”和你的云提供商名称。

接下来重要的一点是要理解计算节点通常会有多个网络接口卡（NIC）。你可以通过运行以下命令来发现这些接口：

```bash
$ sudo ifconfig
```

其中一个接口通常用于用户通过 ssh 连接到节点或其他非计算相关的服务，例如发送邮件或下载数据。这个接口通常被称为 `eth0`，其中 `eth` 表示以太网，但它也可能被命名为其他名称。

然后是节点间的接口，可能是 Infiniband、EFA、OPA、HPE Slingshot 等。可能会有一个或多个这样的接口。

以下是 `ifconfig` 输出的一些示例：

```bash
$ sudo ifconfig
enp5s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.23  netmask 255.255.255.0  broadcast 10.0.0.255
        [...]
```

我移除了大部分输出只显示一些信息。这里的关键信息是在 `inet` 之后列出的 IP 地址。在上面的例子中，它是 `10.0.0.23`。这是 `enp5s0` 接口的 IP 地址。

如果有另一个节点，它的 IP 地址可能是 `10.0.0.24` 或 `10.0.0.21` 或类似的东西——最后一段数字会有所不同。

再看一个例子：

```bash
$ sudo ifconfig
ib0     Link encap:UNSPEC  HWaddr 00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00
        inet addr:172.0.0.50  Bcast: 172.0.0.255  Mask:255.255.255.0
        [...]
```

这里 `ib` 通常表示这是一个 InfiniBand 卡，但实际上它可以是任何其他供应商。我见过 [OmniPath](../network#omni-path) 使用 `ib`。同样，`inet` 告诉我们此接口的 IP 地址是 `172.0.0.50`。

如果你跟不上思路，我们希望获取 IP 地址，以便可以测试每个节点上的 `ip:port` 是否已打开。

最后回到我们的 `10.0.0.1:6000` 和 `10.0.0.2:6000`，让我们使用 2 个终端进行 `all_reduce` 测试，其中选择 `10.0.0.1` 作为主主机来协调其他节点。
为了测试，我们将使用这个辅助调试程序 [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py)。

在终端 A 中：

```bash
$ ssh 10.0.0.1
$ python -m torch.distributed.run --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --master_addr 10.0.0.1 --master_port 6000 torch-distributed-gpu-test.py
```

在终端 B 中：

```bash
$ ssh 10.0.0.2
$ python -m torch.distributed.run --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --master_addr 10.0.0.1 --master_port 6000 torch-distributed-gpu-test.py
```

注意我在两种情况下都使用了相同的 `--master_addr 10.0.0.1 --master_port 6000`，因为我们在之前检查过端口 6000 已经打开，并且我们使用 `10.0.0.1` 作为协调主机。

这种从每个节点手动运行的方法很痛苦，因此有一些工具可以自动在多个节点上启动相同的命令。

**pdsh**

`pdsh` 就是一个这样的解决方案——类似于 `ssh`，但会自动在同一组节点上运行相同的命令：

```bash
PDSH_RCMD_TYPE=ssh pdsh -w 10.0.0.1,10.0.0.2 \
"python -m torch.distributed.run --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --master_addr 10.0.0.1 --master_port 6000 torch-distributed-gpu-test.py"
```

你可以看到我将两组命令合并成了一个。如果有更多的节点，只需添加更多的节点作为 `-w` 参数。

**SLURM**

如果你使用 SLURM，几乎可以肯定的是，设置的人已经为你打开了所有端口，所以应该可以直接工作。但如果不行，本节中的信息应该有助于调试。

以下是使用 SLURM 的方法。

```bash
#!/bin/bash
#SBATCH --job-name=test-nodes        # 名称
#SBATCH --nodes=2                    # 节点数
#SBATCH --ntasks-per-node=1          # 关键 - 每个节点仅一个任务！
#SBATCH --cpus-per-task=10           # 每个任务的内核数
#SBATCH --gres=gpu:8                 # GPU 数量
#SBATCH --time 0:05:00               # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000
#
srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
torch-distributed-gpu-test.py'
```
如果你有超过两个节点，只需要更改节点数量，上述脚本将自动适用于任意数量的节点。

**MPI**:

另一种流行的方法是使用 [消息传递接口 (MPI)](https://zh.wikipedia.org/wiki/消息传递接口)。有几个开源实现可用。

要使用此工具，你首先要创建一个包含目标节点及其应在此主机上运行的进程数的 `hostfile`。在这个部分的例子中，对于 2 个节点和每节点 8 个 GPU，它将是：

```bash
$ cat hostfile
10.0.0.1:8
10.0.0.2:8
```
运行时，只需：
```bash
$ mpirun --hostfile  -np 16 -map-by ppr:8:node python my-program.py
```

请注意，我在这里使用了 `my-program.py`，因为 [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py) 是为与 `torch.distributed.run`（也称为 `torchrun`）一起工作而编写的。使用 `mpirun` 时，你必须检查你的具体实现来查看使用哪个环境变量传递程序的 rank，并替换 `LOCAL_RANK`，其余部分应该基本相同。

注意事项：
- 你可能需要明确指定要使用的接口，通过添加 `--mca btl_tcp_if_include 10.0.0.0/24` 来匹配我们的示例。如果你有许多网络接口，它可能会使用未打开的接口或错误的接口。
- 你也可以反过来排除某些接口。例如，如果你有 `docker0` 和 `lo` 接口，可以通过添加 `--mca btl_tcp_if_exclude docker0,lo` 来排除它们。

`mpirun` 有很多标志，我建议阅读其手册页以获取更多信息。我的目的是仅仅展示你可以如何使用它。不同的 `mpirun` 实现可能使用不同的命令行选项。

