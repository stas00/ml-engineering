# README - 中文翻译

## 网络调试

很多时候，你不需要成为网络工程师就能解决网络问题。一些常见的问题可以通过阅读以下笔记来解决。

---

## 术语表

- OOB：带外（通常是较慢的以太网网卡）
- 绑定：使用多个网卡一起以获得更快的速度或作为备份
- IB：InfiniBand（最初由Mellanox开发，后被NVIDIA收购）
- NIC：网络接口卡

---

## 如何诊断NCCL多GPU和多节点连接问题

这一部分并不是详尽无遗的，而是旨在涵盖我经常遇到的一些最常见的设置问题。对于更复杂的问题，请查阅[NCCL仓库问题](https://github.com/NVIDIA/nccl/issues)，或者如果你找不到匹配的情况，可以提交一个新的Issue。NCCL还包含了一个简要的[故障排除部分](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html)，但通常从阅读[问题](https://github.com/NVIDIA/nccl/issues)中学到的东西更多。

对于网络诊断工作，建议使用这个专门设计的测试脚本，而不是使用可能需要很长时间启动并且存在无关问题的完整应用程序：[torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py)。

首先，运行nccl程序并设置：

```bash
export NCCL_DEBUG=INFO
```

这会打印关于NCCL设置及其网络流量的大量调试信息。

例如，如果你正在使用上述调试脚本，在一个有8个GPU的单节点上，你可能会这样做：

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 torch-distributed-gpu-test.py
```

要在多个节点上启动它，你需要使用一些编排软件如SLURM或Kubernetes，或者手动在每个节点上启动（`pdsh`会非常有帮助）——参见[torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py)中的详细说明。但为了理解如何操作，我建议先从一个节点开始，然后逐步过渡到两个节点，最后再到更多的节点。

现在，检查程序的输出并查找以以下内容开头的行：

```bash
NCCL INFO NET/
```

然后检查它正在使用哪个协议和哪些接口。

例如，以下输出：

```bash
NCCL INFO NET/FastSocket : Using [0]ibs108:10.0.19.12<0> [1]ibs109:10.0.19.13<0> [2]ibs110:10.0.19.14<0> [3]ibs111:10.0.19.15<0> [4]ibs112:10.0.19.16<0> [5]ibs113:10.0.19.17<0> [6]ibs114:10.0.19.18<0> [7]ibs115:10.0.19.19<0>
```

告诉我们使用了[nccl-fastsocket](https://github.com/google/nccl-fastsocket)传输层插件，并且它发现了8个`ibs*`网络接口（网卡）。如果你使用的是Google云，这是正确的，你的NCCL配置很可能正确。但如果使用的是InfiniBand (IB)，并且你得到了上述输出，那么你很可能会遇到非常低的节点间速度，因为这意味着你激活了错误的插件。

对于IB，你希望看到的是`NET/IB`和它的IB接口：

```bash
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/IB [RO]; OOB eno1:101.262.0.9<0>
```

在这里，你可以看到IB被用于8个`mlx5_*`接口进行集体通信，并且有一个OOB（带外），通常使用一个较慢的以太网网卡（有时是几个网卡绑定在一起——如果你想知道接口名称中的`bond`代表什么的话）。

要了解你的节点有哪些TCP/IP接口，可以在其中一个节点上运行`ifconfig`命令（通常所有类似的节点会有相同的接口名称，但不总是如此）。

如果您的集体通信网络是IB，您应该运行`ibstat`而不是`ifconfig`。上面最后一个`NCCL INFO NET`示例将对应以下输出：

```bash
$ ibstat | grep mlx5
CA 'mlx5_0'
CA 'mlx5_1'
CA 'mlx5_2'
CA 'mlx5_3'
CA 'mlx5_4'
CA 'mlx5_5'
CA 'mlx5_6'
CA 'mlx5_7'
```

除了快速节点间连接的NIC之外，你还可能有一个缓慢的管理以太网NIC（甚至有几个这样的NIC），它们的存在是为了能够配置节点、使用共享文件系统、访问互联网，因此`ifconfig`也可能会包括额外的NIC。你也可能有一个docker网络接口、loopback接口等。例如，在我的电脑上，我可能会得到以下输出：

```bash
$ ifconfig
docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.99.0.1  netmask 255.255.0.0  broadcast 172.99.255.255
        inet6 f330::42:fe33:f335:7c94  prefixlen 64  scopeid 0x20<link>
        ether 02:42:fe:15:1c:94  txqueuelen 0  (Ethernet)
        RX packets 219909  bytes 650966314 (650.9 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 262998  bytes 20750134 (20.7 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 1147283113  bytes 138463231270 (138.4 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 1147283113  bytes 138463231270 (138.4 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.23  netmask 255.255.255.0  broadcast 10.0.0.255
        inet6 2601:3108:1c71:600:4224:7e4b:13e4:7b54  prefixlen 64  scopeid 0x0<global>
        ether 04:41:1a:16:17:bd  txqueuelen 1000  (Ethernet)
        RX packets 304675330  bytes 388788486256 (388.7 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 74956770  bytes 28501279127 (28.5 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
        device memory 0xa3b00000-a3bfffff
```

我提到这些的原因是关键在于确保NCCL仅报告正确的接口。如果像`docker0`、`lo`或`eth0`这样的接口被报告，例如：

```bash
NCCL INFO NET/Socket : Using [0]eth0:10.0.0.23<0>
```

如果不是你有更快的网络接口可用，这很可能不是你想要的结果。当然，在某些情况下，以太网NIC可能是你唯一拥有的，这样的话上述情况也是可以接受的——只是会非常慢。

有时，如果错误的接口被使用，应用程序可能会挂起。

如果你有所有正确的接口，再加上一些错误的接口，NCCL可能会工作但速度较慢。

如果是云环境，通常你的云服务商会给你正确的设置步骤。如果没有，你需要至少询问他们你需要使用哪些网络接口来设置NCCL。

虽然NCCL尽力自动发现应该使用的接口，但如果它无法正确完成，你可以通过告诉它使用或不使用哪些接口来帮助它：

- `NCCL_SOCKET_IFNAME` 可用于指定哪些`ifconfig`接口包括或排除，当不使用InfiniBand时。这里有一些例子：

```bash
export NCCL_SOCKET_IFNAME=eth:       # 使用所有以eth开头的接口，如eth0, eth1, ...
export NCCL_SOCKET_IFNAME==eth0:     # 仅使用接口eth0
export NCCL_SOCKET_IFNAME==eth0,eth1: # 仅使用接口eth0和eth1
export NCCL_SOCKET_IFNAME=^docker:   # 不使用任何以docker开头的接口
export NCCL_SOCKET_IFNAME=^=docker0: # 不使用接口docker0。
```
完整的文档在这里：[这里](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname)。

- 当使用IB RDMA（IB Verbs接口）时，而不是使用`NCCL_SOCKET_IFNAME`，使用`NCCL_IB_HCA`环境变量选择用于集体通信的接口。例子：

```bash
export NCCL_IB_HCA=mlx5:              # 使用所有以mlx5开头的卡的所有端口
export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1 # 使用卡mlx5_0和mlx5_1的端口1。
export NCCL_IB_HCA=^=mlx5_1,mlx5_4:   # 不使用卡mlx5_1和mlx5_4。
```
完整的文档在这里：[这里](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca)。

例如，通常使用IB时，还会有一些你不希望包含在NCCL通信中的额外接口，如`mlx5_bond_0`。例如，这个报告表明错误的`[8]mlx5_bond_0:1/RoCE`接口被包括在内，这几乎肯定会导致带宽降低：

```bash
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/I [8]mlx5_bond_0:1/RoCE [RO]; OOB ibp25s0:10.0.12.82<0>
```

在这种情况下，你可以排除它：

```bash
export NCCL_IB_HCA=^mlx5_bond_0:1
```

或者，你可以明确列出你想要的接口，例如：

```bash
export NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

如前所述，在一个用IB互连的节点上运行`ibstat`将显示可用的IB接口。

由于NCCL试图自动选择最佳网络接口，只有在NCCL不能正常工作或速度慢时才需要执行上述操作。在正常情况下，NCCL应该能直接工作，无需用户进行特殊操作。

此外，根据使用的云环境，很可能会有一堆环境变量需要设置。如果你设置了其中的一些变量不正确，NCCL可能会工作得很慢或根本不起作用。

另一个用户经常遇到的问题是尝试重用他们在云A中工作的NCCL设置到云B中。通常情况下，这些设置并不能简单地移植，你必须仔细删除之前设置的所有环境变量，并为新的云环境正确地重新设置。即使你在使用相同的云环境，但不同的实例类型，也会出现这种情况，因为一些网络设置非常特定于给定的实例类型，不会在其他地方工作。