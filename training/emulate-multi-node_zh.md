# emulate-multi-node - 中文翻译

使用单个节点模拟多节点设置

目标是使用具有2个GPU的单个节点来模拟一个2节点环境（用于测试目的）。当然，这可以进一步扩展到更大的设置。

我们在这里使用`deepspeed`启动器。无需实际使用任何`deepspeed`代码，只需利用其更高级的功能即可。您只需要安装`pip install deepspeed`。

完整的设置说明如下：

1. 创建`hostfile`文件：

```bash
$ cat hostfile
worker-0 slots=1
worker-1 slots=1
```

2. 在您的SSH客户端中添加相应的配置：

```bash
$ cat ~/.ssh/config
[...]
Host worker-0
    HostName localhost
    Port 22
Host worker-1
    HostName localhost
    Port 22
```

如果端口不是22或主机名不是`localhost`，请进行相应调整。

3. 由于您的本地设置可能需要密码验证，请确保将您的公钥添加到`~/.ssh/authorized_keys`文件中。

`deepspeed`启动器明确使用无密码连接，例如在worker0上它会运行：`ssh -o PasswordAuthentication=no worker-0 hostname`，因此您可以随时通过以下命令调试SSH设置：

```bash
$ ssh -vvv -o PasswordAuthentication=no worker-0 hostname
```

4. 创建一个测试脚本来检查是否两个GPU都被使用。

```bash
$ cat test1.py
import os
import time
import torch
import deepspeed
import torch.distributed as dist

# 关键的hack，用于使用第二个GPU（否则两个进程都会使用GPU0）
if os.environ["RANK"] == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dist.init_process_group("nccl")
local_rank = int(os.environ.get("LOCAL_RANK"))
print(f'{dist.get_rank()=}, {local_rank=}')

x = torch.ones(2**30, device=f"cuda:{local_rank}")
time.sleep(100)
```

运行：

```bash
$ deepspeed -H hostfile test1.py
[2022-09-08 12:02:15,192] [INFO] [runner.py:415:main] 使用IP地址192.168.0.17作为worker-0的节点
[2022-09-08 12:02:15,192] [INFO] [multinode_runner.py:65:get_cmd] 在以下工作节点上运行：worker-0,worker-1
[2022-09-08 12:02:15,192] [INFO] [runner.py:504:main] cmd = pdsh -S -f 1024 -w worker-0,worker-1 export PYTHONPATH=/mnt/nvme0/code/huggingface/multi-node-emulate-ds;  cd /mnt/nvme0/code/huggingface/multi-node-emulate-ds; /home/stas/anaconda3/envs/py38-pt112/bin/python -u -m deepspeed.launcher.launch --world_info=eyJ3b3JrZXItMCI6IFswXSwgIndvcmtlci0xIjogWzBdfQ== --node_rank=%n --master_addr=192.168.0.17 --master_port=29500 test1.py
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=0
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:156:main] dist_world_size=2
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:158:main] 设置CUDA_VISIBLE_DEVICES=0
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=1
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:156:main] dist_world_size=2
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:158:main] 设置CUDA_VISIBLE_DEVICES=0
worker-1: torch.distributed.get_rank()=1, local_rank=0
worker-0: torch.distributed.get_rank()=0, local_rank=0
worker-1: tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
worker-0: tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
```

如果SSH设置正确，您可以在并行运行`nvidia-smi`时观察到两个GPU分配了约4GB的内存来自`torch.ones`调用。

注意，脚本通过`CUDA_VISIBLE_DEVICES`来告诉第二个进程使用gpu1，但在两种情况下它会被视为`local_rank==0`。

5. 最后，让我们测试一下NCCL集体操作是否也能正常工作

从`torch-distributed-gpu-test.py`脚本中修改而来，仅需调整`os.environ["CUDA_VISIBLE_DEVICES"]`。

```bash
$ cat test2.py
import deepspeed
import fcntl
import os
import socket
import time
import torch
import torch.distributed as dist

# 关键的hack，用于第二个进程使用第二个GPU（否则两个进程都会使用GPU0）
if os.environ["RANK"] == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def printflock(*msgs):
    """ 解决多进程交错打印问题 """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
hostname = socket.gethostname()

gpu = f"[{hostname}-{local_rank}]"

try:
    # 测试分布式
    dist.init_process_group("nccl")
    dist.all_reduce(torch.ones(1).to(device), op=dist.ReduceOp.SUM)
    dist.barrier()
    print(f'{dist.get_rank()=}, {local_rank=}')

    # 测试CUDA可用且能分配内存
    torch.cuda.is_available()
    torch.ones(1).cuda(local_rank)

    # 全局rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    printflock(f"{gpu} 是正常的（全局rank: {rank}/{world_size}）")

    dist.barrier()
    if rank == 0:
        printflock(f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
        printflock(f"设备计算能力={torch.cuda.get_device_capability()}")
        printflock(f"PyTorch计算能力={torch.cuda.get_arch_list()}")

except Exception:
    printflock(f"{gpu} 出现故障")
    raise
```

运行：

```bash
$ deepspeed -H hostfile test2.py
[2022-09-08 12:07:09,336] [INFO] [runner.py:415:main] 使用IP地址192.168.0.17作为worker-0的节点
[2022-09-08 12:07:09,337] [INFO] [multinode_runner.py:65:get_cmd] 在以下工作节点上运行：worker-0,worker-1
[2022-09-08 12:07:09,337] [INFO] [runner.py:504:main] cmd = pdsh -S -f 1024 -w worker-0,worker-1 export PYTHONPATH=/mnt/nvme0/code/huggingface/multi-node-emulate-ds;  cd /mnt/nvme0/code/huggingface/multi-node-emulate-ds; /home/stas/anaconda3/envs/py38-pt112/bin/python -u -m deepspeed.launcher.launch --world_info=eyJ3b3JrZXItMCI6IFswXSwgIndvcmtlci0xIjogWzBdfQ== --node_rank=%n --master_addr=192.168.0.17 --master_port=29500 test2.py
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=0
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:156:main] dist_world_size=2
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:158:main] 设置CUDA_VISIBLE_DEVICES=0
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=1
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:156:main] dist_world_size=2
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:158:main] 设置CUDA_VISIBLE_DEVICES=0
worker-0: dist.get_rank()=0, local_rank=0
worker-1: dist.get_rank()=1, local_rank=0
worker-0: [hope-0] 是正常的（全局rank: 0/2）
worker-1: [hope-0] 是正常的（全局rank: 1/2）
worker-0: pt=1.12.1+cu116, cuda=11.6, nccl=(2, 10, 3)
worker-0: 设备计算能力=(8, 0)
worker-0: PyTorch计算能力=['sm_37', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86']
worker-1: [2022-09-08 12:07:13,642] [INFO] [launch.py:318:main] 进程576485成功退出。
worker-0: [2022-09-08 12:07:13,642] [INFO] [launch.py:318:main] 进程576484成功退出。
```

大功告成。

我们测试了NCCL集体操作是否正常工作，但它们使用的是本地NVLink/PCIe而不是真正的多节点中的IB/ETH连接，所以根据需要测试的内容，这可能足以满足测试需求，也可能不足够。


## 更大的设置

假设您有4个GPU，并且想要模拟2x2个节点。则只需更改`hostfile`为：

```bash
$ cat hostfile
worker-0 slots=2
worker-1 slots=2
```

并将`CUDA_VISIBLE_DEVICES` hack改为：

```bash
if os.environ["RANK"] in ["2", "3"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
```

其余部分应保持不变。


## 自动化过程

如果您希望有一个自动处理任何拓扑形状的方法，可以使用类似以下的方法：

```python
def set_cuda_visible_devices():
    """
    通过调整CUDA_VISIBLE_DEVICES环境变量自动分配每个模拟节点正确的GPU组
    """

    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    emulated_node_size = int(os.environ["LOCAL_SIZE"])
    emulated_node_rank = int(global_rank // emulated_node_size)
    gpus = list(map(str, range(world_size)))
    emulated_node_gpus = ",".join(gpus[emulated_node_rank*emulated_node_size:(emulated_node_rank+1)*emulated_node_size])
    print(f"设置CUDA_VISIBLE_DEVICES={emulated_node_gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = emulated_node_gpus

set_cuda_visible_devices()
```


## 使用单个GPU模拟多个GPU

以下内容与本文讨论的主题正交，但相关性较高，因此我认为分享一些见解是有用的：

使用NVIDIA A100，您可以使用[MIG](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/)在单个真实GPU上最多模拟7个GPU实例，但遗憾的是，这些实例只能用于独立使用——例如，您不能使用这些GPU进行DDP或任何NCCL通信。我希望我可以用我的A100模拟7个实例并添加一个真实的GPU以拥有8个GPU进行开发，但不行，这不起作用。向NVIDIA工程师询问过这个问题，目前没有计划支持这种用例。