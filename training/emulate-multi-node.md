# Emulate a Multi-Node Setup Using Just a Single Node

The goal is to emulate a 2-node environment using a single node with 2 GPUs (for testing purposes). This, of course, can be further expanded to [larger set ups](#larger-set-ups).

We use the `deepspeed` launcher here. There is no need to actually use any of the deepspeed code, it's just easier to use its more advanced capabilities. You will just need to install `pip install deepspeed`.

The full setup instructions follow:

1. Create a `hostfile`:

```bash
$ cat hostfile
worker-0 slots=1
worker-1 slots=1
```

2. Add a matching config to your ssh client

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

Adapt the port if it's not 22 and the hostname if `localhost` isn't it.


3. As your local setup is probably password protected ensure to add your public key to `~/.ssh/authorized_keys`

The `deepspeed` launcher explicitly uses no-password connection, e.g. on worker0 it'd run: `ssh -o PasswordAuthentication=no worker-0 hostname`, so you can always debug ssh setup using:

```bash
$ ssh -vvv -o PasswordAuthentication=no worker-0 hostname
```

4. Create a test script to check both GPUs are used.

```bash
$ cat test1.py
import os
import time
import torch
import deepspeed
import torch.distributed as dist

# critical hack to use the 2nd gpu (otherwise both processes will use gpu0)
if os.environ["RANK"] == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dist.init_process_group("nccl")
local_rank = int(os.environ.get("LOCAL_RANK"))
print(f'{dist.get_rank()=}, {local_rank=}')

x = torch.ones(2**30, device=f"cuda:{local_rank}")
print(x)
time.sleep(100)
```

Run:

```bash
$ deepspeed -H hostfile test1.py
[2022-09-08 12:02:15,192] [INFO] [runner.py:415:main] Using IP address of 192.168.0.17 for node worker-0
[2022-09-08 12:02:15,192] [INFO] [multinode_runner.py:65:get_cmd] Running on the following workers: worker-0,worker-1
[2022-09-08 12:02:15,192] [INFO] [runner.py:504:main] cmd = pdsh -S -f 1024 -w worker-0,worker-1 export PYTHONPATH=/mnt/nvme0/code/huggingface/multi-node-emulate-ds;  cd /mnt/nvme0/code/huggingface/multi-node-emulate-ds; /home/stas/anaconda3/envs/py38-pt112/bin/python -u -m deepspeed.launcher.launch --world_info=eyJ3b3JrZXItMCI6IFswXSwgIndvcmtlci0xIjogWzBdfQ== --node_rank=%n --master_addr=192.168.0.17 --master_port=29500 test1.py
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=0
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:156:main] dist_world_size=2
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:158:main] Setting CUDA_VISIBLE_DEVICES=0
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=1
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:156:main] dist_world_size=2
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:158:main] Setting CUDA_VISIBLE_DEVICES=0
worker-1: dist.get_rank()=1, local_rank=0
worker-0: dist.get_rank()=0, local_rank=0
worker-1: tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
worker-0: tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
```

If the ssh set up works you can run `nvidia-smi` in parallel and observe that both GPUs allocated ~4GiB of memory from `torch.ones` call.

Note that the script hacks in `CUDA_VISIBLE_DEVICES` to tell the 2nd process to use gpu1, but it'll be seen as `local_rank==0` in both cases.

5. Finally, let's test that NCCL collectives work as well

Script adapted from [torch-distributed-gpu-test.py](../debug/torch-distributed-gpu-test.py) to just tweak `os.environ["CUDA_VISIBLE_DEVICES"]`

```bash
$ cat test2.py
import deepspeed
import fcntl
import os
import socket
import time
import torch
import torch.distributed as dist

# a critical hack to use the 2nd GPU by the 2nd process (otherwise both processes will use gpu0)
if os.environ["RANK"] == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def printflock(*msgs):
    """ solves multi-process interleaved print problem """
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
    # test distributed
    dist.init_process_group("nccl")
    dist.all_reduce(torch.ones(1).to(device), op=dist.ReduceOp.SUM)
    dist.barrier()
    print(f'{dist.get_rank()=}, {local_rank=}')

    # test cuda is available and can allocate memory
    torch.cuda.is_available()
    torch.ones(1).cuda(local_rank)

    # global rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    printflock(f"{gpu} is OK (global rank: {rank}/{world_size})")

    dist.barrier()
    if rank == 0:
        printflock(f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
        printflock(f"device compute capabilities={torch.cuda.get_device_capability()}")
        printflock(f"pytorch compute capabilities={torch.cuda.get_arch_list()}")

except Exception:
    printflock(f"{gpu} is broken")
    raise
```

Run:

```bash
$ deepspeed -H hostfile test2.py
[2022-09-08 12:07:09,336] [INFO] [runner.py:415:main] Using IP address of 192.168.0.17 for node worker-0
[2022-09-08 12:07:09,337] [INFO] [multinode_runner.py:65:get_cmd] Running on the following workers: worker-0,worker-1
[2022-09-08 12:07:09,337] [INFO] [runner.py:504:main] cmd = pdsh -S -f 1024 -w worker-0,worker-1 export PYTHONPATH=/mnt/nvme0/code/huggingface/multi-node-emulate-ds;  cd /mnt/nvme0/code/huggingface/multi-node-emulate-ds; /home/stas/anaconda3/envs/py38-pt112/bin/python -u -m deepspeed.launcher.launch --world_info=eyJ3b3JrZXItMCI6IFswXSwgIndvcmtlci0xIjogWzBdfQ== --node_rank=%n --master_addr=192.168.0.17 --master_port=29500 test2.py
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=0
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:156:main] dist_world_size=2
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:158:main] Setting CUDA_VISIBLE_DEVICES=0
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=1
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:156:main] dist_world_size=2
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:158:main] Setting CUDA_VISIBLE_DEVICES=0
worker-0: dist.get_rank()=0, local_rank=0
worker-1: dist.get_rank()=1, local_rank=0
worker-0: [hope-0] is OK (global rank: 0/2)
worker-1: [hope-0] is OK (global rank: 1/2)
worker-0: pt=1.12.1+cu116, cuda=11.6, nccl=(2, 10, 3)
worker-0: device compute capabilities=(8, 0)
worker-0: pytorch compute capabilities=['sm_37', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86']
worker-1: [2022-09-08 12:07:13,642] [INFO] [launch.py:318:main] Process 576485 exits successfully.
worker-0: [2022-09-08 12:07:13,642] [INFO] [launch.py:318:main] Process 576484 exits successfully.
```

Voila, mission accomplished.

We tested that the NCCL collectives work, but they use local NVLink/PCIe and not the IB/ETH connections like in real multi-node, so it may or may not be good enough for testing depending on what needs to be tested.


## Larger set ups

Now, let's say you have 4 GPUs and you want to emulate 2x2 nodes. Then simply change the `hostfile` to be:

```bash
$ cat hostfile
worker-0 slots=2
worker-1 slots=2
```
and the `CUDA_VISIBLE_DEVICES` hack to:

```bash
if os.environ["RANK"] in ["2", "3"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
```

Everything else should be the same.


## Automating the process

If you want an automatic approach to handle any shape of topology, you could use something like this:

```python
def set_cuda_visible_devices():
    """
    automatically assign the correct groups of gpus for each emulated node by tweaking the
    CUDA_VISIBLE_DEVICES env var
    """

    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    emulated_node_size = int(os.environ["LOCAL_SIZE"])
    emulated_node_rank = int(global_rank // emulated_node_size)
    gpus = list(map(str, range(world_size)))
    emulated_node_gpus = ",".join(gpus[emulated_node_rank*emulated_node_size:(emulated_node_rank+1)*emulated_node_size])
    print(f"Setting CUDA_VISIBLE_DEVICES={emulated_node_gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = emulated_node_gpus

set_cuda_visible_devices()
```


## Emulating multiple GPUs with a single GPU

The following is an orthogonal need to the one discussed in this document, but it's related so I thought it'd be useful to share some insights here:

With NVIDIA A100 you can use [MIG](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/) to emulate up to 7 instances of GPUs on just one real GPU, but for years you couldn't use those instances for anything but standalone use - e.g. you couldn't do DDP or any NCCL comms over those GPUs. I hoped I could use my A100 to emulate 7 instances and add one more real GPU and to have 8x GPUs to do development with - but nope it didn't work. Asking NVIDIA engineers about it at the time, there were no plans to have this use-case supported.

That has changed - [NCCL 2.31 (2026) added experimental support for communicators that span MIG instances](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-mig-instances), so a single physical GPU can now stand in for a small multi-GPU node and you can run DDP, FSDP, ZeRO and any NCCL collective across its instances. What follows is a working recipe on an H200, and measurements of how fast those emulated GPUs actually talk to each other.


### Getting a new enough NCCL

You need an NVIDIA driver with CUDA 13.0 or newer, and NCCL 2.31 or newer.

As of Aug 2026, Pytorch's latest 2.13.0 bundles a lower version. Until a PyTorch build arrives with NCCL>=2.31 in it, install NCCL yourself and load it ahead of the bundled one:

```bash
pip install -U --target $HOME/nccl-new 'nvidia-nccl-cu13>=2.31'
LD_PRELOAD=$HOME/nccl-new/nvidia/nccl/lib/libnccl.so.2 python train.py
```

It goes into its own directory rather than into the environment because PyTorch pins an exact NCCL version (`nvidia-nccl-cu13==2.29.7` for 2.13.0) and installing over it would put pip's dependency resolver at odds with torch's metadata. `LD_PRELOAD` then makes the dynamic linker resolve NCCL's symbols from the new library before it reaches the one that ships inside the torch wheel.

Just to make sure run the program once with `NCCL_DEBUG=INFO` and check that the banner says `NCCL version 2.31.2+cuda13.3` or newer. Don't use `torch.cuda.nccl.version()` for this check - it reports the version PyTorch was built against, not the preloaded one that is actually doing the work.


### The NCCL settings MIG needs

```bash
export NCCL_MULTI_RANK_GPU_ENABLE=1
export NCCL_NVLS_ENABLE=0
```

The NCCL docs say `NCCL_MULTI_RANK_GPU_ENABLE` isn't required, since each MIG instance is a distinct CUDA device - but on an H200 with NCCL 2.31.2 the communicator won't come up without it, so set it anyway. NCCL identifies a rank's device by the parent GPU's NVML index and PCI bus id rather than by the MIG instance, so four ranks on four instances of one GPU look to it like four ranks sharing a single device, and without the flag init stops with `Multiple Ranks are using the same GPU/Partition`.

`NCCL_NVLS_ENABLE=0` is a guard rather than a requirement. NCCL's default already leaves NVLink SHARP off in this situation, so if nothing in your environment sets that variable you can leave it alone. But many cluster images and pod specs export `NCCL_NVLS_ENABLE=1`, and when NVLink SHARP is explicitly demanded NCCL refuses to start with `NCCL_NVLS_ENABLE has been set to "1" and communicator has multiple ranks using the same NVML device`. Setting it to 0 costs nothing and makes the recipe work regardless of what the environment already exported. You can also set `NCCL_NVLS_ENABLE=2` which will automatically use NVLS when possible.

The docs also tell you to set `NCCL_MNNVL_ENABLE=0`. On a node with no NVLink fabric that flag changes nothing - the same benchmark returned 142.28, 142.42 and 142.23 GBps with it unset, set to `0` and set to `1` - so it only matters on fabric-attached systems like GB200 NVL72.


### Carving up a GPU

List the profiles the GPU offers, enable MIG on it, then create the instances. On H200 we can create up to 7 instances. Let's create 4 instances of 35GB each on GPU 4 which will allows us to do for example SP=2, EP=2 model parallelism on a single GPU.

```bash
nvidia-smi mig -lgip             # show profiles available on this GPU
sudo nvidia-smi -i 4 -mig 1      # enable MIG on GPU 4
sudo nvidia-smi mig -i 4 -cgi 1g.35gb,1g.35gb,1g.35gb,1g.35gb -C
```

To undo MIG and to regain the full GPU:

```bash
sudo nvidia-smi mig -i 4 -dci && sudo nvidia-smi mig -i 4 -dgi
sudo nvidia-smi -i 4 -mig 0
```


### Finding the MIG UUIDs

Each rank must be given exactly one instance through `CUDA_VISIBLE_DEVICES`, and MIG instances are addressed by UUID rather than by index. `nvidia-smi -L` prints them indented under their parent GPU:

```bash
$ nvidia-smi -L
[...]
GPU 4: NVIDIA H200 (UUID: GPU-31f70a1b-dee6-617e-13c5-5c2a2fbe0013)
  MIG 1g.35gb  Device 0: (UUID: MIG-e26b1e11-8424-5ef0-97a8-d87e519f7824)
  MIG 1g.35gb  Device 1: (UUID: MIG-1a7d7808-5dcc-50fa-842c-2ceaa9f6e8ee)
  MIG 1g.35gb  Device 2: (UUID: MIG-46c30315-dfa5-56d4-82f6-345f84b4d52d)
  MIG 1g.35gb  Device 3: (UUID: MIG-6b4ff05d-689c-5bf3-8a51-3ff519c617ac)
```

Collect every rank-sized device on the node into one comma-separated variable - a GPU that is in MIG mode contributes its instances, a GPU that isn't contributes itself:

```bash
export MIG_UUIDS=$(nvidia-smi -L | perl -ne 'if (/\(UUID: (GPU-[^)]+)\)/) { push @u, $1 } elsif (/\(UUID: (MIG-[^)]+)\)/) { my $m = $1; pop @u if $u[-1] =~ /^GPU-/; push @u, $m } END { print join ",", @u }')
```


### Launching the ranks

`torchrun` hands each process a `LOCAL_RANK`, which is all that's needed to give each one its own instance:

```bash
torchrun --nproc_per_node 4 --no-python bash -c '
export CUDA_VISIBLE_DEVICES=$(echo $MIG_UUIDS | cut -d, -f$((LOCAL_RANK+1)))
export LOCAL_RANK=0
exec python train.py'
```

Resetting `LOCAL_RANK` to `0` is required, see [Caveats](#caveats).

`all_reduce`, `all_gather`, `reduce_scatter`, `broadcast` and a DDP training step all work across 4 such instances.


### How fast do the emulated GPUs talk to each other?

The emulation is only useful if the collectives aren't crawling, so here is [all_reduce_bench.py](../network/benchmarks/all_reduce_bench.py) run on MIG instances of one H200, measured against the same number of whole H200s - 4 instances against 4 real GPUs, 2 instances against 2 real GPUs.

The profiles used below, all on one H200 141GB:

| MIG profile | compute slices | memory | instances per GPU |
| :---------- | -------------: | -----: | ----------------: |
| `1g.18gb`   |         1 of 7 |   18GB |                 7 |
| `1g.35gb`   |         1 of 7 |   35GB |                 4 |
| `2g.35gb`   |         2 of 7 |   35GB |                 3 |
| `3g.71gb`   |         3 of 7 |   71GB |                 2 |

Every results table below starts with the whole-GPU row it is measured against, then the MIG rows in descending `busbw`.

4 ranks, 512MiB payload:

| what the 4 ranks run on |     busbw | vs whole GPUs |
| :---------------------- | --------: | ------------: |
| 4x whole H200 (NVLink)  | 341.1GBps |          100% |
| 4x `1g.35gb` of one GPU | 273.8GBps |           80% |
| 4x `1g.18gb` of one GPU | 142.3GBps |           42% |

4 ranks, 2GiB payload:

| what the 4 ranks run on |     busbw | vs whole GPUs |
| :---------------------- | --------: | ------------: |
| 4x whole H200 (NVLink)  | 354.7GBps |          100% |
| 4x `1g.35gb` of one GPU | 281.3GBps |           79% |
| 4x `1g.18gb` of one GPU | 144.7GBps |           41% |

2 ranks, 512MiB payload:

| what the 2 ranks run on |     busbw | vs whole GPUs |
| :---------------------- | --------: | ------------: |
| 2x whole H200 (NVLink)  | 308.2GBps |          100% |
| 2x `3g.71gb` of one GPU | 413.8GBps |          134% |
| 2x `2g.35gb` of one GPU | 245.3GBps |           80% |
| 2x `1g.35gb` of one GPU | 230.9GBps |           75% |
| 2x `1g.18gb` of one GPU | 122.0GBps |           40% |

2 ranks, 2GiB payload:

| what the 2 ranks run on |     busbw | vs whole GPUs |
| :---------------------- | --------: | ------------: |
| 2x whole H200 (NVLink)  | 331.7GBps |          100% |
| 2x `3g.71gb` of one GPU | 429.5GBps |          129% |
| 2x `2g.35gb` of one GPU | 250.4GBps |           75% |
| 2x `1g.35gb` of one GPU | 235.0GBps |           71% |
| 2x `1g.18gb` of one GPU | 123.6GBps |           37% |

In the profile table above, the compute-slice count is the digit in the profile name (`2g.35gb` gets 2 of the GPU's 7 compute slices) and the memory figure says how many of its 8 memory slices the instance owns - 18GB is 1 slice, 35GB is 2, 71GB is 4.

Setup: H200 141GB SXM, `torch==2.9.1+cu130`, NCCL 2.31.2 preloaded as above, CUDA driver 13.0.

The big surprise here is how much the choice of profile matters. Traffic between MIG instances never leaves the GPU - NCCL logs every channel as `P2P/CUMEM` with no socket or PCIe fallback anywhere in the run - so the limit is the memory bandwidth the instance owns, which is set by its memory slices and barely at all by its SMs. The two 512MiB tables isolate this: in the 4-rank one, going from `1g.18gb` to `1g.35gb` keeps compute identical at one slice and doubles the memory, and bandwidth nearly doubles (142 -> 274GBps), while in the 2-rank one, going from `1g.35gb` to `2g.35gb` doubles the compute at the same memory and buys only 6% (231 -> 245GBps).

So 4x `1g.35gb` instances reproduce about 79% of the interconnect bandwidth of 4 real H200s, while 4x `1g.18gb` instances - same rank count, smaller slices - give only 41%. Pick the profile with the most memory per instance that still gives you the rank count you need.

And with only 2 ranks, `3g.71gb` instances are actually *faster* than 2 whole GPUs (429 against 332GBps in the 2-rank 2GiB table), because a copy inside one chip's HBM beats a hop over NVLink.

### Spreading the instances over 2 GPUs

Until NVIDIA overcomes the odd [7-instance per GPU limit](#only-7-instances-per-gpu), if you want to do a [3D parallelism](model-parallelism/README.md#dppptp), you need at least 8 GPUs or instances. So for a time being we have to use 2 GPUs to get to 8.

Everything above keeps the whole emulated world inside one physical GPU. Partitioning 2 GPUs into 4 instances each gives 8 ranks, and the setup plus the benchmark is:

```bash
for i in 0 1; do
    sudo nvidia-smi -i $i -mig 1
    sudo nvidia-smi mig -i $i -cgi 1g.35gb,1g.35gb,1g.35gb,1g.35gb -C
done

export MIG_UUIDS=$(nvidia-smi -L | perl -ne 'if (/\(UUID: (GPU-[^)]+)\)/) { push @u, $1 } elsif (/\(UUID: (MIG-[^)]+)\)/) { my $m = $1; pop @u if $u[-1] =~ /^GPU-/; push @u, $m } END { print join ",", @u }')

torchrun --nproc_per_node 8 --no-python bash -c '
export CUDA_VISIBLE_DEVICES=$(echo $MIG_UUIDS | cut -d, -f$((LOCAL_RANK+1)))
export LOCAL_RANK=0
exec python all_reduce_bench.py --payload_size_in_gib 2'
```

8 ranks, 512MiB payload:

| what the 8 ranks run on        |     busbw | vs whole GPUs |
| :----------------------------- | --------: | ------------: |
| 8x whole H200 (NVLink)         | 450.2GBps |          100% |
| 4x `1g.35gb` on each of 2 GPUs |  26.8GBps |            6% |

8 ranks, 2GiB payload:

| what the 8 ranks run on        |     busbw | vs whole GPUs |
| :----------------------------- | --------: | ------------: |
| 8x whole H200 (NVLink)         | 465.9GBps |          100% |
| 4x `1g.35gb` on each of 2 GPUs |  25.6GBps |            5% |

That is a collapse, and it is not a small-message effect - the 2GiB payload is no better than the 512MiB one. NCCL's own log says why: of the ring's channels, 24 hops went `via P2P/CUMEM` and 6 went `via SHM/direct`. The hops that stay inside a GPU keep the fast path, but the ones that cross to the other GPU fall back to host shared memory, because the CUDA driver doesn't support NVLink P2P for MIG instances - the same limitation that rules out NVLink SHARP here. A ring runs at the speed of its slowest link, so those few host-memory hops set the pace for all 8 ranks.


### Mixing MIG instances and full GPUs

Not every GPU has to be partitioned. Since 7 instances is the most one GPU will give you, the cheapest route to an 8-rank world is to carve one H200 (or another high end GPU) into its maximum 7 instances and leave a second GPU whole. The `MIG_UUIDS` command from above already produces the right list for that mix - the partitioned GPU contributes its 7 instances, the whole one contributes itself:

```bash
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi mig -i 0 -cgi 1g.18gb,1g.18gb,1g.18gb,1g.18gb,1g.18gb,1g.18gb,1g.18gb -C

export MIG_UUIDS=$(nvidia-smi -L | perl -ne 'if (/\(UUID: (GPU-[^)]+)\)/) { push @u, $1 } elsif (/\(UUID: (MIG-[^)]+)\)/) { my $m = $1; pop @u if $u[-1] =~ /^GPU-/; push @u, $m } END { print join ",", @u }')

torchrun --nproc_per_node 8 --no-python bash -c '
export CUDA_VISIBLE_DEVICES=$(echo $MIG_UUIDS | cut -d, -f$((LOCAL_RANK+1)))
export LOCAL_RANK=0
exec python all_reduce_bench.py --payload_size_in_gib 2'
```

8 ranks, 512MiB payload:

| what the 8 ranks run on                |     busbw | vs whole GPUs |
| :------------------------------------- | --------: | ------------: |
| 8x whole H200 (NVLink)                 | 450.2GBps |          100% |
| 7x `1g.18gb` of one GPU + 1 whole H200 |  28.5GBps |            6% |

8 ranks, 2GiB payload:

| what the 8 ranks run on                |     busbw | vs whole GPUs |
| :------------------------------------- | --------: | ------------: |
| 8x whole H200 (NVLink)                 | 465.9GBps |          100% |
| 7x `1g.18gb` of one GPU + 1 whole H200 |  27.5GBps |            6% |

The whole GPU rescues nothing: 24 hops ran `via P2P/CUMEM` inside the partitioned GPU and 6 `via SHM/direct` to the whole one, and the ring again runs at the speed of the host-memory hops. The useful corollary is that the second device's own bandwidth doesn't matter - a whole H200 in the eighth slot gave 28.5GBps, no better than the 26.8GBps of the run where that eighth rank was a MIG instance on a second partitioned H200. So if the goal is only to get past the 7-instances-per-GPU ceiling, the second device can be any GPU CUDA will talk to, and it needs no MIG support of its own since it joins whole - one H200 plus one modest GPU is an 8-rank world.

The practical rule: keep an emulated multi-GPU job inside a single physical GPU. Crossing to a second GPU costs a factor of 10 against 4 instances on one GPU (26.8 against 273.8GBps at 512MiB), and leaves you at 5% of what 8 real GPUs do. Though if you're working with a tiny model and payload it probably doesn't matter much for dev purposes.

Do keep in mind that this emulates the *topology* of a multi-GPU job and not its performance - each rank also gets only a fraction of the SMs and a fraction of the memory, so it's a development and testing tool, not a way to measure throughput.


### Caveats

#### `LOCAL_RANK` is `0` in every rank

Once `CUDA_VISIBLE_DEVICES` names a single instance, the process sees exactly one device and CUDA numbers it `0`, so code doing `torch.cuda.set_device(local_rank)` with the value torchrun handed out would fail with `invalid device ordinal`. There is no way around it: CUDA enumerates only a single compute instance per process even when several MIG devices are visible. It stays true when the ranks span 2 partitioned GPUs - which physical GPU a rank landed on is carried by the UUID, not by the ordinal, so every rank says `cuda:0` and every rank is on different hardware.

That leaves every rank reporting `LOCAL_RANK=0`, which is worth being precise about.

Nothing in `torch.distributed` is affected. `dist.get_rank()`, `dist.get_world_size()` and the process group are built from `RANK` and `WORLD_SIZE`, which the launch wrapper doesn't touch. The usual device-selection idiom also keeps working unchanged:

```python
local_rank = int(os.environ["LOCAL_RANK"])   # 0 in every process
torch.cuda.set_device(local_rank)            # binds to this rank's own instance
model = DDP(model.to(local_rank), device_ids=[local_rank])
```

Each process asks for device 0 and each one gets a different slice, because `CUDA_VISIBLE_DEVICES` differs per rank.

What breaks is code that treats `LOCAL_RANK` as an identity rather than as a device ordinal, and the node-leader guard is the common case:

```python
if local_rank == 0:
    download_dataset()   # meant to run once per node
```

Under MIG that condition is true in every process, so all the ranks download the dataset, write the same log file, or race to populate the same cache directory.

In a single-node emulation the fix is to use `RANK`, which is still unique across the job:

```python
if dist.get_rank() == 0:
    download_dataset()
```

If the code must keep working unmodified on real multi-node runs, preserve the original slot number under a different name instead. The launch wrapper stashes it before overwriting:

```bash
torchrun --nproc_per_node 8 --no-python bash -c '
export MIG_SLOT=$LOCAL_RANK
export CUDA_VISIBLE_DEVICES=$(echo $MIG_UUIDS | cut -d, -f$((MIG_SLOT+1)))
export LOCAL_RANK=0
exec python train.py'
```

and the guard reads whichever variable is present:

```python
node_slot = int(os.environ.get("MIG_SLOT", os.environ["LOCAL_RANK"]))
if node_slot == 0:
    download_dataset()
```

#### Only 7 instances per GPU

MIG cuts along boundaries the silicon already has. A compute slice comes from a GPC (Graphics Processing Cluster), the top-level block that bundles a set of SMs with its own work distribution, and a memory slice comes from the memory system attached to the HBM stacks. Ampere, Hopper, Blackwell, and recently Rubin, according to NVIDIA sources, each contain exactly 8 GPCs in the full GPU implementation (though a shipping SKU may have fewer enabled).

MIG can have a maximum of 7 instances and not 8 because the eighth compute unit is reserved to manage the instances - it is held back the moment MIG mode is enabled, whether you then create 7 instances or one. The memory system has no such tenant, so all 8 memory slices stay available, and the MIG documentation states the resulting ratio outright: each memory slice is 1/8 of the memory, each SM slice is 1/7 of the SMs.

That asymmetry is what the profile table above shows. Seven `1g` instances consume all 7 compute slices but only 7 of the 8 memory slices, so one slice worth of HBM is stranded on a fully partitioned GPU - 18GB of the H200's 141GB. It is also why a `4g` profile allows just one instance: a second would need 8 compute slices and there are only 7.

#### `torch.cuda.device_count()` disagrees with `LOCAL_WORLD_SIZE`

A process can see only its own instance, so `torch.cuda.device_count()` returns 1 in every rank. On whole GPUs that call happens to equal the number of ranks on the node, and plenty of code relies on it - to size a shard count, to build a list of devices to iterate, or to check that the requested world fits on the node. Take the number from the environment instead:

```python
ranks_on_this_node = torch.cuda.device_count()               # 1 in every rank under MIG
ranks_on_this_node = int(os.environ["LOCAL_WORLD_SIZE"])     # what torchrun was given as --nproc_per_node
```

`LOCAL_WORLD_SIZE` is set by torchrun in every worker, alongside `RANK`, `LOCAL_RANK` and `WORLD_SIZE`, and it keeps reporting the full local rank count under MIG.
