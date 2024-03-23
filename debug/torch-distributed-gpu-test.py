#!/usr/bin/env python

"""

This a `torch.distributed` diagnostics script that checks that all GPUs in the cluster (one or
many nodes) can talk to each other via nccl and allocate gpu memory. It also prints other useful information like NUMA affinities.

To run it you just need to adjust the number of processes and nodes according to your use case:

```
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

You may need to add `--master_addr $MASTER_ADDR --master_port $MASTER_PORT` if using a custom addr:port

You can also use the rdzv API: `--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d`

If you get a hanging in `barrier` calls you have some network issues, you may try to debug this with:

```
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

which should tell you what's going on behind the scenes.

This script can be run via `srun` in the SLURM environment as well. Here is a SLURM script that
runs on 2 nodes of 8 gpus per node:

```
#!/bin/bash
#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # EDIT to the number of nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per node for this script
#SBATCH --cpus-per-task=10           # EDIT this to how many cpu cores the node has
#SBATCH --gres=gpu:8                 # EDIT this if it's not an 8-GPUs node setup
#SBATCH --partition=dev              # EDIT to the desired partition name
#SBATCH --time 0:05:00               # 5 min should be enough
#SBATCH --output=%x-%j.out           # output file name

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
torch-distributed-gpu-test.py'
```

You can also add this to the launcher for automatic prefixing of all logs with `[hostname:rank] ` (e.g. after `--master_addr`):

```
--role `hostname -s`: --tee 3
```

"""

import builtins
import fcntl
import os
import socket
import torch
import torch.distributed as dist

def print(*args, **kwargs):
    """ solves multi-process interleaved print problem """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
hostname = socket.gethostname()

gpu = f"[{hostname}:{local_rank}]"

try:
    # XXX: possibly change the dist timeout to something much shorter to get this script to fail
    # fast if there is a problem and not wait for the default 30min

    # test distributed
    dist.init_process_group("nccl")

    # global rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # reduction test
    t = torch.ones(1, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    dist.barrier()
    print(f"{gpu} Reduction op=sum result: {t.item()}")

    # test cuda is available and can allocate memory
    torch.cuda.is_available()
    torch.ones(1).cuda(local_rank)

    print(f"{gpu} is OK (global rank: {rank}/{world_size})")

    dist.barrier()
    if rank == 0:
        print(f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
        print(f"device compute capabilities={torch.cuda.get_device_capability()}")
        print(f"pytorch compute capabilities={torch.cuda.get_arch_list()}")

except Exception:
    print(f"{gpu} is broken (but it could also mean that it failed because another gpu didn't respond)")
    raise
