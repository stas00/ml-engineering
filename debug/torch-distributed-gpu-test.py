#!/usr/bin/env python

#
# This a `torch.distributed` diagnostics script that checks that all GPUs in the cluster (one or
# many nodes) can talk to each other via nccl and allocate gpu memory.
#
# To run first adjust the number of processes and nodes:
#
# python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
#
# You may need to add --master_addr $MASTER_ADDR --master_port $MASTER_PORT if using a custom addr:port
#
# You can also use the rdzv API: --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d
#
# use torch.distributed.launch instead of torch.distributed.run for torch < 1.9
#
# If you get a hanging in `barrier` calls you have some network issues, you may try to debug this with:
#
# NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
#
# which should tell you what's going on behind the scenes.
#
#
# This script can be run via `srun` in the SLURM environment as well. Here is a SLURM script that
# runs on 2 nodes of 4 gpus per node:

# #!/bin/bash
# #SBATCH --job-name=test-nodes        # name
# #SBATCH --nodes=2                    # nodes
# #SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
# #SBATCH --cpus-per-task=10           # number of cores per tasks
# #SBATCH --gres=gpu:4                 # number of gpus
# #SBATCH --time 0:05:00               # maximum execution time (HH:MM:SS)
# #SBATCH --output=%x-%j.out           # output file name
#
# export GPUS_PER_NODE=4
# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=6000
#
# srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
# --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# torch-distributed-gpu-test.py'
#
# can also add this for automatic prefixing of all logs with [hostname:rank] (in addition to `--master_addr` etc)
# --role `hostname -s`: --tee 3 \
#

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

gpu = f"[{hostname}-{local_rank}]"

try:
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
    print(f"{gpu} is broken")
    raise
