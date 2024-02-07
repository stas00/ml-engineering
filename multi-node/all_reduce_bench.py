#!/usr/bin/env python

"""

The latest version of this program can be found at https://github.com/stas00/ml-engineering

This benchmark is very similar to https://github.com/NVIDIA/nccl-tests but it's much easier to set
up as it only requires PyTorch to be installed

This version:
- has been derived from @jeffra's gist: https://gist.github.com/jeffra/b5e80466b4c86be00ea3b6f130fb7a36
- which in turn is derived from the logic in https://github.com/NVIDIA/nccl-tests
- with contributions from:
  * Indu Thangakrishnan https://github.com/indhub to handle timing correctly using cuda events


Important notes:

- when you finished running this benchmark you want to pay attention to the busbw result (not
  algbw) as explained here https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bandwidth

- similar to NVIDIA/nccl-tests this benchmark measures a unidirectional bandwidth - so compare the
  outcome against the advertised unidirectional peak throughput and not bi-directional (duplex)

- currently this benchmark tests a payload of 4GB (M * N * 4). If your target application uses a
  much smaller payload you want to modify M*N*4 to match the target payload. To calculate the
  payload use the number of parameters sent in each reduction multiplied by 2 (bf16/fp16) or 4
  (fp32). e.g., if a reduction is of a single layer of 1B params, and you use bf16 grads it'd be
  2GB of payload. depending on the framework you use (DDP, FSDP, DeepSpeed ZeRO) they all use
  different logic to how much of a message size they send.

- if you are wondering whether you need to also run https://github.com/NVIDIA/nccl-tests - I
  already validated that I got very similar results with ./build/all_reduce_perf -b 4G -e 4G
  (tested with mpirun on 4 nodes). It should be either on par or slightly slower because it uses a
  blocking approach - that is it wait for each new all_reduce to finish before firing the next
  one, whereas nccl-tests fires them all in an async fashion (you can add `-z` to nccl-tests to
  emulate blocking)

- to benchmark other collectives use nccl-tests. It's also useful if you want to test a range of
  payloads, e.g. there you'd set -b 8 -e 4G -f 2 and it will test many sizes automatically.

To run on 4 nodes:

GPUS_PER_NODE=8
NNODES=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    all_reduce_bench.py

note: adapt MASTER_ADDR to node rank 0's hostname if it's not a SLURM environment where it's derived automatically

e.g. example to run with salloc+srun:

salloc --partition=mypartition --nodes=4 --ntasks-per-node=1 --cpus-per-task=48 --gres=gpu:8 --time=1:00:00 bash

srun --gres=gpu:8 --nodes=4 --tasks-per-node=1 python -u -m torch.distributed.run --nproc_per_node=8 \
--nnodes 4 --rdzv_endpoint $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):6000 --rdzv_backend \
c10d all_reduce_bench.py

To do a quick test on 2 gpus:

python -u -m torch.distributed.run --nproc_per_node=2 --rdzv_endpoint localhost:6000  --rdzv_backend c10d \
all_reduce_bench.py

"""

import os
import socket
import torch
import torch.distributed as dist

TRIALS = 5

# these emulate the payload which will become a M * N * 4-sized tensor below
N = 500000
M = 2000

def timed_allreduce(mat, start_event, end_event):
    dist.barrier()
    start_event.record()
    dist.all_reduce(mat)
    end_event.record()

    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / 1000

    n = dist.get_world_size()
    size = M * N * 4 # 4 is 4 bytes in fp32
    # note that this is following the same math as NVIDIA/nccl-tests
    algbw = torch.tensor([size / duration]).cuda(local_rank)

    # calculate mean across all ranks
    dist.reduce(algbw, dst=0, op=dist.ReduceOp.SUM)
    algbw /= n

    return algbw

def run(local_rank):
    hostname = socket.gethostname()
    is_global_rank_0 = dist.get_rank() == 0

    mat = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # do a few warm up iterations
    for i in range(2):
        timed_allreduce(mat, start_event, end_event)

    # real benchmark
    algbw_gather = []
    for i in range(TRIALS):
        if is_global_rank_0:
            print(i+1)
        algbw_gather += timed_allreduce(mat, start_event, end_event)

    algbw = torch.mean(torch.stack(algbw_gather))

    # the 2*(n-1)/n busbw correction factor specific to all-reduce is explained here:
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
    # busbw reflects how optimally the hardware is used
    n = dist.get_world_size()
    busbw = algbw * (2*(n - 1) / n)

    if is_global_rank_0:
        print(f"The average bandwidth of all_reduce with a {M*N*4/1e9}GB payload ({TRIALS} trials, {n} ranks):\n",
              f"algbw: {algbw/1e9:.3f} GBps ({algbw*8/1e9:.1f} Gbps)\n",
              f"busbw: {busbw/1e9:.3f} GBps ({busbw*8/1e9:.1f} Gbps)\n",
        )

def init_processes(local_rank, fn, backend='nccl'):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend)
    fn(local_rank)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_processes(local_rank=local_rank, fn=run)
