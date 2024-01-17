#!/usr/bin/env python

# this is derived from the all_reduce_bench.py
# but adjusted to show how 1x 4GB reduction is much faster than 1000x 4MB reduction
#
# to run on 8 gpus:
# python -u -m torch.distributed.run --nproc_per_node=8 all_reduce_latency_comp.py

import os
import socket
import torch
import torch.distributed as dist

TRIALS = 1

# these emulate the payload which will become a M * N * 4-sized tensor below
N = 500000
M = 2000

def timed_allreduce(mat, repeat_times, id, start_event, end_event):
    start_event.record()
    for i in range(repeat_times):
        dist.all_reduce(mat)
    end_event.record()

    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / 1000

    size = M * N * 4 # 4 is fp32
    algbw = (size / duration) * 8 # 8 is bytes to bits
    n = dist.get_world_size()
    # the 2*(n-1)/n busbw correction factor specific to all-reduce is explained here:
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
    # busbw reflects how optimally the hardware is used
    busbw = algbw * (2*(n - 1) / n)

    # gather all data on global-rank-0 and print the results from there to avoid interleaved prints
    data = [id, duration, algbw, busbw]
    output = [None for _ in range(dist.get_world_size())] if dist.get_rank() == 0 else None
    dist.gather_object(data, output, dst=0)
    if dist.get_rank() == 0:
        for data in output:
            id, duration, algbw, busbw = data
            print(f"{id}:\n",
                  f"duration: {duration:.3f} sec\n",
                  f"algbw: {algbw/1e9:.3f} Gbps\n",
                  f"busbw: {busbw / 1e9:.3f} Gbps"
    )



def run(local_rank):
    hostname = socket.gethostname()
    id = f"{hostname}:{local_rank}"
    global_rank = dist.get_rank()

    chunks = 1000
    mat1 = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)
    mat2 = torch.rand(int(N/chunks), M, dtype=torch.float32).cuda(local_rank)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(TRIALS):
        dist.barrier()

        if global_rank == 0:
            print(f"\n\n\n----------- 1x {N*M*4/1e9}GB ----------------")
        timed_allreduce(mat1, 1, id, start_event, end_event)

        if global_rank == 0:
            print(f"\n\n\n----------- {chunks}x {(N*M*4/chunks)/1e9}GB ----------------")
        timed_allreduce(mat2, chunks, id, start_event, end_event)

def init_processes(local_rank, fn, backend='nccl'):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend)
    fn(local_rank)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    print("local_rank: %d" % local_rank)
    init_processes(local_rank=local_rank, fn=run)
