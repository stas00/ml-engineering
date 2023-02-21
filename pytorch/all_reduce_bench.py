# this version has been derived from @jeffra's gist: https://gist.github.com/jeffra/b5e80466b4c86be00ea3b6f130fb7a36
#
# python -m torch.distributed.run --nproc_per_node=2 all_reduce_bench.py

import argparse
import fcntl
import os
import socket
import time
import torch
import torch.distributed as dist

# note: this benchmark doesn't care how many gpus per node one has

TRIALS = 5

N = 500000
M = 2000

def printflock(*msgs):
    """ print """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

def timed_allreduce(mat, id):
    pre = time.perf_counter()
    dist.all_reduce(mat)
    printflock(f"ignore me {int(mat[0][0])}")  # required due to lazy evaluation
    duration = time.perf_counter() - pre
    tput = ((M*N*4*2)/duration)*8 # *2 is for send + receive, *8 for gigabits/second
    size = M * N * 4 # 4 is fp32
    n = dist.get_world_size()
    busbw = (size / duration) * (2 * (n - 1) / n) * 8
    printflock(f"{id}:\n",
               f"duration: {duration:.4f} sec\n",
               f"algo throughput: {tput:.4f} bps, {tput/1e9:.4f} Gbps\n",
               f"busbw: {busbw / 1e9:.4f}  Gbps"
    )

def run(local_rank):
    hostname = socket.gethostname()
    id = f"{hostname}:{local_rank}"
    global_rank = dist.get_rank()

    printflock(f"{id} data size: {M*N*4/1e9} GB")
    mat = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)

    for i in range(TRIALS):
        dist.barrier()
        if global_rank == 0:
            print(f"\n\n\n-----------trial-{i}----------------")
        timed_allreduce(mat, id)

def init_processes(local_rank, fn, backend='nccl'):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend)
    fn(local_rank)


if __name__ == "__main__":
    rank = int(os.environ["LOCAL_RANK"])
    printflock("local_rank: %d" % rank)
    init_processes(local_rank=rank, fn=run)
