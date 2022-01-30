#!/usr/bin/env python

# printflock allows one to print in a non-interleaved fashion when printing from multiple procesess.
# Typically this only the issue within a single node. When processes from different nodes print their
# output it doesn't get interleaved.
#
# This file includes the wrapper and a full example on how to use it.
#
# e.g., if you have 2 gpus run it as:
#
# python -m torch.distributed.run --nproc_per_node 2 multi-gpu-non-interleaved-print.py
#

import fcntl
def printflock(*args, **kwargs):
    """
    non-interleaved print function for using when printing concurrently from many processes,
    like the case under torch.distributed
    """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)



if __name__ == "__main__":

    import torch.distributed as dist
    import torch
    import os
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    printflock(f"This is a very long message from rank {rank} (world_size={world_size})")
