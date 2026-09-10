#!/usr/bin/env python
import torch, torch.distributed as dist
from datetime import timedelta

def buggy(x, rank):
    dist.all_reduce(x)          # both ranks take part - fine
    if rank == 0:
        dist.all_reduce(x)      # BUG: only rank 0 calls this -> everyone hangs

def main():
    dist.init_process_group("nccl", timeout=timedelta(seconds=8))
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    x = torch.ones(4, device="cuda")
    buggy(x, rank)
    dist.barrier()

main()
