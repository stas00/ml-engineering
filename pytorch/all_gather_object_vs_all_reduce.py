#!/usr/bin/env python

#
# all_reduce to gather counts across process group is 23x faster than the same via all_gather_object
#
# python -m torch.distributed.run --nproc_per_node 2 all_gather_object_vs_all_reduce.py
#
# all_gather_object=0.26279118900129106
# all_gather_object=0.2628160299973388
# all_reduce       =0.011241967000387376
# all_reduce       =0.011610440000367817

import torch.distributed as dist
import torch
import os

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

world_size = dist.get_world_size()
rank = dist.get_rank()

flag_pt = torch.tensor(1.0, device=device)
flag_py = 1

def all_gather_object():
    output_objects = [None for _ in range(world_size)]
    dist.all_gather_object(output_objects, flag_py)
    flag = sum(output_objects)
    return flag

def all_reduce():
    dist.all_reduce(flag_pt, op=dist.ReduceOp.SUM)
    return flag_pt

# test
print(f"all_gather_object: {all_gather_object()}\n")
print(f"all_reduce: {all_reduce()}\n")

import timeit
print(f'all_gather_object={timeit.Timer("all_gather_object()", globals=globals()).timeit(number=1000)}')
print(f'all_reduce       ={timeit.Timer("all_reduce()"       , globals=globals()).timeit(number=1000)}')
