#!/usr/bin/env python

#
# all_gather to gather counts across process group is 23x faster than the same via all_gather_object
#
# python -m torch.distributed.run --nproc_per_node 2 all_gather_object_vs_all_gather.py
#
# XXX: in this case the benchmark isn't the most representative since there is almost no data, so
# the overhead of code is huge, shouldn't be as big for bigger data. But I wanted to compare
# all_gather to all_gather_object and used the same setup as all_gather_object_vs_all_reduce.py as
# the base for the benchmark. Probably need to rework it.
#
# all_gather_object=0.2697904680026113
# all_gather_object=0.26981512399652274
# all_gather       =0.05322460600291379
# all_gather       =0.05485054099699482

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

def all_gather():
    tensor_list = [torch.zeros(1, dtype=torch.float, device=device) for _ in range(2)]
    dist.all_gather(tensor_list, flag_pt)
    return tensor_list

# test
print(f"all_gather_object: {all_gather_object()}\n")
print(f"all_gather: {all_gather()}\n")

import timeit
print(f'all_gather_object={timeit.Timer("all_gather_object()", globals=globals()).timeit(number=1000)}')
print(f'all_gather       ={timeit.Timer("all_gather()"       , globals=globals()).timeit(number=1000)}')
