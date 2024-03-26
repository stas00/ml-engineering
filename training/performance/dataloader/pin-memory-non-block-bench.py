#!/usr/bin/env python

"""

This benchmark shows that a combo of:

(1) DataLoader(pin_memory=True, ...)
(2) batch.to(device="cuda", non_blocking=True)

leads to a faster transfer from the workers to the process doing compute and a potential overlap between the compute and the data movement

See:
- https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
- https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/

usage:

./pin-memory-non-block-bench.py

"""

import torch
import time

class MyDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.tensor = torch.ones(1*2**18) # 1 mb tensor

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return self.tensor

n_runs = 10
n_workers = 5
batch_size = 100

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
device = "cuda:0"

for pm in [True, False]:

    ds = MyDataset()
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=pm,
        num_workers=n_workers,
    )
    duration = 0
    for i in range(n_runs):
        slept = 0
        start_event.record()
        for batch in dl:
            # non_blocking=True would further speeds things up in addition to pinned memory
            batch = batch.to(device=device, non_blocking=pm)
            # emulate a compute delay to give workers a chance to reload, otherwise the benchmark
            # will be measuring waiting for workers
            time.sleep(0.2)
            slept += 0.2 # will then subtract this delay from the time measurement
        end_event.record()
        torch.cuda.synchronize()
        duration += start_event.elapsed_time(end_event) / 1000 - slept
    duration /= n_runs
    print(f"pin_memory={pm!s:>5}: average time:", duration)
