#!/usr/bin/env python

"""

This benchmark shows that num_workers>0 leads to a better performance

usage:

./num-workers-bench.py

"""

import torch
import time

class MyDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.tensor = torch.ones(1*2**18) # 1 mb tensor

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        # emulate a slow data transform
        time.sleep(0.005)
        return self.tensor

num_runs = 10
num_workers = 5
batch_size = 100
compute_emulation_time = 0.2

ds = MyDataset()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
device = "cuda:0"

for num_workers in range(5):
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    duration = 0
    for i in range(num_runs):
        slept_time = 0
        start_event.record()
        for batch in dl:
            batch = batch.to(device=device, non_blocking=True)
            # emulate a compute delay to give workers a chance to reload, otherwise the benchmark
            # will be measuring waiting for workers
            time.sleep(compute_emulation_time)
            # will then subtract this artificial delay from the total to try to isolate
            # the iterator's overhead
            slept_time += compute_emulation_time
        end_event.record()
        torch.cuda.synchronize()
        duration += start_event.elapsed_time(end_event) / 1000 - slept_time
    duration /= num_runs
    print(f"num_workers={num_workers}: average time: {duration:0.3f}")
