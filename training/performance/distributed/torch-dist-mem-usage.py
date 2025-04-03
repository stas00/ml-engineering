#!/usr/bin/env python

"""

This script demonstrates that when using `torch.distributed` a few GBs of GPU memory is taken away per GPU.

*** To do a quick test on 2 GPUs:

python -u -m torch.distributed.run --nproc_per_node=2 --rdzv_endpoint localhost:6000  --rdzv_backend c10d \
torch-dist-mem-usage.py

Watch the NV column (which is the equivalent of memory usage in `nvidia-smi`).


"""

import gc
import os
import psutil
import pynvml
import torch
import torch.distributed as dist

def see_memory_usage(message, force=False, ranks=[0]):
    """
    Arguments:
        message: a pre-amble message to print before the counter dumps - useful for annotating where each measurement has been taken - e.g. "before foo" and later "after foo"
        force: allows you to leave see_memory_usage in the code w/o running the code, force=True to activate
        ranks: by default prints only on rank 0 but sometimes we need to debug other ranks, so pass the list. Example: ranks=[1,3]
    """

    if not force:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    if not rank in ranks:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # this would be bad for production, only use during debug
    torch.cuda.empty_cache()

    # collect raw memory usage outside pytorch
    pynvml.nvmlInit()
    rank = dist.get_rank() if dist.is_initialized() else 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    nv_mem = memory_info.used

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)

    accelerator_mem_str = " | ".join([
        f"MA {round(torch.cuda.memory_allocated() / 2**30, 2):0.2f} GB",
        f"Max_MA {round(torch.cuda.max_memory_allocated() / 2**30, 2):0.2f} GB",
        f"CA {round(torch.cuda.memory_reserved() / 2**30, 2):0.2f} GB",
        f"Max_CA {round(torch.cuda.max_memory_reserved() / 2**30, 2):0.2f} GB",
        f"NV {round(nv_mem / 2**30, 2):0.2f} GB",
    ])
    cpu_mem_str = f"CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%"

    # add '[rank] mp' prefix to enable easy grep
    print(f"[{rank}] mp: {message}")
    print(f"[{rank}] mp: " + " | ".join([accelerator_mem_str, cpu_mem_str]))

    # get the peak memory to report correct data, so reset the counter for the next call
    torch.cuda.reset_peak_memory_stats()


def init_processes(local_rank, backend='nccl'):
    torch.cuda.set_device(local_rank)

    # if we don't pass `device_id` arg, the memory allocation won't happen till the first `barrier` call in this example.
    dist.init_process_group(backend)
    # if passing device_id arg, some memory will get used earlier already in `init_process_group`
    # device = torch.device("cuda", local_rank)
    # dist.init_process_group(backend, device_id=device)
    see_memory_usage("before barrier", force=True)
    dist.barrier()
    see_memory_usage("after barrier", force=True)
    dist.barrier()
    see_memory_usage("after 2nd barrier", force=True)
    dist.destroy_process_group()
    see_memory_usage("after dist destroy", force=True)

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_processes(local_rank=local_rank)
