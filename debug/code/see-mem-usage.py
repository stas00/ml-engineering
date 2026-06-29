"""

a small debug library for tracing cpu and gpu (cuda) memory, see the `see_memory_usage` doc for usage patterns.

requires: `pip install nvidia-ml-py`

"""

import gc
import os
import psutil
import torch
import torch.distributed as dist

can_run_pynvml = True
try:
    import pynvml

    pynvml.nvmlInit()
except Exception:
    can_run_pynvml = False

pynvml_handle = None
def get_nvml_mem():
    global pynvml_handle

    if not can_run_pynvml:
        return 0

    if pynvml_handle is None:
        device_id = get_device_id()
        if device_id is None:
            return 0
        pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    memory_info = pynvml.nvmlDeviceGetMemoryInfo(pynvml_handle)
    return memory_info.used


def get_device_id():
    """
    Derive the device id running this rank with the help of LOCAL_RANK and CUDA_VISIBLE_DEVICES env vars. The device id is
    needed for applications like pynvml.

    returns `None` if CUDA_VISIBLE_DEVICES is set to ""
    """

    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    if cuda_visible_devices == "":
        return None
    visible_device_ids = list(map(int, cuda_visible_devices.split(",")))

    if dist.is_initialized():
        local_rank = int(os.getenv("LOCAL_RANK", 0))
    else:
        local_rank = 0

    return visible_device_ids[local_rank]


def see_memory_usage(message, force=False, ranks=[0]):
    """
    Arguments:
        - `message`: a pre-amble message to print before the counter dumps - useful for annotating where each measurement has been taken - e.g. "before foo" and later "after foo"
        - `force`: allows you to leave see_memory_usage in the code w/o running the code, set `force=True` to activate
        - `ranks`: by default prints only on rank 0 but if needing to debug other ranks, pass the list of desirable ranks, e.g., `ranks=[1,3]`

    You want to make sure `pip install nvidia-ml-py` is run, so that the report include not only the CUDA memory report but the total gpu memory usage, since CUDA memory allocator is not always used. e.g. NCCL memory allocations aren't visible by CUDA and thus aren't reported, but can consume GBs of gpu memory.

    Pattern of usage:

    see_memory_usage("before fwd", force=True)
    output = model(**inputs)
    see_memory_usage("before bwd", force=True)
    output.loss.backward()
    see_memory_usage("before step", force=True)
    optimizer.step()
    see_memory_usage("after step", force=True)

    """
    if not force:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank not in ranks:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # In some situations we want to flush the cache but not others, so for now let the developer
    # override this manually - by default it should not be called. when it's not enabled use the
    # MA_* numbers to get the real memory usage, rather than CA_* ones
    # torch.cuda.empty_cache()

    # collect raw memory usage outside pytorch
    nv_mem = get_nvml_mem()

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)

    accelerator_mem_str = " | ".join(
        [
            f"MA {round(torch.cuda.memory_allocated() / 2**30, 2):0.2f} GB",
            f"Max_MA {round(torch.cuda.max_memory_allocated() / 2**30, 2):0.2f} GB",
            f"CA {round(torch.cuda.memory_reserved() / 2**30, 2):0.2f} GB",
            f"Max_CA {round(torch.cuda.max_memory_reserved() / 2**30, 2):0.2f} GB",
            f"NV {round(nv_mem / 2**30, 2):0.2f} GB",
        ]
    )
    cpu_mem_str = f"CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%"

    # add '[rank] mp' prefix to enable easy grep
    print(f"[{rank}] mp: {message}")
    print(f"[{rank}] mp: " + " | ".join([accelerator_mem_str, cpu_mem_str]))

    # get the peak memory to report correct data, so reset the counter for the next call
    torch.cuda.reset_peak_memory_stats()



if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    see_memory_usage("before alloc", force=True)
    t1 = torch.zeros(100000,10000, device=device)
    t2 = torch.zeros(100000,10000, device=device)
    del t2
    see_memory_usage("after alloc", force=True)
    c1 = t1.cpu()
    see_memory_usage("after copy to cpu", force=True)
    del t1
    see_memory_usage("after freeing on gpu", force=True)
    del c1
    see_memory_usage("after freeing on cpu", force=True)