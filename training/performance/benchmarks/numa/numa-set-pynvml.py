# this helper util will assign the cpu-cores belonging to the same NUMA node as the GPU

# derived from
# https://github.com/NVIDIA/DeepLearningExamples/blob/9dd9fcb98f56187e49c5ee280cf8dbd530dde57b/TensorFlow2/LanguageModeling/BERT/gpu_affinity.py

import os
import math
import pynvml as nvml

nvml.nvmlInit()

def set_numa_affinity(gpu_index, verbose=False):
    """This util will assign to the current process the cpu cores set that resides on the same NUMA
    node as the GPU. Typically if you have 8 GPUs, then the first 4 are on the first NUMA node and
    the remaining 4 are on the second.

    `gpu_index` is typically the same as `LOCAL_RANK` in the distributed training, but beware that
    `CUDA_VISIBLE_DEVICES` could impact that. e.g. `CUDA_VISIBLE_DEVICES=0,7` won't do the right
    thing - then you will probably want to remap the ids with something like:

    ```
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
        gpu_index = ids[gpu_index] # remap
    ```

    """


    num_elements = math.ceil(os.cpu_count() / 64)
    handle = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
    affinity_string = ""
    for j in nvml.nvmlDeviceGetCpuAffinity(handle, num_elements):
        # assume nvml returns list of 64 bit ints
        affinity_string = f"{j:064b}{affinity_string}"
    affinity_list = [int(x) for x in affinity_string]
    affinity_list.reverse()  # so core 0 is the 0th element
    affinity_to_set = [i for i, e in enumerate(affinity_list) if e != 0]

    if verbose:
        cores = os.sched_getaffinity(0)
        print(f"before: {len(cores)} visible cpu cores: {cores}")
    os.sched_setaffinity(0, affinity_to_set)
    if verbose:
        cores = os.sched_getaffinity(0)
        print(f"after: {len(cores)} visible cpu cores: {cores}")

if __name__ == "__main__":

    # pretend we are process that drives gpu 0
    set_numa_affinity(0, verbose=True)
