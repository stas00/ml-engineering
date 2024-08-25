import os
import socket
import torch
import torch.distributed as dist
import numpy as np

TRIALS = 5

# Emulate payload size. Adjust dynamically based on system load.
N = 500000
M = 2000

def adaptive_warmup_iterations(prev_durations):
    """
    Adaptive logic to determine optimal number of warm-up iterations.
    """
    if not prev_durations:
        return 2  # Default warm-up iterations if no previous data
    mean_duration = np.mean(prev_durations)
    if mean_duration < 100:  # Threshold for adjusting warm-ups
        return 1
    return 3

def timed_allreduce(mat, start_event, end_event):
    dist.barrier()
    start_event.record()
    dist.all_reduce(mat)
    end_event.record()

    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / 1000

    n = dist.get_world_size()
    size = M * N * 4  # 4 bytes in fp32
    algbw = torch.tensor([size / duration]).cuda(local_rank)

    dist.reduce(algbw, dst=0, op=dist.ReduceOp.SUM)
    algbw /= n

    return algbw, duration

def gather_performance_metrics():
    """
    Collect additional performance metrics like GPU utilization and memory bandwidth.
    """
    import psutil

    cpu_usage = psutil.cpu_percent(interval=1)
    gpu_mem = torch.cuda.memory_allocated(local_rank) / 1e9  # GB
    return cpu_usage, gpu_mem

def run(local_rank):
    hostname = socket.gethostname()
    is_global_rank_0 = dist.get_rank() == 0

    mat = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    prev_durations = []
    warmups = adaptive_warmup_iterations(prev_durations)
    for _ in range(warmups):
        _, duration = timed_allreduce(mat, start_event, end_event)
        prev_durations.append(duration)

    algbw_gather = []
    for i in range(TRIALS):
        if is_global_rank_0:
            print(f"Trial {i+1}")
        algbw, duration = timed_allreduce(mat, start_event, end_event)
        algbw_gather.append(algbw)

    algbw = torch.mean(torch.stack(algbw_gather))

    n = dist.get_world_size()
    busbw = algbw * (2*(n - 1) / n)

    cpu_usage, gpu_mem = gather_performance_metrics()

    if is_global_rank_0:
        print(f"The average bandwidth of all_reduce with a {M*N*4/1e9:.3f}GB payload ({TRIALS} trials, {n} ranks):\n",
              f"algbw: {algbw/1e9:.3f} GBps ({algbw*8/1e9:.1f} Gbps)\n",
              f"busbw: {busbw/1e9:.3f} GBps ({busbw*8/1e9:.1f} Gbps)\n",
              f"CPU Usage: {cpu_usage:.2f}%\n",
              f"GPU Memory Usage: {gpu_mem:.2f} GB\n")

def init_processes(local_rank, fn, backend='nccl'):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend)
    fn(local_rank)

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_processes(local_rank, fn=run)
