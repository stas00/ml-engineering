#!/usr/bin/env python

"""

This is Maximum Achievable Matmul FLOPS (MAMAF) finder

While some accelerator manufacturers publish the theoretical TFLOPS these usually can't be reached. As a result of this when we try to optimize our software we have no realistic performance bar to compare ourselves to. The Model FLOPS Utilization (MFU) metric measures TFLOPS achieved against theoretical TFLOPS. Usually when one scores around 50% MFU it's considered a win. But this gives us no indication how far are we from the real achievable throughput.

This benchmark scans various large shapes of matmul and reports the highest achievable TFLOPS it registered. As transformers training and inference workloads are dominated by large matmul operations it's safe to use the best matmul TFLOPS one can measure on each accelerator as a rough estimation that this is the Maximum Achievable Matmul FLOPS (MAMAF). Now instead of the previously used MFU, one can use Model Achievable Matmul FLOPS Utilization (MAMFU).

Therefore now you can compare the TFLOPS you measured for your training or inference against a realistic number. As you will now be much closer to 100% it'll be much easier to know when to stop optimizing.

Currently supported high end architectures:
- NVIDIA: V100, A100, H100, ...
- AMD: MI250, MI300X, ...
- Intel Gaudi2+

Fairness notes:
- if you can find a better and more efficient way to detect the best matmul TFLOPS by approaching each new accelerator as a black box, please kindly send a PR with the improvement.
- also if you know that this benchmark should be run under special conditions to show the best results, such some some kernel settings or similar, please submit a PR to add such special instructions. For example, for AMD I'm being told disabling the numa_balancing is supposed to help

Architecture specific notes:

Follow the special setup instructions before running the benchmark to achieve the best results:

** MI300x **

Turn numa_balancing off for better performance:

sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'



Credits:
- Parts of this benchmark have been derived from https://github.com/EleutherAI/cookbook/tree/main/benchmarks/sizing (highly recommended!)
- Imtiaz Sajwani: HPU porting

Examples of usage:

clear; ./mamaf-finder.py --m_range 0 30480 1024 -n 1024 -k 1024 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
clear; ./mamaf-finder.py --m_range 0 21480 256 -n 2048 -k 2048 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
clear; ./mamaf-finder.py --m_range 0 21480 256 -n 4096 -k 4096 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt

"""

from pathlib import Path
import argparse
import datetime
import numpy as np
import os
import platform
import shlex
import signal
import sys
import time
import torch


has_hpu = False
try:
    import habana_frameworks.torch as ht
    if torch.hpu.is_available():
        has_hpu = True
except ModuleNotFoundError:
    pass

file_dir = os.path.abspath(os.path.dirname(__file__))



### Architecture specific helper classes ###

class Arch:
    def __init__(self):
        self.arch = "unknown"

    def __repr__(self):
        return self.arch

class CUDAArch(Arch):
    """ shared with CUDA and ROCm: NVIDIA + AMD """
    def __init__(self):
        if torch.version.hip is not None:
            self.arch = "rocm"
        else:
            self.arch = "cuda"

    def device(self):
        return torch.device('cuda:0')

    def name(self):
        return self.arch

    def device_info(self):
        return torch.cuda.get_device_properties(device)

    def compute_info(self):
        if self.arch == "rocm":
            return f"hip={torch.version.hip}, cuda={torch.version.cuda}"
        else:
            return f"cuda={torch.version.cuda}"

    def event(self, enable_timing=True):
        return torch.cuda.Event(enable_timing)

    def synchronize(self):
        torch.cuda.synchronize()

class HPUArch(Arch):
    """ Intel Gaudi* """
    def __init__(self):
        self.arch = "hpu"

    def device(self):
        return torch.device('hpu')

    def name(self):
        return self.arch

    def device_info(self):
        return torch.hpu.get_device_properties(device)

    def compute_info(self):
        return f"hpu={torch.version.hpu}"

    def event(self, enable_timing=True):
        return ht.hpu.Event(enable_timing)

    def synchronize(self):
        ht.hpu.synchronize()


def get_accelerator_arch():
    """
    returns: CUDAArch or HPUArch object
    """
    # cuda / rocm
    if torch.cuda.is_available():
        return CUDAArch()

    # hpu
    if has_hpu:
        return HPUArch()

    raise ValueError("Currently only cuda, rocm and hpu are supported")

arch = get_accelerator_arch()



### Helper classes ###

class Tee(object):
    def __init__(self, filename, verbose):
        Path(filename).resolve().parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filename, "w")
        self.verbose = verbose
        if self.verbose:
            self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        if self.verbose:
            self.stdout.write(message)

    def flush(self):
        self.file.flush()
        if self.verbose:
            self.stdout.flush()


def print_benchmark_header(dtype, device, notes="None"):

    device_info = arch.device_info()
    compute_info = arch.compute_info()

    print(f"""
Benchmark started on {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}

** Command line:
{sys.executable} {" ".join(map(shlex.quote, sys.argv))}

** Dtype: {dtype}

** Platform/Device info:
{" ".join(platform.uname())}
{device_info}

** Critical software versions:
torch={torch.__version__}
{compute_info}

** Additional notes:
{notes}

{"-" * 80}

""")

# Benchmark of a basic GEMM
def benchmark_mm(m, n, k, dtype, device, num_iterations, num_warmup_iterations):
    start = arch.event(enable_timing=True)
    end = arch.event(enable_timing=True)

    A = torch.randn(m, n, dtype=dtype, device=device)
    B = torch.randn(n, k, dtype=dtype, device=device)
    C = torch.empty(m, k, dtype=dtype, device=device)

    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            torch.mm(A, B, out=C)
            end.record()
        arch.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times)/1000 # want the fastest
    tflops = (2 * m * n * k) / (elapsed_time * 10**12)
    return tflops


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    m_group = parser.add_mutually_exclusive_group(required=True)
    m_group.add_argument("-m", nargs="+", type=int, help='The first dimension of the GEMM, enter any number of arguments')
    m_group.add_argument("--m_range", nargs='+', type=int, help="The first dimension of the GEMM, [start,stop,step]")

    n_group = parser.add_mutually_exclusive_group(required=True)
    n_group.add_argument("-n", nargs="*", type=int, help='The shared dimension of the GEMM, enter any number of arguments')
    n_group.add_argument("--n_range", nargs='+', type=int, help="The shared dimension of the GEMM, [start,stop,step]")

    k_group = parser.add_mutually_exclusive_group(required=True)
    k_group.add_argument("-k", nargs="*", type=int, help='The last dimension of the GEMM, enter any number of arguments')
    k_group.add_argument("--k_range", nargs='+', type=int, help="The last dimension of the GEMM, [start,stop,step]")

    parser.add_argument("--num_iterations", type=int, default=100, help='The number of iterations used to benchmark each GEMM')
    parser.add_argument("--num_warmup_iterations", type=int, default=50, help='The number of warmup iterations')
    parser.add_argument("--cuda_device", type=int, default=0, help="The cuda device to run the benchmark on")
    parser.add_argument("--output_file", type=str, default=f"{file_dir}/results/mm.out")
    parser.add_argument("--notes", type=str, default="", help="benchmark-specific notes to add to the output_file's header")
    parser.add_argument("--verbose", default=True, action=argparse.BooleanOptionalAction, help='log to stdout besides output_file?')
    args = parser.parse_args()

    m = args.m
    n = args.n
    k = args.k

    dtype = torch.bfloat16
    device = arch.device()

    if m is None:
        start, stop, step = args.m_range
        m = np.arange(start, stop, step)
    if n is None:
        start, stop, step = args.n_range
        n = np.arange(start, stop, step)
    if k is None:
        start, stop, step = args.k_range
        k = np.arange(start, stop, step)

    sys.stdout = Tee(args.output_file, args.verbose)
    print_benchmark_header(dtype, device, args.notes)

    # this is useful for when one wants to interrupt the run - and still report the best outcome so far
    def sigkill_handler(signum, frame):
         finish()
         sys.exit(1)

    signal.signal(signal.SIGINT, sigkill_handler)

    best_tflops = 0
    best_config = ""
    num_shapes = 0
    start_time = time.time()

    def finish():
        time_delta = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
        print("", end="\033[K")
        print(f"The best outcome was {best_tflops:.1f}TFLOPS @ {best_config} (tried {num_shapes} shapes)")
        print(f"Elapsed time: {time_str}")

    # XXX: the transpose version seemed to work better for MI300X

    # loop through all sizes to benchmark
    for M in m:
        for N in n:
            for K in k:
                num_shapes += 1
                tflops = benchmark_mm(M, N, K, dtype, device, args.num_iterations, args.num_warmup_iterations)
                if tflops > best_tflops:
                    best_tflops = tflops
                    best_config = f"{M}x{N}x{K} (MxNxK)"
                print(f"{num_shapes:>5} | {tflops:6.1f} TFLOPS @ {M}x{N}x{K} | best: {best_tflops:6.1f} TFLOPS @ {best_config}", end="\r")
    finish()
