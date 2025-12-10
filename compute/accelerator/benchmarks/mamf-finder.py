#!/usr/bin/env python

"""

This is Maximum Achievable Matmul FLOPS (MAMF) Finder

For a quick run use:

python mamf-finder.py --m_range 0 20480 256 --n 4096 --k 4096 --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt

But this usually is an insufficient range to get the best results, therefore for multiple examples, discussion and multiple important nuances please refer to
https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks#maximum-achievable-matmul-flops-finder

The results are shared here: https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#maximum-achievable-matmul-flops-comparison-table

Credits:
- Parts of this benchmark have been derived from https://github.com/EleutherAI/cookbook/tree/main/benchmarks/sizing (highly recommended!)
- Imtiaz Sajwani: HPU porting
- Xiaoyu Zhang https://github.com/BBuf - flexible dtype support
- Oren Leung https://github.com/OrenLeung - flagging the lack of cache/dest-matrix reset and suggesting a fix - also proposing geomean
- Ivan Fioravanti https://github.com/ivanfioravanti - MPS support
"""

from pathlib import Path

import argparse
import datetime
import numpy as np
import os
import platform
import re
import shlex
import signal
import sys
import time
import torch
from packaging import version
from warnings import warn

# important: when changing how the benchmark measures things bump up its version, so that the old
# reports could be differentiated from the new ones
benchmark_version = 2

has_hpu = False
try:
    import habana_frameworks.torch as ht
    if torch.hpu.is_available():
        has_hpu = True
except ModuleNotFoundError:
    pass

file_dir = os.path.abspath(os.path.dirname(__file__))

def get_torch_dtype(dtype_str):
    """Convert string dtype to torch dtype object."""
    try:
        return getattr(torch, dtype_str)
    except AttributeError:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Must be a valid torch dtype name.")



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

    @property
    def device(self):
        return torch.device('cuda:0')

    @property
    def name(self):
        return self.arch

    @property
    def device_info(self):
        return torch.cuda.get_device_properties(device)

    @property
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

    @property
    def device(self):
        return torch.device('hpu')

    @property
    def name(self):
        return self.arch

    @property
    def device_info(self):
        return torch.hpu.get_device_properties(device)

    @property
    def compute_info(self):
        return f"hpu={torch.hpu}"

    def event(self, enable_timing=True):
        return ht.hpu.Event(enable_timing)

    def synchronize(self):
        ht.hpu.synchronize()

class XPUArch(Arch):
    """ Intel dGPUs (like ARC A770) """
    def __init__(self):
        self.arch = "xpu"

    @property
    def device(self):
        return torch.device('xpu')

    @property
    def name(self):
        return self.arch

    @property
    def device_info(self):
        return torch.xpu.get_device_properties(device)

    @property
    def compute_info(self):
        return f"xpu={torch.version.xpu}"

    def event(self, enable_timing=True):
        return torch.xpu.Event(enable_timing)

    def synchronize(self):
        torch.xpu.synchronize()

class MPSEvent:
    """Fallback event implementation for Apple's MPS backend."""
    def __init__(self):
        self._timestamp = None

    def record(self):
        torch.mps.synchronize()
        self._timestamp = time.perf_counter()

    def elapsed_time(self, other):
        if self._timestamp is None or other._timestamp is None:
            raise RuntimeError("Attempted to measure elapsed time before events were recorded")
        return (other._timestamp - self._timestamp) * 1000.0

class MPSArch(Arch):
    """ Apple Silicon GPUs via Metal Performance Shaders """
    def __init__(self):
        self.arch = "mps"

    @property
    def device(self):
        return torch.device('mps')

    @property
    def name(self):
        return self.arch

    @property
    def device_info(self):
        return "Apple Metal Performance Shaders (MPS)"

    @property
    def compute_info(self):
        driver_version = None
        if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "driver_version"):
            try:
                driver_version = torch.backends.mps.driver_version()
            except TypeError:
                # driver_version may be a property on some torch releases
                driver_version = torch.backends.mps.driver_version
        if driver_version:
            return f"mps={driver_version}"
        return "mps"

    def event(self, enable_timing=True):
        return MPSEvent()

    def synchronize(self):
        torch.mps.synchronize()

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

    if torch.xpu.is_available():
        return XPUArch()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return MPSArch()

    raise ValueError("Currently only cuda, rocm, hpu, xpu and mps are supported")

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

        if self.verbose:
            self.stdout.write(message)
        # replace `\r` and `033\[K` which are nice in the console, but we don't want those in the log file
        message = re.sub(r"(\r|\033\[K)", "\n", message)
        self.file.write(message)

    def flush(self):
        self.file.flush()
        if self.verbose:
            self.stdout.flush()


def print_benchmark_header(dtype, device, notes="None"):

    device_info = arch.device_info
    compute_info = arch.compute_info

    print(f"""
Benchmark started on {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}

** Command line:
{sys.executable} {" ".join(map(shlex.quote, sys.argv))}

** Dtype: {dtype}

** Platform/Device info:
- {" ".join(platform.uname())}
- {device_info}

** Critical software versions:
- torch={torch.__version__}
- {compute_info}

** Critical environment variables:
- PYTORCH_TUNABLEOP_ENABLED={os.environ.get("PYTORCH_TUNABLEOP_ENABLED", "0")}

** Additional notes:
- benchmark version: {benchmark_version}
{notes}

{"-" * 80}

""")

# Benchmark of a basic GEMM
def benchmark_mm(m, n, k, dtype, device, num_iterations, num_warmup_iterations):
    start = arch.event(enable_timing=True)
    end = arch.event(enable_timing=True)

    # this will be used to write to the accelerator between each benchmark iteration to emulate cache reset.
    # On AMD this will really be an l3/LLC cache - later need to figure out how to get the maximum cache
    # size automatically, according to this table 256MB is the highest value so far across all
    # recent accelerators:
    # https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#caches
    l2_cache_size_in_mbs = 256
    l2_cache = torch.empty(int(l2_cache_size_in_mbs * 2**20 / 4), dtype=torch.int, device=device)

    C = torch.empty(m, n, dtype=dtype, device=device).contiguous()
    # this random matrix will be used in the loop to ensure that C gets actually written to, as
    # otherwise the rerun results will be always the same and no power will be drawn to write - would lead
    # to invalid emulation of a real use case
    C_rand = torch.randn(m, n, device=device).to(dtype=dtype).contiguous()

    def time_it(iters=1):
        def decorator(func):
            def func_wrapper(*args, **kwargs):
                start_events = [arch.event(enable_timing=True) for _ in range(iters)]
                end_events = [arch.event(enable_timing=True) for _ in range(iters)]

                for i in range(iters):
                    with torch.no_grad():
                        l2_cache.zero_() # clear accelerator cache
                        C.copy_(C_rand)  # re-randomize the target matrix
                        start_events[i].record()
                        ret = func(*args, **kwargs)
                        end_events[i].record()
                arch.synchronize()
                times = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
                return times
            return func_wrapper
        return decorator

    total_iterations = num_iterations + num_warmup_iterations

    # fp8 requires special handling depending on the vendor:
    # float8_e4m3fn for nvidia, float8_e4m3fnuz for amd
    fp8_dtypes = [torch.float8_e4m3fn, torch.float8_e4m3fnuz]
    if dtype in fp8_dtypes:
        # torch._scaled_mm is different before pt-2.5
        if version.parse(torch.__version__) < version.parse("2.5"):
            raise ValueError("float8 dtypes require torch>=2.5")
        if dtype == torch.float8_e4m3fn and arch.name == "rocm":
            raise ValueError("ROCm doesn't support float8_e4m3fn, use --dtype float8_e4m3fnuz instead")

        A = torch.randn(m, k, dtype=torch.float32, device=device).contiguous()
        B = torch.randn(n, k, dtype=torch.float32, device=device).contiguous().t()
        scale = torch.tensor([1.0]).to(device)
        A = A.to(dtype)
        B = B.to(dtype)

        # Simplified call for PyTorch 2.5+
        @time_it(total_iterations)
        def time_iterations():
            # must not move `out=C` as `C = ...` as Gaudi needs it this way to work
            torch._scaled_mm(A, B, scale, scale, out=C)

    else:
        A = torch.randn(m, k, dtype=dtype, device=device).contiguous()
        B = torch.randn(n, k, dtype=dtype, device=device).contiguous().t()

        @time_it(total_iterations)
        def time_iterations():
            torch.mm(A, B, out=C)

    times = time_iterations()[num_warmup_iterations:]
    flos = 2 * m * n * k

    mean_elapsed_time = np.mean(times)/1000
    mean_tflops = flos / (mean_elapsed_time * 10**12)

    median_elapsed_time = np.median(times)/1000
    median_tflops = flos / (median_elapsed_time * 10**12)

    min_elapsed_time = np.amin(times)/1000
    max_tflops = flos / (min_elapsed_time * 10**12)

    return mean_tflops, median_tflops, max_tflops

def setup_checks():
    if arch.name == "rocm":
        if int(os.environ.get("PYTORCH_TUNABLEOP_ENABLED", "0")) == 0:
            warn("AMD GPUs usually require `export PYTORCH_TUNABLEOP_ENABLED=1` to measure the best possible compute, but it hasn't been set. Proceeding as is - expect potentially bad/invalid results.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    m_group = parser.add_mutually_exclusive_group(required=True)
    m_group.add_argument("--m", nargs="+", type=int, help='The first dimension of the GEMM, enter any number of arguments')
    m_group.add_argument("--m_range", nargs='+', type=int, help="The first dimension of the GEMM, [start,stop,step]")

    n_group = parser.add_mutually_exclusive_group(required=True)
    n_group.add_argument("--n", nargs="*", type=int, help='The last dimension of the GEMM, enter any number of arguments')
    n_group.add_argument("--n_range", nargs='+', type=int, help="The last dimension of the GEMM, [start,stop,step]")

    k_group = parser.add_mutually_exclusive_group(required=True)
    k_group.add_argument("--k", nargs="*", type=int, help='The shared (reduction) dimension of the GEMM, enter any number of arguments')
    k_group.add_argument("--k_range", nargs='+', type=int, help="The shared (reduction) dimension of the GEMM, [start,stop,step]")

    parser.add_argument("--num_iterations", type=int, default=100, help='The number of iterations used to benchmark each GEMM')
    parser.add_argument("--num_warmup_iterations", type=int, default=50, help='The number of warmup iterations')
    parser.add_argument("--cuda_device", type=int, default=0, help="The cuda device to run the benchmark on")
    parser.add_argument("--output_file", type=str, default=f"{file_dir}/results/mm.out")
    parser.add_argument("--notes", type=str, default="", help="benchmark-specific notes to add to the output_file's header")
    parser.add_argument("--verbose", default=True, action=argparse.BooleanOptionalAction, help='log to stdout besides output_file?')
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type to use for the benchmark (e.g., float32, float16, bfloat16, float8_e4m3fn, torch.float8_e4m3fnuz)")
    args = parser.parse_args()

    m = args.m
    n = args.n
    k = args.k

    dtype = get_torch_dtype(args.dtype)
    device = arch.device

    setup_checks()

    range_info = (
        f"m={args.m_range if m is None else args.m} | "
        f"n={args.n_range if n is None else args.n} | "
        f"k={args.k_range if k is None else args.k}"
    )

    if m is None:
        start, stop, step = args.m_range
        if start == 0: # can't have a 0 dimension
            start = step
        m = np.arange(start, stop, step)
    if n is None:
        start, stop, step = args.n_range
        if start == 0: # can't have a 0 dimension
            start = step
        n = np.arange(start, stop, step)
    if k is None:
        start, stop, step = args.k_range
        if start == 0: # can't have a 0 dimension
            start = step
        k = np.arange(start, stop, step)

    sys.stdout = Tee(args.output_file, args.verbose)
    print_benchmark_header(dtype, device, args.notes)

    # this is useful for when one wants to interrupt the run - and still report the best outcome so far
    def sigkill_handler(signum, frame):
         finish()
         sys.exit(1)

    signal.signal(signal.SIGINT, sigkill_handler)

    best_tflops = dict(max=0, median=0, mean=0)
    best_config = dict(max="", median="", mean="")
    num_shapes = 0
    all_mean_tflops = []
    start_time = time.time()

    def finish():

        all_tried_shapes_geometric_mean_tflops  = np.exp(np.log(all_mean_tflops).mean())
        all_tried_shapes_arithmetic_mean_tflops = np.mean(all_mean_tflops)

        time_delta = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
        print("", end="\033[K")
        print(f"""
Tried {num_shapes} shapes => the best outcomes were:
mean:   {best_tflops["mean"]:.1f} TFLOPS @ {best_config["mean"]}
median: {best_tflops["median"]:.1f} TFLOPS @ {best_config["median"]}
max:    {best_tflops["max"]:.1f} TFLOPS @ {best_config["max"]}

Across {num_shapes} shapes in range: {range_info} in this run:
arithmetic mean: {all_tried_shapes_arithmetic_mean_tflops:.1f} TFLOPS
geometric mean:  {all_tried_shapes_geometric_mean_tflops:.1f} TFLOPS
""")
        print(f"Legend: TFLOPS = 10**12 FLOPS")
        print(f"Elapsed time: {time_str}")

    # XXX: the transpose version seemed to work better for MI300X

    # always start with additional warmup iterations to give fare results, otherwise based on
    # rerunning this benchmark many times - a cold accelerator gives a higher score on say a single
    # shape, than the same shape run after a dozen of other shapes
    accelerator_warmup_seconds = 30
    end_time = time.monotonic() + accelerator_warmup_seconds
    print(f"Warming up the accelerator for {accelerator_warmup_seconds} secs ... ", end="", flush=True)
    while time.monotonic() < end_time:
        _ = benchmark_mm(m[0], n[0], k[0], dtype, device, args.num_iterations, args.num_warmup_iterations)
    print("accelerator warmup finished")

    # loop through all sizes to benchmark
    for M in m:
        for N in n:
            for K in k:
                num_shapes += 1
                mean_tflops, median_tflops, max_tflops = benchmark_mm(M, N, K, dtype, device, args.num_iterations, args.num_warmup_iterations)
                all_mean_tflops.append(mean_tflops)

                cur_config = f"{M}x{N}x{K}"
                if median_tflops > best_tflops["median"]:
                    best_tflops["median"] = median_tflops
                    best_config["median"] = f"{cur_config} (MxNxK)"
                if mean_tflops > best_tflops["mean"]:
                    best_tflops["mean"] = mean_tflops
                    best_config["mean"] = f"{cur_config} (MxNxK)"
                if max_tflops > best_tflops["max"]:
                    best_tflops["max"] = max_tflops
                    best_config["max"] = f"{cur_config} (MxNxK)"

                print(f"{num_shapes:>6} | {mean_tflops:6.1f}(mean) {median_tflops:6.1f}(median) {max_tflops:6.1f}(max) @ {cur_config:<20} | best: {best_tflops['mean']:6.1f}(mean) {best_tflops['median']:6.1f}(median) {best_tflops['max']:6.1f}(max) TFLOPS", end="\r")
    finish()
