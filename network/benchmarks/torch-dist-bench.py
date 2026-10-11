#!/usr/bin/env python

r"""

The latest version of this program can be found at
https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/torch-dist-bench.py

This benchmark is very similar to https://github.com/NVIDIA/nccl-tests, but much easier to set up, since all
it needs to run is PyTorch, and you want to use it if your program calls torch.distributed, as it calls the
collectives the way your program does: through PyTorch, which runs some of them with different NCCL calls than
nccl-tests does (e.g. scatter, batch_isend_irecv), and with --with-host-overhead it also includes PyTorch's per-call
host overhead. If you write a custom comms kernel, which calls the NCCL/RCCL C API directly, then you want to use
nccl-tests instead.

For detailed documentation please refer to:
https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/README.md#torch-dist-bench

This script:
- has been derived from @jeffra's gist: https://gist.github.com/jeffra/b5e80466b4c86be00ea3b6f130fb7a36
- which in turn is derived from the logic in https://github.com/NVIDIA/nccl-tests
- with contributions from:
  * Indu Thangakrishnan https://github.com/indhub to handle timing correctly using cuda events

Examples:

The recipes below run on:
1. a single node - using `torch.distributed.run` (aka `torchrun`), which can be easily replaced with `deepspeed`,
   `accelerate` and other distributed launchers
2. multiple nodes - using SLURM or `pdsh` (e.g. on k8s)

*** To do a quick test on 2 GPUs:

python -u -m torch.distributed.run --nproc_per_node 2 --rdzv_endpoint localhost:6000 --rdzv_backend c10d \
    torch-dist-bench.py

*** To benchmark all_gather and reduce_scatter, or all the collectives, on 8 GPUs:

python -u -m torch.distributed.run --nproc_per_node 8 --rdzv_endpoint localhost:6000 --rdzv_backend c10d \
    torch-dist-bench.py --collectives all_gather,reduce_scatter

python -u -m torch.distributed.run --nproc_per_node 8 --rdzv_endpoint localhost:6000 --rdzv_backend c10d \
    torch-dist-bench.py --collectives all

*** To run on 4 nodes on SLURM:

GPUS_PER_NODE=8
NNODES=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    torch-dist-bench.py

note: the MASTER_ADDR line above gets the first node's hostname from the SLURM allocation - outside SLURM, set it to
the hostname of the node with node rank 0.

Or, to run with salloc+srun:

salloc --partition=mypartition --nodes=4 --ntasks-per-node=1 --cpus-per-task=48 --gres=gpu:8 --time=1:00:00 bash

srun --gres=gpu:8 --nodes=4 --tasks-per-node=1 \
    python -u -m torch.distributed.run --nproc_per_node 8 --nnodes 4 \
    --rdzv_endpoint $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):6000 --rdzv_backend c10d \
    torch-dist-bench.py

*** To run on 2 nodes with pdsh

This approach requires passwordless ssh between the participating nodes.

Either hardcode the ips or hostnames:

MASTER_HOST=10.0.0.10
HOSTS=10.0.0.10,10.0.0.11
NNODES=2

or, if you already have a deepspeed-style hostfile, with a "hostname slots=x" entry per line, or an mpi-style one,
with a "hostname" per line, take them from it:

MASTER_HOST=$(cat ~/hostfile | cut -d " " -f1 | head -1)
HOSTS=$(cat ~/hostfile | cut -d " " -f1 | tr '\n' ',' | sed 's/,*$//g')
NNODES=2

You can first check that your pdsh setup works with this quick command, which prints the hostname of each
participating node:

PDSH_RCMD_TYPE=ssh pdsh -w $HOSTS hostname

Then run the benchmark, after setting `DIR` to the directory that holds this script - pdsh doesn't run the command
in your current working directory, so the script has to be given by its full path:

DIR=/change/the/path/benchmarks
PDSH_RCMD_TYPE=ssh pdsh -w $HOSTS \
    python -u -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES \
    --rdzv_endpoint $MASTER_HOST:6000 --rdzv_backend c10d \
    $DIR/torch-dist-bench.py


"""

from pathlib import Path
import argparse
import datetime
import gc
import math
import os
import signal
import socket
import sys
import textwrap
import time
import torch
import torch.distributed as dist

has_hpu = False
try:
    import habana_frameworks.torch as ht
    if torch.hpu.is_available():
        has_hpu = True
except ModuleNotFoundError:
    pass

args = None

# calls timed per trial: enough to amortize ranks reaching the first call at slightly different times, which
# matters at small payloads where a call takes 10s of us, few enough that one trial's in-place sums of [0, 1) values
# can't overflow fp32 even on 1000s of ranks. Large payloads get fewer calls (see calls_per_trial)
MAX_CALLS_PER_TRIAL = 10

# how long the device is kept busy before the queued calls - some 10x longer than the host takes to queue them. It must
# be the same on all ranks, otherwise the ranks with a shorter wait time their calls waiting for the others to arrive
BUSY_WAIT_MS = 2

# latency samples per payload, as Intel MPI Benchmarks counts its repetitions: up to 1000 calls, so that the p99 is the
# 10th slowest of them, but fewer once they'd move more than 40MiB, where calls get slow and their time steady
LATENCY_MAX_SAMPLES = 1000
LATENCY_MAX_VOLUME = 40 * 2**20

# the payload's 1/ranks pieces are rounded down to this, as nccl-tests does
CHUNK_ALIGN_BYTES = 16

# all_reduce first - it's the default
COLLECTIVES = ["all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast", "reduce", "gather", "scatter",
               "batch_isend_irecv"]

# the busbw correction factors of nccl-tests, explained in
# https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bus-bandwidth - busbw reflects how optimally
# the hardware is used, and is comparable across collectives and numbers of ranks
BUSBW_FACTOR = dict(
    all_reduce        = lambda n: 2 * (n - 1) / n,
    all_gather        = lambda n: (n - 1) / n,
    reduce_scatter    = lambda n: (n - 1) / n,
    all_to_all        = lambda n: (n - 1) / n,
    broadcast         = lambda n: 1,
    reduce            = lambda n: 1,
    gather            = lambda n: (n - 1) / n,
    scatter           = lambda n: (n - 1) / n,
    batch_isend_irecv = lambda n: 1,
)

# 8 -> 8B, 2**20 -> 1MiB, 768 -> 768B, 3*2**19 -> 1.5MiB
def fmt_bytes(v):
    unit = max(v.bit_length() - 1, 0) // 10
    return f"{v / 2**(unit * 10):g}" + ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"][unit]
fmt_us = lambda v : f"{v * 10**6:.1f}us"
# following the common networking hw spec convention which uses base 10, instead of 2 for bps/Bps (it makes speed look bigger than it is)
conv_to_GBps = lambda v : v/10**9


### Architecture specific helper classes ###

class Arch:
    def __init__(self):
        self.arch = "unknown"

    def __repr__(self):
        return self.arch

    @property
    def device_info(self):
        return "Unknown accelerator"

    def busy_wait(self, ms):
        """Keep the device busy for about `ms` milliseconds, so the work queued behind it runs back to back however
        long the host takes to queue it"""
        raise NotImplementedError(f"{type(self).__name__} doesn't implement busy_wait() yet - run with --with-host-overhead")

class CudaLikeArch(Arch):
    """ CUDA and ROCm - both are torch device 'cuda' """
    def __init__(self):
        self.arch = "rocm" if torch.version.hip is not None else "cuda"
        self.sleep_cycles_per_ms = None

    @property
    def device_info(self):
        return repr(torch.cuda.get_device_properties('cuda'))

    def busy_wait(self, ms):
        """Keep the GPU busy for `ms` milliseconds. torch.cuda._sleep spins for a number of device clock cycles, which
        tick at a different rate on each GPU, so the cycles per ms are measured on the first call."""
        if self.sleep_cycles_per_ms is None:
            cycles = 1_000_000
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(cycles)
            start.record()
            torch.cuda._sleep(cycles)
            end.record()
            torch.cuda.synchronize()
            self.sleep_cycles_per_ms = cycles / start.elapsed_time(end)
        torch.cuda._sleep(int(ms * self.sleep_cycles_per_ms))

class HPUArch(Arch):
    """ Intel Gaudi* """
    def __init__(self):
        self.arch = "hpu"

    @property
    def device_info(self):
        return repr(torch.hpu.get_device_properties('hpu'))

def get_accelerator_arch():
    """
    returns: an Arch subclass instance for the active accelerator
    """
    if torch.cuda.is_available():
        return CudaLikeArch()
    if has_hpu:
        return HPUArch()
    return Arch()

arch = get_accelerator_arch()

def new_plot(path):
    """Returns matplotlib.pyplot with a new figure, or None if matplotlib isn't installed"""
    try:
        import matplotlib.pyplot as plt
    except:
        print("!!! Can't generate plot. Please run `pip install matplotlib` to enable plotting. !!!\n")
        return None

    print(f"\n*** Plotting results into {path}\n")
    plt.figure(dpi=500)
    return plt

def save_plot(plt, path):
    """Writes the device info under the plot and saves it"""
    # wrap notes - this can now handle several lines of text.
    notes = "\n".join(textwrap.wrap(arch.device_info, width=60))

    plt.annotate(notes,
                 xy=(0.001, -0.3),
                 xycoords='axes fraction',
                 ha='left',
                 va="center",
                 fontsize=10)

    plt.savefig(path, bbox_inches='tight')

def set_size_axis(plt, sizes):
    plt.xscale("log", base=2)
    plt.xticks(sizes, [fmt_bytes(x) for x in sizes], rotation=45, fontsize=6)
    plt.xlabel(f"Message size")

def set_title_and_legend(plt, results, what, ranks):
    """One line per collective gets a legend outside the plot, where it can't hide the lines"""
    if len(results) > 1:
        plt.title(f"{what} of {len(results)} collectives on ranks={ranks}")
        plt.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    else:
        plt.title(f"{next(iter(results))} {what} on ranks={ranks}")

def plot_averages(path, results, ranks):
    """results: {collective: its results} - one line per collective"""
    results = {name: res for name, res in results.items() if res["busbw"]}
    if not results:
        return
    plt = new_plot(path)
    if plt is None:
        return

    for name, res in results.items():
        plt.plot(list(res["busbw"].keys()), [conv_to_GBps(x) for x in res["busbw"].values()], label=name)
    set_size_axis(plt, sorted({size for res in results.values() for size in res["busbw"].keys()}))
    plt.ylabel("Bus bandwidth (GBps)")
    set_title_and_legend(plt, results, "bus bandwidth", ranks)
    save_plot(plt, path)

def plot_latency(path, results, ranks):
    """Log-log, as latency barely changes over the small payloads and then grows with the payload size. The p99 is
    shaded for one collective, and left out for several, which would hide each other's medians"""
    plt = new_plot(path)
    if plt is None:
        return

    for name, res in results.items():
        sizes = list(res["latency"].keys())
        median_us = [x["median"] * 10**6 for x in res["latency"].values()]
        if len(results) > 1:
            plt.plot(sizes, median_us, marker=".", label=name)
            continue
        p99_us = [x["p99"] * 10**6 for x in res["latency"].values()]
        plt.plot(sizes, median_us, marker=".", label="median")
        plt.plot(sizes, p99_us, linewidth=0.5, label="p99")
        plt.fill_between(sizes, median_us, p99_us, alpha=0.3)
    set_size_axis(plt, sorted({size for res in results.values() for size in res["latency"].keys()}))
    plt.yscale("log")
    from matplotlib.ticker import LogFormatter

    class ENotation(LogFormatter):
        """1e1, 1e2, 2e1 rather than 10, 100, 20 or 10^1, 10^2, 2x10^1 - LogFormatter still decides which ticks get a
        label, which for the minor ones is only when the axis spans less than a decade or so"""
        def __call__(self, x, pos=None):
            if not super().__call__(x, pos):
                return ""
            exp = math.floor(math.log10(x) + 1e-9)
            return f"{x / 10**exp:g}e{exp}"

    plt.gca().yaxis.set_major_formatter(ENotation())
    plt.gca().yaxis.set_minor_formatter(ENotation(labelOnlyBase=False, minor_thresholds=(1, 0.4)))
    plt.grid(which="both", linewidth=0.2)
    plt.ylabel(f"Latency per call (us){', median' if len(results) > 1 else ''}")
    if len(results) == 1:
        plt.legend()
    set_title_and_legend(plt, results, "latency", ranks)
    save_plot(plt, path)

def plot_profile(path, name, y, ranks):
    plt = new_plot(path)
    if plt is None:
        return

    plt.plot(y)
    plt.xlabel(f"Iteration")
    plt.ylabel("Bus bandwidth (GBps)")
    plt.title(f"{name} bus bandwidth profile for {args.payload_size_in_gib}GiB payload on ranks={ranks}")
    save_plot(plt, path)

def calls_per_trial(size):
    """MAX_CALLS_PER_TRIAL up to 64MiB, then fewer, down to 1 from 1GiB up, where a single call takes milliseconds
    and the per-trial costs the extra calls amortize are noise"""
    if args.with_host_overhead:
        return 1
    return max(1, min(MAX_CALLS_PER_TRIAL, 2**30 // size))

def latency_samples_per_trial(calls):
    """The first of several queued calls also waits for the ranks that reach it later, so it's left out"""
    return calls - 1 if calls > 1 else 1

def num_trials(size, calls):
    """--num_iterations trials, or more at small payloads, enough for up to LATENCY_MAX_SAMPLES latency samples"""
    samples = min(LATENCY_MAX_SAMPLES, LATENCY_MAX_VOLUME // size)
    return max(args.num_iterations, math.ceil(samples / latency_samples_per_trial(calls)))

def collective_call(name, size, buffer):
    """
    Returns (call, inputs, size) - call() runs one `name` collective on a `size` bytes payload, `inputs` are the
    tensors to refill before each trial, and `size` is the payload it moves, smaller than the one asked for when the
    1/ranks pieces had to be rounded down. Returns None if the payload is too small to split into ranks pieces.
    `buffer(slot, numel)` returns an fp32 tensor - slot 0 for the input, slot 1 for a separate output.
    """
    ranks, rank = dist.get_world_size(), dist.get_rank()
    numel = size // 4

    if name in ("all_reduce", "broadcast", "reduce"):
        tensor = buffer(0, numel)
        call = dict(all_reduce=lambda: dist.all_reduce(tensor),
                    broadcast=lambda: dist.broadcast(tensor, src=0),
                    reduce=lambda: dist.reduce(tensor, dst=0))[name]
        return call, [tensor], size

    if name == "batch_isend_irecv":
        send, recv = buffer(0, numel), buffer(1, numel)
        def call():
            ops = [dist.P2POp(dist.isend, send, (rank + 1) % ranks), dist.P2POp(dist.irecv, recv, (rank - 1) % ranks)]
            for work in dist.batch_isend_irecv(ops):
                work.wait()
        return call, [send], size

    chunk = size // ranks // CHUNK_ALIGN_BYTES * CHUNK_ALIGN_BYTES // 4
    if chunk == 0:
        return None
    whole = chunk * ranks
    if name == "all_gather":
        input, output = buffer(0, chunk), buffer(1, whole)
        call = lambda: dist.all_gather_into_tensor(output, input)
    elif name == "reduce_scatter":
        input, output = buffer(0, whole), buffer(1, chunk)
        call = lambda: dist.reduce_scatter_tensor(output, input)
    elif name == "all_to_all":
        input, output = buffer(0, whole), buffer(1, whole)
        call = lambda: dist.all_to_all_single(output, input)
    elif name == "gather":
        input, output = buffer(0, chunk), buffer(1, whole)
        # gather_single gathers into one tensor, with NCCL's own gather when torch was built with NCCL>=2.28.3, as
        # nccl-tests does - the older list-based gather sends and receives each piece separately
        if hasattr(dist, "gather_single"):
            gather_tensor = output if rank == 0 else None
            call = lambda: dist.gather_single(input, gather_tensor, dst=0)
        else:
            gather_list = list(output.chunk(ranks)) if rank == 0 else None
            call = lambda: dist.gather(input, gather_list, dst=0)
    elif name == "scatter":
        input, output = buffer(0, whole), buffer(1, chunk)
        scatter_list = list(input.chunk(ranks)) if rank == 0 else None
        call = lambda: dist.scatter(output, scatter_list, src=0)
    return call, [input], whole * 4

def timed_calls(call, inputs, size, calls, events):
    """Times one trial of `calls` calls - with an event after each call if there are calls+1 `events`, or only at the
    trial's start and end if there are 2. Returns the algbw averaged over the ranks, valid on rank 0 only, this rank's
    latency samples in seconds, none without the per-call events, and whether this rank's host took longer to queue
    the calls than the busy wait kept the device busy - then the device waited for the host, and the trial timed the
    host rather than the calls"""
    per_call = len(events) == calls + 1
    # refill outside the timed region - each in-place sum grows the values by up to `ranks` times, which would overflow
    # fp32 to inf over the trials
    for tensor in inputs:
        tensor.uniform_()
    dist.barrier()
    host_fell_behind = False
    if args.with_host_overhead:
        events[0].record()
        call()
        events[1].record()
    else:
        align = torch.zeros(1, device=torch.device("cuda", local_rank))
        queue_start = time.perf_counter()
        arch.busy_wait(BUSY_WAIT_MS)
        dist.all_reduce(align)
        events[0].record()
        for i in range(calls):
            call()
            if per_call:
                events[i + 1].record()
        if not per_call:
            events[-1].record()
        host_fell_behind = (time.perf_counter() - queue_start) * 1000 > BUSY_WAIT_MS
    torch.cuda.synchronize()
    duration = events[0].elapsed_time(events[-1]) / 1000 / calls
    latencies = []
    if per_call:
        latencies = [a.elapsed_time(b) / 1000 for a, b in zip(events, events[1:])]
        latencies = latencies[-latency_samples_per_trial(calls):]

    n = dist.get_world_size()
    # note that this is following the same math as NVIDIA/nccl-tests
    algbw = torch.tensor([size / duration]).cuda(local_rank)

    # calculate mean across all ranks
    dist.reduce(algbw, dst=0, op=dist.ReduceOp.SUM)
    algbw /= n

    return algbw, latencies, host_fell_behind

def sym_mem_buffers(numel, count, local_rank):
    """
    Returns (backend, pool, buffers) where buffers are `count` `numel`-long fp32 tensors allocated with ncclMemAlloc
    and registered as NCCL symmetric memory windows on the default process group's communicator.
    """
    from packaging import version
    # .release so that 2.9 pre-release/nightly builds pass too
    if version.parse(torch.__version__).release < (2, 9):
        sys.exit(f"--sym-mem needs torch>=2.9, this is torch=={torch.__version__}")
    if torch.cuda.nccl.version() < (2, 27):
        sys.exit(f"--sym-mem needs NCCL>=2.27, this torch uses nccl={torch.cuda.nccl.version()}")

    device = torch.device("cuda", local_rank)
    backend = dist.group.WORLD._get_backend(device)
    pool = torch.cuda.MemPool(backend.mem_allocator)
    with torch.cuda.use_mem_pool(pool):
        buffers = [torch.empty(numel, dtype=torch.float32, device=device) for _ in range(count)]
    backend.register_mem_pool(pool, symm=True)
    return backend, pool, buffers

def run(local_rank):

    start_time = time.time()

    hostname = socket.gethostname()
    is_global_rank_0 = dist.get_rank() == 0
    ranks = dist.get_world_size()

    if args.payload_size_in_gib is None:
        # powers of 2, e.g. 2**3 to 2**34 => 8B to 16GiB
        sizes = [2**x for x in range(args.min_payload.bit_length()-1, args.max_payload.bit_length())]
    else:
        sizes = [int(args.payload_size_in_gib * 2**30)]

    # windows sized for the largest payload and registered once - one for the inputs, and one for the outputs if a
    # collective has separate ones. Each buffer is an offset-0 view of its window, so it sits at the same window offset
    # on every rank, as NCCL's symmetric kernels require
    sym_mem_pool = None
    if args.sym_mem:
        count = 1 if set(args.collectives) <= {"all_reduce", "broadcast", "reduce"} else 2
        sym_mem_backend, sym_mem_pool, sym_mem_bufs = sym_mem_buffers(max(sizes)//4, count, local_rank)

    def buffer(slot, numel):
        if args.sym_mem:
            return sym_mem_bufs[slot][:numel]
        return torch.empty(numel, dtype=torch.float32, device=torch.device("cuda", local_rank))

    # this is useful for when one wants to interrupt the run - and still report the best outcome so far
    def sigkill_handler(signum, frame):
         finish()
         sys.exit(1)

    signal.signal(signal.SIGINT, sigkill_handler)

    def finish():
        torch.cuda.synchronize()
        if sym_mem_pool is not None:
            sym_mem_backend.deregister_mem_pool(sym_mem_pool)
        dist.destroy_process_group()

        if not is_global_rank_0:
            return

        print(f"\nEnvironment:")
        print(f"- software: torch={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
        print(f"- hardware: {arch.device_info}\n")

        trials = f"at least {args.num_warmup_iterations} warmups / {args.num_iterations} trials"
        plot_suffix = ""
        if args.with_host_overhead:
            trials += ", 1 call per trial, host overhead included"
            plot_suffix += "-with-host-overhead"
        else:
            trials += f", up to {MAX_CALLS_PER_TRIAL} queued calls per trial"
        if args.sym_mem:
            trials += ", symmetric memory"
            plot_suffix += "-sym-mem"

        if args.profile_stability:
            for name, res in results.items():
                print_profile(name, res, trials, plot_suffix)
        elif args.separate_tables or len(results) == 1:
            for name, res in results.items():
                print_results(name, res, trials, plot_suffix)
        else:
            print_combined_results(results, trials, plot_suffix)
        measured = {name: res for name, res in results.items() if res["latency"]}
        if not args.profile_stability and len(measured) > 1:
            print_summary(measured, plot_suffix)

        time_delta = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
        print(f"Legend: 1KiB = 2**10 Bytes, 1MiB = 2**20 Bytes, 1GiB = 2**30 Bytes")
        print(f"        1GBps = 10**9 Bytes per second (networking bw spec convention)")
        print(f"Elapsed time: {time_str}")

    def print_profile(name, res, trials, plot_suffix):
        print(f"The {args.payload_size_in_gib}GiB payload bandwidth of {name} over {ranks} ranks ({trials}):\n")
        print(f"|    busbw   |    algbw   |")
        print(f"| ---------: | ---------: |")
        for busbw, algbw in zip(res["busbw_points"], res["algbw_points"]):
            print(f"| {conv_to_GBps(busbw):6.2f}GBps | {conv_to_GBps(algbw):6.2f}GBps |")
        print()
        plot_path = f"busbw-profile-{name}-{hostname}-{ranks}{plot_suffix}.png"
        plot_profile(plot_path, name, [conv_to_GBps(x) for x in res["busbw_points"]], ranks)

    def print_results(name, res, trials, plot_suffix):
        busbw, algbw, latency = res["busbw"], res["algbw"], res["latency"]
        print(f"The average bandwidth and the per-call latency of {name} over {ranks} ranks ({trials}):\n")
        if not latency:
            print(f"!!! No payload was measured !!!\n")
            return
        print(f"| payload |    busbw   |    algbw   | latency median |    p99    |    mean   | calls |")
        print(f"| ------: | ---------: | ---------: | -------------: | --------: | --------: | ----: |")
        for size in latency.keys():
            lat = latency[size]
            bandwidth = f"{'':10} | {'':10}"
            if size in busbw:
                bandwidth = f"{conv_to_GBps(busbw[size]):6.2f}GBps | {conv_to_GBps(algbw[size]):6.2f}GBps"
            print(f"| {fmt_bytes(size):>7} | {bandwidth} | {fmt_us(lat['median']):>14} "
                  f"| {fmt_us(lat['p99']):>9} | {fmt_us(lat['mean']):>9} | {lat['calls']:5} |")
        print(f"\nlatency: of one call, in microseconds, each call's time being the slowest rank's")
        print_busbw_start({name: res})
        print_notes({name: res})

        if len(latency) > 1:
            size0 = min(latency.keys())
            print(f"\nThe smallest payload, {fmt_bytes(size0)}, takes {fmt_us(latency[size0]['median'])} at the "
                  f"median - about the least {name} costs on this setup.")
        if len(busbw) > 1:
            peak = max(busbw.values())
            print(f"The smallest payload reaching half the peak busbw of {conv_to_GBps(peak):.2f}GBps is "
                  f"{fmt_bytes(half_peak_size(busbw))} - smaller ones get less than half the bandwidth this setup "
                  f"can give, and are better judged by their latency.")

        plot_averages(f"busbw-mean-{name}-{hostname}-{ranks}{plot_suffix}.png", {name: res}, ranks)
        plot_latency(f"latency-{name}-{hostname}-{ranks}{plot_suffix}.png", {name: res}, ranks)
        print()

    def print_combined_results(results, trials, plot_suffix):
        print(f"The average bandwidth and the per-call latency of {len(results)} collectives over {ranks} ranks "
              f"({trials}):\n")
        sizes = sorted({size for res in results.values() for size in res["latency"]})
        if not sizes:
            print(f"!!! No payload was measured !!!\n")
            return
        header = ["payload"] + [f"{name} {column}" for name in results for column in ("busbw", "median")]
        rows = []
        for size in sizes:
            row = [fmt_bytes(size)]
            for res in results.values():
                row.append(f"{conv_to_GBps(res['busbw'][size]):.2f}" if size in res["busbw"] else "")
                row.append(f"{res['latency'][size]['median'] * 10**6:.1f}" if size in res["latency"] else "")
            rows.append(row)
        widths = [len(column) for column in header]
        print("| " + " | ".join(header) + " |")
        print("| " + " | ".join("-" * (width - 1) + ":" for width in widths) + " |")
        for row in rows:
            print("| " + " | ".join(cell.rjust(width) for cell, width in zip(row, widths)) + " |")
        print(f"\nbusbw: in GBps")
        print(f"median: the latency median of one call, in microseconds, each call's time being the slowest rank's")
        print(f"a blank cell: a payload that collective didn't measure")
        print(f"--separate-tables gives each collective its own table, with algbw and the latency's p99 and mean")
        print_busbw_start(results)
        print_notes(results)
        print()

        for name, res in results.items():
            if res["latency"]:
                plot_averages(f"busbw-mean-{name}-{hostname}-{ranks}{plot_suffix}.png", {name: res}, ranks)
                plot_latency(f"latency-{name}-{hostname}-{ranks}{plot_suffix}.png", {name: res}, ranks)

    def print_busbw_start(results):
        if any(size < args.min_busbw_payload for res in results.values() for size in res["latency"]):
            print(f"\nThe bandwidth is measured from {fmt_bytes(args.min_busbw_payload)} up (--min-busbw-payload), "
                  f"below which a call's time is nearly all latency.")

    def print_notes(results):
        """The skipped payloads, the out of memory stops and the left out trials of one or more collectives - naming
        the collective only when there is more than one"""
        several = len(results) > 1
        listed = lambda names: names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
        of = lambda names: f" of {listed(names)}" if several else ""

        skipped_below = {}
        for name, res in results.items():
            if res["skipped"] and res["latency"]:
                skipped_below.setdefault(min(res["latency"].keys()), []).append(name)
        for size, names in skipped_below.items():
            print(f"\nSkipped the payloads below {fmt_bytes(size)}{of(names)}, too small to give each of the "
                  f"{ranks} ranks {CHUNK_ALIGN_BYTES} bytes.")

        for name, res in results.items():
            if not res["latency"] and several:
                print(f"\n!!! No payload of {name} was measured !!!")
            if res["out_of_memory"]:
                print(f"\n!!! Stopped{' ' + name if several else ''} before {fmt_bytes(res['out_of_memory'])}, "
                      f"which ran out of memory on some rank !!!")

        left_out = {}
        for name, res in results.items():
            latency = res["latency"]
            dropped = sum(latency[size]["dropped"] for size in latency.keys())
            if dropped:
                left_out[name] = (dropped, sum(latency[size]["trials"] for size in latency.keys()))
        if left_out:
            why = (f"in which a rank's host took longer than the {BUSY_WAIT_MS}ms busy wait to queue its calls, so "
                   f"its device waited for the host")
            if several:
                counts = ", ".join(f"{name} {dropped} of {total}" for name, (dropped, total) in left_out.items())
                print(f"\nLeft out the trials {why}: {counts}.")
            else:
                (dropped, total), = left_out.values()
                print(f"\nLeft out {dropped} of {total} trials, {why}.")
        for name in left_out:
            res = results[name]
            all_dropped = [fmt_bytes(size) for size in res["latency"].keys() if res["latency"][size]["all_dropped"]]
            if all_dropped:
                print(f"!!! The host fell behind in every trial of {', '.join(all_dropped)}{of([name])}, which are "
                      f"reported anyway - their numbers include the host's time !!!")

    def print_summary(measured, plot_suffix):
        print(f"Summary of {len(measured)} collectives over {ranks} ranks:\n")
        print(f"| collective     | smallest payload | latency median | peak busbw | at payload | half-peak payload |")
        print(f"| :------------- | ---------------: | -------------: | ---------: | ---------: | ----------------: |")
        for name, res in measured.items():
            busbw, latency = res["busbw"], res["latency"]
            size0 = min(latency.keys())
            if busbw:
                peak_size = max(busbw.keys(), key=lambda size: busbw[size])
                bandwidth = (f"{conv_to_GBps(busbw[peak_size]):6.2f}GBps | {fmt_bytes(peak_size):>10} "
                             f"| {fmt_bytes(half_peak_size(busbw)):>17}")
            else:
                bandwidth = f"{'':10} | {'':10} | {'':17}"
            print(f"| {name:<14} | {fmt_bytes(size0):>16} | {fmt_us(latency[size0]['median']):>14} | {bandwidth} |")
        print(f"\nhalf-peak payload: the smallest payload reaching half the collective's peak busbw")

        plot_averages(f"busbw-mean-collectives-{hostname}-{ranks}{plot_suffix}.png", measured, ranks)
        plot_latency(f"latency-collectives-{hostname}-{ranks}{plot_suffix}.png", measured, ranks)
        print()

    results = {}
    for name in args.collectives:
        results[name] = res = dict(algbw={}, busbw={}, latency={}, algbw_points=[], busbw_points=[], skipped=False,
                                   out_of_memory=None)
        for size in sizes:
            # clear prev-iteration memory for cards w/ ~24GiB
            call = inputs = None
            gc.collect()

            # every rank asks for the same buffers, so normally all or none of them run out of memory - but should only
            # some, the others must not go on to a collective they'd wait in forever
            try:
                benchmark = collective_call(name, size, buffer)
                out_of_memory = 0
            except torch.cuda.OutOfMemoryError:
                benchmark = None
                out_of_memory = 1
            out_of_memory = torch.tensor([out_of_memory], device=torch.device("cuda", local_rank))
            dist.all_reduce(out_of_memory, op=dist.ReduceOp.MAX)
            if out_of_memory.item():
                res["out_of_memory"] = size
                break
            if benchmark is None:
                res["skipped"] = True
                continue
            call, inputs, size = benchmark
            if size in res["latency"]:
                continue
            measure(name, res, call, inputs, size)
            call = inputs = benchmark = None
        if is_global_rank_0 and res["out_of_memory"]:
            print(f"{name} ran out of memory at {fmt_bytes(res['out_of_memory'])}, skipping the larger payloads")

    finish()

def half_peak_size(busbw):
    """The smallest payload reaching half the peak busbw"""
    peak = max(busbw.values())
    return [size for size in busbw.keys() if busbw[size] >= peak / 2][0]

def measure(name, res, call, inputs, size):
    """Benchmarks one payload of one collective and adds its bandwidth and latency to `res`"""
    is_global_rank_0 = dist.get_rank() == 0
    calls = calls_per_trial(size)
    trials = num_trials(size, calls)
    latency_events = [torch.cuda.Event(enable_timing=True) for _ in range(calls + 1)]
    # an event recorded between two queued calls slows them down - a 1MiB all_reduce on B200 takes 35us with one
    # and 30us without, in nccl-tests too (-I 1 vs its default) - so the bandwidth is timed by separate trials, with
    # events only at their start and end. With 1 call per trial both are the same trials
    with_bandwidth = size >= args.min_busbw_payload
    bandwidth_trials = 0 if calls == 1 or not with_bandwidth else args.num_iterations
    bandwidth_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    # do a few warm up iterations - a fifth of the trials when there are many, as OSU warms up 200 of its 1000
    # small-payload calls
    for i in range(max(args.num_warmup_iterations, trials // 5)):
        timed_calls(call, inputs, size, calls, latency_events)

    # real benchmark
    algbw_gather = []
    latency_gather = []
    fell_behind_gather = []
    for i in range(bandwidth_trials + trials):
        if is_global_rank_0:
            print(f"{name} {fmt_bytes(size):>6}: {i+1}", end="\r")
        events = bandwidth_events if i < bandwidth_trials else latency_events
        trial_algbw, trial_latencies, trial_fell_behind = timed_calls(call, inputs, size, calls, events)
        algbw_gather += trial_algbw
        latency_gather += trial_latencies
        fell_behind_gather.append(float(trial_fell_behind))
    if is_global_rank_0:
        print()

    # all ranks time the same calls in the same order, and a call is done when the slowest rank is done with it.
    # A trial in which any rank's host fell behind is left out on all of them
    latencies = torch.tensor(latency_gather + fell_behind_gather, dtype=torch.float64,
                             device=torch.device("cuda", torch.cuda.current_device()))
    dist.reduce(latencies, dst=0, op=dist.ReduceOp.MAX)
    latencies, fell_behind = latencies.cpu().split([len(latency_gather), len(fell_behind_gather)])

    def kept(fell_behind):
        keep = fell_behind == 0
        if not keep.any():
            keep[:] = True
        return keep
    keep_bandwidth = kept(fell_behind[:bandwidth_trials]) if bandwidth_trials else None
    keep = kept(fell_behind[bandwidth_trials:])
    dropped = len(fell_behind) - (fell_behind == 0).sum().item()
    latencies = latencies.view(trials, -1)[keep].flatten()
    if bandwidth_trials:
        algbw_gather = [x for x, k in zip(algbw_gather[:bandwidth_trials], keep_bandwidth) if k]
    else:
        algbw_gather = [x for x, k in zip(algbw_gather, keep) if k]
    res["latency"][size] = dict(median=latencies.quantile(0.5).item(), p99=latencies.quantile(0.99).item(),
                                mean=latencies.mean().item(), calls=len(latencies),
                                trials=len(fell_behind), dropped=dropped,
                                all_dropped=(fell_behind[bandwidth_trials:] != 0).all().item())
    if not with_bandwidth:
        return

    busbw_coeff = BUSBW_FACTOR[name](dist.get_world_size())
    if args.profile_stability:
        res["algbw_points"] = [x.item() for x in algbw_gather]
        res["busbw_points"] = [x * busbw_coeff for x in res["algbw_points"]]
    res["algbw"][size] = torch.mean(torch.stack(algbw_gather)).item()
    res["busbw"][size] = res["algbw"][size] * busbw_coeff

def device_id_kwargs(local_rank):
    """
    torch.dist in recent pytorch versions loudly complains about device_id not being set, but it's a very problematic setting.
    this util returns a dict to be passed to `dist.init_process_group` to set `device_id` if it's safe to do so.
    """

    from packaging import version
    import inspect
    # 1. device_id arg was added in torch==2.3
    # 2. setting device_id leads to hanging in 2.6.0<torch<2.7.1 https://github.com/pytorch/pytorch/issues/153960
    if 'device_id' in inspect.signature(torch.distributed.init_process_group).parameters and not (version.parse("2.6.0") < version.parse(torch.__version__) < version.parse("2.7.1")):
        return dict(device_id=torch.device(local_rank))
    else:
        return dict()

def payload_size(s):
    """'8' -> 8, '32K' -> 32768, '16G' -> 2**34 - K, M and G are powers of 2, as in nccl-tests' -b and -e"""
    units = dict(K=2**10, M=2**20, G=2**30)
    s = s.strip().upper().removesuffix("B").removesuffix("I")
    size = int(s[:-1]) * units[s[-1]] if s[-1:] in units else int(s)
    if size < 4 or size & (size - 1):
        raise argparse.ArgumentTypeError(f"{s} must be a power of 2 and at least 4 bytes (one fp32 element)")
    return size

def collective_names(value):
    """'all_gather, reduce_scatter,alltoall' -> ['all_gather', 'reduce_scatter', 'all_to_all'], and `all` -> all the
    collectives. Dashes and missing underscores are fine too, e.g. all-reduce and allreduce"""
    canonical = {name.replace("_", ""): name for name in COLLECTIVES}
    names = []
    for given in value.split(","):
        name = given.strip().lower().replace("-", "").replace("_", "")
        if not name:
            continue
        if name == "all":
            names += COLLECTIVES
        elif name in canonical:
            names.append(canonical[name])
        else:
            raise ValueError(f"--collectives: unknown collective '{given.strip()}', the names are comma separated, "
                             f"pick from: all, {', '.join(COLLECTIVES)}")
    return list(dict.fromkeys(names))

def parse_args():
    global args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # this arg is not used directly, but a launcher may pass it
    parser.add_argument("--local_rank", type=int, default=0, help='local rank')
    parser.add_argument("--collectives", default="all_reduce",
                        help=f"the collectives to benchmark, comma separated, e.g. all_gather,reduce_scatter, or "
                             f"`all` for all of them: {', '.join(COLLECTIVES)}")
    parser.add_argument("--separate-tables", action="store_true",
                        help="give each collective its own table, with algbw and the latency's p99 and mean, instead "
                             "of one table of all the collectives' busbw and latency median")
    parser.add_argument("--num_iterations", type=int, default=20, help='The number of bandwidth trials per payload, and the minimal number of latency trials - small payloads get more of those, to sample the latency of up to 1000 calls')
    parser.add_argument("--num_warmup_iterations", type=int, default=5, help='The minimal number of warmup trials per payload - payloads with many trials warm up for a fifth of them')
    parser.add_argument("--min-payload", type=payload_size, default="8", help='the smallest payload in bytes, a power of 2, K/M/G suffixes allowed, e.g. 32K')
    parser.add_argument("--max-payload", type=payload_size, default="16G", help='the largest payload in bytes, a power of 2, K/M/G suffixes allowed, e.g. 16K for a quick latency-only run')
    parser.add_argument("--min-busbw-payload", type=payload_size, default="32K", help="the smallest payload whose bandwidth is measured - the smaller ones get only their latency measured, as a call's time is nearly all latency there")
    parser.add_argument("--payload_size_in_gib", type=float, default=None, help='payload size in GiBs, e.g. 4 (4GiB). If specified, only this payload is benchmarked instead of the --min-payload .. --max-payload range')
    parser.add_argument("--profile_stability", action="store_true", help="Reports individual results for each non-warmup iteration. Requires --payload_size_in_gib. This is used to test the stability of performance, rather than reporting an averaged outcome.")
    parser.add_argument("--with-host-overhead", action="store_true", help="time 1 call per trial on an idle device, which adds PyTorch's per-call host overhead to every call - what a collective that the program waits on (a sync point) costs")
    parser.add_argument("--sym-mem", action="store_true", help="use buffers registered as an NCCL symmetric memory window (torch>=2.9, NCCL>=2.27), the equivalent of nccl-tests' -R 2. Use only if the target workload uses symmetric memory buffers too")

    args = parser.parse_args()
    args.collectives = collective_names(args.collectives)

    if args.min_payload > args.max_payload:
        raise ValueError(f"--min-payload {fmt_bytes(args.min_payload)} is larger than --max-payload {fmt_bytes(args.max_payload)}")

    if args.profile_stability and args.payload_size_in_gib is None:
        raise ValueError("--profile_stability requires --payload_size_in_gib to profile one specific payload, e.g. to profile a 0.5GiB payload use: --profile_stability --payload_size_in_gib 0.5")


if __name__ == "__main__":
    parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", **device_id_kwargs(local_rank))
    run(local_rank)
