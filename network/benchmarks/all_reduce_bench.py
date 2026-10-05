#!/usr/bin/env python

"""

The latest version of this program can be found at https://github.com/stas00/ml-engineering

This benchmark is very similar to https://github.com/NVIDIA/nccl-tests but it's much easier to set
up as it only requires PyTorch to be installed

This version:
- has been derived from @jeffra's gist: https://gist.github.com/jeffra/b5e80466b4c86be00ea3b6f130fb7a36
- which in turn is derived from the logic in https://github.com/NVIDIA/nccl-tests
- with contributions from:
  * Indu Thangakrishnan https://github.com/indhub to handle timing correctly using cuda events

Important notes:

- when you finished running this benchmark you want to pay attention to the busbw result (not
  algbw) as explained here https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bandwidth

- similar to NVIDIA/nccl-tests this benchmark measures a unidirectional bandwidth - so compare the
  outcome against the advertised unidirectional peak throughput and not bi-directional (duplex)

- currently this benchmark scans a payload range of 32KiB to 16GiB.

- this benchmark automatically generates a plot of the results if you have `matplotlib` installed.

- each trial keeps the device busy for a moment and queues up to 10 back-to-back dist.all_reduce calls behind
  that, so PyTorch's per-call host overhead of some 20-30us is hidden, as it is in a training loop whose host runs
  ahead of the device. Payloads of 1GiB and up get 1 call per trial, since a single call takes milliseconds there.
  This measures all-reduce as a PyTorch program gets it, through torch.distributed. If you're writing code that
  calls NCCL directly (e.g. a custom communication kernel), use https://github.com/NVIDIA/nccl-tests instead, which
  calls the NCCL C API the same way - it's also the tool for the other collectives.

- add --with-host-overhead to time one call per trial on an idle device instead, which adds the per-call host
  overhead to every call - what an all-reduce costs when the program waits on its result. The results are the same at large payloads, but
  much lower at small ones, where that overhead dominates the call's time.

- to benchmark other collectives use nccl-tests or adapt this benchmark to use the desired collective.

- you can interrupt (Ctrl-C) the benchmark in the middle and it'll complete with the results it has
  measured so far.

- you can also profile a single payload and get a plot with results for each iteration - for that use --profile_stability --payload_size_in_gib 0.5 (change the last value to the desired payload size in GiB)

- add --sym-mem to all-reduce buffers registered as an NCCL symmetric memory window, which lets NCCL>=2.27 use its
  symmetric kernels - what nccl-tests turns on with `-R 2`. Run it with and without the flag to see what
  symmetric memory gains on your setup. It needs torch>=2.9, the first release whose
  ProcessGroupNCCL.register_mem_pool() takes `symm`. Use it only if the workload you're benchmarking for all-reduces
  symmetric memory buffers too, otherwise the numbers won't reflect what that workload will get.

Examples:

The following are recipes to use to run on:
1. single node - using `torchdist`, which can be easily adapted to use `deepspeed`, `accelerate` and other distributed launchers
2. multi-node - using SLURM or `pdsh` (k8s)

*** To do a quick test on 2 GPUs:

python -u -m torch.distributed.run --nproc_per_node=2 --rdzv_endpoint localhost:6000  --rdzv_backend c10d \
all_reduce_bench.py

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
    all_reduce_bench.py

note: adapt MASTER_ADDR to node rank 0's hostname if it's not a SLURM environment where it's derived automatically.

e.g. example to run with salloc+srun:

salloc --partition=mypartition --nodes=4 --ntasks-per-node=1 --cpus-per-task=48 --gres=gpu:8 --time=1:00:00 bash

srun --gres=gpu:8 --nodes=4 --tasks-per-node=1 python -u -m torch.distributed.run --nproc_per_node=8 \
--nnodes 4 --rdzv_endpoint $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):6000 --rdzv_backend \
c10d all_reduce_bench.py

*** To run on 2 nodes with pdsh

This approach requires passwordless ssh between participating nodes:

You can hardcode the ips or hostnames:

MASTER_HOST=10.0.0.10
HOSTS=10.0.0.10,10.0.0.11

or if you already have a deepspeed-style hostfile w/ "hostname slots=x" entries per line entries or mpi-style hostfile w/ "hostname" per line entries:

MASTER_HOST=$(cat ~/hostfile | cut -d " " -f1 | head -1)
HOSTS=$(cat ~/hostfile | cut -d " " -f1 | tr '\n' ',' | sed 's/,*$//g')
NNODES=2

You can first test that your pdsh setup works with this quick command, which will print the hostname of each participating node:

PDSH_RCMD_TYPE=ssh pdsh -w $HOSTS hostname

Now you're ready to run the benchmark after adjusting the `DIR` value - it's critical since your current working dir with `pdsh` won't be the same as where you launched things from:

DIR=/change/the/path/benchmarks
PDSH_RCMD_TYPE=ssh pdsh -w $HOSTS python -u -m torch.distributed.run --nproc_per_node=8 --nnodes=$NNODES --rdzv_endpoint $MASTER_HOST:6003  --rdzv_backend c10d $DIR/all_reduce_bench.py


"""

from pathlib import Path
import argparse
import datetime
import gc
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

# all-reduces timed per trial: enough to amortize ranks reaching the first call at slightly different times, which
# matters at small payloads where a call takes 10s of us, few enough that one trial's in-place sums of [0, 1) values
# can't overflow fp32 even on 1000s of ranks. Large payloads get fewer calls (see calls_per_trial)
MAX_CALLS_PER_TRIAL = 10

# how long the device is kept busy before the queued calls - some 10x longer than the host takes to queue them. It must
# be the same on all ranks, otherwise the ranks with a shorter wait time their calls waiting for the others to arrive
BUSY_WAIT_MS = 2

# https://stackoverflow.com/a/75332100/9201239
fmt_bytes = lambda v : str(v >> ((max(v.bit_length()-1, 0)//10)*10)) +["", "K", "M", "G", "T", "P", "E"][max(v.bit_length()-1, 0)//10]+"iB"
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

def plot_averages(path, x, y, ranks):

    try:
        import matplotlib.pyplot as plt
    except:
        print("!!! Can't generate plot. Please run `pip install matplotlib` to enable plotting. !!!\n")
        return

    print(f"\n*** Plotting results into {path}\n")

    plt.figure(dpi=500)
    plt.plot(x, y)
    plt.xlabel(f"Message size")
    plt.ylabel("Bus bandwidth (GBps)")
    plt.title(f"all-reduce bus bandwidth on ranks={ranks}")
    plt.xticks(rotation=45)

    device_info = arch.device_info

    # wrap notes - this can now handle several lines of text.
    notes = "\n".join(textwrap.wrap(device_info, width=60))

    plt.annotate(notes,
                 xy=(0.001, -0.3),
                 xycoords='axes fraction',
                 ha='left',
                 va="center",
                 fontsize=10)

    plt.savefig(path, bbox_inches='tight')


def plot_profile(path, y, ranks):

    try:
        import matplotlib.pyplot as plt
    except:
        print("!!! Can't generate plot. Please run `pip install matplotlib` to enable plotting. !!!\n")
        return

    print(f"\n*** Plotting results into {path}\n")

    plt.figure(dpi=500)
    plt.plot(y)
    plt.xlabel(f"Iteration")
    plt.ylabel("Bus bandwidth (GBps)")
    plt.title(f"all-reduce bus bandwidth profile for {args.payload_size_in_gib}GiB payload on ranks={ranks}")
    #plt.xticks(rotation=45)

    device_info = arch.device_info

    # wrap notes - this can now handle several lines of text.
    notes = "\n".join(textwrap.wrap(device_info, width=60))

    plt.annotate(notes,
                 xy=(0.001, -0.3),
                 xycoords='axes fraction',
                 ha='left',
                 va="center",
                 fontsize=10)

    plt.savefig(path, bbox_inches='tight')



def calls_per_trial(size):
    """MAX_CALLS_PER_TRIAL up to 64MiB, then fewer, down to 1 from 1GiB up, where a single call takes milliseconds
    and the per-trial costs the extra calls amortize are noise"""
    if args.with_host_overhead:
        return 1
    return max(1, min(MAX_CALLS_PER_TRIAL, 2**30 // size))

def timed_allreduce(tensor, size, calls, start_event, end_event):
    # refill outside the timed region - each in-place sum grows the values by up to `ranks` times, which would overflow
    # fp32 to inf over the trials
    tensor.uniform_()
    dist.barrier()
    if args.with_host_overhead:
        start_event.record()
        dist.all_reduce(tensor)
    else:
        arch.busy_wait(BUSY_WAIT_MS)
        start_event.record()
        for _ in range(calls):
            dist.all_reduce(tensor)
    end_event.record()
    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / 1000 / calls

    n = dist.get_world_size()
    # note that this is following the same math as NVIDIA/nccl-tests
    algbw = torch.tensor([size / duration]).cuda(local_rank)

    # calculate mean across all ranks
    dist.reduce(algbw, dst=0, op=dist.ReduceOp.SUM)
    algbw /= n

    return algbw

def sym_mem_buffer(numel, local_rank):
    """
    Returns (backend, pool, buffer) where buffer is a `numel`-long fp32 tensor allocated with ncclMemAlloc and
    registered as an NCCL symmetric memory window on the default process group's communicator.
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
        buffer = torch.empty(numel, dtype=torch.float32, device=device)
    backend.register_mem_pool(pool, symm=True)
    return backend, pool, buffer

def run(local_rank):

    start_time = time.time()

    hostname = socket.gethostname()
    is_global_rank_0 = dist.get_rank() == 0
    ranks = dist.get_world_size()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if args.payload_size_in_gib is None:
        lower_limit = 15
        upper_limit = 34

        #lower_limit = 32
        #upper_limit = 32
        # 2**15 to 2**34 => 32KiB to 16GiB
        sizes = [2**x for x in range(lower_limit, upper_limit+1)]
    else:
        sizes = [int(args.payload_size_in_gib * 2**30)]

    # one window sized for the largest payload and registered once - each payload is an offset-0 view of it, so it
    # sits at the same window offset on every rank, as NCCL's symmetric kernels require
    sym_mem_pool = None
    if args.sym_mem:
        sym_mem_backend, sym_mem_pool, sym_mem_buf = sym_mem_buffer(max(sizes)//4, local_rank)

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

        trials = f"{args.num_warmup_iterations} warmups / {args.num_iterations} trials"
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
            print(f"The {args.payload_size_in_gib}GiB payload bandwidth of all_reduce over {ranks} ranks ({trials}):\n")
            print(f"|    busbw   |    algbw   |")
            print(f"| ---------: | ---------: |")
            for i in range(len(busbw_points)):
                print(f"| {conv_to_GBps(busbw_points[i]):6.2f}GBps | {conv_to_GBps(algbw_points[i]):6.2f}GBps |")
            busbw_GBps = [conv_to_GBps(x) for x in busbw_points]
            plot_path = f"busbw-profile-{hostname}-{ranks}{plot_suffix}.png"
            plot_profile(plot_path, busbw_GBps, ranks)


        else:
            print(f"The average bandwidth of all_reduce over {ranks} ranks ({trials}):\n")
            print(f"| payload |    busbw   |    algbw   |")
            print(f"| ------: | ---------: | ---------: |")
            for size in busbw.keys():
                print(f"| {fmt_bytes(size):>7} | {conv_to_GBps(busbw[size]):6.2f}GBps | {conv_to_GBps(algbw[size]):6.2f}GBps |")

            busbw_GBps = [conv_to_GBps(x) for x in busbw.values()]
            sizes_fmted = [fmt_bytes(x) for x in busbw.keys()]
            plot_path = f"busbw-mean-{hostname}-{ranks}{plot_suffix}.png"
            plot_averages(plot_path, sizes_fmted, busbw_GBps, ranks)

        time_delta = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
        print(f"Legend: 1KiB = 2**10 Bytes, 1MiB = 2**20 Bytes, 1GiB = 2**30 Bytes")
        print(f"        1GBps = 10**9 Bytes per second (networking bw spec convention)")
        print(f"Elapsed time: {time_str}")

    algbw = {}
    busbw = {}
    for size in sizes:
        # clear prev-iteration memory for cards w/ ~24GiB
        tensor = None
        gc.collect()

        # /4 is for 4 bytes in fp32
        if args.sym_mem:
            tensor = sym_mem_buf[:size//4].uniform_()
        else:
            tensor = torch.rand(size//4, dtype=torch.float32, device=torch.device("cuda", local_rank))

        calls = calls_per_trial(size)

        # do a few warm up iterations
        for i in range(args.num_warmup_iterations):
            timed_allreduce(tensor, size, calls, start_event, end_event)

        # real benchmark
        algbw_gather = []
        for i in range(args.num_iterations):
            if is_global_rank_0:
                print(f"{fmt_bytes(size):>6}: {i+1}", end="\r")
            algbw_gather += timed_allreduce(tensor, size, calls, start_event, end_event)
        if is_global_rank_0:
            print()


        # the 2*(n-1)/n busbw correction factor specific to all-reduce is explained here:
        # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
        # busbw reflects how optimally the hardware is used
        busbw_coeff = (2*(ranks - 1) / ranks)

        if args.profile_stability:
            algbw_points = [x.item() for x in algbw_gather]
            busbw_points = [x * busbw_coeff for x in algbw_points]
        else:
            algbw[size] = torch.mean(torch.stack(algbw_gather)).item()
            busbw[size] = algbw[size] * busbw_coeff

    finish()


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


def parse_args():
    global args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # this arg is not used directly, but a launcher may pass it
    parser.add_argument("--local_rank", type=int, default=0, help='local rank')
    parser.add_argument("--num_iterations", type=int, default=20, help='The number of iterations used to benchmark each collective call')
    parser.add_argument("--num_warmup_iterations", type=int, default=5, help='The number of warmup iterations')
    parser.add_argument("--payload_size_in_gib", type=float, default=None, help='payload size in GiBs, e.g. 4 (4GiB). If not specified the full range 2**15 .. 2**34 will be benchmarked')
    parser.add_argument("--profile_stability", action="store_true", help="Reports individual results for each non-warmup iteration. Requires --payload_size_in_gib. This is used to test the stability of performance, rather than reporting an averaged outcome.")
    parser.add_argument("--with-host-overhead", action="store_true", help="time 1 all-reduce per trial on an idle device, which adds PyTorch's per-call host overhead to every call - what an all-reduce that the program waits on (a sync point) costs")
    parser.add_argument("--sym-mem", action="store_true", help="all-reduce buffers registered as an NCCL symmetric memory window (torch>=2.9, NCCL>=2.27), the equivalent of nccl-tests' -R 2. Use only if the target workload all-reduces symmetric memory buffers too")

    args = parser.parse_args()

    if args.profile_stability and args.payload_size_in_gib is None:
        raise ValueError("--profile_stability requires --payload_size_in_gib to profile one specific payload, e.g. to profile a 0.5GiB payload use: --profile_stability --payload_size_in_gib 0.5")


if __name__ == "__main__":
    parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", **device_id_kwargs(local_rank))
    run(local_rank)
