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

- currently this benchmark scans a payload range of 32KB to 16GB.

- this benchmark automatically generates a plot of the results if you have `matplotlib` installed.

- if you are wondering whether you need to also run https://github.com/NVIDIA/nccl-tests - I
  already validated that I got very similar results with ./build/all_reduce_perf -b 4G -e 4G
  (tested with mpirun on 4 nodes). It should be either on par or slightly slower because it uses a
  blocking approach - that is it waits for each new all_reduce to finish before firing the next
  one, whereas nccl-tests fires them all in an async fashion (you can add `-z` to nccl-tests to
  emulate blocking)

- to benchmark other collectives use nccl-tests or adapt this benchmark to use the desired collective.

- you can interrupt (Ctrl-C) the benchmark in the middle and it'll complete with the results it has
  measured so far.

- you can also profile a single payload and get a plot with results for each iteration - for that use --profile_stability --payload_size_in_gib 0.5 (change the last value to the desired payload size in GiB)

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

# https://stackoverflow.com/a/75332100/9201239
fmt_bytes = lambda v : str(v >> ((max(v.bit_length()-1, 0)//10)*10)) +["", "K", "M", "G", "T", "P", "E"][max(v.bit_length()-1, 0)//10]+"iB"
# following the common networking hw spec convention which uses base 10, instead of 2 for bps/Bps (it makes speed look bigger than it is)
conv_to_GBps = lambda v : v/10**9

def get_device_info():
    if torch.cuda.is_available():
        return repr(torch.cuda.get_device_properties('cuda'))
    elif has_hpu:
        return repr(torch.hpu.get_device_properties('hpu'))
    else:
        return "Unknown accelerator"

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

    device_info = get_device_info()

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

    device_info = get_device_info()

    # wrap notes - this can now handle several lines of text.
    notes = "\n".join(textwrap.wrap(device_info, width=60))

    plt.annotate(notes,
                 xy=(0.001, -0.3),
                 xycoords='axes fraction',
                 ha='left',
                 va="center",
                 fontsize=10)

    plt.savefig(path, bbox_inches='tight')



def timed_allreduce(tensor, size, start_event, end_event):
    dist.barrier()
    start_event.record()
    dist.all_reduce(tensor)
    end_event.record()
    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / 1000

    n = dist.get_world_size()
    # note that this is following the same math as NVIDIA/nccl-tests
    algbw = torch.tensor([size / duration]).cuda(local_rank)

    # calculate mean across all ranks
    dist.reduce(algbw, dst=0, op=dist.ReduceOp.SUM)
    algbw /= n

    return algbw

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
        # 2**15 to 2**34 => 32KB to 16GB
        sizes = [2**x for x in range(lower_limit, upper_limit+1)]
    else:
        sizes = [int(args.payload_size_in_gib * 2**30)]

    # this is useful for when one wants to interrupt the run - and still report the best outcome so far
    def sigkill_handler(signum, frame):
         finish()
         sys.exit(1)

    signal.signal(signal.SIGINT, sigkill_handler)

    def finish():
        dist.destroy_process_group()

        if not is_global_rank_0:
            return

        print(f"\nEnvironment:")
        print(f"- software: torch={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
        print(f"- hardware: {get_device_info()}\n")

        if args.profile_stability:
            print(f"The {args.payload_size_in_gib}GiB payload bandwidth of all_reduce over {ranks} ranks ({args.num_warmup_iterations} warmups / {args.num_iterations} trials):\n")
            print(f"|    busbw   |    algbw   |")
            print(f"| ---------: | ---------: |")
            for i in range(len(busbw_points)):
                print(f"| {conv_to_GBps(busbw_points[i]):6.2f}GBps | {conv_to_GBps(algbw_points[i]):6.2f}GBps |")
            busbw_GBps = [conv_to_GBps(x) for x in busbw_points]
            plot_path = f"busbw-profile-{hostname}-{ranks}.png"
            plot_profile(plot_path, busbw_GBps, ranks)


        else:
            print(f"The average bandwidth of all_reduce over {ranks} ranks ({args.num_warmup_iterations} warmups / {args.num_iterations} trials):\n")
            print(f"| payload |    busbw   |    algbw   |")
            print(f"| ------: | ---------: | ---------: |")
            for size in busbw.keys():
                print(f"| {fmt_bytes(size):>7} | {conv_to_GBps(busbw[size]):6.2f}GBps | {conv_to_GBps(algbw[size]):6.2f}GBps |")

            busbw_GBps = [conv_to_GBps(x) for x in busbw.values()]
            sizes_fmted = [fmt_bytes(x) for x in busbw.keys()]
            plot_path = f"busbw-mean-{hostname}-{ranks}.png"
            plot_averages(plot_path, sizes_fmted, busbw_GBps, ranks)

        time_delta = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
        print(f"Legend: 1KiB = 2**10 Bytes, 1MiB = 2**20 Bytes, 1GiB = 2**30 Bytes")
        print(f"        1GBps = 10**9 Bytes per second (networking bw spec convention)")
        print(f"Elapsed time: {time_str}")

    algbw = {}
    busbw = {}
    for size in sizes:
        # clear prev-iteration memory for cards w/ ~24GB
        tensor = None
        gc.collect()

        # /4 is for 4 bytes in fp32
        tensor = torch.rand(size//4, 1, dtype=torch.float32).cuda(local_rank)

        # do a few warm up iterations
        for i in range(args.num_warmup_iterations):
            timed_allreduce(tensor, size, start_event, end_event)

        # real benchmark
        algbw_gather = []
        for i in range(args.num_iterations):
            if is_global_rank_0:
                print(f"{fmt_bytes(size):>6}: {i+1}", end="\r")
            algbw_gather += timed_allreduce(tensor, size, start_event, end_event)
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
    parser = argparse.ArgumentParser()

    # this arg is not used directly, but a launcher may pass it
    parser.add_argument("--local_rank", type=int, default=0, help='local rank')
    parser.add_argument("--num_iterations", type=int, default=20, help='The number of iterations used to benchmark each collective call')
    parser.add_argument("--num_warmup_iterations", type=int, default=5, help='The number of warmup iterations')
    parser.add_argument("--payload_size_in_gib", type=float, default=None, help='payload size in GiBs, e.g. 4 (4GiB). If not specified the full range 2**15 .. 2**32 will be benchmarked')
    parser.add_argument("--profile_stability", action="store_true", help="Reports individual results for each non-warmup iteration. Requires --payload_size_in_gib. This is used to test the stability of performance, rather than reporting an averaged outcome.")

    args = parser.parse_args(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    if args.profile_stability and args.payload_size_in_gib is None:
        raise ValueError("--profile_stability requires --payload_size_in_gib to profile one specific payload, e.g. to profile a 0.5GiB payload use: --profile_stability --payload_size_in_gib 0.5")


if __name__ == "__main__":
    parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", **device_id_kwargs(local_rank))
    run(local_rank)
