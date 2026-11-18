#!/usr/bin/env python

r"""

The latest version of this program can be found at
https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/comm-compute-overlap-bench.py

This benchmark measures how much of a training step's communication overlaps with its compute - and, through the
forward phase of the model-parallel patterns, how much of an inference prefill's (not decode's) communication does -
for a parallelism pattern such as DDP, DeepSpeed ZeRO, FSDP2, tensor, sequence, context or expert parallelism, without
setting up the framework and a model.

For detailed documentation please refer to:
https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/README.md#comm-compute-overlap-bench

To list the supported patterns and their settings:

python comm-compute-overlap-bench.py --list-patterns



### Single node examples

- FSDP2 with Llama-3.1-8B layers and 8192 tokens per rank, on 8 GPUs:

python -u -m torch.distributed.run --nproc_per_node 8 --rdzv_endpoint localhost:6000 --rdzv_backend c10d \
    comm-compute-overlap-bench.py --pattern fsdp --model llama-3.1-8b --tokens 8192

- DeepSpeed ZeRO-2 with your own model's layers at 2 token counts, with overlap_comm on and the gradients reduced
in fp32:

python -u -m torch.distributed.run --nproc_per_node 8 --rdzv_endpoint localhost:6000 --rdzv_backend c10d \
    comm-compute-overlap-bench.py --pattern ds-zero2 --model path/to/config.json --tokens 4096,16384 \
    --set overlap_comm=1,communication_data_type=fp32

- Megatron-LM tensor parallelism with sequence parallelism over 8 GPUs, 16384 tokens per tensor-parallel group:

python -u -m torch.distributed.run --nproc_per_node 8 --rdzv_endpoint localhost:6000 --rdzv_backend c10d \
    comm-compute-overlap-bench.py --pattern sp-megatron --set tp=8 --tokens 16384

- all_gather alone and queued before a bf16 8192x8192x8192 matmul, from 1MiB to 1GiB:

python -u -m torch.distributed.run --nproc_per_node 8 --rdzv_endpoint localhost:6000 --rdzv_backend c10d \
    comm-compute-overlap-bench.py --pattern contention \
    --set collective=all_gather,kernel=matmul:8192x8192x8192,min_payload=1M,max_payload=1G


### Multi-node examples

Adapting the single-node FSDP2 example from above to multi-node

- using an salloc allocation or an sbatch script - srun starts the launcher on each node, and the rendezvous endpoint
is the allocation's first node:

salloc --partition=mypartition --nodes=4 --ntasks-per-node=1 --cpus-per-task=48 --gres=gpu:8 --time=1:00:00 bash

srun --gres=gpu:8 --nodes=4 --tasks-per-node=1 \
    python -u -m torch.distributed.run --nproc_per_node 8 --nnodes 4 \
    --rdzv_endpoint $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):6000 --rdzv_backend c10d \
    comm-compute-overlap-bench.py --pattern fsdp --model llama-3.1-8b --tokens 8192


- with pdsh, which needs passwordless ssh between the nodes:

Either hardcode the IPs or hostnames:

MASTER_HOST=10.0.0.10
HOSTS=10.0.0.10,10.0.0.11,10.0.0.12,10.0.0.13
NNODES=4

or, if you already have a DeepSpeed-style hostfile, with a "hostname slots=x" entry per line, or an MPI-style one,
with a "hostname" per line, take them from it:

MASTER_HOST=$(cat ~/hostfile | cut -d " " -f1 | head -1)
HOSTS=$(cat ~/hostfile | cut -d " " -f1 | tr '\n' ',' | sed 's/,*$//g')
NNODES=4

You can first check that your pdsh setup works with this quick command, which prints the hostname of each
participating node:

PDSH_RCMD_TYPE=ssh pdsh -w $HOSTS hostname

Then run the benchmark, after setting `DIR` to the directory that holds this script - pdsh doesn't run the command
in your current working directory, so the script has to be given by its full path:

DIR=/change/the/path/benchmarks
PDSH_RCMD_TYPE=ssh pdsh -w $HOSTS \
    python -u -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES \
    --rdzv_endpoint $MASTER_HOST:6000 --rdzv_backend c10d \
    $DIR/comm-compute-overlap-bench.py --pattern fsdp --model llama-3.1-8b --tokens 8192


- you can also launch the same script on all participating nodes manually - the c10d rendezvous assigns the node
ranks, so the command is the same on each node, as long as MASTER_HOST is the first node on all of them, taken from
the same hostfile as here or set by hand:

MASTER_HOST=$(cat ~/hostfile | cut -d " " -f1 | head -1) # set it explicitly if you don't have hostfile
NNODES=4
python -u -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --rdzv_endpoint $MASTER_HOST:6000 \
    --rdzv_backend c10d comm-compute-overlap-bench.py --pattern fsdp --model llama-3.1-8b --tokens 8192


"""

import argparse
import atexit
import collections
import contextlib
import datetime
import functools
import gc
import itertools
import json
import os
import socket
import sys
import textwrap
import threading
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F

args = None

# how long the device is kept busy before a trial's work is queued, at least - each pattern measures how long its host
# takes to queue a trial and waits half as long again. It must be the same on all ranks, otherwise the ranks with a
# shorter wait time their work waiting for the others to arrive
BUSY_WAIT_MS = 2

# how long the device idles before each trial. The compute runs at the GPU's power cap, where its speed depends on what
# the device ran in the last 10s of ms: on an 8x B200 node 20ms of bf16 matmuls took up to 18% longer right after
# other matmuls, or after the busy wait's spin, than after an idle device, and after 100ms of idling they took as long
# whatever came before. Without it each kind of trial would start from whatever the kind before it left behind
IDLE_MS = 100

# the payload's 1/ranks pieces are rounded down to this, as nccl-tests does
CHUNK_ALIGN_BYTES = 16

# torch 2.13 renamed all_gather_into_tensor and reduce_scatter_tensor, deprecating the old names
all_gather_single = getattr(dist, "all_gather_single", dist.all_gather_into_tensor)
reduce_scatter_single = getattr(dist, "reduce_scatter_single", dist.reduce_scatter_tensor)

# contention: the most collective calls one trial queues, the calls and kernel runs of a probe, and how many times a
# payload's trials are halved while the host falls behind in its warmups
MAX_CALLS = 5000
PROBE_CALLS = 10
PROBE_RUNS = 5
MAX_HALVINGS = 4
# contention: a matmul's speed varies by up to 1.5% from one trial to the next on a B200, so an overlapped trial is
# reported as too short only below this share of the longer of the two alone
SHORTEST_OVERLAPPED = 0.95

BUSBW_FACTOR = dict(
    all_reduce     = lambda n: 2 * (n - 1) / n,
    all_gather     = lambda n: (n - 1) / n,
    reduce_scatter = lambda n: (n - 1) / n,
    all_to_all     = lambda n: (n - 1) / n,
)

# the layer shapes of a few models, for --model without a config.json
MODEL_PRESETS = {
    "llama-3.1-8b":  dict(hidden_size=4096, intermediate_size=14336, num_attention_heads=32, num_key_value_heads=8,
                          head_dim=128),
    "qwen3-32b":     dict(hidden_size=5120, intermediate_size=25600, num_attention_heads=64, num_key_value_heads=8,
                          head_dim=128),
    "llama-3.1-70b": dict(hidden_size=8192, intermediate_size=28672, num_attention_heads=64, num_key_value_heads=8,
                          head_dim=128),
    # MoE, for the ep pattern
    "qwen3-30b-a3b": dict(hidden_size=2048, intermediate_size=6144, num_attention_heads=32, num_key_value_heads=4,
                          head_dim=128, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768),
    "mixtral-8x7b":  dict(hidden_size=4096, intermediate_size=14336, num_attention_heads=32, num_key_value_heads=8,
                          head_dim=128, num_local_experts=8, num_experts_per_tok=2),
}

# 8 -> 8B, 2**20 -> 1MiB, 768 -> 768B, 3*2**19 -> 1.5MiB
def fmt_bytes(v):
    unit = max(v.bit_length() - 1, 0) // 10
    return f"{v / 2**(unit * 10):.4g}" + ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"][unit]
fmt_ms = lambda v: f"{v * 1000:.2f}ms"

def fmt_share(part, whole, digits=0):
    """`part` as a share of `whole`, e.g. 35%, 0% if `part` is below 0"""
    return f"{max(part / whole, 0) * 100:.{digits}f}%"
# following the common networking hw spec convention which uses base 10, instead of 2 for bps/Bps
conv_to_GBps = lambda v: v / 10**9

def table_line(cells, widths, aligns=None):
    aligns = aligns or ">" * len(widths)
    def pad(cell, width, align):
        if align == "<":
            return cell.ljust(width)
        if align == "^":
            return cell.center(width)
        return cell.rjust(width)
    return "| " + " | ".join(pad(cell, width, align) for cell, width, align in zip(cells, widths, aligns)) + " |"

def table_rule(widths, aligns=None):
    aligns = aligns or ">" * len(widths)
    cells = [":" + "-" * (width - 1) if align == "<" else "-" * (width - 1) + ":"
             for width, align in zip(widths, aligns)]
    return "| " + " | ".join(cells) + " |"

def format_table(title, heads, rows, aligns):
    """`heads` are (top, bottom) label pairs - the bottom line is printed only if a bottom label is set, and a top
    label of None continues the top label of the column before it, which then spans both. A title of None prints
    none"""
    widths = [max(len(head[1]), *(len(row[i]) for row in rows)) for i, head in enumerate(heads)]
    # each top label's [first, last + 1) columns
    spans = []
    for i, head in enumerate(heads):
        if head[0] is None:
            spans[-1][1] = i + 1
        else:
            spans.append([i, i + 1])
    # a span's width is its columns' and the " | " between them
    span_width = lambda start, end: sum(widths[start:end]) + len(" | ") * (end - start - 1)
    for start, end in spans:
        for i in range(len(heads[start][0]) - span_width(start, end)):
            widths[start + i % (end - start)] += 1
    lines = [] if title is None else [title]
    lines.append("|" + "|".join(f" {heads[start][0].center(span_width(start, end))} " for start, end in spans) + "|")
    if any(head[1] for head in heads):
        lines.append(table_line([head[1] for head in heads], widths, "^" * len(heads)))
    lines.append(table_rule(widths, aligns))
    lines += [table_line(row, widths, aligns) for row in rows]
    return lines

def device():
    return torch.device("cuda", torch.cuda.current_device())

def device_info():
    return repr(torch.cuda.get_device_properties(device()))

def nccl_version():
    """The NCCL that is loaded, which LD_PRELOAD can make another than the one torch was built with and reports"""
    import ctypes
    try:
        code = ctypes.c_int()
        ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(code))
        return (code.value // 10000, code.value // 100 % 100, code.value % 100)
    except (OSError, AttributeError):
        return torch.cuda.nccl.version()

_sleep_cycles_per_ms = None
def busy_wait(ms):
    """Keep the GPU busy for `ms` milliseconds. torch.cuda._sleep spins for a number of device clock cycles, which
    tick at a different rate on each GPU, so the cycles per ms are measured on the first call."""
    global _sleep_cycles_per_ms
    if _sleep_cycles_per_ms is None:
        cycles = 1_000_000
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(cycles)
        start.record()
        torch.cuda._sleep(cycles)
        end.record()
        torch.cuda.synchronize()
        _sleep_cycles_per_ms = cycles / start.elapsed_time(end)
    torch.cuda._sleep(int(ms * _sleep_cycles_per_ms))

def start_trial(busy_ms):
    """Lets the device idle for IDLE_MS, lines the ranks up and keeps the device busy for `busy_ms`, so that the work
    queued next starts on all ranks at once, from the same state of the device whatever ran before it, and runs back to
    back however long the host takes to queue it. Returns the host's start time and the event the trial's streams start
    from"""
    torch.cuda.synchronize()
    time.sleep(IDLE_MS / 1000)
    dist.barrier()
    align = torch.zeros(1, device=device())
    queue_start = time.perf_counter()
    busy_wait(busy_ms)
    dist.all_reduce(align)
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    return queue_start, start

_progress = ""
_progress_since = 0.0
_progress_lock = threading.Lock()
_spinner = None

def draw_progress(text, overwrite):
    r"""torchrun tails the log with readline, which holds everything until \n, so off a terminal each update is its
    own line, which \033[1A moves back onto the previous one"""
    if sys.stdout.isatty():
        sys.stdout.write(f"\r\033[K{text}")
    elif overwrite:
        sys.stdout.write(f"\r\033[K\033[1A\r\033[K{text}\n")
    else:
        sys.stdout.write(f"\r\033[K{text}\n")
    sys.stdout.flush()

def spin():
    """Redraws the status line with a spinner and how long it has stood, so that a hang shows as a count that keeps
    climbing - the spinner alone would keep turning through a hung collective"""
    for frame in itertools.cycle("|/-\\"):
        time.sleep(1 if sys.stdout.isatty() else 10)
        with _progress_lock:
            if _progress:
                draw_progress(f"{_progress} {frame} {time.monotonic() - _progress_since:.0f}s", True)

def show_progress(text):
    r"""One status line, rewritten with \r"""
    global _progress, _progress_since, _spinner
    if dist.get_rank() != 0:
        return
    with _progress_lock:
        draw_progress(text, bool(_progress))
        _progress, _progress_since = text, time.monotonic()
    if _spinner is None:
        _spinner = threading.Thread(target=spin, daemon=True)
        _spinner.start()

def end_progress():
    global _progress
    if dist.get_rank() != 0 or not _progress:
        return
    with _progress_lock:
        draw_progress(_progress, True)
        if sys.stdout.isatty():
            sys.stdout.write("\n")
            sys.stdout.flush()
        _progress = ""

def new_plot(path):
    """Returns matplotlib.pyplot with a new figure, or None if matplotlib isn't installed"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if not new_plot.warned:
            print("!!! Can't generate plots. Please run `pip install matplotlib` to enable plotting. !!!")
            new_plot.warned = True
        return None
    print(f"\n*** Plotting results into {path}\n")
    plt.figure(dpi=500)
    return plt
new_plot.warned = False

def save_plot(plt, path):
    notes = "\n".join(textwrap.wrap(device_info(), width=60))
    plt.annotate(notes, xy=(0.001, -0.3), xycoords="axes fraction", ha="left", va="center", fontsize=10)
    plt.savefig(path, bbox_inches="tight")


### Settings ###

def byte_size(s):
    """'25M' -> 25*2**20 - K, M and G are powers of 2, as in DDP's bucket_cap_mb"""
    units = dict(K=2**10, M=2**20, G=2**30)
    s = s.strip().upper().removesuffix("B").removesuffix("I")
    size = int(s[:-1]) * units[s[-1]] if s[-1:] in units else int(s)
    if size < 1:
        raise ValueError(f"{s} must be at least 1 byte")
    return size

def payload_size(s):
    """A byte_size that's a power of 2 of at least 4 bytes, one fp32 element"""
    size = byte_size(s)
    if size < 4 or size & (size - 1):
        raise ValueError(f"{s} must be a power of 2 and at least 4 bytes (one fp32 element)")
    return size

def dtype_setting(s):
    dtypes = dict(bf16=torch.bfloat16, fp32=torch.float32)
    if s not in dtypes:
        raise ValueError(f"{s}: pick one of {', '.join(dtypes)}")
    return dtypes[s]

def bool_setting(s):
    if s.lower() in ("1", "true", "yes"):
        return True
    if s.lower() in ("0", "false", "no"):
        return False
    raise ValueError(f"{s}: use 1 or 0")

def count_setting(s):
    if not s.isdigit():
        raise ValueError(f"{s} must be a whole number")
    return int(s)

def degree_setting(s):
    """A parallelism degree, the ranks of each group - at least 2, as a degree of 1 is no parallelism - or `node`
    for the GPUs of a node"""
    if s == "node":
        return int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    degree = count_setting(s)
    if degree < 2:
        raise ValueError(f"{s} must be at least 2, as a degree of 1 is no parallelism")
    return degree

def calls_setting(s):
    return "auto" if s == "auto" else max(1, count_setting(s))

def choice_setting(*choices):
    def parse(s):
        if s not in choices:
            raise ValueError(f"{s}: pick one of {', '.join(choices)}")
        return s
    return parse

def kernel_setting(value):
    """'matmul' -> ('matmul', (8192, 8192, 8192)), 'matmul:16384x4096x14336' -> ('matmul', (16384, 4096, 14336)),
    'copy' -> ('copy', 2**30), 'copy:2G' -> ('copy', 2**31)"""
    kind, _, shape = value.strip().lower().partition(":")
    if kind == "matmul":
        if not shape:
            return kind, (8192, 8192, 8192)
        dims = shape.split("x")
        if len(dims) != 3 or not all(dim.isdigit() and int(dim) > 0 for dim in dims):
            raise ValueError(f"{value}: the matmul shape is MxNxK, e.g. matmul:8192x8192x8192")
        return kind, tuple(int(dim) for dim in dims)
    if kind == "copy":
        return kind, payload_size(shape) if shape else 2**30
    raise ValueError(f"{value}: the kernel is matmul, matmul:MxNxK, copy or copy:SIZE")

def elements_setting(s):
    """'5e8' -> 500000000, as DeepSpeed's config gives its bucket sizes in elements"""
    try:
        value = int(float(s))
    except ValueError:
        raise ValueError(f"{s} must be a number of elements, e.g. 5e8")
    if value < 1:
        raise ValueError(f"{s} must be at least 1 element")
    return value

def mib_setting(s):
    """'25' -> 25*2**20, as DDP's bucket_cap_mb is in MiB"""
    try:
        value = float(s)
    except ValueError:
        raise ValueError(f"{s} must be a number of MiB, e.g. 25")
    if value <= 0:
        raise ValueError(f"{s} must be more than 0")
    return int(value * 2**20)

DS_BUCKET = ("5e8", elements_setting, "reduce_bucket_size: the most gradient elements one reduction takes, 953.7MiB "
             "in bf16")
DS_COMM_DTYPE = ("bf16", dtype_setting, "communication_data_type: the dtype the gradients are reduced in, bf16 "
                 "being DeepSpeed's default with bf16 enabled - fp32 doubles the bytes reduced")
DS_OVERLAP_COMM = ("0", bool_setting, "overlap_comm: 1 reduces each bucket on a stream of its own while backward "
                   "goes on, the compute waiting for the bucket before it when it gets to the next one, 0 reduces on "
                   "the compute stream - DeepSpeed defaults to 0 for ZeRO-1 and ZeRO-2")
DS_ABOUT = ("All-reduces the gradients in buckets, each rank keeping its partition of the result, as DeepSpeed does "
            "with its defaults reduce_scatter=true and use_multi_rank_bucket_allreduce=true, and after the optimizer "
            "step all-gathers the updated parameters in one call, as DeepSpeed does whatever allgather_bucket_size is")
GAS = ("1", count_setting, "gradient_accumulation_steps: the micro-batches of each optimizer step, each a forward "
       "and a backward of --tokens tokens per rank")
TP_DEGREE = ("node", degree_setting, "the tensor-parallel degree, `node` for the GPUs of a node")

PATTERNS = {
    "ddp": dict(
        about="PyTorch DDP: as each gradient bucket fills in backward, it's copied into the bucket and all-reduced on "
              "a stream of its own while backward goes on, and the results are copied back into the gradients at "
              "the end of backward. A bucket is closed once it reaches the cap, in the reverse order of the "
              "parameters",
        settings=dict(
            bucket_cap_mb=("25", mib_setting, "the bucket cap in MiB"),
            first_bucket_mb=("1", mib_setting, "the cap of the first bucket, which DDP keeps small so that the "
                             "first all-reduce starts early"),
            gradient_as_bucket_view=("0", bool_setting, "1 makes the gradients views into the buckets, which "
                                     "saves the 2 copies"),
            gas=GAS,
            no_sync=("1", bool_setting, "1 runs the micro-batches before the last under DDP's no_sync(), so that "
                     "only the last one's backward all-reduces, as HF Trainer and Accelerate do - 0 all-reduces in "
                     "each one's backward"))),
    "ds-zero1": dict(
        about="DeepSpeed ZeRO-1. " + DS_ABOUT + ". With overlap_comm=0, ZeRO-1 reduces all the gradients after "
              "backward, otherwise as each bucket fills in backward - with gas above 1 only in the last "
              "micro-batch's backward",
        settings=dict(reduce_bucket_size=DS_BUCKET, overlap_comm=DS_OVERLAP_COMM,
                      communication_data_type=DS_COMM_DTYPE, gas=GAS)),
    "ds-zero2": dict(
        about="DeepSpeed ZeRO-2. " + DS_ABOUT + ". ZeRO-2 reduces each bucket as it fills in backward, on the "
              "compute stream unless overlap_comm=1 - with gas above 1 in each micro-batch's backward",
        settings=dict(reduce_bucket_size=DS_BUCKET, overlap_comm=DS_OVERLAP_COMM,
                      communication_data_type=DS_COMM_DTYPE, gas=GAS)),
    "ds-zero3": dict(
        about="DeepSpeed ZeRO-3: each linear layer's weight is all-gathered when its module needs it, together with "
              "the next ones up to stage3_prefetch_bucket_size, which a module's weights stay gathered for if "
              "they're used again within stage3_max_reuse_distance elements. The gradients are copied into a bucket "
              "and reduce-scattered on a stream of its own, a bucket at a time. With gas above 1 each micro-batch "
              "fetches the weights anew, as DeepSpeed resets the fetching at the start of each forward",
        settings=dict(
            stage3_prefetch_bucket_size=("5e7", elements_setting, "the most elements prefetched past the module "
                                         "being fetched"),
            stage3_max_live_parameters=("1e9", elements_setting, "the most gathered elements at a time, which "
                                        "limits the prefetching"),
            stage3_max_reuse_distance=("1e9", elements_setting, "a module's weights stay gathered if they're used "
                                       "again within this many elements - which keeps the last layers' weights "
                                       "gathered from forward into backward"),
            reduce_bucket_size=DS_BUCKET,
            communication_data_type=DS_COMM_DTYPE,
            gas=GAS)),
    "fsdp": dict(
        about="PyTorch FSDP2 with fully_shard on each transformer layer: each layer's parameters are all-gathered "
              "into a buffer and copied out of it before the layer's compute, and its gradients are copied into a "
              "buffer and reduce-scattered after its backward",
        settings=dict(
            reshard_after_forward=("1", bool_setting, "1 frees the gathered parameters after forward, so backward "
                                   "all-gathers them again, 0 keeps them until the end of backward, as FSDP1's "
                                   "ShardingStrategy.SHARD_GRAD_OP does"),
            forward_prefetch=("1", count_setting, "how many layers ahead forward all-gathers - 1 is FSDP2's "
                              "default, in which each layer's all-gather waits for the compute to start on the layer "
                              "2 before it, FSDP2's set_modules_to_forward_prefetch() method prefetches more"),
            backward_prefetch=("1", count_setting, "how many layers ahead backward all-gathers - 1 is FSDP2's "
                               "default, FSDP2's set_modules_to_backward_prefetch() method sets others"),
            reduce_dtype=("bf16", dtype_setting, "MixedPrecisionPolicy.reduce_dtype - fp32 doubles the bytes "
                          "reduced"),
            separate_rs_group=("0", bool_setting, "1 reduce-scatters on a process group of its own, as FSDP2's "
                               "set_separate_reduce_scatter_group() method does, so that the reduce-scatters can run "
                               "while an all-gather does - one NCCL communicator runs one collective at a time"),
            high_priority=("1", bool_setting, "1 queues the collectives on high-priority streams, as FSDP2 does"),
            gas=GAS,
            requires_gradient_sync=("1", bool_setting, "0 reduce-scatters only in the last micro-batch's backward, "
                                    "the others keeping their gradients unsharded, as FSDP2's "
                                    "set_requires_gradient_sync(False) method does for them"),
            reshard_after_backward=("1", bool_setting, "0 keeps the parameters gathered from each micro-batch's "
                                    "backward into the next one's forward, as FSDP2's "
                                    "set_reshard_after_backward(False) method does for all but the last - with "
                                    "reshard_after_forward=1 backward still all-gathers them again"))),
    "tp": dict(
        about="Megatron-LM's tensor parallelism over groups of tp ranks, which compute the same tokens: q, k, v, "
              "gate and up are split by their output features, o and down by their input features, whose partial "
              "outputs are all-reduced, blocking the compute, in forward and in backward. Backward all-reduces the "
              "input gradients of q, k, v, gate and up while it computes their weight gradients. --tokens is per "
              "tensor-parallel group. The CUDA_DEVICE_MAX_CONNECTIONS env var is set to 1 unless it's set already, "
              "as Megatron-LM recommends",
        settings=dict(tp=TP_DEGREE)),
    "sp-megatron": dict(
        about="Megatron-LM's sequence parallelism, which runs only together with its tensor parallelism, over the "
              "same groups: the tp pattern, with the tokens split over the tensor-parallel group between the split "
              "matmuls, turning each all_reduce into a reduce_scatter and an all_gather, with the same bytes in all, "
              "and backward all-gathers the input of q, k, v, gate and up again while it computes their input "
              "gradients. --tokens is per tensor-parallel group, and CUDA_DEVICE_MAX_CONNECTIONS is set as with tp",
        settings=dict(tp=TP_DEGREE)),
    "sp-ulysses": dict(
        about="DeepSpeed Ulysses sequence parallelism over groups of sp ranks, each with a piece of one sequence "
              "and all the weights: all_to_all calls switch q, k and v to all the sequence with a piece of the "
              "heads before attention, and its output back after it, blocking the compute, and backward does the "
              "reverse. The sequence is --tokens times sp, and --seqlen doesn't apply",
        settings=dict(sp=("node", degree_setting, "the sequence-parallel degree, `node` for the GPUs of a node"))),
    "cp-rsa": dict(
        about="Context parallelism with ring attention, blockwise as Transformer Engine's context parallelism with "
              "KV P2P runs it, over groups of cp ranks, each with a load-balanced piece of one sequence and all the "
              "weights: each rank sends its k and v around the ring, one step per rank, while it computes its q's "
              "attention to the k and v it has, and backward sends k and v around again with their gradients. The "
              "sequence is --tokens times cp, and --seqlen doesn't apply",
        settings=dict(
            cp=("node", degree_setting, "the context-parallel degree, `node` for the GPUs of a node"),
            cuda_graphs=("1", bool_setting, "1 queues each ring step's attention, forward and backward, as one CUDA "
                         "graph, as the host takes longer to queue a step's SDPA and autograd kernels one by one than "
                         "the GPU takes to run them with 2048 tokens per rank - 0 queues them one by one"))),
    "ep": dict(
        about="MoE expert parallelism over groups of ep ranks, each with its own tokens and attention weights and an "
              "equal share of the experts, as Megatron-LM's alltoall token dispatcher runs it: an all_to_all sends "
              "each token to its top_k experts' ranks and another sends their outputs back, with a routing that "
              "sends each expert the same number of tokens. --model has to be an MoE model, e.g. qwen3-30b-a3b",
        settings=dict(
            ep=("node", degree_setting, "the expert-parallel degree, `node` for the GPUs of a node"),
            micro_batches=("1", count_setting, "2 splits the tokens into 2 micro-batches, and runs the all_to_all "
                           "calls of each while the other computes, as DeepSeek's DualPipe and SGLang's two-batch "
                           "overlap do - 1 blocks the compute on each all_to_all"),
            dispatcher=("nccl", choice_setting("nccl", "deepep"), "nccl sends the tokens with NCCL all_to_all calls, "
                        "deepep with DeepEP's dispatch and combine kernels, which send each token once to each rank "
                        "of its experts and copy it into the expert's rows themselves - deepep needs DeepEP "
                        "(https://github.com/deepseek-ai/DeepEP) installed, which no other pattern or setting needs"),
            deepep_sms=("0", count_setting, "the SMs of DeepEP's kernels, 0 for DeepEP's own estimate for the "
                        "topology"))),
    "pp": dict(
        about="Megatron-LM's interleaved 1F1B pipeline schedule over groups of pp ranks, world size / pp ranks apart "
              "as in Megatron-LM, so that with pp=2 over 2 nodes each sends to the other node: the layers are split "
              "into pp times virtual_stages chunks of consecutive layers, chunk i on the group's rank i % pp, and each "
              "micro-batch of --tokens tokens goes through the chunks in order, each sending its output to the next "
              "chunk's rank, and in backward the gradient of its input to the previous one. Besides the compute "
              "alone, the comm alone and both, a fourth trial runs the schedule with 1-element sends, whose ranks "
              "wait for each other as in training while moving almost no bytes, which times the pipeline bubble",
        settings=dict(
            pp=("node", degree_setting, "the pipeline-parallel degree, `node` for the GPUs of a node"),
            virtual_stages=("2", count_setting, "the chunks of layers on each rank, Megatron-LM's "
                            "virtual_pipeline_model_parallel_size - at least 2, as the schedule is the interleaved "
                            "one"),
            micro_batches=("16", count_setting, "the micro-batches of each step, a multiple of pp - the bubble is "
                           "about (pp - 1) / (micro_batches * virtual_stages) of the step"),
            overlap_p2p_comm=("1", bool_setting, "1 queues each send and receive as soon as its tensor or buffer "
                              "is ready and waits for it only before the compute that needs it, as Megatron-LM does "
                              "by default with virtual stages, 0 batches them and waits for them after each forward, "
                              "or each forward and backward, as its batch_p2p_comm does"))),
    "contention": dict(
        about="one collective queued before a kernel running on another stream, swept over payloads - how much each "
              "slows the other down. --model and --tokens don't apply",
        settings=dict(
            collective=("all_gather", choice_setting(*BUSBW_FACTOR), "the collective"),
            kernel=("matmul", kernel_setting, "`matmul` for a bf16 8192x8192x8192 matmul, which competes mostly for "
                    "the SMs, `matmul:MxNxK` for another shape, `copy` for an elementwise bf16 copy of 1GiB, which "
                    "competes mostly for the memory bandwidth, or `copy:SIZE` for another size"),
            calls=("1", calls_setting, "the collective calls per trial - 1, as a training step issues one collective "
                   "ahead of the compute it overlaps with, or `auto` for back-to-back calls lasting about "
                   "kernel_ms, which shows how collectives queued behind one another fare"),
            kernel_ms=("20", float, "how long the kernel runs of one trial take alone"),
            order=("comm-first", choice_setting("comm-first", "kernel-first"), "which is queued first - a "
                   "collective queued after a kernel that fills all the SMs waits for that kernel to end"),
            min_payload=("1M", payload_size, "the smallest payload, a power of 2"),
            max_payload=("1G", payload_size, "the largest payload, a power of 2"))),
}

def parse_settings(pattern, text):
    """The pattern's settings, each the default unless `text`, e.g. 'overlap_comm=1,reduce_bucket_size=2.5e8', sets
    it"""
    specs = PATTERNS[pattern]["settings"]
    given = {}
    for item in filter(None, (piece.strip() for piece in (text or "").split(","))):
        key, sep, value = item.partition("=")
        if not sep or key not in specs:
            sys.exit(f"--set: '{item}' isn't a setting of {pattern}, which has: {', '.join(specs)} - see "
                     f"--list-patterns")
        given[key] = value.strip()
    settings = {}
    for key, (default, parse, _) in specs.items():
        try:
            settings[key] = parse(given.get(key, default))
        except ValueError as e:
            sys.exit(f"--set {key}: {e}")
    settings["_given"] = given
    return settings

def list_patterns():
    for name, pattern in PATTERNS.items():
        print(f"{name}: " + "\n    ".join(textwrap.wrap(pattern["about"], width=112)))
        for key, (default, _, help) in pattern["settings"].items():
            label = f"{key}={default}" if default else f"{key}, unset by default"
            print("\n".join(textwrap.wrap(f"{label}: {help}", width=112, initial_indent="  - ",
                                          subsequent_indent="    ")))
        print()


### Communication setup ###

def nccl_pg_options():
    """init_process_group kwargs for --cta-policy"""
    if args.cta_policy == "default":
        return {}
    pg = dist.ProcessGroupNCCL
    policy = f"NCCL_CTA_POLICY_{args.cta_policy.upper()}"
    if not hasattr(pg, policy):
        sys.exit(f"--cta-policy needs a torch with ProcessGroupNCCL.{policy}, this is torch=={torch.__version__}")
    if torch.cuda.nccl.version() < (2, 28):
        sys.exit(f"--cta-policy needs NCCL>=2.28, this torch uses nccl={torch.cuda.nccl.version()}")
    options = pg.Options()
    options.config.cta_policy = getattr(pg, policy)
    return dict(pg_options=options)

def device_id_kwargs(local_rank):
    """device_id for init_process_group if this torch has it and doesn't hang with it"""
    from packaging import version
    import inspect
    # with device_id, NCCL serializes each send and recv with all the group's other ops, while without it each pair of
    # ranks gets a communicator of its own - what Megatron-LM, which doesn't set device_id, overlaps the pipeline's with
    if args.pattern == "pp":
        return {}
    # setting device_id leads to hanging in 2.6.0<torch<2.7.1 https://github.com/pytorch/pytorch/issues/153960
    if ("device_id" in inspect.signature(dist.init_process_group).parameters and
            not version.parse("2.6.0") < version.parse(torch.__version__) < version.parse("2.7.1")):
        return dict(device_id=torch.device(local_rank))
    return {}

class OOMGuard:
    """Has all the ranks skip a token count that some of them ran out of memory measuring. Those ranks stop at
    different points of the step, which leaves the others waiting in collectives that never complete, until NCCL's
    watchdog times them out. So a rank that runs out of memory says so in the rendezvous store, a thread on every rank
    that watches the store aborts the rank's communicators, which ends the collectives waiting, and once all the ranks
    are done with the token count they start new communicators for the next one"""
    # the run's guard, which the P2P batches go through
    active = None

    def __init__(self, local_rank):
        self.local_rank, self.rank, self.world = local_rank, dist.get_rank(), dist.get_world_size()
        # the groups aborted on the token count
        self.count, self.aborted, self.lock = 0, set(), threading.Lock()
        # held by a P2P batch, see batch_isend_irecv
        self.batch_lock = threading.Lock()
        self.store = self.connect()
        # a communicator of this rank alone, which isn't aborted, to keep NCCL's net plugin running: the aws-ofi-nccl
        # one shuts down with the last communicator using it, and fails to start again in the same process, which
        # leaves NCCL to fall back to its own IB or socket transport
        self.keeper = dist.ProcessGroupNCCL(dist.PrefixStore(f"keeper{self.rank}/", self.store), 0, 1)
        self.keeper.eager_connect_single_device(torch.device("cuda", local_rank))
        self.stopping = threading.Event()
        self.watcher = threading.Thread(target=self.watch, args=(self.connect(),), daemon=True)
        self.watcher.start()
        atexit.register(self.close)
        OOMGuard.active = self

    def close(self):
        """Stops the watcher before the interpreter exits, which would abort the process if it caught the watcher
        inside a store call: Python 3.12 and older end a daemon thread that takes the GIL back meanwhile with a
        pthread_exit, which C++ turns into a std::terminate. And then shuts the keeper down"""
        self.stopping.set()
        self.watcher.join()
        self.keeper.shutdown()

    @staticmethod
    def connect():
        """A client of the store that init_process_group's env:// rendezvous uses - one per thread"""
        store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]), is_master=False)
        return dist.PrefixStore("comm-compute-overlap-bench/", store)

    def watch(self, store):
        try:
            while not self.stopping.wait(1):
                with self.lock:
                    # again every second, for the groups created since
                    if store.check([f"oom{self.count}"]):
                        self.abort()
        except BaseException:
            # without the watcher, a rank running out of memory would hang the others again
            sys.excepthook(*sys.exc_info())
            os._exit(1)

    def abort(self):
        """Ends this rank's collectives, the waiting ones included, by aborting the groups not aborted yet. Called
        with the lock held. The groups are aborted at once, each in a thread of its own, as aborting one can wait for
        the device, which a collective waiting in another one keeps busy. torch's _abort_process_group, which aborts
        them one by one, isn't used here, as the collectives that the main thread queues meanwhile would end up in
        its coalescing manager"""
        if not hasattr(dist.distributed_c10d, "_abort_process_group"):
            print(f"rank {self.rank}: a rank ran out of memory, and torch=={torch.__version__} can't abort the "
                  f"collectives waiting for it", flush=True)
            os._exit(1)
        groups = [pg for pg in list(dist.distributed_c10d._world.pg_map) if pg not in self.aborted]
        # a P2P batch that holds its lock for longer than that waits for a peer to connect, in the end of its NCCL
        # group - where the thread has already left the group, and which only aborting ends
        batch = self.batch_lock.acquire(timeout=1)
        try:
            threads = [threading.Thread(target=pg.abort) for pg in groups]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.aborted.update(groups)
        finally:
            if batch:
                self.batch_lock.release()

    @classmethod
    def batch_isend_irecv(cls, ops):
        """dist.batch_isend_irecv, which the guard doesn't abort the groups in the middle of queueing: a P2P op raising
        then would leave the thread inside an NCCL group, as torch's _coalescing_manager doesn't end it, and NCCL
        defers all that the thread queues after that - the new communicators' start included - for good"""
        guard = cls.active
        if guard is None:
            return dist.batch_isend_irecv(ops)
        with guard.batch_lock:
            if guard.aborted:
                raise RuntimeError(f"rank {guard.rank}: the guard aborted the groups, a rank having run out of memory")
            return dist.batch_isend_irecv(ops)

    def ran_out(self):
        """This rank ran out of memory"""
        self.store.add(f"oom{self.count}", 1)
        with self.lock:
            self.abort()

    def skipping(self):
        """Whether a rank ran out of memory on the token count - which it says before any rank aborts a collective,
        and so before a collective raises for it"""
        return self.store.check([f"oom{self.count}"])

    def finish(self):
        """Waits for all the ranks to be done with the token count and returns how many of them ran out of memory on
        it, after starting new communicators if any did"""
        done, oom = f"done{self.count}", f"oom{self.count}"
        self.store.add(done, 1)
        while int(self.store.get(done)) < self.world:
            time.sleep(0.01)
        ooms = int(self.store.get(oom)) if self.store.check([oom]) else 0
        with self.lock:
            if ooms:
                # no collective waits anymore, so this aborts the rest one by one, and drops all the groups
                dist.distributed_c10d._abort_process_group()
                PARALLEL_GROUPS.clear()
                dist.init_process_group("nccl", store=dist.PrefixStore(f"pg{self.count}/", self.store),
                                        rank=self.rank, world_size=self.world, **device_id_kwargs(self.local_rank),
                                        **nccl_pg_options())
                # a rank whose thread was left inside an NCCL group sees its own 1 back, as NCCL defers this all_reduce
                # for good, along with the communicator's start - rather than measuring the next token count without
                # communicating
                ones = torch.ones(1, device=torch.device("cuda", self.local_rank))
                dist.all_reduce(ones)
                if ones.item() != self.world:
                    raise RuntimeError(f"rank {self.rank}: the communicators started after the skip don't communicate")
            self.count, self.aborted = self.count + 1, set()
        return ooms

class SymMem:
    """With --sym-mem, the communication buffers are allocated with ncclMemAlloc and registered as NCCL symmetric
    memory windows on the communicators of `groups`. Every rank allocates the same buffers in the same order, so each
    sits at the same window offset on every rank, as NCCL's symmetric kernels require"""
    def __init__(self, groups):
        self.pool = None
        if not args.sym_mem:
            return
        from packaging import version
        # on an 8x B200 the cu130 sym-mem path hangs in broadcast on all versions until torch 2.12.0/NCCL 2.29.7
        if version.parse(torch.__version__).release < (2, 12):
            sys.exit(f"--sym-mem needs torch>=2.12, this is torch=={torch.__version__}")
        if torch.cuda.nccl.version() < (2, 29):
            sys.exit(f"--sym-mem needs NCCL>=2.29, this torch uses nccl={torch.cuda.nccl.version()}")
        self.backends = list({id(b): b for b in (group._get_backend(device()) for group in groups)}.values())
        self.pool = torch.cuda.MemPool(self.backends[0].mem_allocator)

    def alloc(self):
        """The context to allocate the communication buffers in"""
        return torch.cuda.use_mem_pool(self.pool) if self.pool is not None else contextlib.nullcontext()

    def register(self):
        if self.pool is not None:
            for backend in self.backends:
                backend.register_mem_pool(self.pool, symm=True)

    def deregister(self):
        if self.pool is not None:
            for backend in self.backends:
                backend.deregister_mem_pool(self.pool)

def trial_desc():
    desc = f"{args.num_warmup_iterations} warmups / {args.num_iterations} trials of each"
    if args.sym_mem:
        desc += ", symmetric memory"
    if args.cta_policy != "default":
        desc += f", NCCL CTA policy {args.cta_policy}"
    return desc


### The training-step patterns ###

def load_model_shape(model, moe=False):
    """The layer shape of a preset, a local config.json, or a Hugging Face Hub model's config.json - with `moe` that
    of an MoE layer, whose experts each have an intermediate size of fe"""
    if model in MODEL_PRESETS:
        config = MODEL_PRESETS[model]
    else:
        path = model
        if not os.path.isfile(path):
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                sys.exit(f"--model {model}: not a preset ({', '.join(MODEL_PRESETS)}) nor a config.json, and "
                         f"huggingface_hub isn't installed to download it from the Hub")
            path = hf_hub_download(model, "config.json")
        config = json.load(open(path))
        config = config.get("text_config", config)
    experts = next((config[key] for key in ("num_experts", "num_local_experts", "n_routed_experts")
                    if config.get(key)), None)
    if experts and not moe:
        sys.exit(f"--model {model} has MoE layers of {experts} experts, which only the ep pattern models")
    if moe and not experts:
        sys.exit(f"--model {model} has no MoE layers, which the ep pattern needs - e.g. qwen3-30b-a3b")
    if moe and (config.get("n_shared_experts") or config.get("shared_expert_intermediate_size")):
        sys.exit(f"--model {model} has shared experts, which aren't modeled yet")
    h, heads = config["hidden_size"], config["num_attention_heads"]
    shape = dict(h=h, f=config["intermediate_size"], heads=heads, kv=config.get("num_key_value_heads") or heads,
                 hd=config.get("head_dim") or h // heads)
    if moe:
        shape.update(experts=experts, top_k=config["num_experts_per_tok"],
                     fe=config.get("moe_intermediate_size") or config["intermediate_size"])
    return shape

# a layer's weights in the order of its parameters, as Hugging Face's Llama and Qwen layers have them
WEIGHTS = ("q", "k", "v", "o", "gate", "up", "down")
# the stages of a layer's compute and the weights each uses: forward runs them in this order, backward in reverse, in
# which order autograd also produces a stage's weight gradients
STAGES = dict(qkv=("q", "k", "v"), o=("o",), gu=("gate", "up"), d=("down",))

def weight_numels(shape):
    h, f, heads, kv, hd = (shape[k] for k in ("h", "f", "heads", "kv", "hd"))
    return dict(q=heads * hd * h, k=kv * hd * h, v=kv * hd * h, o=h * heads * hd, gate=f * h, up=f * h, down=h * f)

def layer_numel(shape):
    return sum(weight_numels(shape).values())

def layer_flops(shape, tokens, seqlen):
    """forward and backward of one layer: 2 FLOPs per weight per token for forward and twice that for backward, plus
    causal attention's 2 matmuls over half of each sequence's seqlen x seqlen scores"""
    attention = 2 * (tokens // seqlen) * seqlen**2 * shape["heads"] * shape["hd"]
    return 3 * (2 * tokens * layer_numel(shape) + attention)

def describe_calls(calls):
    """What the (phase, op, bytes) `calls` of the last trial with collectives communicated, for the report"""
    by_op = {}
    for phase, op, size in calls:
        by_op.setdefault((phase, op), []).append(size)
    desc = []
    for (phase, op), sizes in by_op.items():
        each = (fmt_bytes(sizes[0]) if min(sizes) == max(sizes) else
                f"{fmt_bytes(min(sizes))} to {fmt_bytes(max(sizes))}")
        desc.append(f"{phase}: {len(sizes)} {op} calls of {each}, {fmt_bytes(sum(sizes))} in all")
    return "; ".join(desc)

def no_hook(when, stage):
    pass

def rms_norm(x):
    """A weightless RMSNorm of x, and what rms_norm_backward needs. Llama's norms before attention and before the MLP
    keep the values finite, which without them grow with each layer until bf16 overflows into inf and NaN, whose
    matmuls draw less power than those of finite values and so run faster. These are the fused kernels that F.rms_norm
    and its autograd backward run, called directly: a call of torch.autograd.grad takes the host longer than a small
    layer's norm takes the GPU, which left the GPU waiting for the host"""
    out, rstd = torch.ops.aten._fused_rms_norm(x, [x.shape[-1]], None, None)
    return out, (x, rstd)

def rms_norm_backward(saved, dout):
    x, rstd = saved
    return torch.ops.aten._fused_rms_norm_backward(dout, x, [x.shape[-1]], rstd, None, [True, False])[0]

class Layer:
    """A dense transformer layer whose weights are views into a flat parameter buffer, and whose weight gradients
    are written into views of a flat gradient buffer, as FSDP and DeepSpeed lay them out, the weights in WEIGHTS
    order. Its norms are weightless. The backward is written out with the weight gradients' matmuls, apart from
    attention's, which autograd does. `hook(when, stage)` is called "before" and "after" each of the
    STAGES, where a framework waits for the stage's weights to be gathered, and after which it frees them, or takes
    their gradients"""
    def __init__(self, shape, params, grads, seqlen):
        h, f, heads, kv, hd = (shape[k] for k in ("h", "f", "heads", "kv", "hd"))
        self.shape, self.seqlen = shape, seqlen
        shapes = [((heads + 2 * kv) * hd, h), (h, heads * hd), (2 * f, h), (h, f)]
        def views(flat):
            out, offset = [], 0
            for rows, cols in shapes:
                out.append(flat[offset:offset + rows * cols].view(rows, cols))
                offset += rows * cols
            return out
        self.w, self.g = views(params), views(grads)
        self.saved = None

    def forward(self, x, hook=no_hook):
        heads, kv, hd = self.shape["heads"], self.shape["kv"], self.shape["hd"]
        Wqkv, Wo, Wgu, Wd = self.w
        T, S = x.shape[0], self.seqlen
        hook("before", "qkv")
        xn, xg = rms_norm(x)
        q, k, v = F.linear(xn, Wqkv).split([heads * hd, kv * hd, kv * hd], dim=-1)
        q, k, v = [t.view(T // S, S, -1, hd).transpose(1, 2).detach().requires_grad_() for t in (q, k, v)]
        with torch.enable_grad():
            # the KV heads repeated for each group of q heads, as Hugging Face's repeat_kv does, so that SDPA takes
            # its flash attention path
            attn = F.scaled_dot_product_attention(q, k.repeat_interleave(heads // kv, dim=1),
                                                  v.repeat_interleave(heads // kv, dim=1), is_causal=True)
        a = attn.detach().transpose(1, 2).reshape(T, heads * hd)
        hook("after", "qkv")
        hook("before", "o")
        h1 = x + F.linear(a, Wo)
        hook("after", "o")
        hook("before", "gu")
        hn, hg = rms_norm(h1)
        gu = F.linear(hn, Wgu)
        g, u = gu.chunk(2, dim=-1)
        act = F.silu(g) * u
        hook("after", "gu")
        hook("before", "d")
        out = h1 + F.linear(act, Wd)
        hook("after", "d")
        self.saved = (xn, xg, q, k, v, attn, a, hn, hg, gu, act)
        return out

    def backward(self, dout, hook=no_hook):
        heads, kv, hd = self.shape["heads"], self.shape["kv"], self.shape["hd"]
        Wqkv, Wo, Wgu, Wd = self.w
        gWqkv, gWo, gWgu, gWd = self.g
        xn, xg, q, k, v, attn, a, hn, hg, gu, act = self.saved
        self.saved = None
        T, S = xn.shape[0], self.seqlen
        hook("before", "d")
        torch.matmul(dout.t(), act, out=gWd)
        dact = dout @ Wd
        hook("after", "d")
        hook("before", "gu")
        g, u = gu.chunk(2, dim=-1)
        sig = torch.sigmoid(g)
        dgu = torch.cat([dact * u * sig * (1 + g * (1 - sig)), dact * g * sig], dim=-1)
        torch.matmul(dgu.t(), hn, out=gWgu)
        dh1 = dout + rms_norm_backward(hg, dgu @ Wgu)
        hook("after", "gu")
        hook("before", "o")
        torch.matmul(dh1.t(), a, out=gWo)
        dattn = (dh1 @ Wo).view(T // S, S, heads, hd).transpose(1, 2)
        hook("after", "o")
        hook("before", "qkv")
        dq, dk, dv = torch.autograd.grad(attn, (q, k, v), dattn)
        dqkv = torch.cat([t.transpose(1, 2).reshape(T, -1) for t in (dq, dk, dv)], dim=-1)
        torch.matmul(dqkv.t(), xn, out=gWqkv)
        dx = dh1 + rms_norm_backward(xg, dqkv @ Wqkv)
        hook("after", "qkv")
        return dx

class Step:
    """The buffers, layers and streams of one training step. All the layers' parameters and gradients are each one
    flat buffer, layer i's at [i*n, (i+1)*n) and its weights in WEIGHTS order within it, so that what a framework
    gathers or reduces in one call is one contiguous piece of it. Each subclass queues the step's collectives the way
    its framework does, from the hooks the queue calls around each layer and each stage of a layer"""
    # whether forward has collectives, and whether there's a post-step all-gather, for the report
    forward_comm = False
    post_step_comm = False

    def __init__(self, settings, shape, tokens, seqlen, layers, rs_group=None, priority=0):
        self.settings, self.L = settings, layers
        self.gas = settings.get("gas", 1)
        if self.gas < 1:
            sys.exit("--set gas: use 1 or more")
        self.last = True
        self.ranks = ranks = dist.get_world_size()
        # each rank's share of each weight a whole number of 128-byte bf16 pieces
        align = ranks * 64
        self.numels = weight_numels(shape)
        for name, numel in self.numels.items():
            if numel % align:
                sys.exit(f"--model's {name} weight of {numel} elements doesn't split into {ranks} shares of a "
                         f"multiple of 64 elements")
        self.n = n = layer_numel(shape)
        self.total = total = layers * n
        self.offsets, offset = {}, 0
        for name in WEIGHTS:
            self.offsets[name] = offset
            offset += self.numels[name]

        self.ag_group = dist.group.WORLD
        self.rs_group = rs_group or dist.group.WORLD
        self.sym_mem = SymMem([self.ag_group, self.rs_group])
        bf16 = torch.bfloat16
        with self.sym_mem.alloc():
            self.params = torch.empty(total, dtype=bf16, device=device())
            self.param_shards = torch.empty(total // ranks, dtype=bf16, device=device())
            self.grads = torch.empty(total, dtype=bf16, device=device())
            self.alloc_buffers()
        self.sym_mem.register()
        # random values, as a matmul of zeros or of repeating values draws less power, and so keeps a higher clock
        self.params.normal_(std=0.02)
        self.param_shards.normal_(std=0.02)
        self.layers = [Layer(shape, self.params[i * n:(i + 1) * n], self.grads[i * n:(i + 1) * n], seqlen)
                       for i in range(layers)]
        self.x = torch.randn(tokens, shape["h"], dtype=bf16, device=device())
        self.dout = torch.randn(tokens, shape["h"], dtype=bf16, device=device()) * 0.01
        self.ag_stream = torch.cuda.Stream(priority=priority)
        self.rs_stream = torch.cuda.Stream(priority=priority)
        self.calls = []

    def alloc_buffers(self):
        """Allocates the subclass's communication buffers, in the symmetric memory pool with --sym-mem"""

    def weight(self, i, name):
        """(lo, hi) of layer i's weight `name` in the flat buffers"""
        lo = i * self.n + self.offsets[name]
        return lo, lo + self.numels[name]

    def collective(self, op, out, inp=None, stream=None, group=None):
        """Queues `op` on `stream`, the current stream if None, and records it for the report"""
        with torch.cuda.stream(stream or torch.cuda.current_stream()):
            if op == "all_gather":
                all_gather_single(out, inp, group=group or self.ag_group)
            elif op == "reduce_scatter":
                reduce_scatter_single(out, inp, group=group or self.rs_group)
            else:
                dist.all_reduce(out, group=group or self.rs_group)
        size = out if op == "all_gather" else inp if op == "reduce_scatter" else out
        self.calls.append((self.phase, op, size.numel() * size.element_size()))

    def collectives(self):
        return describe_calls(self.calls)

    def gradients(self):
        return [self.grads]

    def queue(self, comm, compute, events, marks):
        """Queues one step - its collectives if `comm`, its compute if `compute` - after events[0], recording the end
        of each micro-batch's forward and backward and of the step on the compute stream into events[1:]. After each
        layer's forward and backward it appends to `marks` the host's time and an event the compute stream records
        then"""
        self.comm, self.cs = comm, torch.cuda.current_stream()

        def mark():
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            marks.append((time.perf_counter(), event))

        def run(layer, i, pass_, x):
            hook = lambda when, stage: self.stage_hook(i, pass_, when, stage)
            if compute:
                return getattr(layer, pass_)(x, hook)
            for stage in (STAGES if pass_ == "forward" else reversed(STAGES)):
                hook("before", stage)
                hook("after", stage)
            return x

        if comm:
            self.calls = []
            self.ag_stream.wait_event(events[0])
            self.rs_stream.wait_event(events[0])
        self.begin()
        for m in range(self.gas):
            # whether this micro-batch's backward is the one that ends the gradient accumulation
            self.last = m == self.gas - 1
            self.begin_micro()
            self.phase = "forward"
            x = self.x
            for i in range(self.L):
                self.pre_forward(i)
                x = run(self.layers[i], i, "forward", x)
                self.post_forward(i)
                mark()
            events[2 * m + 1].record()
            self.phase = "backward"
            dx = self.dout
            for i in reversed(range(self.L)):
                self.pre_backward(i)
                dx = run(self.layers[i], i, "backward", dx)
                self.post_backward(i)
                mark()
            self.out, self.dx = x, dx
            if comm:
                self.end_backward()
                # each framework waits for its reductions at the end of each backward, and the optimizer step needs
                # all the gradients reduced
                self.cs.wait_stream(self.rs_stream)
                self.cs.wait_stream(self.ag_stream)
            events[2 * m + 2].record()
        self.phase = "post-step"
        if comm:
            self.post_step()
        events[-1].record()
        marks.append((time.perf_counter(), events[-1]))

    # the hooks, which only run in a trial with collectives, apart from begin and begin_micro
    def begin(self):
        """Resets the state of a step"""

    def begin_micro(self):
        """Resets the state of a micro-batch, self.last telling whether it's the step's last"""

    def pre_forward(self, i):
        pass

    def post_forward(self, i):
        pass

    def pre_backward(self, i):
        pass

    def post_backward(self, i):
        pass

    def stage_hook(self, i, pass_, when, stage):
        pass

    def end_backward(self):
        pass

    def post_step(self):
        pass

class DDPStep(Step):
    """PyTorch DDP, from its Reducer: each gradient is copied into its bucket as it's produced, a full bucket is
    all-reduced on the process group's stream, and at the end of backward the results are copied back. Under
    no_sync() a micro-batch's backward does none of that"""
    def __init__(self, settings, shape, tokens, seqlen, layers):
        self.as_view = settings["gradient_as_bucket_view"]
        self.no_sync = settings["no_sync"]
        super().__init__(settings, shape, tokens, seqlen, layers)
        # the buckets, in the reverse order of the parameters, each closed once it reaches its cap
        self.buckets, bucket, size, cap = [], [], 0, settings["first_bucket_mb"]
        for i in reversed(range(layers)):
            for name in reversed(WEIGHTS):
                bucket.append((i, name))
                size += self.numels[name] * 2
                if size >= cap:
                    self.buckets.append(bucket)
                    bucket, size, cap = [], 0, settings["bucket_cap_mb"]
        if bucket:
            self.buckets.append(bucket)
        self.bucket_of = {key: b for b, keys in enumerate(self.buckets) for key in keys}

    def alloc_buffers(self):
        self.bucket_buf = self.grads if self.as_view else torch.empty_like(self.grads)

    def begin_micro(self):
        self.sync = self.last or not self.no_sync
        self.pending = [len(bucket) for bucket in self.buckets]

    def stage_hook(self, i, pass_, when, stage):
        if not self.comm or not self.sync or pass_ != "backward" or when != "after":
            return
        for name in reversed(STAGES[stage]):
            lo, hi = self.weight(i, name)
            if not self.as_view:
                self.bucket_buf[lo:hi].copy_(self.grads[lo:hi])
            b = self.bucket_of[(i, name)]
            self.pending[b] -= 1
            if self.pending[b] == 0:
                lo = min(self.weight(*key)[0] for key in self.buckets[b])
                hi = max(self.weight(*key)[1] for key in self.buckets[b])
                self.rs_stream.wait_stream(self.cs)
                self.collective("all_reduce", self.bucket_buf[lo:hi], stream=self.rs_stream)

    def end_backward(self):
        if self.sync and not self.as_view:
            self.cs.wait_stream(self.rs_stream)
            self.grads.copy_(self.bucket_buf)

class DSZeroStep(Step):
    """DeepSpeed ZeRO-1 and ZeRO-2, from stage_1_and_2.py: each gradient is copied into the current bucket (the
    "IPG" buffer) on the compute stream, a bucket that would overflow is all-reduced first, and after the optimizer
    step the parameters are all-gathered in one call. ZeRO-1 reduces only in the last micro-batch's backward, ZeRO-2
    in each one's, as stage_1_and_2.py's reduce_ready_partitions_and_remove_grads does. The micro-batches' gradients
    aren't added up, which is an elementwise add per micro-batch"""
    post_step_comm = True

    def __init__(self, stage, settings, shape, tokens, seqlen, layers):
        self.stage = stage
        self.bucket = settings["reduce_bucket_size"]
        self.overlap = settings["overlap_comm"]
        self.comm_dtype = settings["communication_data_type"]
        self.largest = max(weight_numels(shape).values())
        super().__init__(settings, shape, tokens, seqlen, layers)

    def alloc_buffers(self):
        # with overlap_comm 2 buffers take turns, so that one fills while the other is reduced
        self.ipg = [torch.empty(max(self.bucket, self.largest), dtype=self.comm_dtype, device=device())
                    for _ in range(2 if self.overlap else 1)]

    def begin(self):
        self.elements, self.index = 0, 0

    def add(self, i, name):
        lo, hi = self.weight(i, name)
        if self.elements + hi - lo > self.bucket and self.elements > 0:
            self.reduce()
        self.ipg[self.index][self.elements:self.elements + hi - lo].copy_(self.grads[lo:hi])
        self.elements += hi - lo

    def reduce(self):
        bucket = self.ipg[self.index][:self.elements]
        if self.overlap:
            # the compute waits for the bucket before this one, and the reduction for the compute
            self.rs_stream.wait_stream(self.cs)
            self.cs.wait_stream(self.rs_stream)
            self.collective("all_reduce", bucket, stream=self.rs_stream)
            self.index = 1 - self.index
        else:
            self.collective("all_reduce", bucket)
        self.elements = 0

    def stage_hook(self, i, pass_, when, stage):
        if self.comm and pass_ == "backward" and when == "after" and (self.stage == 2 or self.overlap and self.last):
            for name in reversed(STAGES[stage]):
                self.add(i, name)

    def end_backward(self):
        if self.stage == 1 and not self.overlap and self.last:
            for i in range(self.L):
                for name in WEIGHTS:
                    self.add(i, name)
        if self.elements:
            self.reduce()

    def post_step(self):
        self.collective("all_gather", self.params, self.param_shards)

class DSZero3Step(Step):
    """DeepSpeed ZeRO-3, from partitioned_param_coordinator.py and stage3.py, each linear layer being a module with
    one weight. A module fetches its weight and then prefetches the next not yet gathered weights in the order of
    the step's modules, up to stage3_prefetch_bucket_size elements, in one call; the compute waits for all the
    all-gathers queued before the fetch, and the host for the oldest of more than 2 fetches in flight. After its use
    a weight is freed unless it's used again within
    stage3_max_reuse_distance elements. Each gradient is copied into the bucket on the reduce stream, and a bucket
    that would overflow is reduce-scattered first"""
    forward_comm = True

    def __init__(self, settings, shape, tokens, seqlen, layers):
        self.prefetch_size = settings["stage3_prefetch_bucket_size"]
        self.max_live = settings["stage3_max_live_parameters"]
        self.bucket = settings["reduce_bucket_size"]
        self.comm_dtype = settings["communication_data_type"]
        self.largest = max(weight_numels(shape).values())
        super().__init__(settings, shape, tokens, seqlen, layers)
        # the modules in the order the step uses them
        self.trace = ([(i, name) for i in range(layers) for name in WEIGHTS] +
                      [(i, name) for i in reversed(range(layers)) for stage in reversed(STAGES)
                       for name in reversed(STAGES[stage])])
        numel = lambda key: self.numels[key[1]]
        self.keep, self.next_use = [], []
        for s, key in enumerate(self.trace):
            traversed, keep, later = 0, False, None
            for t in range(s + 1, len(self.trace)):
                if self.trace[t] == key:
                    later = t if later is None else later
                    keep = keep or traversed < settings["stage3_max_reuse_distance"]
                    break
                traversed += numel(self.trace[t])
            self.keep.append(keep)
            self.next_use.append(later)

    def alloc_buffers(self):
        size = max(self.bucket, self.largest)
        self.ipg = torch.empty(size, dtype=self.comm_dtype, device=device())
        self.rs_out = torch.empty(size // self.ranks + 1, dtype=self.comm_dtype, device=device())

    def begin_micro(self):
        self.status = {key: "released" for key in self.trace}
        self.ongoing = collections.deque()
        self.available, self.head, self.step_id, self.elements = 0, 0, 0, 0

    def gather(self, keys):
        """All-gathers the weights of `keys`, one call per contiguous run of them"""
        runs = []
        for lo, hi, key in sorted(self.weight(*key) + (key,) for key in keys):
            if runs and runs[-1][1] == lo:
                runs[-1][1] = hi
                runs[-1][2].append(key)
            else:
                runs.append([lo, hi, [key]])
        for lo, hi, run in runs:
            self.collective("all_gather", self.params[lo:hi],
                            self.param_shards[lo // self.ranks:hi // self.ranks], stream=self.ag_stream)
            for key in run:
                self.status[key] = "in flight"
                self.available += self.numels[key[1]]

    def fetch(self, s):
        key = self.trace[s]
        if self.status[key] == "released":
            self.gather([key])
        if self.status[key] == "in flight":
            while self.ongoing and self.ongoing[0].query():
                self.ongoing.popleft()
            if len(self.ongoing) > 2:
                self.ongoing.popleft().synchronize()
            event = torch.cuda.Event()
            event.record(self.ag_stream)
            self.ongoing.append(event)
            self.status[key] = "gathered"
        # the compute waits for every all-gather queued so far, the prefetched ones too
        self.cs.wait_stream(self.ag_stream)
        self.head = max(self.head, s + 1)
        limit = min(self.max_live - self.available, self.prefetch_size)
        prefetch, numel = [], 0
        while self.head < len(self.trace) and numel < limit:
            later = self.trace[self.head]
            self.head += 1
            if self.status[later] == "released" and later not in prefetch:
                prefetch.append(later)
                numel += self.numels[later[1]]
        if prefetch:
            self.gather(prefetch)

    def release(self, s):
        key = self.trace[s]
        # kept if used again soon, or if a prefetch already went past its next use while it was gathered
        if self.keep[s] or (self.next_use[s] is not None and self.next_use[s] < self.head):
            return
        self.status[key] = "released"
        self.available -= self.numels[key[1]]

    def add(self, i, name):
        lo, hi = self.weight(i, name)
        if self.elements + hi - lo > self.bucket and self.elements > 0:
            self.reduce()
        self.rs_stream.wait_stream(self.cs)
        with torch.cuda.stream(self.rs_stream):
            self.ipg[self.elements:self.elements + hi - lo].copy_(self.grads[lo:hi])
        self.elements += hi - lo

    def reduce(self):
        self.collective("reduce_scatter", self.rs_out[:self.elements // self.ranks], self.ipg[:self.elements],
                        stream=self.rs_stream)
        self.elements = 0

    def stage_hook(self, i, pass_, when, stage):
        if not self.comm:
            return
        names = STAGES[stage] if pass_ == "forward" else tuple(reversed(STAGES[stage]))
        if when == "before":
            self.used = []
            for _ in names:
                self.fetch(self.step_id)
                self.used.append(self.step_id)
                self.step_id += 1
            return
        for s in self.used:
            self.release(s)
        if pass_ == "backward":
            for name in names:
                self.add(i, name)

    def end_backward(self):
        if self.elements:
            self.reduce()

class FSDPStep(Step):
    """PyTorch FSDP2, from _fsdp_param_group.py and _fsdp_collectives.py, with fully_shard on each layer. A layer's
    all-gather lands in a buffer, which the compute stream copies out into the parameters before the layer's
    compute. In forward each all-gather waits for the copy-out of the layer forward_prefetch+1 before it, and in
    backward for the copy-out of the layer before it. After a layer's backward the compute stream waits for the
    previous layer's reduce-scatter, copies the gradients into a buffer, and the reduce-scatter stream reduces it.
    With requires_gradient_sync=0 only the last micro-batch's backward reduce-scatters, and with
    reshard_after_backward=0 the layers backward gathered stay gathered into the next micro-batch"""
    forward_comm = True

    def __init__(self, settings, shape, tokens, seqlen, layers):
        self.reshard = settings["reshard_after_forward"]
        self.sync = settings["requires_gradient_sync"]
        self.reshard_after_backward = settings["reshard_after_backward"]
        self.forward_prefetch = max(1, settings["forward_prefetch"])
        self.backward_prefetch = settings["backward_prefetch"]
        self.reduce_dtype = settings["reduce_dtype"]
        self.slots = max(self.forward_prefetch, self.backward_prefetch) + 2
        rs_group = (dist.new_group(list(range(dist.get_world_size())), **nccl_pg_options())
                    if settings["separate_rs_group"] else None)
        super().__init__(settings, shape, tokens, seqlen, layers, rs_group=rs_group,
                         priority=-1 if settings["high_priority"] else 0)

    def alloc_buffers(self):
        n, ranks = self.n, self.ranks
        self.ag_bufs = [torch.empty(n, dtype=torch.bfloat16, device=device()) for _ in range(self.slots)]
        self.rs_bufs = [torch.empty(n, dtype=self.reduce_dtype, device=device()) for _ in range(2)]
        self.rs_out = torch.empty(n // ranks, dtype=self.reduce_dtype, device=device())

    def begin(self):
        self.pending, self.unsharded, self.reduce_events = {}, set(), []
        self.slot, self.copied_out = 0, None

    def layer(self, i):
        return slice(i * self.n, (i + 1) * self.n)

    def unshard(self, i):
        """Queues layer i's all-gather unless it's queued or gathered already"""
        if not 0 <= i < self.L or i in self.pending or i in self.unsharded:
            return
        buf = self.ag_bufs[self.slot]
        self.slot = (self.slot + 1) % self.slots
        n, ranks = self.n, self.ranks
        self.collective("all_gather", buf, self.param_shards[i * n // ranks:(i + 1) * n // ranks],
                        stream=self.ag_stream)
        event = torch.cuda.Event()
        event.record(self.ag_stream)
        self.pending[i] = (buf, event)

    def wait_for_unshard(self, i):
        if i not in self.pending:
            return
        if self.phase == "forward" and self.copied_out is not None:
            # the next all-gather waits for the previous layer's copy-out
            self.ag_stream.wait_event(self.copied_out)
        buf, event = self.pending.pop(i)
        self.cs.wait_event(event)
        self.params[self.layer(i)].copy_(buf)
        self.copied_out = torch.cuda.Event()
        self.copied_out.record()
        if self.phase == "backward":
            self.ag_stream.wait_event(self.copied_out)
        self.unsharded.add(i)

    def pre_forward(self, i):
        if self.comm:
            self.unshard(i)
            self.wait_for_unshard(i)
            for j in range(i + 1, i + 1 + self.forward_prefetch):
                self.unshard(j)

    def post_forward(self, i):
        if self.comm and self.reshard:
            self.unsharded.discard(i)

    def pre_backward(self, i):
        if self.comm:
            self.unshard(i)
            self.wait_for_unshard(i)
            for j in range(i - 1, i - 1 - self.backward_prefetch, -1):
                self.unshard(j)

    def post_backward(self, i):
        if not self.comm:
            return
        if self.sync or self.last:
            # reduce_scatter_max_input_buffers=1: the previous layer's reduce-scatter has to be done
            while self.reduce_events:
                self.cs.wait_event(self.reduce_events.pop(0))
            buf = self.rs_bufs[i % 2]
            buf.copy_(self.grads[self.layer(i)])
            self.rs_stream.wait_stream(self.cs)
            self.collective("reduce_scatter", self.rs_out, buf, stream=self.rs_stream)
            event = torch.cuda.Event()
            event.record(self.rs_stream)
            self.reduce_events.append(event)
        if self.reshard_after_backward or self.last:
            self.unsharded.discard(i)

### The model-parallel patterns ###

class Handle:
    """The works of an async collective, whose wait() makes the current stream wait for them"""
    def __init__(self, works):
        self.works = works if isinstance(works, list) else [works] if works is not None else []

    def wait(self):
        for work in self.works:
            work.wait()

# the model-parallel groups, by degree, each created once
PARALLEL_GROUPS = {}

def parallel_group(name, degree, strided=False):
    """This rank's group of `degree` consecutive ranks, the ranks being split into groups of `degree`, or with
    `strided` of `degree` ranks world / degree apart"""
    world = dist.get_world_size()
    if world % degree:
        sys.exit(f"--set {name}={degree}: the {world} ranks don't split into groups of {degree}")
    key = (degree, strided)
    if key not in PARALLEL_GROUPS:
        n = world // degree
        PARALLEL_GROUPS[key] = (dist.group.WORLD if degree == world else
                                dist.new_subgroups_by_enumeration([list(range(i, world, n)) for i in range(n)],
                                                                  **nccl_pg_options())[0] if strided else
                                dist.new_subgroups(group_size=degree, **nccl_pg_options())[0])
    return PARALLEL_GROUPS[key]

class Comm:
    """The collectives of a model-parallel pattern within `group`, queued as PyTorch queues them on the process
    group's stream: a blocking one makes the current stream wait for it, an async one returns a Handle as well. With
    `real` False they stand in for the collectives without communicating, returning the output of the same call in
    the last step with real collectives - the n-th call of its name and shape since begin_step() - so that the
    compute alone computes on the same matrices as the overlapped step, which keeps every output of a step in memory.
    While `recording` is a list each collective is recorded into it, so that replay() can queue the step's collectives
    alone"""
    def __init__(self, group):
        self.group = group
        self.size, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.global_ranks = dist.get_process_group_ranks(group)
        self.real, self.recording, self.calls, self.phase = True, None, [], "forward"
        self.buffers, self.stand_ins, self.call_index = {}, {}, {}
        # the memory pool of the CUDA graphs that the step's layers capture, if any
        self.graph_pool = None

    def begin_step(self):
        """Starts counting the calls of each name and shape again, so that each call gets its own stand-in"""
        self.call_index = {}

    def peer(self, offset):
        """The global rank `offset` ranks after this one in the group, around the ring"""
        return self.global_ranks[(self.rank + offset) % self.size]

    def buffer(self, name, shape, dtype=torch.bfloat16):
        key = (name, tuple(shape), dtype)
        if key not in self.buffers:
            # random values, as a matmul of zeros or of repeating values draws less power, and so keeps a higher clock
            self.buffers[key] = (torch.randn(shape, device=device()) * 0.02).to(dtype)
        return self.buffers[key]

    def empty(self, name, shape, dtype=torch.bfloat16):
        """The output of a collective: a new tensor, kept as the stand-in for this call, or with `real` False the
        stand-in"""
        key = (name, tuple(shape), dtype)
        index = self.call_index[key] = self.call_index.get(key, -1) + 1
        if not self.real:
            stand_in = self.stand_ins.get(key + (index,))
            return self.buffer(name, shape, dtype) if stand_in is None else stand_in
        out = torch.empty(shape, dtype=dtype, device=device())
        self.stand_ins[key + (index,)] = out
        return out

    def collective(self, op, out, inp, async_=False):
        work = None
        if self.real:
            if op == "all_reduce":
                work = dist.all_reduce(out, group=self.group, async_op=async_)
            elif op == "all_gather":
                work = all_gather_single(out, inp, group=self.group, async_op=async_)
            elif op == "reduce_scatter":
                work = reduce_scatter_single(out, inp, group=self.group, async_op=async_)
            else:
                work = dist.all_to_all_single(out, inp, group=self.group, async_op=async_)
            size = out if op == "all_gather" else inp
            self.calls.append((self.phase, op, size.numel() * size.element_size()))
            if self.recording is not None:
                self.recording.append((self.phase, op, tuple(out.shape), tuple(inp.shape), out.dtype))
        return (out, Handle(work)) if async_ else out

    def all_reduce(self, t, async_=False):
        return self.collective("all_reduce", t, t, async_)

    def all_gather(self, name, shard, async_=False):
        out = self.empty(name, (shard.shape[0] * self.size,) + tuple(shard.shape[1:]), shard.dtype)
        return self.collective("all_gather", out, shard, async_)

    def reduce_scatter(self, name, full, async_=False):
        out = self.empty(name, (full.shape[0] // self.size,) + tuple(full.shape[1:]), full.dtype)
        return self.collective("reduce_scatter", out, full, async_)

    def all_to_all(self, name, inp, async_=False):
        """Sends the i-th of `inp`'s equal pieces along its first dimension to the group's rank i"""
        return self.collective("all_to_all", self.empty(name, inp.shape, inp.dtype), inp, async_)

    def p2p(self, sends, recvs):
        """Queues the sends, (tensor, peer), and the receives, (buffer, peer), in one batch, returning a Handle - each
        peer a global rank"""
        works = None
        if self.real and (sends or recvs):
            works = OOMGuard.batch_isend_irecv([dist.P2POp(dist.isend, t, peer) for t, peer in sends] +
                                               [dist.P2POp(dist.irecv, t, peer) for t, peer in recvs])
            self.calls.append((self.phase, "send/recv", sum(t.numel() * t.element_size() for t, _ in sends)))
            if self.recording is not None:
                spec = lambda pairs: [(tuple(t.shape), t.dtype, peer) for t, peer in pairs]
                self.recording.append((self.phase, "p2p", spec(sends), spec(recvs)))
        return Handle(works)

    def replay(self, recorded, phase):
        """Queues the recorded collectives of `phase` alone, one after the other, on buffers of their shapes"""
        for p, op, *spec in recorded:
            if p != phase:
                continue
            if op == "library":
                Handle(spec[0]()[1]).wait()
                continue
            if op == "p2p":
                sends, recvs = spec
                ops = ([dist.P2POp(dist.isend, self.buffer("send", shape, dtype), peer)
                        for shape, dtype, peer in sends] +
                       [dist.P2POp(dist.irecv, self.buffer("recv", shape, dtype), peer)
                        for shape, dtype, peer in recvs])
                Handle(OOMGuard.batch_isend_irecv(ops)).wait()
                continue
            out_shape, in_shape, dtype = spec
            out = self.buffer("out", out_shape, dtype)
            inp = out if op == "all_reduce" else self.buffer("in", in_shape, dtype)
            real, recording, self.recording = self.real, self.recording, None
            self.real = True
            calls, self.calls = self.calls, []
            self.collective(op, out, inp)
            self.real, self.recording, self.calls = real, recording, calls

    def library_call(self, name, op, run, size):
        """A collective of a library other than torch.distributed: run() queues it and returns its output and what to
        wait() on before using it, or None if it blocks. With `real` False, returns the output of the same call in the
        last real step instead. `size` is the bytes it was given to send, for the report"""
        key = (name, op)
        index = self.call_index[key] = self.call_index.get(key, -1) + 1
        if not self.real:
            return self.stand_ins[key + (index,)], Handle(None)
        out, work = run()
        self.stand_ins[key + (index,)] = out
        self.calls.append((self.phase, op, size))
        if self.recording is not None:
            self.recording.append((self.phase, "library", run))
        return out, Handle(work)

def attention_forward(q, k, v, heads, kv, causal=True):
    """SDPA of (batch, heads, seq, hd) q over (batch, kv, seq, hd) k and v, each a new leaf, so that its backward is
    autograd's of their graph. The KV heads are repeated for each group of q heads, as Hugging Face's repeat_kv does,
    so that SDPA takes its flash attention path"""
    q, k, v = [t.detach().requires_grad_() for t in (q, k, v)]
    with torch.enable_grad():
        attn = F.scaled_dot_product_attention(q, k.repeat_interleave(heads // kv, dim=1),
                                              v.repeat_interleave(heads // kv, dim=1), is_causal=causal)
    return q, k, v, attn

def swiglu_backward(dact, gu):
    g, u = gu.chunk(2, dim=-1)
    sig = torch.sigmoid(g)
    return torch.cat([dact * u * sig * (1 + g * (1 - sig)), dact * g * sig], dim=-1)

class MPStep:
    """One training step of a model-parallel pattern, whose layers queue its collectives between their compute
    through `self.c`. The compute alone runs the layers with stand-ins for the collectives, and the comm alone
    replays the collectives that the first overlapped trial recorded. `parallelism` describes the groups, e.g.
    tp=8"""
    forward_comm, post_step_comm = True, False
    tokens_desc = "tokens per rank"

    def __init__(self, name, degree, parallelism=None, strided=False):
        self.c = Comm(parallel_group(name, degree, strided))
        self.parallelism = parallelism or f"{name}={degree}"
        self.sym_mem = SymMem([])
        self.recorded = None

    def collectives(self):
        return describe_calls(self.c.calls)

    def gradients(self):
        return [self.grads]

    def queue(self, comm, compute, events, marks):
        """As Step.queue"""
        c = self.c

        def mark(event=None):
            if event is None:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
            marks.append((time.perf_counter(), event))

        if comm and not compute:
            if self.recorded is None:
                raise RuntimeError("the collectives alone can only be replayed after an overlapped trial")
            for phase, event in (("forward", events[1]), ("backward", events[2])):
                c.replay(self.recorded, phase)
                event.record()
                mark(event)
            events[3].record()
            mark(events[3])
            return
        c.real = comm
        c.begin_step()
        if comm:
            c.calls = []
            c.recording = [] if self.recorded is None else None
        c.phase = "forward"
        x = self.x
        for layer in self.layers:
            x = layer.forward(x)
            mark()
        events[1].record()
        c.phase = "backward"
        dx = self.dout
        for layer in reversed(self.layers):
            dx = layer.backward(dx)
            mark()
        self.out, self.dx = x, dx
        events[2].record()
        events[3].record()
        mark(events[3])
        if c.recording is not None:
            self.recorded, c.recording = c.recording, None

class TPLayer(Layer):
    """A layer split over the ranks as Megatron-LM's ColumnParallelLinear and RowParallelLinear split it: q, k, v,
    gate and up by their output features, o and down by their input features"""
    def __init__(self, c, sp, shape, params, grads, seqlen):
        super().__init__(shape, params, grads, seqlen)
        self.c, self.sp = c, sp

    def reduce(self, name, t):
        """The partial sums of a row-parallel layer's output"""
        return self.c.reduce_scatter(name, t) if self.sp else self.c.all_reduce(t)

    def gather(self, name, t):
        """A column-parallel layer's input, which sequence parallelism splits over the ranks"""
        return self.c.all_gather(name, t) if self.sp else t

    def column_backward(self, dy, inp, W, gW, name):
        """A column-parallel layer's backward, as Megatron-LM's LinearWithGradAccumulationAndAsyncCommunication does
        it: with sequence parallelism it all-gathers its input again while it computes the input gradient, and it
        reduces the input gradient while it computes the weight gradient"""
        c = self.c
        if self.sp:
            inp, gathered = c.all_gather(name, inp, async_=True)
        dx = dy @ W
        if self.sp:
            gathered.wait()
            dx, reduced = c.reduce_scatter(name + " grad", dx, async_=True)
        else:
            dx, reduced = c.all_reduce(dx, async_=True)
        torch.matmul(dy.t(), inp, out=gW)
        reduced.wait()
        return dx

    def forward(self, x):
        heads, kv, hd = self.shape["heads"], self.shape["kv"], self.shape["hd"]
        Wqkv, Wo, Wgu, Wd = self.w
        # with sequence parallelism each rank norms its piece, which is then gathered
        xn, xg = rms_norm(x)
        xf = self.gather("x", xn)
        T, S = xf.shape[0], self.seqlen
        q, k, v = F.linear(xf, Wqkv).split([heads * hd, kv * hd, kv * hd], dim=-1)
        q, k, v, attn = attention_forward(*[t.view(T // S, S, -1, hd).transpose(1, 2) for t in (q, k, v)], heads, kv)
        a = attn.detach().transpose(1, 2).reshape(T, heads * hd)
        h1 = x + self.reduce("o", F.linear(a, Wo))
        hn, hg = rms_norm(h1)
        gu = F.linear(self.gather("h1", hn), Wgu)
        g, u = gu.chunk(2, dim=-1)
        act = F.silu(g) * u
        out = h1 + self.reduce("d", F.linear(act, Wd))
        # with sequence parallelism only the inputs' pieces are kept, and backward gathers them again
        self.saved = (xn, xg, q, k, v, attn, a, hn, hg, gu, act)
        return out

    def backward(self, dout):
        heads, hd = self.shape["heads"], self.shape["hd"]
        Wqkv, Wo, Wgu, Wd = self.w
        gWqkv, gWo, gWgu, gWd = self.g
        xn, xg, q, k, v, attn, a, hn, hg, gu, act = self.saved
        self.saved = None
        doutf = self.gather("dout", dout)
        T, S = doutf.shape[0], self.seqlen
        torch.matmul(doutf.t(), act, out=gWd)
        dgu = swiglu_backward(doutf @ Wd, gu)
        dh1 = dout + rms_norm_backward(hg, self.column_backward(dgu, hn, Wgu, gWgu, "h1"))
        dh1f = self.gather("dh1", dh1)
        torch.matmul(dh1f.t(), a, out=gWo)
        dattn = (dh1f @ Wo).view(T // S, S, heads, hd).transpose(1, 2)
        dq, dk, dv = torch.autograd.grad(attn, (q, k, v), dattn)
        dqkv = torch.cat([t.transpose(1, 2).reshape(T, -1) for t in (dq, dk, dv)], dim=-1)
        return dh1 + rms_norm_backward(xg, self.column_backward(dqkv, xn, Wqkv, gWqkv, "x"))

def split_shape(shape, ranks, keys):
    for key in keys:
        if shape[key] % ranks:
            sys.exit(f"--model's {key}={shape[key]} doesn't split over {ranks} ranks")
    return dict(shape, **{key: shape[key] // ranks for key in keys})

def random_bf16(*shape):
    return (torch.randn(*shape, device=device()) * 0.02).to(torch.bfloat16)

class TPStep(MPStep):
    """Megatron-LM's tensor parallelism over groups of tp ranks, each group's ranks computing the same tokens, with
    its sequence parallelism if `sp`"""
    tokens_desc = "tokens per tensor-parallel group"

    def __init__(self, settings, shape, tokens, seqlen, layers, sp=False):
        tp = settings["tp"]
        if sp and tokens % tp:
            sys.exit(f"--tokens {tokens} doesn't split over the {tp} ranks of the tensor-parallel group")
        super().__init__("tp", tp)
        local = split_shape(shape, tp, ("heads", "kv", "f"))
        n = layer_numel(local)
        self.params = random_bf16(layers * n)
        self.grads = torch.empty_like(self.params)
        self.layers = [TPLayer(self.c, sp, local, self.params[i * n:(i + 1) * n], self.grads[i * n:(i + 1) * n],
                               seqlen) for i in range(layers)]
        rows = tokens // tp if sp else tokens
        self.x = torch.randn(rows, shape["h"], dtype=torch.bfloat16, device=device())
        self.dout = torch.randn(rows, shape["h"], dtype=torch.bfloat16, device=device()) * 0.01
        self.flops = layers * layer_flops(local, tokens, seqlen)

class UlyssesLayer(Layer):
    """A layer of DeepSpeed Ulysses, from DeepSpeed's DistributedAttention: each rank has a piece of the sequence,
    and all the weights. Around attention, all_to_all calls switch q, k and v from a piece of the sequence with all
    the heads to all the sequence with a piece of the heads, one call each, and the attention's output back"""
    def __init__(self, c, shape, params, grads, seqlen):
        super().__init__(shape, params, grads, seqlen)
        self.c, self.ranks = c, c.size

    def seq_to_heads(self, name, t, n):
        """(rows, n*hd), a piece of the sequence with n heads, to (rows*ranks, n/ranks*hd)"""
        sp, rows, width = self.ranks, t.shape[0], n // self.ranks * self.shape["hd"]
        t = t.reshape(rows, sp, width).transpose(0, 1).contiguous()
        return self.c.all_to_all(name, t).reshape(sp * rows, width)

    def heads_to_seq(self, name, t, n):
        """(rows*ranks, n/ranks*hd) back to (rows, n*hd)"""
        sp, width = self.ranks, n // self.ranks * self.shape["hd"]
        rows = t.shape[0] // sp
        t = self.c.all_to_all(name, t.reshape(sp, rows, width))
        return t.transpose(0, 1).reshape(rows, sp * width)

    def forward(self, x):
        heads, kv, hd, sp = self.shape["heads"], self.shape["kv"], self.shape["hd"], self.ranks
        Wqkv, Wo, Wgu, Wd = self.w
        xn, xg = rms_norm(x)
        q, k, v = F.linear(xn, Wqkv).split([heads * hd, kv * hd, kv * hd], dim=-1)
        q, k, v = [self.seq_to_heads(name, t, n) for name, t, n in (("q", q, heads), ("k", k, kv), ("v", v, kv))]
        T = q.shape[0]
        q, k, v, attn = attention_forward(*[t.view(1, T, -1, hd).transpose(1, 2) for t in (q, k, v)], heads // sp,
                                          kv // sp)
        a = self.heads_to_seq("attn", attn.detach().transpose(1, 2).reshape(T, -1), heads)
        h1 = x + F.linear(a, Wo)
        hn, hg = rms_norm(h1)
        gu = F.linear(hn, Wgu)
        g, u = gu.chunk(2, dim=-1)
        act = F.silu(g) * u
        self.saved = (xn, xg, q, k, v, attn, a, hn, hg, gu, act)
        return h1 + F.linear(act, Wd)

    def backward(self, dout):
        heads, kv, hd = self.shape["heads"], self.shape["kv"], self.shape["hd"]
        Wqkv, Wo, Wgu, Wd = self.w
        gWqkv, gWo, gWgu, gWd = self.g
        xn, xg, q, k, v, attn, a, hn, hg, gu, act = self.saved
        self.saved = None
        torch.matmul(dout.t(), act, out=gWd)
        dgu = swiglu_backward(dout @ Wd, gu)
        torch.matmul(dgu.t(), hn, out=gWgu)
        dh1 = dout + rms_norm_backward(hg, dgu @ Wgu)
        torch.matmul(dh1.t(), a, out=gWo)
        da = self.seq_to_heads("attn grad", dh1 @ Wo, heads)
        T = da.shape[0]
        dq, dk, dv = torch.autograd.grad(attn, (q, k, v), da.view(1, T, -1, hd).transpose(1, 2))
        dqkv = torch.cat([self.heads_to_seq(name, t.transpose(1, 2).reshape(T, -1), n)
                          for name, t, n in (("q grad", dq, heads), ("k grad", dk, kv), ("v grad", dv, kv))], dim=-1)
        torch.matmul(dqkv.t(), xn, out=gWqkv)
        return dh1 + rms_norm_backward(xg, dqkv @ Wqkv)

class GraphedAttention:
    """One ring step of one layer's attention - SDPA of the layer's q against one rank's k and v, and autograd's
    backward of it - each captured once as a CUDA graph and replayed after, so that the host queues one launch for
    each instead of their SDPA and autograd kernels one by one, which takes the host longer than a small step takes the
    GPU. The graphs compute on the layer's buffers: the forward adds the step's output into the layer's, and the
    backward adds the step's gradients into the layer's q gradient and writes those of k and v into the layer's
    buffer, so that nothing a graph allocates is used after it, and all the graphs of a step share one memory pool,
    as they replay one after the other on one stream, in the order of their capture. The layer is passed in rather
    than kept, as a reference cycle would keep the graphs' memory after the step is gone"""
    def __init__(self, layer, s, kv):
        if layer.c.graph_pool is None:
            layer.c.graph_pool = torch.cuda.graph_pool_handle()
        self.pool, self.bwd = layer.c.graph_pool, None
        self.kv = kv.clone()
        heads, kvh = layer.shape["heads"], layer.shape["kv"]
        # the warmup that capturing needs, which also has cuDNN build its plans for these shapes
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            qs, ks, vs, _ = layer.block(layer.q_buf, self.kv, s)
            qs, ks, vs, attn = attention_forward(qs, ks, vs, heads, kvh, causal=s == 0)
            torch.autograd.grad(attn, (qs, ks, vs), torch.ones_like(attn))
        torch.cuda.current_stream().wait_stream(side)
        self.fwd = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.fwd, pool=self.pool, capture_error_mode="thread_local"):
            qs, ks, vs, self.rows_of = layer.block(layer.q_buf, self.kv, s)
            self.qs, self.ks, self.vs, self.attn = attention_forward(qs, ks, vs, heads, kvh, causal=s == 0)
            if s == 0:
                layer.out_buf.zero_()
            layer.out_buf[:, :, self.rows_of] += self.attn.detach()

    def forward(self, kv):
        self.kv.copy_(kv)
        self.fwd.replay()

    def backward(self, layer):
        """Adds the step's q gradient into the layer's and writes its k and v gradients into the layer's buffer"""
        dout = layer.dattn_buf[:, :, self.rows_of]
        if self.bwd is None:
            torch.autograd.grad(self.attn, (self.qs, self.ks, self.vs), dout, retain_graph=True)
            torch.cuda.synchronize()
            self.bwd = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.bwd, pool=self.pool, capture_error_mode="thread_local"):
                gq, gk, gv = torch.autograd.grad(self.attn, (self.qs, self.ks, self.vs), dout, retain_graph=True)
                layer.dq_buf[:, :, self.rows_of] += gq
                rows = layer.dkv_buf.shape[1]
                kv_rows = slice(None) if gk.shape[2] == rows else slice(None, rows // 2)
                layer.dkv_buf.zero_()
                layer.dkv_buf[:, kv_rows] = torch.stack([gk[0].transpose(0, 1), gv[0].transpose(0, 1)])
        self.bwd.replay()

class RingLayer(Layer):
    """A layer of ring attention as Transformer Engine's context parallelism with KV P2P runs it, the sequence split
    over the ranks in load-balanced pieces: rank r has the r-th and the (2*ranks-1-r)-th of 2*ranks chunks, so that
    with causal attention each rank has the same compute. Each attention step computes the local q against one
    rank's k and v while they're sent on to the next rank, and backward sends them around again, with the gradients
    of k and v, which go on to the next rank one step later, until they reach the rank whose k and v they are. With
    `cuda_graphs` each step's attention runs as CUDA graphs, see GraphedAttention"""
    def __init__(self, c, shape, params, grads, seqlen, cuda_graphs=False):
        super().__init__(shape, params, grads, seqlen)
        self.c = c
        self.ranks, self.rank = c.size, c.rank
        self.next, self.prev = c.peer(1), c.peer(-1)
        self.cuda_graphs, self.graphs = cuda_graphs, {}

    def block(self, q, kv, s):
        """q and the k and v of step s: the local chunks against themselves, causally, and then either both local q
        chunks against the first k and v chunk of an earlier rank, or the second q chunk against both of a later
        rank's, whose first chunk is earlier and its second chunk later in the sequence than the local ones"""
        half = q.shape[2] // 2
        k, v = (t.transpose(0, 1).unsqueeze(0) for t in kv)
        src = (self.rank - s) % self.ranks
        if s == 0:
            return q, k, v, slice(None)
        if src < self.rank:
            return q, k[:, :, :half], v[:, :, :half], slice(None)
        return q[:, :, half:], k, v, slice(half, None)

    def forward(self, x):
        heads, kv, hd = self.shape["heads"], self.shape["kv"], self.shape["hd"]
        Wqkv, Wo, Wgu, Wd = self.w
        rows = x.shape[0]
        xn, xg = rms_norm(x)
        q, k, v = F.linear(xn, Wqkv).split([heads * hd, kv * hd, kv * hd], dim=-1)
        q = q.view(1, rows, heads, hd).transpose(1, 2)
        kvs = [torch.stack([k.view(rows, kv, hd), v.view(rows, kv, hd)])]
        steps = []
        if self.cuda_graphs:
            if not self.graphs:
                self.q_buf, self.out_buf = torch.empty_like(q), torch.empty_like(q)
            self.q_buf.copy_(q)
            out = self.out_buf
        else:
            out = torch.zeros_like(q)
        for s in range(self.ranks):
            if s < self.ranks - 1:
                kvs.append(self.c.empty("kv", kvs[0].shape))
                sent = self.c.p2p([(kvs[s], self.next)], [(kvs[s + 1], self.prev)])
            if self.cuda_graphs:
                if s not in self.graphs:
                    self.graphs[s] = GraphedAttention(self, s, kvs[s])
                self.graphs[s].forward(kvs[s])
                steps.append(self.graphs[s])
            else:
                qs, ks, vs, rows_of = self.block(q, kvs[s], s)
                qs, ks, vs, attn = attention_forward(qs, ks, vs, heads, kv, causal=s == 0)
                # stands in for the merge of the steps' outputs by their softmax statistics, which is elementwise too
                out[:, :, rows_of] += attn.detach()
                steps.append((qs, ks, vs, attn, rows_of))
            if s < self.ranks - 1:
                sent.wait()
        a = out.transpose(1, 2).reshape(rows, heads * hd)
        h1 = x + F.linear(a, Wo)
        hn, hg = rms_norm(h1)
        gu = F.linear(hn, Wgu)
        g, u = gu.chunk(2, dim=-1)
        act = F.silu(g) * u
        self.saved = (xn, xg, kvs[0], steps, a, hn, hg, gu, act)
        return h1 + F.linear(act, Wd)

    def backward(self, dout):
        heads, kv, hd = self.shape["heads"], self.shape["kv"], self.shape["hd"]
        Wqkv, Wo, Wgu, Wd = self.w
        gWqkv, gWo, gWgu, gWd = self.g
        xn, xg, own_kv, steps, a, hn, hg, gu, act = self.saved
        self.saved = None
        rows = xn.shape[0]
        torch.matmul(dout.t(), act, out=gWd)
        dgu = swiglu_backward(dout @ Wd, gu)
        torch.matmul(dgu.t(), hn, out=gWgu)
        dh1 = dout + rms_norm_backward(hg, dgu @ Wgu)
        torch.matmul(dh1.t(), a, out=gWo)
        dattn = (dh1 @ Wo).view(1, rows, heads, hd).transpose(1, 2)
        if self.cuda_graphs:
            if self.graphs[0].bwd is None:
                self.dattn_buf, self.dq_buf = torch.empty_like(dattn), torch.empty_like(dattn)
                self.dkv_buf = torch.empty_like(own_kv)
            self.dattn_buf.copy_(dattn)
            self.dq_buf.zero_()
            dq = self.dq_buf
        else:
            dq = torch.zeros_like(dattn)
        kv_now, dkv = own_kv, None
        for s, step in enumerate(steps):
            sends, recvs = [], []
            if s < self.ranks - 1:
                kv_next = self.c.empty("kv", own_kv.shape)
                sends.append((kv_now, self.next))
                recvs.append((kv_next, self.prev))
            if s > 0:
                dkv_in = self.c.empty("dkv", own_kv.shape)
                sends.append((dkv, self.next))
                recvs.append((dkv_in, self.prev))
            sent = self.c.p2p(sends, recvs)
            if self.cuda_graphs:
                step.backward(self)
                # the next step's graph overwrites the buffer while this step's gradients are sent on
                contribution = self.dkv_buf if s > 0 else self.dkv_buf.clone()
            else:
                qs, ks, vs, attn, rows_of = step
                gq, gk, gv = torch.autograd.grad(attn, (qs, ks, vs), dattn[:, :, rows_of])
                dq[:, :, rows_of] += gq
                contribution = torch.zeros_like(own_kv)
                kv_rows = slice(None) if gk.shape[2] == rows else slice(None, rows // 2)
                contribution[:, kv_rows] = torch.stack([gk[0].transpose(0, 1), gv[0].transpose(0, 1)])
            sent.wait()
            dkv = contribution + dkv_in if s > 0 else contribution
            if s < self.ranks - 1:
                kv_now = kv_next
        # the gradients of the k and v held last go on to their rank, as those of the local ones come back
        own_dkv = self.c.empty("dkv", own_kv.shape)
        self.c.p2p([(dkv, self.next)], [(own_dkv, self.prev)]).wait()
        dqkv = torch.cat([dq.transpose(1, 2).reshape(rows, -1), own_dkv[0].reshape(rows, -1),
                          own_dkv[1].reshape(rows, -1)], dim=-1)
        torch.matmul(dqkv.t(), xn, out=gWqkv)
        return dh1 + rms_norm_backward(xg, dqkv @ Wqkv)

class SeqStep(MPStep):
    """Sequence or context parallelism over groups of `degree` ranks, each of which has a piece of one sequence and
    all the weights"""
    def __init__(self, layer_cls, name, degree, shape, tokens, layers, split):
        super().__init__(name, degree)
        ranks = degree
        split_shape(shape, ranks, split)
        if name == "cp" and tokens % 2:
            sys.exit(f"--tokens {tokens}: cp-rsa splits each rank's tokens into 2 chunks, so they have to be even")
        n = layer_numel(shape)
        self.params = random_bf16(layers * n)
        self.grads = torch.empty_like(self.params)
        self.layers = [layer_cls(self.c, shape, self.params[i * n:(i + 1) * n], self.grads[i * n:(i + 1) * n],
                                 tokens * ranks) for i in range(layers)]
        self.x = torch.randn(tokens, shape["h"], dtype=torch.bfloat16, device=device())
        self.dout = torch.randn(tokens, shape["h"], dtype=torch.bfloat16, device=device()) * 0.01
        # each rank computes its tokens' share of the weights' FLOPs and 1/ranks of the causal attention's
        attention = 2 * (tokens * ranks)**2 * shape["heads"] * shape["hd"] // ranks
        self.flops = layers * 3 * (2 * tokens * layer_numel(shape) + attention)
        self.tokens_desc = f"tokens per rank, of one sequence of {tokens * ranks} tokens"

class EPLayer:
    """An MoE layer with its experts split over the ranks, as Megatron-LM's alltoall token dispatcher runs it: the
    attention computes the rank's tokens, the router sends each token to top_k experts, an all_to_all sends the
    tokens to the experts' ranks and another one sends the experts' outputs back. The routing is balanced, so that
    each rank sends the same number of tokens to every rank, each expert computing the same number of tokens in one
    batched matmul, which stands in for a grouped GEMM. The router's backward is left out. With 2 micro-batches the
    all_to_all calls of one overlap with the compute of the other, as DeepSeek's DualPipe and SGLang's two-batch
    overlap have them do"""
    def __init__(self, c, shape, tokens, seqlen, micro_batches):
        h, heads, kv, hd = shape["h"], shape["heads"], shape["kv"], shape["hd"]
        ranks = c.size
        self.c, self.shape, self.seqlen, self.mb = c, shape, seqlen, micro_batches
        self.top_k, self.local = shape["top_k"], shape["experts"] // ranks
        fe = shape["fe"]
        self.Wqkv, self.Wo = random_bf16((heads + 2 * kv) * hd, h), random_bf16(h, heads * hd)
        self.Wr = random_bf16(shape["experts"], h)
        self.Wgu, self.Wd = random_bf16(self.local, 2 * fe, h), random_bf16(self.local, h, fe)
        self.gWqkv, self.gWo = torch.empty_like(self.Wqkv), torch.empty_like(self.Wo)
        self.gWgu, self.gWd = torch.empty_like(self.Wgu), torch.empty_like(self.Wd)
        rows = tokens // micro_batches
        self.copies = rows * self.top_k
        # each token's top_k copies, in the order they're sent to the ranks, and the received ones' order by expert
        self.order = torch.randperm(self.copies, device=device())
        self.token_of = torch.arange(rows, device=device()).repeat_interleave(self.top_k)[self.order]
        self.by_expert = torch.randperm(self.copies, device=device())
        self.unsort = torch.argsort(self.by_expert)

    def attn(self, m):
        heads, kv, hd = self.shape["heads"], self.shape["kv"], self.shape["hd"]
        x = m["x"]
        T, S = x.shape[0], self.seqlen
        m["xn"], m["xg"] = rms_norm(x)
        q, k, v = F.linear(m["xn"], self.Wqkv).split([heads * hd, kv * hd, kv * hd], dim=-1)
        m["q"], m["k"], m["v"], m["attn"] = attention_forward(
            *[t.view(T // S, S, -1, hd).transpose(1, 2) for t in (q, k, v)], heads, kv)
        m["a"] = m["attn"].detach().transpose(1, 2).reshape(T, heads * hd)
        m["h1"] = x + F.linear(m["a"], self.Wo)
        m["hn"], m["hg"] = rms_norm(m["h1"])
        self.route(m, F.linear(m["hn"], self.Wr).softmax(dim=-1).topk(self.top_k, dim=-1).values)

    def route(self, m, weights):
        """Puts the tokens to send, a copy per expert, into m["send"], given their top_k router weights"""
        m["weights"] = weights.reshape(-1)[self.order].unsqueeze(1)
        m["send"] = m["hn"].index_select(0, self.token_of)

    def gather_grad(self, m):
        """The gradient of the tokens from that of their copies in m["recv"]"""
        return torch.zeros_like(m["h1"]).index_add_(0, self.token_of, m["recv"])

    def experts(self, m):
        xe = m["recv"].index_select(0, self.by_expert).view(self.local, -1, self.shape["h"])
        gu = torch.bmm(xe, self.Wgu.transpose(1, 2))
        g, u = gu.chunk(2, dim=-1)
        act = F.silu(g) * u
        m["xe"], m["gu"], m["act"] = xe, gu, act
        m["send"] = torch.bmm(act, self.Wd.transpose(1, 2)).reshape(self.copies, -1).index_select(0, self.unsort)

    def combine(self, m):
        return m["h1"] + torch.zeros_like(m["h1"]).index_add_(0, self.token_of, m["recv"] * m["weights"])

    def combine_backward(self, m):
        m["send"] = m["dout"].index_select(0, self.token_of) * m["weights"]

    def experts_backward(self, m):
        dye = m["recv"].index_select(0, self.by_expert).view(self.local, -1, self.shape["h"])
        torch.bmm(dye.transpose(1, 2), m["act"], out=self.gWd)
        dgu = swiglu_backward(torch.bmm(dye, self.Wd), m["gu"])
        torch.bmm(dgu.transpose(1, 2), m["xe"], out=self.gWgu)
        m["send"] = torch.bmm(dgu, self.Wgu).reshape(self.copies, -1).index_select(0, self.unsort)

    def attn_backward(self, m):
        heads, hd = self.shape["heads"], self.shape["hd"]
        dh1 = m["dout"] + rms_norm_backward(m["hg"], self.gather_grad(m))
        T, S = dh1.shape[0], self.seqlen
        torch.matmul(dh1.t(), m["a"], out=self.gWo)
        dattn = (dh1 @ self.Wo).view(T // S, S, heads, hd).transpose(1, 2)
        dq, dk, dv = torch.autograd.grad(m["attn"], (m["q"], m["k"], m["v"]), dattn)
        dqkv = torch.cat([t.transpose(1, 2).reshape(T, -1) for t in (dq, dk, dv)], dim=-1)
        torch.matmul(dqkv.t(), m["xn"], out=self.gWqkv)
        return dh1 + rms_norm_backward(m["xg"], dqkv @ self.Wqkv)

    def run(self, mbs, stages):
        """Runs the stages, each a function of a micro-batch followed by an all_to_all of its m["send"] into
        m["recv"], or None for the last stage. With 1 micro-batch each all_to_all blocks; with 2 one micro-batch's
        all_to_all is in flight while the other one computes"""
        if len(mbs) == 1:
            m = mbs[0]
            for stage, name in stages[:-1]:
                stage(m)
                m["recv"], _ = self.exchange(name, 0, m, False)
            return [stages[-1][0](m)]
        pending = [None, None]
        for stage, name in stages[:-1]:
            for i, m in enumerate(mbs):
                if pending[i] is not None:
                    pending[i].wait()
                stage(m)
                m["recv"], pending[i] = self.exchange(name, i, m, True)
        outs = []
        for i, m in enumerate(mbs):
            pending[i].wait()
            outs.append(stages[-1][0](m))
        return outs

    def exchange(self, name, i, m, async_):
        """Sends micro-batch i's m["send"] with the all_to_all of stage `name`, returning what it received and, if
        `async_`, the Handle to wait for before using it"""
        if not async_:
            return self.c.all_to_all(name, m["send"]), None
        return self.c.all_to_all(f"{name} {i}", m["send"], async_=True)

    def forward(self, x):
        self.mbs = [dict(x=t) for t in x.chunk(self.mb)]
        return torch.cat(self.run(self.mbs, [(self.attn, "dispatch"), (self.experts, "combine"),
                                             (self.combine, None)]))

    def backward(self, dout):
        for m, t in zip(self.mbs, dout.chunk(self.mb)):
            m["dout"] = t
        dx = torch.cat(self.run(self.mbs, [(self.combine_backward, "combine grad"),
                                           (self.experts_backward, "dispatch grad"), (self.attn_backward, None)]))
        self.mbs = None
        return dx

class DeepEPLayer(EPLayer):
    """The MoE layer with DeepEP's dispatch and combine in place of the all_to_all calls, as Megatron-Core's flex
    token dispatcher runs them: dispatch sends each token once to each rank of its experts, along with its router
    weights, and copies it into the rows of each of its experts there, in DeepEP's expanded layout, and combine sums
    the weighted expert outputs of each token back on its rank. The backward of dispatch is a combine, and that of
    combine a dispatch. The routing is balanced, as for the all_to_all calls, and the same in every step, so each
    dispatch reuses the layout of the first one, which DeepEP computes with a host sync - in training each routing
    needs one"""
    def __init__(self, c, shape, tokens, seqlen, micro_batches, buffer, num_sms):
        super().__init__(c, shape, tokens, seqlen, micro_batches)
        import deep_ep
        self.buffer, self.num_sms = buffer, num_sms
        rows, experts = tokens // micro_batches, shape["experts"]
        # each column of experts is a shifted copy of a shuffled round-robin over the experts, so that every expert
        # gets the same number of tokens and each token top_k distinct ones, spread over the ranks by the random shifts
        base = (torch.arange(rows, device=device()) % experts)[torch.randperm(rows, device=device())]
        shifts = torch.randperm(experts, device=device())[:self.top_k]
        self.topk_idx = ((base[:, None] + shifts[None, :]) % experts).to(deep_ep.topk_idx_t)
        self.handles = []
        for _ in range(micro_batches):
            x = torch.zeros(rows, shape["h"], dtype=torch.bfloat16, device=device())
            w = torch.full((rows, self.top_k), 1 / self.top_k, dtype=torch.float32, device=device())
            *_, handle, _ = buffer.dispatch(x, topk_idx=self.topk_idx, topk_weights=w, num_experts=experts,
                                            num_sms=self.num_sms, do_cpu_sync=True, do_expand=True)
            self.handles.append(handle)
        torch.cuda.synchronize()

    def exchange(self, name, i, m, async_):
        handle, buffer, send = self.handles[i], self.buffer, m["send"]
        if name in ("dispatch", "combine grad"):
            weights = m.get("topk") if name == "dispatch" else None
            def run():
                recv, _, recv_weights, _, event = buffer.dispatch(
                    send, handle=handle, topk_weights=weights, num_sms=self.num_sms,
                    async_with_compute_stream=async_, do_expand=True)
                return (recv, recv_weights), event if async_ else None
            op = "dispatch"
        else:
            def run():
                out, _, event = buffer.combine(send, handle=handle, num_sms=self.num_sms,
                                               async_with_compute_stream=async_)
                return out, event if async_ else None
            op = "combine"
        # the token copies each delivers, as all_to_all sends them, though DeepEP sends a token once per rank
        copies = self.top_k if op == "dispatch" else 1
        return self.c.library_call(f"{name} {i}", op, run, send.numel() * send.element_size() * copies)

    def route(self, m, weights):
        m["topk"] = weights.float()
        m["send"] = m["hn"]

    def gather_grad(self, m):
        return m["recv"]

    def experts(self, m):
        recv, weights = m["recv"]
        m["w"] = weights.to(torch.bfloat16).unsqueeze(1)
        xe = recv.view(self.local, -1, self.shape["h"])
        gu = torch.bmm(xe, self.Wgu.transpose(1, 2))
        g, u = gu.chunk(2, dim=-1)
        act = F.silu(g) * u
        m["xe"], m["gu"], m["act"] = xe, gu, act
        m["send"] = torch.bmm(act, self.Wd.transpose(1, 2)).reshape(-1, self.shape["h"]) * m["w"]

    def combine(self, m):
        return m["h1"] + m["recv"]

    def combine_backward(self, m):
        m["send"] = m["dout"]

    def experts_backward(self, m):
        dye = (m["recv"][0] * m["w"]).view(self.local, -1, self.shape["h"])
        torch.bmm(dye.transpose(1, 2), m["act"], out=self.gWd)
        dgu = swiglu_backward(torch.bmm(dye, self.Wd), m["gu"])
        torch.bmm(dgu.transpose(1, 2), m["xe"], out=self.gWgu)
        m["send"] = torch.bmm(dgu, self.Wgu).reshape(-1, self.shape["h"])

class EPStep(MPStep):
    """Expert parallelism over all the ranks, each with its own tokens"""
    def __init__(self, settings, shape, tokens, seqlen, layers):
        ranks, mb = settings["ep"], settings["micro_batches"]
        if mb not in (1, 2):
            sys.exit(f"--set micro_batches={mb}: use 1 or 2")
        super().__init__("ep", ranks)
        split_shape(shape, ranks, ("experts",))
        local = shape["experts"] // ranks
        copies = tokens // mb * shape["top_k"]
        if tokens % mb or copies % (ranks * local):
            sys.exit(f"--tokens {tokens}: each of the {mb} micro-batches' tokens times top_k={shape['top_k']} has "
                     f"to split evenly over {ranks} ranks and {local} experts each")
        if settings["dispatcher"] == "nccl":
            self.layers = [EPLayer(self.c, shape, tokens, seqlen, mb) for _ in range(layers)]
        else:
            import deep_ep
            if tokens // mb % shape["experts"]:
                sys.exit(f"--tokens {tokens}: each micro-batch's tokens have to be a multiple of the "
                         f"{shape['experts']} experts, for the balanced routing")
            # one buffer for all the layers, as Megatron-Core's flex dispatcher has
            buffer = deep_ep.EPBuffer(self.c.group, num_max_tokens_per_rank=tokens // mb, hidden=shape["h"],
                                      num_topk=shape["top_k"])
            # set here, as a dispatch with a cached layout doesn't make its own estimate
            num_sms = settings["deepep_sms"] or buffer.get_theoretical_num_sms(shape["experts"], shape["top_k"])
            self.parallelism = f"ep={ranks} with DeepEP's dispatch and combine on {num_sms} SMs"
            self.layers = [DeepEPLayer(self.c, shape, tokens, seqlen, mb, buffer, num_sms) for _ in range(layers)]
        self.x = torch.randn(tokens, shape["h"], dtype=torch.bfloat16, device=device())
        self.dout = torch.randn(tokens, shape["h"], dtype=torch.bfloat16, device=device()) * 0.01
        h, heads, kv, hd = shape["h"], shape["heads"], shape["kv"], shape["hd"]
        attention_numel = (heads + 2 * kv) * hd * h + h * heads * hd
        attention = 2 * (tokens // seqlen) * seqlen**2 * heads * hd
        experts = 2 * tokens * shape["top_k"] * 3 * h * shape["fe"]
        self.flops = layers * 3 * (2 * tokens * attention_numel + attention + experts)

    def gradients(self):
        return [g for layer in self.layers for g in (layer.gWqkv, layer.gWo, layer.gWgu, layer.gWd)]

class PPStep(MPStep):
    """Megatron-LM's interleaved 1F1B pipeline schedule over groups of pp ranks, as forward_backward_pipelining_with_
    interleaving in its pipeline_parallel/schedules.py runs it with the default microbatch_group_size_per_vp_stage of
    pp and without overlap_p2p_comm_warmup_flush. The sends and receives are those of its p2p_communication.py: with
    overlap_p2p_comm each is an isend or irecv of its own, queued in the order that _p2p_ops queues them on even and
    odd ranks, and without it they're batched as _batched_p2p_ops batches them and waited for at once - leaving out
    the device synchronize that batch_p2p_sync adds after each batch. The step's 3 phases are the warmup forwards, the
    1F1B pairs and the cooldown backwards. The "bubble" trial runs the schedule with 1-element sends"""
    tokens_desc = "tokens per micro-batch"
    phase_names = ("warmup", "1F1B", "cooldown")
    extra_kinds = dict(bubble=("tiny", True))

    def __init__(self, settings, shape, tokens, seqlen, layers):
        p, v, m = settings["pp"], settings["virtual_stages"], settings["micro_batches"]
        if v < 2:
            sys.exit(f"--set virtual_stages={v}: the interleaved schedule needs at least 2")
        if layers % (p * v):
            sys.exit(f"--layers {layers} doesn't split into pp={p} times virtual_stages={v} chunks")
        if m % p:
            sys.exit(f"--set micro_batches={m} has to be a multiple of pp={p}")
        apart = dist.get_world_size() // p
        # Megatron-LM's default order of the parallelisms, tp-cp-ep-dp-pp, puts pp outermost, so that with tp within
        # a node each pipeline stage is on another node
        super().__init__("pp", p, f"pp={p} with {v} virtual stages" + (f", the ranks of each group {apart} apart"
                                                                         if apart > 1 else ""), strided=True)
        self.p, self.v, self.m, self.overlap = p, v, m, settings["overlap_p2p_comm"]
        n, per_chunk = layer_numel(shape), layers // (p * v)
        self.params = random_bf16(v * per_chunk * n)
        self.grads = torch.empty_like(self.params)
        layer = lambda i: Layer(shape, self.params[i * n:(i + 1) * n], self.grads[i * n:(i + 1) * n], seqlen)
        self.chunks = [[layer(c * per_chunk + i) for i in range(per_chunk)] for c in range(v)]
        self.act_shape = (tokens, shape["h"])
        self.x = torch.randn(*self.act_shape, dtype=torch.bfloat16, device=device())
        self.dout = torch.randn(*self.act_shape, dtype=torch.bfloat16, device=device()) * 0.01
        self.tiny = torch.zeros(1, dtype=torch.bfloat16, device=device())
        self.tiny_sends = False
        self.flops = m * v * per_chunk * layer_flops(shape, tokens, seqlen)
        self.tokens_desc = f"tokens per micro-batch, {m} micro-batches"
        # Megatron-LM's schedule table: groups of pp micro-batches, each group through all the chunks in turn
        table = [(mb, c) for g in range(0, m, p) for c in range(v) for mb in range(g, g + p)]
        self.mb_table, self.chunk_table = [t[0] for t in table], [t[1] for t in table]
        self.total = m * v
        self.warmup = min(self.total, (p - self.c.rank - 1) * 2 + (v - 1) * p)
        if not self.overlap:
            # NCCL connects two ranks in the first batch that has them both, and the host waits in it until the peer
            # joins, so the schedule's first batches, each with only some of the group's ranks, waited for each other
            # in a circle when the group crossed nodes - every rank connecting to both its neighbors at once doesn't
            send, recv = self.c.buffer("connect", self.act_shape), self.c.empty("connect", self.act_shape)
            self.p2p([("send_next", send), ("recv_prev", recv), ("send_prev", send),
                      ("recv_next", torch.empty_like(recv))], wait=True)

    def p2p(self, ops, wait):
        """Queues `ops`, (name, tensor) in the order of Megatron-LM's _batched_p2p_ops, as it or as _p2p_ops queues
        them, the names being send_next, recv_prev, send_prev and recv_next. Returns a Handle per name"""
        c = self.c
        peer = dict(send_next=c.peer(1), recv_prev=c.peer(-1), send_prev=c.peer(-1), recv_next=c.peer(1))
        if not self.overlap:
            works = OOMGuard.batch_isend_irecv(
                [dist.P2POp(dist.isend if name.startswith("send") else dist.irecv, t, peer[name], c.group)
                 for name, t in ops])
            handles = {name: Handle(None) for name, _ in ops}
            if ops:
                handles[ops[0][0]] = Handle(works)
        else:
            ops = dict(ops)
            # with 2 ranks the previous and next rank are the same, and the receives go through the world group,
            # so that they can run alongside the sends
            groups = (c.group, dist.group.WORLD if c.size == 2 else c.group)
            order = (("send_next", 0), ("recv_prev", 1), ("send_prev", 0), ("recv_next", 1)) if c.rank % 2 == 0 \
                else (("recv_prev", 0), ("send_next", 1), ("recv_next", 0), ("send_prev", 1))
            handles = {}
            for name, g in order:
                if name in ops:
                    op = dist.isend if name.startswith("send") else dist.irecv
                    handles[name] = Handle(op(ops[name], peer[name], group=groups[g]))
        if wait:
            for h in handles.values():
                h.wait()
        return handles

    def communicate(self, send_next=None, send_prev=None, recv_prev=False, recv_next=False, wait=True):
        """Megatron-LM's _communicate: returns the tensors received from the previous and the next rank and the
        Handles of the ops not waited for. With `real` False the received tensors are the stand-ins, and in the
        bubble trial 1-element tensors are sent and received instead"""
        c = self.c
        got_prev = c.empty("recv_prev", self.act_shape) if recv_prev else None
        got_next = c.empty("recv_next", self.act_shape) if recv_next else None
        ops = [(name, t) for name, t in (("send_next", send_next), ("recv_prev", got_prev),
                                         ("send_prev", send_prev), ("recv_next", got_next)) if t is not None]
        if not ops or not (c.real or self.tiny_sends):
            return got_prev, got_next, {}
        if self.tiny_sends:
            ops = [(name, self.tiny if name.startswith("send") else torch.empty_like(self.tiny)) for name, _ in ops]
        else:
            c.calls.append((c.phase, "send/recv", sum(t.numel() * t.element_size() for name, t in ops
                                                      if name.startswith("send"))))
            if self.recording is not None:
                self.recording.append((c.phase, [name for name, _ in ops], wait))
        handles = self.p2p(ops, wait)
        return got_prev, got_next, {} if wait else handles

    def replay(self, phase):
        """Queues the recorded sends and receives of `phase` alone, waiting where the step waited at once, and for
        the rest at the end of the phase"""
        pending = []
        for p, names, wait in self.recorded:
            if p == phase:
                ops = [(name, self.c.buffer(name, self.act_shape)) for name in names]
                pending += self.p2p(ops, wait).values()
        for h in pending:
            h.wait()

    def queue(self, comm, compute, events, marks):
        """As Step.queue, with `comm` "tiny" for the bubble trial"""
        c, p, v, total, warmup = self.c, self.p, self.v, self.total, self.warmup
        first, last = c.rank == 0, c.rank == p - 1

        def mark(event=None):
            if event is None:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
            marks.append((time.perf_counter(), event))

        if comm is True and not compute:
            if self.recorded is None:
                raise RuntimeError("the sends and receives alone can only be replayed after an overlapped trial")
            for i, phase in enumerate(self.phase_names):
                self.replay(phase)
                events[i + 1].record()
                mark(events[i + 1])
            return
        c.real, self.tiny_sends = comm is True, comm == "tiny"
        c.begin_step()
        if c.real:
            c.calls = []
        self.recording = [] if c.real and self.recorded is None else None

        inputs, saved, grads = [[] for _ in range(v)], [[] for _ in range(v)], [[] for _ in range(v)]

        def chunk_of(k, forward):
            ch = self.chunk_table[k % total]
            return ch if forward else v - 1 - ch

        def recv_from_previous_stage(k, forward):
            """Whether the step after `k` receives from the previous stage, and the chunk it receives for"""
            if not (first if forward else last):
                return True, chunk_of(k + 1, forward)
            recv = k >= p - 1
            nxt = chunk_of(k - (p - 1) if recv else k + 1, forward)
            if nxt == (v - 1 if forward else 0):
                recv = False
            return recv, nxt + (1 if forward else -1)

        def forward_step(k):
            ch, mb = chunk_of(k, True), self.mb_table[k]
            if ch == 0 and first and len(inputs[0]) == len(saved[0]):
                inputs[0].append(None)
            # the inputs of the micro-batches whose backward is done have been popped
            done = 0 if k < warmup else self.chunk_table[:k - warmup].count(v - 1 - ch)
            x = inputs[ch][mb - done]
            x = self.x if x is None else x
            acts = []
            for layer in self.chunks[ch]:
                x = layer.forward(x)
                acts.append(layer.saved)
                mark()
            saved[ch].append(acts)
            self.out = x
            return None if ch == v - 1 and last else x

        def backward_step(k):
            ch = chunk_of(k, False)
            if ch == v - 1 and last and not grads[ch]:
                grads[ch].append(None)
            inputs[ch].pop(0)
            acts, dx = saved[ch].pop(0), grads[ch].pop(0)
            dx = self.dout if dx is None else dx
            for layer, a in zip(reversed(self.chunks[ch]), reversed(acts)):
                layer.saved = a
                dx = layer.backward(dx)
                mark()
            self.dx = dx
            return None if ch == 0 and first else dx

        recv_prev_handles, recv_next_handles = [], []
        send_next_handle = send_prev_handle = None
        remaining = total - warmup

        c.phase = "warmup"
        inputs[0].append(None if first else self.communicate(recv_prev=True)[0])
        for k in range(warmup):
            recv_prev, nxt = recv_from_previous_stage(k, True)
            recv_prev &= k != total - 1
            out = forward_step(k)
            if k == warmup - 1 and not self.overlap and remaining:
                got, got_grad, _ = self.communicate(send_next=out, recv_prev=recv_prev, recv_next=not last)
                grads[v - 1].append(got_grad)
            else:
                got, _, _ = self.communicate(send_next=out, recv_prev=recv_prev)
            if recv_prev:
                inputs[nxt].append(got)
            if self.overlap and k == warmup - 1 and remaining:
                _, got_grad, h = self.communicate(recv_next=not last, wait=False)
                if "recv_next" in h:
                    recv_next_handles.append(h["recv_next"])
                if not last:
                    grads[v - 1].append(got_grad)
        events[1].record()
        mark(events[1])

        c.phase = "1F1B"
        for k in range(remaining):
            fk = k + warmup
            fch, bch = chunk_of(fk, True), chunk_of(k, False)
            recv_prev, nxt = recv_from_previous_stage(fk, True)
            recv_prev &= k != remaining - 1
            recv_next, nxt_grad = recv_from_previous_stage(k, False)
            if self.overlap:
                if not (fch == 0 and first) and recv_prev_handles:
                    recv_prev_handles.pop(0).wait()
                out = forward_step(fk)
                got, _, h = self.communicate(send_next=out, recv_prev=recv_prev, wait=False)
                if send_next_handle is not None:
                    send_next_handle.wait()
                send_next_handle = h.get("send_next")
                if "recv_prev" in h:
                    recv_prev_handles.append(h["recv_prev"])
                if recv_prev:
                    inputs[nxt].append(got)
                if not (bch == v - 1 and last) and recv_next_handles:
                    recv_next_handles.pop(0).wait()
                dx = backward_step(k)
                _, got_grad, h = self.communicate(send_prev=dx, recv_next=recv_next, wait=False)
                if send_prev_handle is not None:
                    send_prev_handle.wait()
                send_prev_handle = h.get("send_prev")
                if "recv_next" in h:
                    recv_next_handles.append(h["recv_next"])
            else:
                out = forward_step(fk)
                dx = backward_step(k)
                got, got_grad, _ = self.communicate(send_next=out, send_prev=dx, recv_prev=recv_prev,
                                                    recv_next=recv_next)
                if recv_prev:
                    inputs[nxt].append(got)
            if recv_next:
                grads[nxt_grad].append(got_grad)
        events[2].record()
        mark(events[2])

        c.phase = "cooldown"
        if remaining == 0:
            grads[v - 1].append(None if last else self.communicate(recv_next=True)[1])
        for k in range(remaining, total):
            ch = chunk_of(k, False)
            if not (ch == v - 1 and last) and k != 0 and recv_next_handles:
                recv_next_handles.pop(0).wait()
            recv_next, nxt_grad = recv_from_previous_stage(k, False)
            recv_next &= k != total - 1
            dx = backward_step(k)
            _, got_grad, _ = self.communicate(send_prev=dx, recv_next=recv_next)
            if recv_next:
                grads[nxt_grad].append(got_grad)
        for h in (send_next_handle, send_prev_handle):
            if h is not None:
                h.wait()
        if recv_prev_handles or recv_next_handles or any(inputs) or any(saved) or any(grads):
            raise RuntimeError(f"rank {c.rank}: the pipeline schedule left micro-batches behind")
        events[3].record()
        mark(events[3])
        if self.recording is not None:
            self.recorded, self.recording = self.recording, None

STEPS = {
    "ddp": DDPStep,
    "ds-zero1": lambda *a: DSZeroStep(1, *a),
    "ds-zero2": lambda *a: DSZeroStep(2, *a),
    "ds-zero3": DSZero3Step,
    "fsdp": FSDPStep,
    "tp": TPStep,
    "sp-megatron": lambda *a: TPStep(*a, sp=True),
    "sp-ulysses": lambda settings, shape, tokens, seqlen, layers: SeqStep(UlyssesLayer, "sp", settings["sp"], shape,
                                                                          tokens, layers, ("heads", "kv")),
    "cp-rsa": lambda settings, shape, tokens, seqlen, layers: SeqStep(
        functools.partial(RingLayer, cuda_graphs=settings["cuda_graphs"]), "cp", settings["cp"], shape, tokens, layers,
        ()),
    "ep": EPStep,
    "pp": PPStep,
}
# the patterns of one sequence split over the ranks, which --seqlen doesn't apply to
SPLIT_SEQUENCE = ("sp-ulysses", "cp-rsa")
# the patterns of Megatron-LM's tensor parallelism
TENSOR_PARALLEL = ("tp", "sp-megatron")

def zero_trial(step, comm, compute, busy_ms):
    """Times one step. Returns this rank's forward, backward, post-step and whole-step seconds, forward and backward
    summed over the micro-batches, the host's seconds to queue the step, and whether the host fell behind the GPU:
    queued a layer later than the GPU reached it. A step of
    many layers queues more kernels than CUDA's launch queue holds, which blocks the host until the GPU makes room, so
    the host finishing after the busy wait ends doesn't by itself mean that the GPU waited for it"""
    queue_start, start = start_trial(busy_ms)
    events = [start] + [torch.cuda.Event(enable_timing=True) for _ in range(2 * getattr(step, "gas", 1) + 1)]
    marks = []
    step.queue(comm, compute, events, marks)
    host = time.perf_counter() - queue_start
    torch.cuda.synchronize()
    times = [a.elapsed_time(b) / 1000 for a, b in zip(events, events[1:])]
    times = [sum(times[0:-1:2]), sum(times[1:-1:2]), times[-1]]
    # the GPU reaches a mark after the busy wait, the align all_reduce and the step up to the mark - leaving out the
    # all_reduce makes the check stricter
    fell_behind = any(t - queue_start > busy_ms / 1000 + start.elapsed_time(e) / 1000 for t, e in marks)
    return times + [start.elapsed_time(events[-1]) / 1000], host, fell_behind

# each trial kind as (comm, compute)
KINDS = dict(overlapped=(True, True), compute=(False, True), comm=(True, False))

def check_finite(step, kind):
    """Exits if the last trial of `kind` computed inf or NaN on any rank, in the step's output, its input gradient
    or a weight gradient. A matmul of inf and NaN draws less power than one of finite values, and so runs faster at
    the power cap, which makes the times invalid. Every pattern's step has to keep its last output and input gradient
    in `out` and `dx`, and return its weight gradients from gradients()"""
    bad = torch.stack([~step.out.isfinite().all(), ~step.dx.isfinite().all(),
                       torch.stack([~g.isfinite().all() for g in step.gradients()]).any()]).to(torch.int32)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.any():
        what = [name for name, b in zip(("the output", "the input gradient", "a weight gradient"), bad.tolist()) if b]
        end_progress()
        if dist.get_rank() == 0:
            print(f"the {kind} trial computed inf or NaN in {', '.join(what)} - its times aren't valid", flush=True)
        sys.exit(1)

def measure_zero(step, label):
    """The median forward, backward, post-step and step seconds of each trial kind, each trial's time being the
    slowest rank's, and how many trials were left out"""
    kinds = {**KINDS, **getattr(step, "extra_kinds", {})}
    # a step queues 100s of kernels and collectives, which takes the host a while, so the busy wait is sized from the
    # host time of a step - the second, the first having paid for the warmup
    show_progress(f"{label}: sizing the busy wait")
    hosts = [zero_trial(step, True, True, BUSY_WAIT_MS)[1] for _ in range(2)]
    host = torch.tensor([hosts[-1]], dtype=torch.float64, device=device())
    dist.all_reduce(host, op=dist.ReduceOp.MAX)
    busy_ms = BUSY_WAIT_MS + 1.5 * host.item() * 1000

    for i in range(args.num_warmup_iterations):
        show_progress(f"{label}: warmup {i + 1}")
        for comm, compute in kinds.values():
            zero_trial(step, comm, compute, busy_ms)
    samples = []
    for i in range(args.num_iterations):
        show_progress(f"{label}: {i + 1}")
        row, behind = [], False
        for kind, (comm, compute) in kinds.items():
            times, _, fell_behind = zero_trial(step, comm, compute, busy_ms)
            row += times
            behind |= fell_behind
            if compute and i == args.num_iterations - 1:
                check_finite(step, kind)
        samples.append(row + [float(behind)])
    end_progress()

    samples = torch.tensor(samples, dtype=torch.float64, device=device())
    dist.reduce(samples, dst=0, op=dist.ReduceOp.MAX)
    samples = samples.cpu()
    fell_behind = samples[:, -1] != 0
    keep = ~fell_behind if not fell_behind.all() else torch.ones_like(fell_behind)
    medians = samples[keep, :-1].quantile(0.5, dim=0).view(len(kinds), 4).tolist()
    return dict(times=dict(zip(kinds, medians)), trials=len(samples), dropped=fell_behind.sum().item(),
                all_dropped=fell_behind.all().item(), busy_ms=busy_ms)

def run_zero(pattern, settings):
    shape = load_model_shape(args.model, moe=pattern == "ep")
    if args.sym_mem and pattern in TENSOR_PARALLEL + ("ep", "pp") + SPLIT_SEQUENCE:
        sys.exit(f"--sym-mem isn't supported by the {pattern} pattern, whose collectives' outputs are new tensors")
    if args.seqlen and pattern in SPLIT_SEQUENCE:
        sys.exit(f"--seqlen doesn't apply to {pattern}, which splits one sequence over the ranks")
    guard = OOMGuard(int(os.environ["LOCAL_RANK"]))
    # per token count, its results or how many ranks ran out of memory on it
    results = {}
    for tokens in args.tokens:
        # the ep pattern's micro-batches are each a sequence of their own unless --seqlen splits them
        seqlen = args.seqlen or (tokens // settings["micro_batches"] if pattern == "ep" else tokens)
        if tokens % seqlen or (pattern == "ep" and tokens // settings["micro_batches"] % seqlen):
            sys.exit(f"--tokens {tokens} isn't a multiple of --seqlen {seqlen}, per micro-batch for ep")
        step = None
        gc.collect()
        torch.cuda.empty_cache()
        try:
            show_progress(f"{pattern} {tokens} tokens: setup")
            step = STEPS[pattern](settings, shape, tokens, seqlen, args.layers)
            res = measure_zero(step, f"{pattern} {tokens} tokens")
            res["collectives"] = step.collectives()
            res["forward_comm"], res["post_step_comm"] = step.forward_comm, step.post_step_comm
            res["flops"] = getattr(step, "flops",
                                   getattr(step, "gas", 1) * args.layers * layer_flops(shape, tokens, seqlen))
            res["tokens_desc"] = getattr(step, "tokens_desc", "tokens per rank")
            res["parallelism"] = getattr(step, "parallelism", None)
            res["phase_names"] = getattr(step, "phase_names", None)
            res["seqlen"] = None if pattern in SPLIT_SEQUENCE else seqlen
            results[tokens] = res
        except torch.cuda.OutOfMemoryError:
            guard.ran_out()
        except Exception:
            # a collective that the guard aborted, or a P2P batch it refused, another rank having run out of memory
            if not guard.skipping():
                raise
        finally:
            if step is not None:
                torch.cuda.synchronize()
                if not guard.skipping():
                    step.sym_mem.deregister()
            step = None
        ooms = guard.finish()
        if ooms:
            results[tokens] = ooms
            end_progress()
            if dist.get_rank() == 0:
                print(f"{tokens} tokens ran out of memory on {ooms} of {guard.world} ranks - skipped. The NCCL errors "
                      f"logged meanwhile are those of the collectives that the skip aborted", flush=True)
    return shape, results

RESULT_HEADS = [("compute", "alone"), ("comm", "alone"), ("overlapped", ""), ("exposed comm", "ms"),
                ("comm", "cost"), ("compute", "slowdown")]

def setting_desc(pattern, settings, key, text):
    """'reduce_bucket_size', '1.31e7' -> 'reduce_bucket_size=1.31e7 (24.99MiB in bf16)': a setting in elements also
    gets its bytes, in the communication dtype for the gradient buckets and in bf16 for the parameters"""
    if PATTERNS[pattern]["settings"][key][1] is not elements_setting:
        return f"{key}={text}"
    dtype = settings.get("communication_data_type", torch.bfloat16) if key == "reduce_bucket_size" else torch.bfloat16
    name = {torch.bfloat16: "bf16", torch.float32: "fp32"}[dtype]
    return f"{key}={text} ({fmt_bytes(settings[key] * dtype.itemsize)} in {name})"

def print_zero(pattern, settings, shape, results):
    ranks = dist.get_world_size()
    given = settings["_given"]
    config = ", ".join(setting_desc(pattern, settings, k, v) for k, v in given.items()) or "the defaults"
    if pattern in TENSOR_PARALLEL:
        config += f", CUDA_DEVICE_MAX_CONNECTIONS={os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS')}"
    ffn = (f"{shape['experts']} experts of intermediate {shape['fe']}, top {shape['top_k']}" if "experts" in shape
           else f"intermediate {shape['f']}")
    parallelism = next((res["parallelism"] for res in results.values()
                        if isinstance(res, dict) and res["parallelism"]), None)
    groups = f" in groups of {parallelism}" if parallelism else ""
    print(f"{pattern} over {ranks} ranks{groups}, {args.layers} layers of {args.model} (hidden {shape['h']}, {ffn}, "
          f"{shape['heads']} heads, {shape['kv']} KV heads of {shape['hd']}), bf16, with {config} "
          f"({trial_desc()}):\n")
    for tokens, res in results.items():
        if not isinstance(res, dict):
            print(f"!!! {tokens} tokens ran out of memory on {res} of {ranks} ranks !!!\n")
            continue
        times = res["times"]
        seqs = f", in sequences of {res['seqlen']}" if res["seqlen"] else ""
        gas = settings.get("gas", 1)
        micro = f", in each of the optimizer step's {gas} micro-batches" if gas > 1 else ""
        print(f"{tokens} {res['tokens_desc']}{seqs}{micro} - {res['collectives']}:\n")
        # (phase, its index in the times, whether it has collectives, whether it has compute)
        if res["phase_names"]:
            phases = [(name, i, True, True) for i, name in enumerate(res["phase_names"])] + [("step", 3, True, True)]
        else:
            phases = ([("forward", 0, res["forward_comm"], True), ("backward", 1, True, True)] +
                      ([("post-step all_gather", 2, True, False)] if res["post_step_comm"] else []) +
                      [("step", 3, True, True)])
        # with a bubble trial, the exposed comm is the time the sends' bytes add to the bubble trial
        bubbles = "bubble" in times
        rows = []
        for phase, i, has_comm, has_compute in phases:
            compute, comm, both = (times[kind][i] for kind in ("compute", "comm", "overlapped"))
            base = times["bubble"][i] if bubbles else compute
            exposed = both - base if has_compute else both
            bubble_cells = [fmt_ms(base - compute)] if bubbles else []
            if has_comm:
                comm_cells = [fmt_ms(comm), fmt_ms(both), fmt_ms(exposed), fmt_share(exposed, comm),
                              fmt_share(exposed, compute, 1) if has_compute else ""]
            else:
                comm_cells = ["", fmt_ms(both), "", "", ""]
            rows.append([phase, fmt_ms(compute) if has_compute else ""] + bubble_cells + comm_cells)
        heads = [("phase", ""), RESULT_HEADS[0]] + [("bubble", "")] * bubbles + RESULT_HEADS[1:]
        print("\n".join(format_table(None, heads, rows, "<" + ">" * (len(heads) - 1))))
        tflops = lambda t: f"{res['flops'] / t / 10**12:.0f}"
        print(f"\nThe compute alone runs at {tflops(times['compute'][3])} TFLOPS, and at "
              f"{tflops(times['overlapped'][3])} TFLOPS of step time overlapped.")
        if bubbles:
            compute, bubble = times["compute"][3], times["bubble"][3]
            p, v, m = settings["pp"], settings["virtual_stages"], settings["micro_batches"]
            print(f"The pipeline bubble adds {fmt_share(bubble - compute, compute, 1)} to the compute alone, where "
                  f"(pp - 1) / (micro_batches * virtual_stages) is {fmt_share(p - 1, m * v, 1)}.")
        if res["dropped"]:
            print(f"Left out {res['dropped']} of {res['trials']} trials, in which a rank's host queued a layer later "
                  f"than its GPU reached it, despite the {res['busy_ms']:.1f}ms busy wait.")
        if res["all_dropped"]:
            print(f"!!! The host fell behind in every trial, which are reported anyway - their times include the "
                  f"host's !!!")
        print()

    print("compute alone, comm alone: the step with only its compute, and with only its collectives and the copies "
          "into and out of their buffers, which still wait for what they wait for in the overlapped step")
    print("exposed comm: how much longer the overlapped step takes than its compute alone, in ms - the time the comm "
          "added to it, which up to all of the comm's time is the communication that didn't overlap with the "
          "compute, together with how much the two slowed each other down")
    print("comm cost: the same time as a percentage of the comm alone - 0% when the comm overlaps fully with the "
          "compute, 100% when all of it is exposed and the step takes as long as its compute and its comm one after "
          "the other, and above 100% when the step takes longer still, as when the ranks wait at each collective for "
          "the slowest one, which the compute alone and the comm alone don't do")
    print("compute slowdown: the same time as a percentage of the compute alone - how much longer the step takes "
          "than its compute alone")
    print("each time is the slowest rank's, at the median of the trials")
    if settings.get("gas", 1) > 1:
        print("forward, backward: summed over the optimizer step's micro-batches, as are the collectives above")
    if len(results) > 1:
        rows = []
        for tokens, res in results.items():
            if not isinstance(res, dict):
                continue
            compute, comm, both = (res["times"][kind][3] for kind in ("compute", "comm", "overlapped"))
            rows.append([str(tokens), fmt_ms(compute), fmt_ms(comm), fmt_ms(both), fmt_ms(both - compute),
                         fmt_share(both - compute, comm), fmt_share(both - compute, compute, 1)])
        heads = [("tokens", "")] + RESULT_HEADS
        print("\n" + "\n".join(format_table("The whole step:", heads, rows, ">" * 7)))
    print()


### The contention pattern ###

def collective_call(name, size, buffer):
    """Returns (call, size) - call() runs one `name` collective on a `size` bytes payload of fp32, smaller than the
    one asked for when the 1/ranks pieces had to be rounded down - or None if the payload is too small to split into
    ranks pieces. `buffer(slot, numel)` returns an fp32 tensor - slot 0 for the input, 1 for a separate output"""
    ranks = dist.get_world_size()
    if name == "all_reduce":
        tensor = buffer(0, size // 4)
        return lambda: dist.all_reduce(tensor), size
    chunk = size // ranks // CHUNK_ALIGN_BYTES * CHUNK_ALIGN_BYTES // 4
    if chunk == 0:
        return None
    whole = chunk * ranks
    if name == "all_gather":
        input, output = buffer(0, chunk), buffer(1, whole)
        call = lambda: all_gather_single(output, input)
    elif name == "reduce_scatter":
        input, output = buffer(0, whole), buffer(1, chunk)
        call = lambda: reduce_scatter_single(output, input)
    else:
        input, output = buffer(0, whole), buffer(1, whole)
        call = lambda: dist.all_to_all_single(output, input)
    return call, whole * 4

def kernel_load(spec):
    """Returns (op, work, desc): op() runs the kernel once, work is what one run does - its FLOPs for a matmul, the
    bytes it reads and writes for a copy - and desc names it in the report"""
    kind, value = spec
    if kind == "matmul":
        m, n, k = value
        a = torch.randn(m, k, dtype=torch.bfloat16, device=device())
        b = torch.randn(k, n, dtype=torch.bfloat16, device=device())
        c = torch.empty(m, n, dtype=torch.bfloat16, device=device())
        return lambda: torch.matmul(a, b, out=c), 2 * m * n * k, f"a bf16 {m}x{n}x{k} matmul"
    src = torch.randn(value // 2, dtype=torch.bfloat16, device=device())
    dst = torch.empty_like(src)
    # an elementwise kernel, so that it runs on the SMs as a training step's memory-bound kernels do
    return lambda: torch.neg(src, out=dst), 2 * value, f"a bf16 elementwise copy of {fmt_bytes(value)}"

def contention_trial(call, load, calls, loads, busy_ms, comm_first, events, streams):
    """One trial of `calls` collectives on one stream and `loads` kernel runs on another, both starting when the busy
    wait ends - either count can be 0, to time the other alone. Returns each stream's seconds from that start to the
    end of its last op, the host's seconds per queued call and per queued run, and whether this rank's host took
    longer to queue them than the busy wait kept the device busy"""
    comm_end, load_end = events
    comm_stream, load_stream = streams
    queue_start, start = start_trial(busy_ms)
    comm_stream.wait_event(start)
    load_stream.wait_event(start)
    host = {}
    def queue_calls():
        t = time.perf_counter()
        with torch.cuda.stream(comm_stream):
            for _ in range(calls):
                call()
            comm_end.record()
        host["call"] = (time.perf_counter() - t) / max(calls, 1)
    def queue_loads():
        t = time.perf_counter()
        with torch.cuda.stream(load_stream):
            for _ in range(loads):
                load()
            load_end.record()
        host["load"] = (time.perf_counter() - t) / max(loads, 1)
    for queue in ((queue_calls, queue_loads) if comm_first else (queue_loads, queue_calls)):
        queue()
    host_fell_behind = (time.perf_counter() - queue_start) * 1000 > busy_ms
    torch.cuda.synchronize()
    return (start.elapsed_time(comm_end) / 1000, start.elapsed_time(load_end) / 1000, host["call"], host["load"],
            host_fell_behind)

def measure_contention(settings, call, size, load, streams):
    """The median times of one payload's collective alone, the kernel alone, and both at once"""
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    comm_first = settings["order"] == "comm-first"
    op = load[0]
    trial = lambda c, l, busy_ms: contention_trial(call, op, c, l, busy_ms, comm_first, events, streams)

    # a few calls and runs alone give how many of each take kernel_ms, and how long the host takes to queue them.
    # Every rank must queue as many calls and busy wait as long, so all ranks take the slowest rank's numbers
    probe = []
    for i in range(args.num_warmup_iterations + 3):
        t_comm, _, host_call, _, _ = trial(PROBE_CALLS, 0, BUSY_WAIT_MS)
        _, t_load, _, host_load, _ = trial(0, PROBE_RUNS, BUSY_WAIT_MS)
        probe.append([t_comm / PROBE_CALLS, t_load / PROBE_RUNS, host_call, host_load])
    probe = torch.tensor(probe[-3:], dtype=torch.float64, device=device()).quantile(0.5, dim=0)
    dist.all_reduce(probe, op=dist.ReduceOp.MAX)
    per_call, per_load, host_call, host_load = probe.tolist()

    # each trial of the collective alone is followed by one of the kernel alone and one of both, so that a drift in
    # the clocks over the run reaches all three alike. A host can block once CUDA's queue of launched work is full,
    # as it is after a few 100 copy-engine collectives, and then no busy wait is long enough - so while the host falls
    # behind in the warmups, the trials are halved
    kernel_ms = settings["kernel_ms"]
    for halvings in range(MAX_HALVINGS + 1):
        if settings["calls"] == "auto":
            calls = min(MAX_CALLS, max(1, round(kernel_ms / 1000 / per_call)))
        else:
            calls = settings["calls"]
        loads = max(1, round(kernel_ms / 1000 / per_load))
        busy_ms = BUSY_WAIT_MS + 1.5 * (calls * host_call + loads * host_load) * 1000
        kinds = [(calls, 0), (0, loads), (calls, loads)]
        fell_behind = []
        for i in range(args.num_warmup_iterations):
            for c, l in kinds:
                fell_behind.append(trial(c, l, busy_ms)[4])
        fell_behind = torch.tensor([float(any(fell_behind[-3:]))], device=device())
        dist.all_reduce(fell_behind, op=dist.ReduceOp.MAX)
        if not fell_behind.item() or (calls == 1 and loads == 1):
            break
        kernel_ms /= 2
    samples = []
    for i in range(args.num_iterations):
        show_progress(f"{settings['collective']} {fmt_bytes(size):>6}: {i + 1}")
        (comm_alone, _, _, _, f1), (_, load_alone, _, _, f2), (comm_both, load_both, _, _, f3) = [
            trial(c, l, busy_ms) for c, l in kinds]
        samples.append([comm_alone, load_alone, comm_both, load_both, float(f1 or f2 or f3)])

    # an op is done when it's done on every rank, and the trials in which any rank's host fell behind are left out on
    # all of them
    samples = torch.tensor(samples, dtype=torch.float64, device=device())
    dist.reduce(samples, dst=0, op=dist.ReduceOp.MAX)
    samples = samples.cpu()
    fell_behind = samples[:, 4] != 0
    keep = ~fell_behind if not fell_behind.all() else torch.ones_like(fell_behind)
    comm_alone, load_alone, comm_both, load_both = samples[keep, :4].quantile(0.5, dim=0).tolist()
    return dict(calls=calls, loads=loads, kernel_ms=kernel_ms, comm_alone=comm_alone, load_alone=load_alone,
                comm_both=comm_both, load_both=load_both, trials=len(samples), dropped=fell_behind.sum().item(),
                all_dropped=fell_behind.all().item())

def run_contention(settings):
    name = settings["collective"]
    sizes = [2**x for x in range(settings["min_payload"].bit_length() - 1, settings["max_payload"].bit_length())]
    sym_mem = SymMem([dist.group.WORLD])
    with sym_mem.alloc():
        bufs = [torch.empty(max(sizes) // 4, dtype=torch.float32, device=device())
                for _ in range(1 if name == "all_reduce" else 2)]
    sym_mem.register()
    # zeros, as the in-place sums of the 100s of calls a trial can queue would grow any other values to inf
    for buf in bufs:
        buf.zero_()
    buffer = lambda slot, numel: bufs[slot][:numel]
    load = kernel_load(settings["kernel"])
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    results = {}
    for size in sizes:
        benchmark = collective_call(name, size, buffer)
        if benchmark is None:
            continue
        call, size = benchmark
        if size not in results:
            results[size] = measure_contention(settings, call, size, load, streams)
    end_progress()
    torch.cuda.synchronize()
    sym_mem.deregister()
    return load, results

def print_contention(settings, load, results):
    ranks = dist.get_world_size()
    name, kind = settings["collective"], settings["kernel"][0]
    _, work, desc = load
    unit, scale = ("TFLOPS", 10**12) if kind == "matmul" else ("GBps", 10**9)
    order = "queued before" if settings["order"] == "comm-first" else "queued after"
    print(f"The bandwidth of {name} over {ranks} ranks alone and {order} {desc} on another stream ({trial_desc()}, "
          f"about {settings['kernel_ms']:g}ms of the kernel per trial):\n")
    coeff = BUSBW_FACTOR[name](ranks)
    slowdown = lambda overlapped, alone: f"{(overlapped / alone - 1) * 100:+.1f}%"
    rows, outside = [], []
    for size, r in results.items():
        busbw = lambda t: f"{conv_to_GBps(size * r['calls'] / t * coeff):.2f}GBps"
        rate = lambda t: f"{work * r['loads'] / t / scale:.1f}"
        both, longer = max(r["comm_both"], r["load_both"]), max(r["comm_alone"], r["load_alone"])
        if both < SHORTEST_OVERLAPPED * longer:
            outside.append(fmt_bytes(size))
        rows.append([fmt_bytes(size), str(r["calls"]), busbw(r["comm_alone"]), busbw(r["comm_both"]),
                     slowdown(r["comm_both"], r["comm_alone"]), str(r["loads"]), rate(r["load_alone"]),
                     rate(r["load_both"]), slowdown(r["load_both"], r["load_alone"])])
    heads = [("payload", ""), (name, "calls"), ("busbw", "alone"), ("busbw", "overlapped"), (name, "slowdown"),
             (kind, "runs"), (unit, "alone"), (unit, "overlapped"), (kind, "slowdown")]
    print("\n".join(format_table(None, heads, rows, ">" * len(heads))))
    print(f"\ncalls and runs: how many {name} calls and {kind} runs one trial queues")
    print(f"slowdown: how much longer the same calls or runs take overlapped than alone, each time being the slowest "
          f"rank's, at the median of the trials")
    if outside:
        print(f"!!! At {', '.join(outside)} the overlapped trial is shorter than the longer of the two alone, which "
              f"the same work can't be - rerun with more --num_iterations !!!")
    halved = [f"{fmt_bytes(size)} to {r['kernel_ms']:g}ms" for size, r in results.items()
              if r["kernel_ms"] < settings["kernel_ms"]]
    if halved:
        print(f"\nThe host fell behind with {settings['kernel_ms']:g}ms trials, so they were cut at "
              f"{', '.join(halved)}.")
    dropped = sum(r["dropped"] for r in results.values())
    if dropped:
        print(f"\nLeft out {dropped} of {sum(r['trials'] for r in results.values())} trials, in which a rank's host "
              f"took longer than the busy wait to queue its calls and runs, so its device waited for the host.")
    all_dropped = [fmt_bytes(size) for size, r in results.items() if r["all_dropped"]]
    if all_dropped:
        print(f"!!! The host fell behind in every trial of {', '.join(all_dropped)}, which are reported anyway - "
              f"their numbers include the host's time !!!")
    print()

    plt = new_plot(path := f"contention-{name}-{kind}-{socket.gethostname()}-{ranks}.png")
    if plt is not None:
        sizes = list(results)
        for key, label in (("comm_alone", "alone"), ("comm_both", f"{order} {desc}")):
            plt.plot(sizes, [conv_to_GBps(size * r["calls"] / r[key] * coeff) for size, r in results.items()],
                     marker=".", label=label)
        plt.xscale("log", base=2)
        plt.xticks(sizes, [fmt_bytes(x) for x in sizes], rotation=45, fontsize=6)
        plt.xlabel("Message size")
        plt.ylabel("Bus bandwidth (GBps)")
        plt.title(f"{name} bus bandwidth on ranks={ranks}")
        plt.legend(fontsize=7)
        save_plot(plt, path)


def run():
    start_time = time.time()
    settings = parse_settings(args.pattern, args.set)
    if args.pattern == "contention":
        load, results = run_contention(settings)
    else:
        shape, results = run_zero(args.pattern, settings)
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"\nEnvironment:")
        print(f"- software: torch={torch.__version__}, cuda={torch.version.cuda}, nccl={nccl_version()}")
        print(f"- hardware: {device_info()}\n")
        if args.pattern == "contention":
            print_contention(settings, load, results)
        else:
            print_zero(args.pattern, settings, shape, results)
        time_str = str(datetime.timedelta(seconds=time.time() - start_time)).split(".")[0]
        print(f"Legend: 1KiB = 2**10 Bytes, 1MiB = 2**20 Bytes, 1GiB = 2**30 Bytes")
        print(f"        1GBps = 10**9 Bytes per second (networking bw spec convention)")
        print(f"Elapsed time: {time_str}")
    dist.destroy_process_group()

def parse_args():
    global args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pattern", choices=list(PATTERNS), help="the parallelism pattern - see --list-patterns")
    parser.add_argument("--list-patterns", action="store_true", help="describe each pattern and its settings, and exit")
    parser.add_argument("--set", default=None, help="the pattern's settings, comma separated, e.g. overlap_comm=1,reduce_bucket_size=2.5e8 - see --list-patterns")
    parser.add_argument("--model", default="llama-3.1-8b", help=f"the layer shape: a preset ({', '.join(MODEL_PRESETS)}), a config.json, or a Hugging Face Hub model name, whose config.json is downloaded")
    parser.add_argument("--tokens", default="8192", help="tokens per rank per step - per tensor-parallel group with tp - comma separated for several, e.g. 4096,16384")
    parser.add_argument("--seqlen", type=int, default=None, help="the sequence length the tokens are split into, which sets the attention's cost (default: one sequence of all the tokens)")
    parser.add_argument("--layers", type=int, default=8, help="the layers of a step - enough for the layers in the middle to reach the steady state, in which one layer's collectives overlap with another's compute")
    parser.add_argument("--num_iterations", type=int, default=10, help="the trials of each kind - compute alone, comm alone, and both")
    parser.add_argument("--num_warmup_iterations", type=int, default=3, help="the warmup trials of each kind")
    parser.add_argument("--sym-mem", action="store_true", help="allocate the communication buffers as NCCL symmetric memory windows (torch>=2.12, NCCL>=2.29)")
    parser.add_argument("--cta-policy", choices=["default", "efficiency", "zero"], default="default", help="the NCCL CTA policy of the process groups (NCCL>=2.28), which sets how many SMs NCCL gives each collective, up to the limit that the NCCL_MAX_CTAS env var can lower: `default` the ones that give it its best speed, `efficiency` the fewest that still give it a decent one, `zero` none whenever it can, which with --sym-mem runs all_gather and all_to_all on the copy engines within a node. The SMs a collective doesn't take are left to the compute kernels running alongside it")
    args = parser.parse_args()
    if args.list_patterns:
        list_patterns()
        sys.exit(0)
    if args.pattern is None:
        parser.error("--pattern is required - see --list-patterns")
    args.tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    if args.pattern == "ep" and parse_settings(args.pattern, args.set)["dispatcher"] == "deepep":
        # before NCCL loads its network plugins, whose file names DeepEP's import takes for a second NCCL
        try:
            import deep_ep
        except ImportError as e:
            sys.exit(f"--set dispatcher=deepep needs DeepEP, https://github.com/deepseek-ai/DeepEP: {e}")


if __name__ == "__main__":
    parse_args()
    if args.pattern in TENSOR_PARALLEL:
        # as Megatron-LM recommends with sequence parallelism: one hardware queue keeps the kernels in the order
        # they're queued, so that a collective's kernel gets its SMs before the matmul queued after it fills them
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    # a rank whose collective sees another rank's OOMGuard abort as a network error, before its own OOMGuard aborts its
    # communicators, would have torch's NCCL watchdog take the process down a minute later - with torchrun's default of
    # 1 and torch's of 3 alike. With 2 the watchdog only aborts the communicators, and a real NCCL error still ends the
    # run, as the next collective raises
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "2"
    # the watchdog first waits 4 times this for a flight recorder dump, a minute by default, and OOMGuard deleting the
    # group waits for the watchdog
    os.environ.setdefault("TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC", "250")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", **device_id_kwargs(local_rank), **nccl_pg_options())
    run()
