#!/usr/bin/env python
"""
Find straggler GPUs: GPUs that work correctly but run slower than their peers.

In any job that runs on several GPUs in lockstep - training, and inference with tensor, expert or pipeline
parallelism - every GPU waits for the slowest one at each collective or stage boundary, so one GPU that is 5% slow
makes every training step or every generated token 5% slower. Such a GPU fails no diagnostic, and a single-matmul
benchmark may not single it out either: GPU 3 of an 8x B200 node ran a training step 3% slower than its 7
siblings, yet a sustained matmul and memory copies at their speed. So this script measures every GPU under 4
workloads, which between them stand in for training and inference - a training step, a matmul like prefill's, and
2 memory-bound ones like decode's - and reports how each GPU compares with the others in each:

  1. step: runs the same training step on every GPU at once - 32 Llama-3.1-8B layers of one 8192-token sequence, forward
     and backward in bf16, the layer of comm-compute-overlap-bench.py - with no communication between the GPUs
     while it's timed, so that each GPU's time is its own and not the slowest one's. Each step is followed by a
     100ms idle, as a training step's compute is by waits for communication or data, which a GPU's power management
     reacts to - a straggler that was 3-4% slow under this load was only 0.6-1.6% slow under the same step with no
     idle. The GPUs run at once so that each runs under its neighbors' load, which shares the node's power and
     cooling, as in a real job. Its forward runs the same kinds of kernels as an inference forward, so a GPU slow
     here is slow there too
  2. times the step for 60s from almost the first step, on GPUs that have been idle for a minute or two: the same
     GPU was slow only some of the time - in one run 3-4% in the first 20-25s after the load started from idle and
     1% after, in others 5% throughout - so a measure of a GPU's best or typical speed can miss it, the way
     mamf-finder.py's MAMF, the median of a shape's 5 fastest runs, did
  3. matmul: then runs one matmul shape (--matmul-shape) back to back for --other-duration (30s), with no idle -
     the steady load at the power cap of mamf-finder.py's MSMF, but timed throughout rather than in its best window
  4. copy and stream: then measures the HBM bandwidth 2 ways, for --other-duration each, on --copy-mib (1024MiB)
     buffers: copy is a device-to-device cudaMemcpy, which the driver may run on a copy engine, and stream adds 2
     buffers into a third with a kernel on the SMs. Both are limited by the HBM bandwidth, so a GPU slow in both has
     slow memory, while one slow in stream alone, as the straggler was in its memory-bound kernels, has slow SMs
  5. samples each GPU's clock, power, temperature and clock-limit reasons while each workload runs
  6. cuts each workload into --window (2s) windows, and in each window divides every GPU's time by its node's median
     GPU's in it, which removes what they all share, like the warmup, and how its node differs from the others.
     A window in which the node's median GPU is more than --stall (5%) slower than in the node's median window is
     left out of that node's comparisons and reported as a stall: one GPU can't move the median, so all of the
     node's GPUs slowed at once - on one node all 8 went from about 400ms to 474-794ms a step at the same moment,
     3 times in 10s. The stall hit each GPU at a different point in its step, and left in, it flagged a healthy
     GPU as unsteady. Each GPU is then judged by its 90th
     percentile window - how slow it was in its worst stretches - rather than by its median: the straggler's was
     4.0% slow in a run in which its median was 1.4%, while no healthy GPU's went above 1.9% in 5 runs
  7. prints a line per GPU sorted from the slowest step, with each workload's median and 90th percentile window -
     once that would leave out at least 10 GPUs, only every flagged one, the --unflagged-rows (5) slowest unflagged
     ones in the step, and each workload's slowest and least steady unflagged one, then the range of the rest;
     --unflagged-rows -1 prints every GPU - and flags as a straggler every GPU that in any of the 4 workloads is:
       - slow: its 90th percentile window is an outlier - a modified z-score above --outlier (3.5), its distance
         from the median in units of the median absolute deviation, which a straggler can't inflate as it would a
         standard deviation - among all the GPUs of all the nodes, and the baseline's with --baseline, and more
         than --threshold (1%) slow
       - unsteady: its 90th percentile window is more than --unsteady (1.25%) slower than its own median window,
         and an outlier by the same z-score - slower in a stretch of the run than it is the rest of the time,
         wherever it stands among the others. The straggler's step was 2.3-2.7% slower in its 90th percentile
         window than in its median one in the 6 runs in which it was slow for its first 20-25s, while no healthy
         GPU's went above 0.76% in 16 runs
       - below baseline, with --baseline, the --json files of earlier runs on known-good nodes: its median time is
         more than --threshold slower than its own there. This catches a GPU that slowed by less than the healthy
         GPUs differ - on a node whose healthy GPUs ran 0.8% faster to 1.7% slower than their median, a GPU had to
         slow down by 1-4%, depending on how fast it was to begin with, to be flagged, while a healthy GPU repeats
         its median time within 0.2%
     and flags as slow every node whose median GPU in any of the 4 workloads is:
       - slower than the other nodes': its 90th percentile window is more than --slow-node (3%) slower than
         the median node's in the same windows, or the faster node's if there are only 2. Of 2 nodes whose matmul
         differed by 1.6-1.9% at the median, the slower one's 90th percentile window reached 2.5% in 8 runs
       - below baseline: its median time is more than --slow-node slower than the baseline's median node's
     Comparing a node's GPUs with each other can't catch a node whose GPUs are all equally slow, e.g. at a lower
     power limit or in a hot rack, which slows a job as much as a single slow GPU - only other nodes, measured with
     it or in the baseline, can. A flagged GPU is listed with what it was flagged for, and with the UUID, serial and
     PCI bus id to report it by, and a flagged node with its GPUs' serials
  8. with --repeat N runs it all N times, --repeat-idle (120s) apart so that each run starts from idle, reports each
     run as it ends, and then lists every GPU flagged in any run, and every GPU more than --threshold slower in a
     run than in its own fastest one: a straggler that is slow some of the time may be slow for minutes or hours
     and then not, so a single 3-minute run can miss it
  9. on 2 or more nodes it lists the nodes fastest first, by their pace in a job that runs the GPUs in lockstep:
     the slowest GPU's step in each --window, at the 90th percentile window, in the slowest of the --repeat runs.
     A node with a flagged GPU, or flagged itself, goes after the others. With --hostfile-to-sort it writes that
     hostfile's lines in the same order to the same path with -fast-to-slow appended, so that its first N lines are
     the N fastest nodes. A line is matched to a node by its first word, without any user@, which can be the
     node's hostname or one of its IP addresses

It shows what it's running as it goes - a progress bar per workload, tqdm's if it's installed - and ends with the
runtime of each stage.

The comparisons need a group to judge each GPU and each node by - run it on whole nodes, as many as you can at
once: a single node is the same check on fewer GPUs, without the comparison of the nodes. Record a baseline on
nodes known to be good to check a single node, or 2 nodes, which the comparison can tell apart but not which of
them is normal.

The defaults are tight: they flag a GPU that slows a synchronous job by 1-2%, or a node by 3%, which across a job of
thousands of GPUs costs more than replacing the node. Fleet tools use looser ones to decide what to replace -
SuperBench's example rules flag a 5% regression vs a baseline, and Guard keeps a node under 10% slow in the job
and replaces it right away at 20%. Raise --threshold, --unsteady and --slow-node to match your replacement
policy, or if the defaults flag healthy GPUs on your hardware.

The timings run on any accelerator that torch.accelerator supports. The telemetry needs a vendor backend, and only
NVIDIA's (NVML, `pip install nvidia-ml-py`) is implemented - see the Telemetry class for adding one. Without one
only the timings are reported, which is all that is needed to find a straggler.

On NVIDIA the GPU numbers printed are nvidia-smi's indices, which count from 0. They may differ from the CUDA
ordinals when CUDA_DEVICE_ORDER isn't PCI_BUS_ID - the GPUs are matched by UUID, so they are always nvidia-smi's.

Usage:

  One node, all its GPUs:

    python gpu-straggler-finder.py

  Multiple nodes - run on every node at once, e.g. 4 nodes of 8 GPUs:

    torchrun --nnodes 4 --nproc_per_node 8 --rdzv_backend c10d \\
        --rdzv_endpoint $MASTER_ADDR:29500 gpu-straggler-finder.py

  The node results are gathered on rank 0 after the measurement, over the gloo backend - if it picks the wrong
  network interface set GLOO_SOCKET_IFNAME.

Exit code, when run without torchrun: 0 if no GPU was flagged, 3 if any was, and 2 if a step computed inf or NaN,
so it can run as a node check, e.g. in a job's prologue. Under torchrun it is always 0, since a rank exiting with an
error makes torchrun tear down the other nodes; check the report or the --json file's "stragglers" instead.

Examples:

  python gpu-straggler-finder.py --layers 8 --duration 300 --json straggler.json
      A shorter step, timed for longer, with every GPU's step times written to straggler.json.

  python gpu-straggler-finder.py --hidden 8192 --ffn 28672 --heads 64 --layers 8
      A wider layer, closer to a 70B model's, if your GPUs have the memory.

  python gpu-straggler-finder.py --json good-1.json
  python gpu-straggler-finder.py --baseline good-*.json --repeat 10
      Record a baseline on each of a few nodes known to be good, then check another node against them and against
      itself, 10 times over about 50 minutes.

  torchrun --nnodes 5 ... gpu-straggler-finder.py --hostfile-to-sort hostfile
  head -4 hostfile-fast-to-slow > hostfile-fast
      Sort the hostfile of 5 nodes fastest first, then keep the 4 fastest to run a job on.
"""

import argparse
import contextlib
import datetime
import json
import os
import shutil
import socket
import statistics
import sys
import threading
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def fmt_secs(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{s % 60:02.0f}s"


class Progress:
    """
    Shows what is running, on rank 0 only: a bar for a phase of known length - tqdm's if it's installed - else a
    spinner. On a terminal it draws from a background thread every 0.25s, doing no more than reading the elapsed
    time and the number of calls timed, so that it can't delay the timed calls; in a log file each phase is a line
    when it starts and one when it ends. Keeps each phase's runtime for the summary at the end.
    """

    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, enabled):
        self.enabled = enabled
        self.tty = enabled and sys.stderr.isatty()
        self.stages = []

    def say(self, msg=""):
        if self.enabled:
            print(msg, flush=True)

    @contextlib.contextmanager
    def phase(self, name, seconds=None):
        """Yields a dict for the timed loop to put its start ("t0") and its list of call times ("times") in, which
        the bar is drawn from - until then, or without seconds, a spinner is"""
        live = {}
        t0 = time.time()
        stop = threading.Event()
        thread = None
        if self.tty:
            thread = threading.Thread(target=self._draw, args=(name, seconds, live, t0, stop), daemon=True)
            thread.start()
        elif self.enabled:
            print(f"{name} ...", flush=True)
        ok = False
        try:
            yield live
            ok = True
        finally:
            stop.set()
            if thread is not None:
                thread.join()
            took = time.time() - t0
            self.stages.append((name, took))
            times = live.get("times")
            calls = f", {len(times)} calls, median {statistics.median(times):.2f}ms" if times else ""
            self.say(f"{name}: done in {fmt_secs(took)}{calls}" if ok else f"{name}: failed after {fmt_secs(took)}")

    def _draw(self, name, seconds, live, t0, stop):
        bar, i, err = None, 0, sys.stderr
        while not stop.wait(0.25):
            i += 1
            if seconds and "t0" in live:
                done = min(time.time() - live["t0"], seconds)
                calls = f"{len(live['times'])} calls" if "times" in live else ""
                if tqdm is not None:
                    if bar is None:
                        err.write("\r\033[K")
                        # a width, as tqdm drops the bar and this format in a terminal that reports none
                        bar = tqdm(total=seconds, desc=name, leave=False, file=err,
                                   ncols=min(100, shutil.get_terminal_size((100, 24)).columns),
                                   bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n:.0f}/{total:.0f}s{postfix}")
                    bar.n = done
                    bar.set_postfix_str(calls, refresh=False)
                    bar.refresh()
                    continue
                calls = f", {calls}" if calls else ""
                filled = int(30 * done / seconds)
                err.write(f"\r\033[K{name}: [{'#' * filled}{'.' * (30 - filled)}] {done:.0f}/{seconds:.0f}s{calls}")
            else:
                what = ": warming up" if seconds else ""
                err.write(f"\r\033[K{self.SPINNER[i % len(self.SPINNER)]} {name}{what} {time.time() - t0:.0f}s")
            err.flush()
        if bar is not None:
            bar.close()
        err.write("\r\033[K")
        err.flush()


def get_args():
    p = argparse.ArgumentParser(description="Find straggler GPUs", formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--layers", type=int, default=32, help="decoder layers (default: 32)")
    p.add_argument("--hidden", type=int, default=4096, help="hidden size (default: 4096)")
    p.add_argument("--ffn", type=int, default=14336, help="MLP intermediate size (default: 14336)")
    p.add_argument("--heads", type=int, default=32, help="attention heads (default: 32)")
    p.add_argument("--kv-heads", type=int, default=8, help="key/value heads (default: 8)")
    p.add_argument("--seqlen", type=int, default=8192, help="sequence length (default: 8192)")
    p.add_argument("--batch", type=int, default=1, help="sequences per step (default: 1)")
    p.add_argument("--idle-ms", type=float, default=100, help="idle time after each step, like a wait for communication or data (default: 100)")
    p.add_argument("--warmup-steps", type=int, default=3, help="untimed steps, for the first calls' setup (default: 3)")
    p.add_argument("--duration", type=float, default=60, help="seconds of timed steps (default: 60)")
    p.add_argument("--window", type=float, default=2, help="seconds of each window the GPUs are compared in (default: 2)")
    p.add_argument("--matmul-shape", default="8192x8192x8192", help="MxNxK of the sustained matmul (default: 8192x8192x8192)")
    p.add_argument("--copy-mib", type=int, default=1024, help="MiB of each buffer of the copy and the stream (default: 1024)")
    p.add_argument("--other-duration", type=float, default=30, help="seconds of the timed matmul, copy and stream, each (default: 30)")
    p.add_argument("--outlier", type=float, default=3.5, help="flag GPUs above this modified z-score of a workload's 90th percentile window (default: 3.5)")
    p.add_argument("--threshold", type=float, default=1, help="...and only if they are more than this many %% above the median (default: 1)")
    p.add_argument("--unsteady", type=float, default=1.25, help="also flag GPUs whose 90th percentile window is more than this many %% slower than their own median window, if an outlier by --outlier (default: 1.25)")
    p.add_argument("--stall", type=float, default=5, help="leave out of a node's comparisons, and report as a stall, every window in which its median GPU is more than this many %% slower than in its median window (default: 5)")
    p.add_argument("--slow-node", type=float, default=3, help="flag nodes whose median GPU is more than this many %% slower than the median node's, or the faster node's of 2, or than the baseline's median node's (default: 3)")
    p.add_argument("--baseline", nargs="+", default=None, help="the --json files of runs on known-good nodes: also flag GPUs whose median time in a workload is more than --threshold slower than their own in them, and nodes whose median GPU's is more than --slow-node slower than their median node's")
    p.add_argument("--repeat", type=int, default=1, help="run all the workloads this many times, with --repeat-idle between, report each run, and flag GPUs more than --threshold slower in a run than in their fastest (default: 1)")
    p.add_argument("--repeat-idle", type=float, default=120, help="seconds idle between the repeats, so that each starts from idle (default: 120)")
    p.add_argument("--sample-interval", type=float, default=0.01, help="telemetry sampling period in seconds (default: 0.01)")
    p.add_argument("--json", default=None, help="also write every GPU's results, with each step's time, to this file")
    p.add_argument("--hostfile-to-sort", default=None, help="a hostfile listing the nodes measured, one per line, e.g. `10.0.0.1 slots=8`: write its lines, the fastest node first, to the same path with -fast-to-slow appended, so that e.g. `head -4` of it picks the 4 fastest nodes")
    p.add_argument("--unflagged-rows", type=int, default=5, help="how many unflagged GPUs and nodes to print, the slowest in the step's 90th percentile window, besides every flagged one and each workload's slowest and least steady unflagged one - once that leaves out at least 10 rows; -1 prints every row (default: 5)")
    args = p.parse_args()
    args.matmul_shape = [int(d) for d in args.matmul_shape.lower().split("x")]
    if len(args.matmul_shape) != 3:
        p.error("--matmul-shape takes MxNxK, e.g. 8192x8192x8192")
    if args.repeat < 1:
        p.error("--repeat must be at least 1")
    args.baseline_medians = load_baseline(args, p) if args.baseline else None
    # read and written by rank 0 alone, so it only has to be on rank 0's node
    if args.hostfile_to_sort and os.environ.get("RANK", "0") == "0" and not os.path.isfile(args.hostfile_to_sort):
        p.error(f"--hostfile-to-sort {args.hostfile_to_sort}: no such file")
    return args


# the arguments that change what each workload computes, which must match the baseline's
WORKLOAD_ARGS = ("layers", "hidden", "ffn", "heads", "kv_heads", "seqlen", "batch", "idle_ms", "matmul_shape", "copy_mib")
# each workload by its time, so that + is slower in all 4: the per-call times and starts, and the median time
WORKLOADS = {"step": ("step_ms", "step_starts", "ms"), "matmul": ("matmul_call_ms", "matmul_starts", "matmul_ms"),
             "copy": ("copy_call_ms", "copy_starts", "copy_ms"), "stream": ("stream_call_ms", "stream_starts", "stream_ms")}
MEDIAN_KEYS = {w: key for w, (_, _, key) in WORKLOADS.items()}


def load_baseline(args, p):
    """From the baseline files' GPUs, over all their nodes and repeats: each workload's median time of each GPU by
    UUID, and of the median node's median GPU, every GPU's 90th percentile window and window-to-window variation
    vs its node's median GPU, the number of node runs, and the GPU name"""
    runs = []
    for path in args.baseline:
        with open(path) as f:
            b = json.load(f)
        differ = [k for k in WORKLOAD_ARGS if k in b["args"] and b["args"][k] != getattr(args, k)]
        if differ:
            p.error(f"--baseline {path} measured different workloads: "
                    + ", ".join(f"--{k} {b['args'][k]} there vs {getattr(args, k)} here" for k in differ))
        runs += [b["gpus"]] if "gpus" in b else [run["gpus"] for run in b["repeats"]]
    gpus = [g for run in runs for g in run]
    keys = {w: key for w, key in MEDIAN_KEYS.items() if all(key in g for g in gpus)}
    nodes = [{w: statistics.median(g[key] for g in run if g["host"] == host) for w, key in keys.items()}
             for run in runs for host in sorted(set(g["host"] for g in run))]
    spread = {w: dict(p90=[], stretch=[]) for w in keys}
    for run in runs:
        for host in sorted(set(g["host"] for g in run)):
            mine = [g for g in run if g["host"] == host]
            for w in keys:
                times, starts, _ = WORKLOADS[w]
                if all(starts in g for g in mine):
                    series, _ = window_means(mine, times, starts, args.window)
                    _, keep = node_stalls(series, args.stall)
                    for ratios in vs_reference(series, statistics.median):
                        stats = window_stats([ratios[i] for i in keep])
                        spread[w]["p90"].append(stats["p90"])
                        spread[w]["stretch"].append(stats["stretch"])
    by_uuid = {}
    for g in gpus:
        by_uuid.setdefault(g["uuid"], []).append(g)
    return dict(node={w: statistics.median(n[w] for n in nodes) for w in keys}, nodes=len(nodes), spread=spread,
                own={u: {w: statistics.median(g[key] for g in gs) for w, key in keys.items()}
                     for u, gs in by_uuid.items() if u != "n/a"},
                name=gpus[0]["name"])


def rms_norm(x):
    """A weightless RMSNorm of x, and what rms_norm_backward needs. Without the norms the values grow with each layer
    until bf16 overflows into inf and NaN, whose matmuls draw less power than those of finite values and so run
    faster, which would hide a GPU that is slow at the power cap"""
    out, rstd = torch.ops.aten._fused_rms_norm(x, [x.shape[-1]], None, None)
    return out, (x, rstd)


def rms_norm_backward(saved, dout):
    x, rstd = saved
    return torch.ops.aten._fused_rms_norm_backward(dout, x, [x.shape[-1]], rstd, None, [True, False])[0]


class Layer:
    """A Llama-style decoder layer, the one comm-compute-overlap-bench.py measures: its weights are views into a flat
    parameter buffer and its weight gradients are written into views of a flat gradient buffer, as FSDP and DeepSpeed
    lay them out, and its backward is written out with the weight gradients' matmuls, apart from attention's, which
    autograd does. The norms are weightless"""

    def __init__(self, args, params, grads):
        h, f, heads, kv = args.hidden, args.ffn, args.heads, args.kv_heads
        self.heads, self.kv, self.hd, self.seqlen = heads, kv, h // heads, args.seqlen
        shapes = [((heads + 2 * kv) * self.hd, h), (h, h), (2 * f, h), (h, f)]

        def views(flat):
            out, offset = [], 0
            for rows, cols in shapes:
                out.append(flat[offset:offset + rows * cols].view(rows, cols))
                offset += rows * cols
            return out

        self.w, self.g = views(params), views(grads)

    @staticmethod
    def numel(args):
        h, f, hd = args.hidden, args.ffn, args.hidden // args.heads
        return (args.heads + 2 * args.kv_heads) * hd * h + h * h + 2 * f * h + h * f

    def forward(self, x):
        heads, kv, hd, S = self.heads, self.kv, self.hd, self.seqlen
        Wqkv, Wo, Wgu, Wd = self.w
        T = x.shape[0]
        xn, xg = rms_norm(x)
        q, k, v = F.linear(xn, Wqkv).split([heads * hd, kv * hd, kv * hd], dim=-1)
        q, k, v = [t.view(T // S, S, -1, hd).transpose(1, 2).detach().requires_grad_() for t in (q, k, v)]
        with torch.enable_grad():
            # the KV heads repeated for each group of q heads, as Hugging Face's repeat_kv does, so that SDPA takes
            # its flash attention path
            attn = F.scaled_dot_product_attention(q, k.repeat_interleave(heads // kv, dim=1),
                                                  v.repeat_interleave(heads // kv, dim=1), is_causal=True)
        a = attn.detach().transpose(1, 2).reshape(T, heads * hd)
        h1 = x + F.linear(a, Wo)
        hn, hg = rms_norm(h1)
        gu = F.linear(hn, Wgu)
        g, u = gu.chunk(2, dim=-1)
        out = h1 + F.linear(F.silu(g) * u, Wd)
        self.saved = (xn, xg, q, k, v, attn, a, hn, hg, gu, F.silu(g) * u)
        return out

    def backward(self, dout):
        Wqkv, Wo, Wgu, Wd = self.w
        gWqkv, gWo, gWgu, gWd = self.g
        xn, xg, q, k, v, attn, a, hn, hg, gu, act = self.saved
        self.saved = None
        T = xn.shape[0]
        torch.matmul(dout.t(), act, out=gWd)
        dact = dout @ Wd
        g, u = gu.chunk(2, dim=-1)
        sig = torch.sigmoid(g)
        dgu = torch.cat([dact * u * sig * (1 + g * (1 - sig)), dact * g * sig], dim=-1)
        torch.matmul(dgu.t(), hn, out=gWgu)
        dh1 = dout + rms_norm_backward(hg, dgu @ Wgu)
        torch.matmul(dh1.t(), a, out=gWo)
        dattn = (dh1 @ Wo).view(T // self.seqlen, self.seqlen, self.heads, self.hd).transpose(1, 2)
        dq, dk, dv = torch.autograd.grad(attn, (q, k, v), dattn)
        dqkv = torch.cat([t.transpose(1, 2).reshape(T, -1) for t in (dq, dk, dv)], dim=-1)
        torch.matmul(dqkv.t(), xn, out=gWqkv)
        return dh1 + rms_norm_backward(xg, dqkv @ Wqkv)


class Telemetry:
    """
    Samples one accelerator's clock, power, temperature and clock-limit reasons in a background thread.

    This base class has no vendor backend: it samples nothing, so only the timings are reported - which is all that
    is needed to find a straggler, on any accelerator torch supports. To add a vendor, subclass it the way
    NVIDIATelemetry does: open the device in __init__ and fill self.info, return one sample from read(), and pick
    the subclass in get_telemetry(). A "power_cap" reason gets its own column in the report, others are listed by name.
    """

    backend = None

    def __init__(self, props):
        uuid = getattr(props, "uuid", None)
        self.info = dict(uuid=str(uuid) if uuid is not None else "n/a", index=None, serial="n/a", pci_bus_id="n/a")
        self.samples = []
        # set while a step runs, so that the idle gaps between steps don't count towards the clock and power
        self.busy = False

    def read(self):
        """One sample: (clock MHz, power W, temperature C, the names of the clock-limit reasons that are active)"""
        return None

    def _run(self, interval):
        while not self.stop.is_set():
            if self.busy:
                sample = self.read()
                if sample is not None and self.busy:
                    self.samples.append(sample)
            self.stop.wait(interval)

    def start(self, interval):
        """Samples until finish(), which summarizes these samples only - one start/finish per workload"""
        if self.backend is None:
            return
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(interval,), daemon=True)
        self.thread.start()

    def finish(self):
        if self.backend is None:
            return {}
        self.stop.set()
        self.thread.join()
        if not self.samples:
            return {}
        clocks, power, temps, reasons = zip(*self.samples)
        names = sorted(set().union(*reasons))
        return dict(
            clock_mhz=statistics.median(clocks),
            power_w=statistics.median(power),
            temp_max=max(temps),
            reasons={name: sum(name in r for r in reasons) / len(reasons) for name in names},
        )


class NVIDIATelemetry(Telemetry):
    """NVML, from `pip install nvidia-ml-py`"""

    backend = "nvml"

    # the clock-limit reasons that slow a GPU down, by bit
    REASONS = {
        0x4: "power_cap",
        0x8: "hw_slowdown",
        0x20: "sw_thermal",
        0x40: "hw_thermal",
        0x80: "hw_power_brake",
    }

    def __init__(self, props):
        super().__init__(props)
        import pynvml as m
        self.m = m
        m.nvmlInit()
        self.info["uuid"] = f"GPU-{props.uuid}"
        self.handle = m.nvmlDeviceGetHandleByUUID(self.info["uuid"])
        self.info["index"] = m.nvmlDeviceGetIndex(self.handle)
        bus_id = m.nvmlDeviceGetPciInfo(self.handle).busId
        self.info["pci_bus_id"] = bus_id.decode() if isinstance(bus_id, bytes) else bus_id
        try:
            self.info["serial"] = m.nvmlDeviceGetSerial(self.handle)
        except m.NVMLError:
            pass
        self.reasons_fn = getattr(m, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            m.nvmlDeviceGetCurrentClocksThrottleReasons

    def read(self):
        m = self.m
        try:
            reasons = self.reasons_fn(self.handle)
            return (
                m.nvmlDeviceGetClockInfo(self.handle, m.NVML_CLOCK_SM),
                m.nvmlDeviceGetPowerUsage(self.handle) / 1000,
                m.nvmlDeviceGetTemperature(self.handle, m.NVML_TEMPERATURE_GPU),
                {name for bit, name in self.REASONS.items() if reasons & bit},
            )
        except m.NVMLError:
            return None


def get_telemetry(device_type, props):
    if device_type == "cuda" and torch.version.hip is None:
        try:
            return NVIDIATelemetry(props)
        except Exception as e:
            print(f"no NVML telemetry, reporting the timings only: {e}", file=sys.stderr)
    return Telemetry(props)


def setup(local_rank, args, progress):
    """Allocates and fills every workload's buffers, and returns the function that runs them all once"""
    device_type = torch.accelerator.current_accelerator().type
    torch.accelerator.set_device_index(local_rank)
    dev = torch.device(device_type, local_rank)
    props = getattr(torch, device_type).get_device_properties(local_rank)
    telemetry = get_telemetry(device_type, props)

    torch.manual_seed(42)
    n = Layer.numel(args)
    params = torch.empty(args.layers * n, dtype=torch.bfloat16, device=dev)
    grads = torch.empty_like(params)
    # random values, as a matmul of zeros or of repeating values draws less power, and so keeps a higher clock
    params.normal_(std=0.02)
    layers = [Layer(args, params[i * n:(i + 1) * n], grads[i * n:(i + 1) * n]) for i in range(args.layers)]
    x = torch.randn(args.batch * args.seqlen, args.hidden, dtype=torch.bfloat16, device=dev)
    dout = torch.randn_like(x) * 0.01

    def step():
        y = x
        with torch.no_grad():
            for layer in layers:
                y = layer.forward(y)
            dy = dout
            for layer in reversed(layers):
                dy = layer.backward(dy)
        return y, dy

    def timed(fn, seconds=float("inf"), count=None, idle_ms=0, live=None):
        """The time of each call of fn in ms, and its start in seconds from the first call's - and, for the
        progress bar, the start and the list of times in live"""
        times, starts, out = [], [], None
        t0 = time.time()
        if live is not None:
            live.update(t0=t0, times=times)
        while time.time() < t0 + seconds and (count is None or len(times) < count):
            start, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
            starts.append(time.time() - t0)
            telemetry.busy = True
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            telemetry.busy = False
            times.append(start.elapsed_time(end))
            if idle_ms:
                time.sleep(idle_ms / 1000)
        return times, starts, out

    def measure(fn, seconds, idle_ms=0, live=None):
        dist.barrier()
        telemetry.start(args.sample_interval)
        times, starts, out = timed(fn, seconds, idle_ms=idle_ms, live=live)
        return times, starts, out, telemetry.finish()

    M, N, K = args.matmul_shape
    a = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
    c = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    # several per timed call, so that each call is a few ms long and the launch gaps don't count
    matmuls = 10

    def matmul():
        for _ in range(matmuls):
            torch.matmul(a, b, out=c)

    src = torch.randn(args.copy_mib * 2**20 // 2, dtype=torch.bfloat16, device=dev)
    dst = torch.empty_like(src)
    out = torch.empty_like(src)
    copies = 5

    def copy():
        for _ in range(copies):
            dst.copy_(src)

    def stream():
        for _ in range(copies):
            torch.add(src, dst, out=out)

    def all_workloads(prefix=""):
        # only a few untimed steps: a straggler can be slow mostly in the first tens of seconds after the load
        # starts - GPU 3 of an 8x B200 node was 3-4% slower than its 7 siblings for 20-25s from idle, then 1%
        # slower - so a long warmup can hide it
        with progress.phase(f"{prefix}step", args.duration) as live:
            dist.barrier()
            timed(step, count=args.warmup_steps, idle_ms=args.idle_ms)
            times, starts, (y, dy), tel = measure(step, args.duration, args.idle_ms, live)
        with progress.phase(f"{prefix}matmul", args.other_duration) as live:
            timed(matmul, count=3)
            mm_times, mm_starts, _, mm_tel = measure(matmul, args.other_duration, live=live)
        with progress.phase(f"{prefix}copy", args.other_duration) as live:
            timed(copy, count=3)
            cp_times, cp_starts, _, cp_tel = measure(copy, args.other_duration, live=live)
        with progress.phase(f"{prefix}stream", args.other_duration) as live:
            timed(stream, count=3)
            st_times, st_starts, _, st_tel = measure(stream, args.other_duration, live=live)

        # a matmul of inf/NaN draws less power than one of finite values, and so runs faster at the power cap, which
        # would make the times invalid - checked in the last step's output, input gradient and weight gradients, and
        # in the last matmul's output
        finite = bool(torch.stack([t.isfinite().all() for t in (y, dy, grads, c)]).all())

        matmul_ms = statistics.median(mm_times) / matmuls
        copy_ms = statistics.median(cp_times) / copies
        stream_ms = statistics.median(st_times) / copies
        buffer = args.copy_mib * 2**20
        return dict(
            host=socket.gethostname(),
            addrs=node_addrs(),
            rank=dist.get_rank(),
            local_rank=local_rank,
            name=props.name,
            ms=statistics.median(times),
            step_ms=times,
            step_starts=starts,
            steps=len(times),
            matmul_ms=matmul_ms,
            matmul_tflops=2 * M * N * K / matmul_ms / 1e9,
            matmul_call_ms=mm_times,
            matmul_starts=mm_starts,
            # the bytes read and written, as the HBM bandwidth of a spec sheet counts them
            copy_ms=copy_ms,
            copy_gbps=2 * buffer / copy_ms / 1e6,
            copy_call_ms=cp_times,
            copy_starts=cp_starts,
            stream_ms=stream_ms,
            stream_gbps=3 * buffer / stream_ms / 1e6,
            stream_call_ms=st_times,
            stream_starts=st_starts,
            finite=finite,
            **telemetry.info,
            **tel,
            matmul_telemetry=mm_tel,
            copy_telemetry=cp_tel,
            stream_telemetry=st_tel,
        )

    return all_workloads


def node_addrs():
    """The names and addresses a hostfile may list this node by: its hostname and fully qualified name, the addresses
    they resolve to, and the address it reaches rank 0's node from"""
    names = {socket.gethostname(), socket.getfqdn()}
    addrs = set(names)
    for name in names:
        with contextlib.suppress(OSError):
            addrs.update(a[4][0] for a in socket.getaddrinfo(name, None))
    if "MASTER_ADDR" in os.environ:
        # a UDP connect sends nothing, it only picks the interface that routes there
        with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((os.environ["MASTER_ADDR"], 9))
            addrs.add(s.getsockname()[0])
    return sorted(addrs)


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def window_means(results, times_key, starts_key, window):
    """Per GPU, its mean time in each window, over the windows every GPU has calls in - or just its median time if
    the run was shorter than a window - and the windows' start times in seconds. The ranks start each workload
    together, after a barrier, so the windows of all the nodes line up"""
    bins = []
    for r in results:
        b = {}
        for t, s in zip(r[times_key], r[starts_key]):
            b.setdefault(int(s // window), []).append(t)
        bins.append(b)
    common = sorted(set.intersection(*(set(b) for b in bins)))
    if not common:
        return [[statistics.median(r[times_key])] for r in results], [0]
    return [[statistics.mean(b[w]) for w in common] for b in bins], [w * window for w in common]


def node_stalls(series, stall):
    """A node's median GPU in each window, and the windows to keep: those in which it is no more than stall % slower
    than in the node's median window, which is at least half of them"""
    medians = [statistics.median(column) for column in zip(*series)]
    typical = statistics.median(medians)
    return medians, [i for i, m in enumerate(medians) if m <= typical * (1 + stall / 100)]


def window_stats(ratios):
    """A GPU's window ratios as %: the median, the 90th percentile, and the stretch - how much slower the 90th
    percentile window is than the median one. Unlike a standard deviation, neither moves with a window or 2: a
    stall of the whole node, from the host, moved single steps from one window to the next - a healthy GPU's step
    went +5.1% and then -6.1% in 2 adjacent windows - and tripled its standard deviation"""
    median, p90 = statistics.median(ratios), percentile(ratios, 0.9)
    return dict(median=100 * (median - 1), p90=100 * (p90 - 1), stretch=100 * (p90 - median), windows=len(ratios))


def vs_reference(series, reference):
    """Each series divided, window by window, by the reference - e.g. the median - of all the series in that window"""
    refs = [reference(column) for column in zip(*series)]
    return [[x / ref for x, ref in zip(s, refs)] for s in series]


def banner(text):
    """`text` in a frame of asterisks, so that the outcome stands out in the report"""
    rule = "*" * (len(text) + 6)
    return f"\n{rule}\n*  {text}  *\n{rule}\n"


def counted(n, noun):
    return f"{n} {noun}{'' if n == 1 else 's'}"


def found(parts, where=""):
    """The outcome banner's text, given what was flagged, e.g. ["1 GPU", "1 slow node"]"""
    return f"Stragglers Found{where}: " + ", ".join(parts) if parts else f"No Stragglers Found{where}"


def report(results, args):
    def pct(v, ref):
        return 100 * (v / ref - 1)

    # with --repeat the banner comes once, with the outcome of all the runs
    final = args.repeat == 1
    bad = [r for r in results if not r["finite"]]
    if bad:
        if final:
            print(banner("Invalid Results: inf/NaN"))
        for r in bad:
            print(f"error: {r['host']} GPU {r['index']} computed inf/NaN, so its time is meaningless")
        return 2

    workloads = WORKLOADS

    def zscores(values, healthy=()):
        # the modified z-score, which measures the distance from the median in units of the median absolute
        # deviation, so that the straggler itself can't widen the spread it's judged by the way it would a standard
        # deviation. 3.5 is the usual cutoff (Iglewicz and Hoaglin). The spread is that of all the GPUs measured and
        # of the baseline's: 8 GPUs estimate it poorly - the straggler's step was +3.8% in its 90th percentile window
        # at z 3.2 among its node's 8, while 24 healthy GPU runs on another node never went above +1.8%
        pool = list(values) + list(healthy)
        median = statistics.median(pool)
        mad = statistics.median(abs(x - median) for x in pool)
        return [0.6745 * (x - median) / mad if mad else 0.0 for x in values]

    windows = window_stats
    base = args.baseline_medians
    for r in results:
        r["windows"], r["scores"], r["vs_baseline"], r["vs_itself"] = {}, {}, {}, False
        r["slow_in"], r["unsteady_in"], r["below_baseline"] = [], [], []
    nodes = {}
    for r in results:
        nodes.setdefault(r["host"], dict(host=r["host"], gpus=[], windows={}, vs_baseline={}, slow_in=[],
                                         below_baseline=[], stalls={}, keep={}))["gpus"].append(r)
    # with 2 nodes the median node is halfway between them, so a slow node is judged by the faster one
    node_ref, node_ref_name = (statistics.median, "the median node's") if len(nodes) > 2 else (min, "the faster node's")
    for w, (times, starts, key) in workloads.items():
        # each GPU vs its own node's median GPU, so that the nodes' differences - the medians of healthy nodes span
        # 1.1% - don't widen the spread the GPUs are judged by, and the node's median GPU vs the other nodes'
        series, window_starts = window_means(results, times, starts, args.window)
        means = dict(zip((id(r) for r in results), series))
        node_series = []
        for n in nodes.values():
            mine = [means[id(r)] for r in n["gpus"]]
            medians, keep = node_stalls(mine, args.stall)
            n["keep"][w] = keep
            if len(keep) < len(medians):
                stalled = [i for i in range(len(medians)) if i not in keep]
                n["stalls"][w] = dict(starts=[window_starts[i] for i in stalled], windows=len(medians),
                                      worst=pct(max(medians[i] for i in stalled), statistics.median(medians)))
            for r, ratios in zip(n["gpus"], vs_reference(mine, statistics.median)):
                r["windows"][w] = windows([ratios[i] for i in keep])
            node_series.append(medians)
        if len(nodes) > 1:
            # a stalled node's windows move neither the faster node of 2 nor, by much, the median node of 3 or more
            for n, ratios in zip(nodes.values(), vs_reference(node_series, node_ref)):
                n["windows"][w] = windows([ratios[i] for i in n["keep"][w]])
                if n["windows"][w]["p90"] > args.slow_node:
                    n["slow_in"].append(w)
        healthy = base["spread"][w] if base and w in base["spread"] else dict(p90=(), stretch=())
        for r, z in zip(results, zscores([r["windows"][w]["p90"] for r in results], healthy["p90"])):
            r["scores"][w] = z
            if z > args.outlier and r["windows"][w]["p90"] > args.threshold:
                r["slow_in"].append(w)
        # a GPU slow in a stretch of the run, vs itself the rest of the time, whether or not that's slow among the
        # others: the straggler's step was 2.3-2.7% slower in its 90th percentile window than in its median one in
        # the 6 runs in which it was slow for its first 20-25s, while no healthy GPU's went above 0.76% in 16 runs
        for r, z in zip(results, zscores([r["windows"][w]["stretch"] for r in results], healthy["stretch"])):
            r["windows"][w]["stretch_score"] = z
            if z > args.outlier and r["windows"][w]["stretch"] > args.unsteady:
                r["unsteady_in"].append(w)
        # vs runs on known-good nodes, which catches what comparing the GPUs with each other can't: a GPU slower
        # than it was by less than the healthy GPUs differ - on a node whose healthy GPUs were 0.8% faster to 1.7%
        # slower than their median, a GPU had to slow down by 1-4% to be flagged, while a healthy one repeats its
        # median within 0.2% - and a whole node slow, on its own or with all the nodes measured with it. A GPU is
        # compared with itself and a node with the baseline's median node: a single GPU vs the baseline's median
        # GPU would flag healthy ones, which on one node ran up to 2.3% slower than its median GPU in the matmul
        if base and w in base["node"]:
            for r in results:
                own = base["own"].get(r["uuid"], {})
                r["vs_itself"] = w in own
                if w in own:
                    r["vs_baseline"][w] = pct(r[key], own[w])
                    if r["vs_baseline"][w] > args.threshold:
                        r["below_baseline"].append(w)
            for n in nodes.values():
                n["vs_baseline"][w] = pct(statistics.median(r[key] for r in n["gpus"]), base["node"][w])
                if n["vs_baseline"][w] > args.slow_node:
                    n["below_baseline"].append(w)
    for r in results:
        r["score"] = r["scores"]["step"]
        r["straggler"] = bool(r["slow_in"] or r["unsteady_in"] or r["below_baseline"])
        n = nodes[r["host"]]
        r["node_slow_in"], r["node_below_baseline"], r["node_stalls"] = n["slow_in"], n["below_baseline"], n["stalls"]
    flagged = [r for r in results if r["straggler"]]
    flagged_nodes = [n for n in nodes.values() if n["slow_in"] or n["below_baseline"]]
    mad_step = statistics.median(abs(r["ms"] - statistics.median(x["ms"] for x in results)) for r in results)

    def summary(unit, value):
        v = sorted(value(r) for r in results)
        return (f"median {statistics.median(v):.0f} {unit}, slowest {v[0]:.0f}, fastest {v[-1]:.0f}, "
                f"spread {pct(v[-1], v[0]):.1f}%")

    ms_sorted = sorted(r["ms"] for r in results)
    M, N, K = args.matmul_shape
    print(f"\n{len(results)} GPUs ({results[0]['name']}), {len(set(r['host'] for r in results))} node(s)")
    print(f"step: {args.layers}x decoder layer h={args.hidden} ffn={args.ffn} heads={args.heads}/{args.kv_heads}, "
          f"{args.batch}x{args.seqlen} tokens, bf16, forward+backward, {args.idle_ms:g}ms idle after each step - "
          f"median {statistics.median(ms_sorted):.2f}ms, fastest {ms_sorted[0]:.2f}ms, slowest {ms_sorted[-1]:.2f}ms, "
          f"spread {pct(ms_sorted[-1], ms_sorted[0]):.1f}%, median absolute deviation {mad_step:.2f}ms")
    print(f"matmul: {M}x{N}x{K} bf16 back to back for {args.other_duration:g}s - "
          + summary("TFLOPS", lambda r: r["matmul_tflops"]))
    print(f"copy: a {args.copy_mib}MiB cudaMemcpy within each GPU's memory for {args.other_duration:g}s - "
          + summary("GB/s read+written", lambda r: r["copy_gbps"]))
    print(f"stream: c = a + b on {args.copy_mib}MiB buffers, on the SMs, for {args.other_duration:g}s - "
          + summary("GB/s read+written", lambda r: r["stream_gbps"]))
    width = max(len(r["host"]) for r in results)

    def cell(x, w):
        return f"{x['windows'][w]['median']:+5.1f}/{x['windows'][w]['p90']:+5.1f}%"

    def node_marks(n):
        return [f"{w} slow" for w in n["slow_in"]] + [f"{w} below baseline" for w in n["below_baseline"]]

    def rows(items, flagged_items):
        """The rows to print, slowest step first - every flagged one, the --unflagged-rows slowest unflagged ones in
        the step, and each workload's slowest and least steady unflagged one - and the rest: on a thousand GPUs every
        row would bury the few that matter. The unflagged rows are the spread a flagged one is told apart from"""
        ordered = sorted(items, key=lambda x: -x["windows"]["step"]["p90"])
        flagged_ids = {id(x) for x in flagged_items}
        others = [x for x in ordered if id(x) not in flagged_ids]
        if args.unflagged_rows < 0 or not others:
            return ordered, []
        keep = flagged_ids | {id(x) for x in others[:args.unflagged_rows]}
        for w in workloads:
            keep.add(id(max(others, key=lambda x: x["windows"][w]["p90"])))
            keep.add(id(max(others, key=lambda x: x["windows"][w]["stretch"])))
        if len(items) - len(keep) < 10:
            return ordered, []
        return [x for x in ordered if id(x) in keep], [x for x in ordered if id(x) not in keep]

    def rest(hidden, what):
        if hidden:
            p90 = [x["windows"]["step"]["p90"] for x in hidden]
            print(f"... and {len(hidden)} more {what}, step {min(p90):+.1f} to {max(p90):+.1f}% in their 90th percentile "
                  f"windows (--unflagged-rows -1 prints them)")

    if base:
        note = "" if base["name"] == results[0]["name"] else f", measured on {base['name']}"
        known = sum(r["vs_itself"] for r in results)
        print(f"vs the baseline {', '.join(args.baseline)}{note}, + is slower: {known} of the {len(results)} GPUs "
              f"vs themselves in it, and each node's median GPU vs the median node's of its {base['nodes']} node runs")
        for n in nodes.values():
            print(f"  {n['host']}: " + ", ".join(f"{w} {v:+.1f}%" for w, v in n["vs_baseline"].items()))
    if len(nodes) > 1:
        print(f"each node's median GPU vs {node_ref_name} in the same {args.window:g}s window, + is slower: "
              f"the median window / the 90th percentile window\n")
        print(f"{'node':<{width}}  " + "  ".join(f"{w:>12}" for w in workloads))
        shown, hidden = rows(list(nodes.values()), flagged_nodes)
        for n in shown:
            line = f"{n['host']:<{width}}  " + "  ".join(cell(n, w) for w in workloads)
            if node_marks(n):
                line += f"  <-- slow node: {', '.join(node_marks(n))}"
            print(line)
        rest(hidden, "nodes")
        print()
    stalled = [n for n in nodes.values() if n["stalls"]]
    if stalled:
        print(f"stalls - windows in which a node's median GPU was more than {args.stall:g}% slower than in its median "
              f"window, so all its GPUs slowed at once, which no single GPU can cause; left out of that node's "
              f"comparisons (--stall):")
        for n in stalled:
            print(f"  {n['host']}: " + "; ".join(
                f"{w} {len(s['starts'])} of {s['windows']} windows, up to {s['worst']:+.1f}%, at "
                + ", ".join(f"{t:g}" for t in s["starts"]) + "s" for w, s in n["stalls"].items()))
        print()
    print(f"each workload in {args.window:g}s windows, each GPU's time vs its node's median GPU's in the same window, "
          f"+ is slower: the median window / the 90th percentile window\n")

    header = f"{'node':<{width}}  GPU   step ms" + "".join(f"  {w:>12}" for w in workloads)
    telemetry = any("clock_mhz" in r for r in results)
    if telemetry:
        header += "  step MHz  matmul MHz  power W  max C  power-capped  other limits"
    print(header)

    shown, hidden = rows(results, flagged)
    for r in shown:
        # with telemetry: the vendor's index (nvidia-smi's on NVIDIA), else the process's device ordinal
        idx = r["index"] if r["index"] is not None else r["local_rank"]
        line = f"{r['host']:<{width}}  {idx!s:>3}  {r['ms']:8.2f}  " + "  ".join(cell(r, w) for w in workloads)
        if telemetry:
            mm_clock = r["matmul_telemetry"].get("clock_mhz")
            mm_clock = f"{mm_clock:10.0f}" if mm_clock is not None else f"{'n/a':>10}"
            capped = r.get("reasons", {}).get("power_cap", 0)
            others = ", ".join(f"{k} {100 * v:.0f}%" for k, v in r.get("reasons", {}).items() if k != "power_cap" and v)
            line += (f"  {r.get('clock_mhz', 0):8.0f}  {mm_clock}  {r.get('power_w', 0):7.0f}"
                     f"  {r.get('temp_max', 0):5.0f}  {100 * capped:11.0f}%  {others}")
        if r["straggler"]:
            marks = ([f"{w} slow" for w in r["slow_in"]] + [f"{w} unsteady" for w in r["unsteady_in"]]
                     + [f"{w} below baseline" for w in r["below_baseline"]])
            line += f"  <-- straggler: {', '.join(marks)}"
        print(line.rstrip())
    rest(hidden, "GPUs")

    among = "the GPUs measured and the baseline's" if base else "the GPUs measured"
    rules = [f"slow: a 90th percentile window above a modified z-score of {args.outlier:g} among {among}, and more "
             f"than {args.threshold:g}% slower than its node's median GPU (--outlier, --threshold)",
             f"unsteady: a 90th percentile window more than {args.unsteady:g}% slower than its own median window, "
             f"above a modified z-score of {args.outlier:g} among {among} (--unsteady)"]
    if base:
        rules.append(f"below baseline: a median time more than {args.threshold:g}% slower than its own in the "
                     f"baseline (--threshold)")
    if len(nodes) > 1:
        rules.append(f"slow node: its median GPU's 90th percentile window more than {args.slow_node:g}% slower "
                     f"than {node_ref_name} (--slow-node)")
    if base:
        rules.append(f"node below baseline: its median GPU's median time more than {args.slow_node:g}% slower "
                     f"than the baseline's median node's (--slow-node)")
    caveat = ("\nwith one node and no --baseline a node whose GPUs are all equally slow passes - run on several nodes "
              "at once, or pass --baseline" if len(nodes) == 1 and not base else "")
    if not flagged and not flagged_nodes:
        parts = []
        for w in workloads:
            s = max(results, key=lambda r: r["windows"][w]["p90"])
            u = max(results, key=lambda r: r["windows"][w]["stretch"])
            part = (f"{w}: slowest {s['host']} GPU {s['index']} {s['windows'][w]['p90']:+.1f}% "
                    f"(z {s['scores'][w]:.1f}), least steady {u['host']} GPU {u['index']} "
                    f"+{u['windows'][w]['stretch']:.1f}% (z {u['windows'][w]['stretch_score']:.1f})")
            known = [r for r in results if w in r["vs_baseline"]]
            if known:
                b = max(known, key=lambda r: r["vs_baseline"][w])
                part += f", vs itself in the baseline {b['host']} GPU {b['index']} {b['vs_baseline'][w]:+.1f}%"
            if len(nodes) > 1:
                n = max(nodes.values(), key=lambda n: n["windows"][w]["p90"])
                part += f", slowest node {n['host']} {n['windows'][w]['p90']:+.1f}%"
            if base and w in base["node"]:
                n = max(nodes.values(), key=lambda n: n["vs_baseline"][w])
                part += f", node vs baseline {n['host']} {n['vs_baseline'][w]:+.1f}%"
            parts.append(part)
        print((banner(found([])) + "\n" if final else "\nno straggler - ") +
              "per workload, the slowest 90th percentile window, and the largest gap between a GPU's "
              "90th percentile and median windows:\n  " + "\n  ".join(parts) + "\na straggler is any of\n  " + "\n  ".join(rules) + caveat)
        return 0

    if final:
        print(banner(found([counted(n, noun) for n, noun in ((len(flagged), "GPU"),
                                                            (len(flagged_nodes), "slow node")) if n])), end="")
    if flagged_nodes:
        print(f"\n{len(flagged_nodes)} slow node(s) - report them by their GPUs' serials, which don't change with the "
              f"node name:")
        for n in flagged_nodes:
            reasons = ([f"slow in {w}: {n['windows'][w]['p90']:+.1f}% in its 90th percentile window, "
                        f"{n['windows'][w]['median']:+.1f}% in its median one" for w in n["slow_in"]]
                       + [f"below baseline in {w}: {n['vs_baseline'][w]:+.1f}%" for w in n["below_baseline"]])
            print(f"  {n['host']}: {'; '.join(reasons)} - GPU serials "
                  f"{', '.join(str(r['serial']) for r in sorted(n['gpus'], key=lambda r: str(r['index'])))}")
    if not flagged:
        print("a straggler is any of\n  " + "\n  ".join(rules) + caveat)
        return 3
    print(f"\n{len(flagged)} straggler(s) - report them by UUID and serial, which don't change with the node name or "
          f"the GPU's index:")
    for r in sorted(flagged, key=lambda r: -r["windows"]["step"]["p90"]):
        reasons = [f"slow in {w}: {r['windows'][w]['p90']:+.1f}% in its 90th percentile window, "
                   f"{r['windows'][w]['median']:+.1f}% in its median one (z {r['scores'][w]:.1f})"
                   for w in r["slow_in"]]
        reasons += [f"unsteady in {w}: its 90th percentile window {r['windows'][w]['stretch']:.1f}% slower than its "
                    f"median one (z {r['windows'][w]['stretch_score']:.1f})" for w in r["unsteady_in"]]
        reasons += [f"below baseline in {w}: {r['vs_baseline'][w]:+.1f}%" for w in r["below_baseline"]]
        memory = ""
        if "copy" in r["slow_in"] and "stream" in r["slow_in"]:
            memory = " - slow in both memory workloads, so its HBM is slow"
        elif "stream" in r["slow_in"]:
            memory = " - slow in stream but not in copy, so its SMs are slow, not its HBM"
        print(f"  {r['host']} GPU {r['index']}: {'; '.join(reasons)}{memory} - "
              f"uuid {r['uuid']}, serial {r['serial']}, pci {r['pci_bus_id']}")
    print("a straggler is any of\n  " + "\n  ".join(rules) + caveat)
    return 3


def node_paces(results, window):
    """Each node's pace in a job that runs its GPUs in lockstep, in which every GPU waits for the slowest one: its
    slowest GPU's mean step time in each window, at the 90th percentile window, as the GPUs themselves are judged"""
    series, _ = window_means(results, "step_ms", "step_starts", window)
    by_host = {}
    for r, s in zip(results, series):
        by_host.setdefault(r["host"], []).append(s)
    return {host: percentile([max(column) for column in zip(*gpus)], 0.9) for host, gpus in by_host.items()}


def sort_nodes(repeats, flagged, args):
    """Print the nodes fastest first, and with --hostfile-to-sort write that hostfile's lines in the same order. A
    node with a flagged GPU, or flagged itself, goes after all the others, so that the first lines are a subset of
    nodes the finder passed"""
    paces = {}
    for results in repeats:
        for host, pace in node_paces(results, args.window).items():
            paces[host] = max(pace, paces.get(host, 0))
    order = sorted(paces, key=lambda h: (h in flagged, paces[h]))
    fastest = paces[order[0]]
    runs = f", in the slowest of its {len(repeats)} runs" if len(repeats) > 1 else ""
    width = max(len(h) for h in order)
    print(f"\nnodes fastest first, by their pace in a job that runs the GPUs in lockstep: the slowest GPU's step in each "
          f"{args.window:g}s window, at the 90th percentile window{runs} - a node with a flagged GPU, or flagged "
          f"itself, after the others:")
    for h in order:
        print(f"  {h:<{width}}  {paces[h]:8.2f}ms  {100 * (paces[h] / fastest - 1):+5.1f}%"
              + ("  flagged" if h in flagged else ""))
    if not args.hostfile_to_sort:
        return [dict(host=h, pace_ms=paces[h], flagged=h in flagged) for h in order]

    addrs = {}
    for r in repeats[0]:
        addrs.setdefault(r["host"], set()).update(r.get("addrs", [r["host"]]))
    with open(args.hostfile_to_sort) as f:
        lines = [line.rstrip("\n") for line in f]
    matched, unknown = {}, []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name = line.split()[0].rsplit("@", 1)[-1]
        names = {name}
        with contextlib.suppress(OSError):
            names.update(a[4][0] for a in socket.getaddrinfo(name, None))
        hosts = [h for h in order if names & addrs[h] or name.split(".")[0] == h.split(".")[0]]
        if hosts and hosts[0] not in matched:
            matched[hosts[0]] = line
        else:
            unknown.append(line)
    out = args.hostfile_to_sort + "-fast-to-slow"
    sorted_lines = [matched[h] for h in order if h in matched]
    try:
        with open(out, "w") as f:
            f.write("".join(f"{line}\n" for line in sorted_lines))
        print(f"wrote {len(sorted_lines)} of the lines of {args.hostfile_to_sort} in that order to {out}")
    except OSError as e:
        print(f"couldn't write {out} ({e}) - its lines would be:\n" + "\n".join(sorted_lines))
    if unknown:
        print(f"left out {len(unknown)} line(s) of {args.hostfile_to_sort} that name no node measured, or one "
              f"already listed: " + "; ".join(unknown))
    missing = [h for h in order if h not in matched]
    if missing:
        print(f"no line of {args.hostfile_to_sort} names {', '.join(missing)}, measured but left out of {out}")
    return [dict(host=h, pace_ms=paces[h], flagged=h in flagged, line=matched.get(h)) for h in order]


def worker(local_rank, args, world_size, init_method):
    if init_method is not None:
        dist.init_process_group("gloo", init_method=init_method, rank=local_rank, world_size=world_size,
                                timeout=datetime.timedelta(minutes=30))
    else:
        dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=30))
    t_start = time.time()
    progress = Progress(dist.get_rank() == 0)
    # and about 20s for the setup and the warmups
    total = 20 + args.repeat * (args.duration + 3 * args.other_duration) + (args.repeat - 1) * args.repeat_idle
    runs = f", {args.repeat} times {args.repeat_idle:g}s apart" if args.repeat > 1 else ""
    progress.say(f"\nmeasuring {dist.get_world_size()} GPUs at once: the training step for {args.duration:g}s, then "
                 f"a matmul, a memory copy and a memory stream for {args.other_duration:g}s each{runs} - about "
                 f"{fmt_secs(total)} in all\n")
    with progress.phase("setup: allocating and filling the buffers"):
        all_workloads = setup(local_rank, args, progress)

    repeats, codes = [], []
    for i in range(args.repeat):
        if args.repeat > 1:
            progress.say(f"\n######## run {i + 1} of {args.repeat}\n")
            if i:
                with progress.phase(f"idle before run {i + 1}", args.repeat_idle) as live:
                    dist.barrier()
                    live["t0"] = time.time()
                    time.sleep(args.repeat_idle)
        prefix = f"run {i + 1}: " if args.repeat > 1 else ""
        result = all_workloads(prefix)
        with progress.phase(f"{prefix}gathering the results"):
            results = [None] * dist.get_world_size()
            dist.all_gather_object(results, result)
        repeats.append(results)
        if dist.get_rank() == 0:
            t0 = time.time()
            codes.append(report(results, args))
            progress.stages.append((f"{prefix}report", time.time() - t0))

    rc = 0
    if dist.get_rank() == 0:
        # an inf/NaN invalidates the times, so it outranks a straggler
        rc = 2 if 2 in codes else max(codes)
        stragglers = {}

        def entry(r):
            return stragglers.setdefault(r["rank"], dict(
                {k: r[k] for k in ("host", "index", "uuid", "serial", "pci_bus_id")},
                slow_in=[], unsteady_in=[], below_baseline=[], runs=[], slower_than_itself={}))

        slow_nodes, stalled_nodes = {}, {}
        for i, results in enumerate(repeats):
            for r in results:
                if r.get("node_stalls"):
                    runs = stalled_nodes.setdefault(r["host"], [])
                    if i + 1 not in runs:
                        runs.append(i + 1)
                if r.get("straggler"):
                    s = entry(r)
                    s["runs"].append(i + 1)
                    for k in ("slow_in", "unsteady_in", "below_baseline"):
                        s[k] += [w for w in r[k] if w not in s[k]]
                if r.get("node_slow_in") or r.get("node_below_baseline"):
                    n = slow_nodes.setdefault(r["host"], dict(host=r["host"], slow_in=[], below_baseline=[], runs=[],
                                                              serials={}))
                    n["serials"][r["index"]] = r["serial"]
                    if i + 1 not in n["runs"]:
                        n["runs"].append(i + 1)
                    for k in ("slow_in", "below_baseline"):
                        n[k] += [w for w in r[f"node_{k}"] if w not in n[k]]
        # each GPU vs its own fastest run, which catches one that slowed between runs by less than the healthy GPUs
        # differ from each other, and which a healthy GPU repeats within 0.2% of
        if args.repeat > 1 and rc != 2:
            for rank in range(len(repeats[0])):
                for w, key in MEDIAN_KEYS.items():
                    medians = [results[rank][key] for results in repeats]
                    slower = [i + 1 for i, m in enumerate(medians) if 100 * (m / min(medians) - 1) > args.threshold]
                    if slower:
                        entry(repeats[0][rank])["slower_than_itself"][w] = slower
            if any(s["slower_than_itself"] for s in stragglers.values()):
                rc = 3
        if args.repeat > 1:
            print(f"\n######## all {args.repeat} runs")
            if rc == 2:
                print(banner("Invalid Results: inf/NaN") + "a GPU computed inf/NaN - see the run it was in")
            else:
                every = sum(len(s["runs"]) == args.repeat for s in stragglers.values())
                some = len(stragglers) - every
                parts = ([f"{counted(every, 'GPU')} in all {args.repeat}"] * bool(every) +
                         [f"{counted(some, 'GPU')} in some"] * bool(some) +
                         [counted(len(slow_nodes), "slow node")] * bool(slow_nodes))
                print(banner(found(parts, f" in {args.repeat} Runs")))
                if not parts:
                    print("no straggler in any run, and no GPU more than "
                          f"{args.threshold:g}% slower in a run than in its fastest one")
            for n in slow_nodes.values():
                print(f"  {n['host']}: a slow node in {len(n['runs'])} of {args.repeat} runs "
                      f"(run {', '.join(map(str, n['runs']))}) - GPU serials "
                      f"{', '.join(str(n['serials'][k]) for k in sorted(n['serials'], key=str))}")
            for s in stragglers.values():
                parts = []
                if s["runs"]:
                    parts.append(f"a straggler in {len(s['runs'])} of {args.repeat} runs "
                                 f"(run {', '.join(map(str, s['runs']))})")
                parts += [f"its {w} more than {args.threshold:g}% slower in run {', '.join(map(str, runs))} than in "
                          f"its fastest run" for w, runs in s["slower_than_itself"].items()]
                print(f"  {s['host']} GPU {s['index']}: {'; '.join(parts)} - uuid {s['uuid']}, serial {s['serial']}")
            for host, runs in stalled_nodes.items():
                print(f"  {host}: stalled in {len(runs)} of {args.repeat} runs (run {', '.join(map(str, runs))}) - "
                      f"a node that stalls in most runs holds a synchronous job up as a straggler does")
        fast_to_slow = None
        # an inf/NaN invalidates the times the nodes are sorted by
        if len({r["host"] for r in repeats[0]}) > 1 and rc != 2:
            fast_to_slow = sort_nodes(repeats, {s["host"] for s in stragglers.values()} | set(slow_nodes), args)
        elif args.hostfile_to_sort:
            print(f"\nnot sorting {args.hostfile_to_sort}: " + ("a GPU returned inf/NaN, which invalidates its times"
                                                                if rc == 2 else "only one node was measured"))
        if args.json:
            with open(args.json, "w") as f:
                out = dict(args=vars(args), exit_code=rc, stragglers=list(stragglers.values()),
                           slow_nodes=list(slow_nodes.values()),
                           stalled_nodes=[dict(host=h, runs=runs) for h, runs in stalled_nodes.items()],
                           fast_to_slow=fast_to_slow)
                if args.repeat == 1:
                    out["gpus"] = repeats[0]
                else:
                    out["repeats"] = [dict(exit_code=c, gpus=g) for c, g in zip(codes, repeats)]
                json.dump(out, f, indent=1)
            print(f"\nwrote every GPU's results to {args.json}")
        stages = {}
        for name, took in progress.stages:
            # the repeats' stages added up, e.g. "run 2: step" into "step"
            name = "idle between runs" if name.startswith("idle before run") else name.split(": ", 1)[-1] \
                if args.repeat > 1 and name.startswith("run ") else name
            stages.setdefault(name, []).append(took)
        stages["total"] = [time.time() - t_start]
        width = max(len(name) for name in stages)
        print("\nruntimes:")
        for name, took in stages.items():
            each = f"  ({len(took)}x {fmt_secs(sum(took) / len(took))})" if len(took) > 1 else ""
            print(f"  {name:<{width}}  {fmt_secs(sum(took)):>8}{each}")
    dist.barrier()
    dist.destroy_process_group()
    return rc


def spawned(local_rank, args, world_size, init_method, rc_queue):
    # rank 0's exit code goes back to the parent, as a process exiting with it would make mp.spawn kill the others
    rc = worker(local_rank, args, world_size, init_method)
    if local_rank == 0:
        rc_queue.put(rc)


def main():
    args = get_args()
    if "LOCAL_RANK" in os.environ:
        # under torchrun a rank exiting with an error makes its node's agent tear the rendezvous down while the other
        # nodes still shut down, so every rank exits with 0 and the verdict is in the report and the --json file
        worker(int(os.environ["LOCAL_RANK"]), args, None, None)
        return

    n = torch.accelerator.device_count()
    if n == 0:
        sys.exit("no accelerators visible")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    rc_queue = mp.get_context("spawn").SimpleQueue()
    mp.spawn(spawned, args=(args, n, f"tcp://127.0.0.1:{port}", rc_queue), nprocs=n)
    sys.exit(rc_queue.get())


if __name__ == "__main__":
    main()
