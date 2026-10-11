#!/usr/bin/env python
"""
Measure MAMF/MSMF the way a real workload sees them: while every other GPU of the node computes too.

MSMF (sustainable) is what a GPU holds while every GPU of the node computes and shares the board's power/cooling
budget; measured with idle siblings it gets headroom a full node never has. So this script measures one GPU at a time
while every GPU not being measured runs a continuous bf16 matmul that keeps it at its power limit:

  1. GPU0: the full mamf-finder.py search (console shown) -> MAMF, MSMF and their shapes
  2. every other GPU in turn: those 1-2 shapes pinned (--shapes_file), while the others, GPU0 included, run the matmul
  3. summary: per-GPU MAMF and MSMF; node MAMF and MSMF = the median across GPUs, so one weak or strong chip doesn't
     set the node's figure; the slowest GPU, which synchronous training runs at, and the spread; warnings for GPUs
     whose MSMF window wasn't steady, ran much hotter than the coolest GPU's, or drew well under the highest power

For a single-GPU measurement with idle siblings, run mamf-finder.py on its own. With only one GPU visible (e.g. a
1-GPU VM) this script runs just step 1, which is the same as running mamf-finder.py on its own.

NVIDIA (CUDA) and AMD (ROCm) only; it exits with an error on any other accelerator.

Usage:

  python mamf-finder-all-gpus.py [mamf-finder.py args ...]

Run it with the python whose torch you want measured: mamf-finder.py and the matmuls run with the same interpreter.

Your arguments go to the mamf-finder.py runs:

  step 1, GPU0:          mamf-finder.py with all your arguments, i.e. the full search you asked for.
  step 2, GPU1, GPU2...: one at a time, mamf-finder.py on GPU0's MAMF and MSMF shapes only (--shapes_file), with your
                         remaining arguments (--dtype etc.). Your shape selection (--m/--n/--k, --*_range,
                         --shapes_file, --search) is dropped there, since the shapes are already chosen.

The script sets --cuda_device, --output_file and --verbose/--no-verbose per run itself, so passing any of them is an
error.

Examples:

  python mamf-finder-all-gpus.py
      bf16, --search auto on GPU0, then its MAMF and MSMF shapes on each other visible GPU in turn.

  python mamf-finder-all-gpus.py --dtype float8_e4m3fn
      The same for fp8.

  python mamf-finder-all-gpus.py --m_range 2048 8193 256 --n 4096 --k 4096
      GPU0 searches only your model's shape range; then each other GPU in turn measures the shapes GPU0 found.

  CUDA_VISIBLE_DEVICES=0,1,2,3 python mamf-finder-all-gpus.py
      Only those 4 GPUs are measured or run the matmul; the rest of the node is left alone. GPU0 here means the first
      visible GPU. On AMD, HIP_VISIBLE_DEVICES does the same.

Environment variables:

  OUT_DIR=<dir>    Where the per-GPU logs and summary.txt go. Default: results/all-gpus-<timestamp> next to
                   this script.
  NUM_GPUS=<n>     Use the first n visible GPUs instead of all of them. Default: every visible GPU.

Output:

  $OUT_DIR/gpu<N>.txt   GPU N's full mamf-finder.py log.
  $OUT_DIR/gpu<N>.err   Anything GPU N's run printed before its log opened, e.g. a traceback (GPUs 1+ only).
  $OUT_DIR/shapes.txt   GPU0's MAMF and MSMF shapes, which the other GPUs measure.
  $OUT_DIR/summary.txt  The table printed at the end: per-GPU MAMF and MSMF, the node medians, slowest GPU, spread,
                        and the warnings.

Ctrl-C stops the current mamf-finder.py gracefully (it still prints its results) and skips the remaining GPUs; the
summary covers the GPUs measured so far. If GPU0 had no MSMF yet, there is nothing to summarize and it exits 1.
"""

import multiprocessing
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINDER = HERE / "mamf-finder.py"
SET_PER_RUN = ("--output_file", "--cuda_device", "--verbose", "--no-verbose")
SHAPE_SELECTION = ("--m", "--n", "--k", "--m_range", "--n_range", "--k_range", "--shapes_file", "--search")
HEADLINE = re.compile(r"^(MAMF|MSMF) \(max [^)]*\):\s+([\d.]+) TFLOPS @ (\d+x\d+x\d+) \(MxNxK\)"
                      r"(?:\s+(\d+)W)?(?:\s+(\d+)MHz)?(?:\s+(\d+)C)?")
IDLE_SHARE = re.compile(r"Share of the time each was idle: (.*)")
UNSTEADY = "WARNING: no confirm shape held a steady rate over its window"
# an MSMF window this much hotter than the coolest GPU's, or below this share of the highest power, gets flagged;
# GPUs of a well-cooled node at their power cap stay within ~13C and a few % of each other
HOT_C = 15
LOW_POWER = 0.9
# a cold torch import from a network filesystem can take a minute, and all the matmuls import it at once
MATMUL_START_TIMEOUT_S = 300

finder = None        # the mamf-finder.py run in progress
interrupted = False  # set by Ctrl-C / TERM: let the current run finish and report, skip the remaining GPUs


def die(msg):
    sys.exit(f"error: {msg}")


def warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


def half_up(x):
    """Integer TFLOPS, rounding a half up like mamf-finder.py's headlines (TFLOPS are never negative)."""
    return int(x + 0.5)


def on_signal(signum, frame):
    global interrupted
    interrupted = True
    if finder:
        finder.send_signal(signal.SIGINT)  # mamf-finder.py stops gracefully on SIGINT and still reports


def gpu_count():
    """Visible GPUs. mamf-finder.py and the matmuls use torch.cuda, which only NVIDIA (CUDA) and AMD (ROCm) have."""
    import torch
    if not (torch.cuda.is_available() and (torch.version.cuda or torch.version.hip)):
        die("this script supports NVIDIA (CUDA) and AMD (ROCm) GPUs only, and torch sees neither; "
            "on other accelerators run mamf-finder.py on each device instead")
    n = os.environ.get("NUM_GPUS") or str(torch.cuda.device_count())
    if not n.isdigit() or int(n) < 1:
        die(f"NUM_GPUS must be a positive integer, got {n!r}")
    return int(n)


def without_shape_selection(args):
    """The user's args minus the shape selection, which the pinned --m/--n/--k replaces."""
    kept, skipping = [], False
    for a in args:
        if a.startswith("--"):
            skipping = a in SHAPE_SELECTION
            if skipping or a.split("=")[0] in SHAPE_SELECTION:
                continue
        if not skipping:
            kept.append(a)
    return kept


def matmul(gpu, started):
    """What every GPU not being measured runs: a continuous bf16 8192x8192 matmul, which pins it at its power limit -
    the saturated clock of a node in training. It synchronizes every 20 matmuls so the queue stays short."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # stop_matmuls() ends it; Ctrl-C is for mamf-finder.py
    import torch
    torch.cuda.set_device(gpu)
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    while True:
        for _ in range(20):
            a @ a
        torch.cuda.synchronize()
        started.set()


def start_matmuls(num_gpus, measured_gpu):
    """Start the matmul on every GPU except `measured_gpu` and wait until they all run."""
    ctx = multiprocessing.get_context("spawn")
    procs = []
    for gpu in range(num_gpus):
        if gpu != measured_gpu:
            started = ctx.Event()
            proc = ctx.Process(target=matmul, args=(gpu, started), daemon=True)
            proc.start()
            procs.append((gpu, proc, started))
    deadline = time.time() + MATMUL_START_TIMEOUT_S
    for gpu, proc, started in procs:
        while not started.wait(1):
            if not proc.is_alive():
                warn(f"the matmul on GPU{gpu} exited (code {proc.exitcode}) before it started; continuing")
                break
            if time.time() > deadline:
                warn(f"not every matmul started within {MATMUL_START_TIMEOUT_S}s; continuing")
                return [p for _, p, _ in procs]
    return [p for _, p, _ in procs]


def stop_matmuls(procs):
    for proc in procs:
        proc.terminate()
    for proc in procs:
        proc.join()


def run_finder(gpu, args, out_dir):
    """Run mamf-finder.py on `gpu` and wait for it to exit; return its exit code."""
    global finder
    log = out_dir / f"gpu{gpu}.txt"
    # the child gets this interpreter's flags, e.g. -I, or it may import other packages than this process does
    cmd = [sys.executable, *subprocess._args_from_interpreter_flags(), str(FINDER), *args,
           "--cuda_device", str(gpu), "--output_file", str(log)]
    if gpu == 0:
        finder = subprocess.Popen(cmd)
    else:
        with open(out_dir / f"gpu{gpu}.err", "w") as err:
            finder = subprocess.Popen([*cmd, "--no-verbose"], stdout=err, stderr=subprocess.STDOUT)
    rc = finder.wait()
    finder = None
    return rc


def measure(gpu, args, num_gpus, out_dir):
    """Measure `gpu` with mamf-finder.py while every other GPU runs the matmul; return its exit code."""
    procs = start_matmuls(num_gpus, gpu)
    try:
        return 1 if interrupted else run_finder(gpu, args, out_dir)
    finally:
        stop_matmuls(procs)


def physical_gpus(num_gpus):
    """Driver indices of the GPUs this run uses - the numbering mamf-finder.py's idle-sibling note uses - or None
    when CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES names them some other way (e.g. by UUID)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
    ids = (visible.split(",") if visible else [str(i) for i in range(num_gpus)])[:num_gpus]
    return {int(i) for i in ids} if all(i.strip().isdigit() for i in ids) else None


def parse_log(path):
    """The MAMF/MSMF headlines of a mamf-finder.py log, how long (%) each idle sibling sat idle, and whether MSMF is
    the fallback to the fastest unsteady window."""
    result = {"idle": {}, "unsteady": False}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith(UNSTEADY):
                result["unsteady"] = True
            elif m := HEADLINE.match(line):
                result[m[1]] = dict(tflops=float(m[2]), shape=m[3], W=m[4] or "-", MHz=m[5] or "-", C=m[6] or "-")
            elif m := IDLE_SHARE.search(line):
                result["idle"] = {int(g): int(p) for g, p in re.findall(r"GPU(\d+) (\d+)%", m[1])}
    return result


def msmf_warnings(rows):
    """Warnings for GPUs whose MSMF isn't a steady rate at the power cap: they pull the node median down for reasons
    of this node (cooling, a throttling GPU) rather than of the GPU model."""
    out = []
    gpus = lambda xs: ",".join(str(g) for g in xs)
    if unsteady := [gpu for gpu, r in enumerate(rows) if r["unsteady"] and "MSMF" in r]:
        out.append(f"\nwarning: GPU(s) {gpus(unsteady)} held no MSMF window steady, so their MSMF is the fastest "
                   "unsteady window, not\n         a sustained rate - usually a GPU throttling as it heats up; see "
                   "their logs")
    msmf = [(gpu, r["MSMF"]) for gpu, r in enumerate(rows) if "MSMF" in r]
    temps = {gpu: int(s["C"]) for gpu, s in msmf if s["C"] != "-"}
    if hot := [gpu for gpu, c in temps.items() if c >= min(temps.values()) + HOT_C]:
        lo, hi = min(temps[g] for g in hot), max(temps[g] for g in hot)
        out.append(f"\nwarning: GPU(s) {gpus(hot)} ran their MSMF window at {lo}-{hi}C, {HOT_C}C+ hotter than the "
                   f"coolest GPU ({min(temps.values())}C):\n         the node cools them unevenly, and a hotter GPU "
                   "clocks lower at the same power, so the node MSMF\n         reflects this node's cooling as well "
                   "as the GPU")
    watts = {gpu: int(s["W"]) for gpu, s in msmf if s["W"] != "-"}
    if low := [gpu for gpu, w in watts.items() if w < LOW_POWER * max(watts.values())]:
        out.append(f"\nwarning: GPU(s) {gpus(low)} drew under {LOW_POWER:.0%} of the node's highest MSMF power "
                   f"({max(watts.values())}W), so something\n         other than the power cap held them back, "
                   "usually temperature")
    return out


def summarize(rows, ours):
    """`ours`: driver indices of the GPUs in this run (None = unknown, treat every idle sibling as one of ours)."""
    how = ("the only GPU, so nothing else ran" if len(rows) == 1 else
           f"{len(rows)} GPUs, each measured while all the others ran a continuous matmul")
    out = [f"\n{'-' * 80}\n",
           f"** Node results ({how}):\n",
           f"  {'GPU':>3}  {'MAMF':>7}  {'MxNxK':<18}  {'MSMF':>7}  {'MxNxK':<18} {'W':>4} {'MHz':>5} {'C':>3}  "
           "sibling idle"]
    outside = set()
    for gpu, r in enumerate(rows):
        r["idle_ours"] = max((p for g, p in r["idle"].items() if ours is None or g in ours), default=0)
        outside |= {g for g, p in r["idle"].items() if ours is not None and g not in ours and p >= 10}
        a = f"{half_up(r['MAMF']['tflops']):7d}  {r['MAMF']['shape']:<18}" if "MAMF" in r else f"{'n/a':>7}  {'':<18}"
        if s := r.get("MSMF"):
            msmf = f"{half_up(s['tflops']):7d}  {s['shape']:<18} {s['W']:>4} {s['MHz']:>5} {s['C']:>3}"
        else:
            msmf = f"{'n/a':>7}  {'':<18} {'':>4} {'':>5} {'':>3}"
        out.append(f"  {gpu:>3}  {a}  {msmf}  {r['idle_ours']:>3}%")

    out.append("")
    for key in ("MAMF", "MSMF"):
        measured = [(gpu, r[key]) for gpu, r in enumerate(rows) if key in r]
        if not measured:
            continue
        lo_gpu, lo = min(measured, key=lambda gm: gm[1]["tflops"])
        hi = max(s["tflops"] for _, s in measured)
        median = statistics.median(s["tflops"] for _, s in measured)
        shape = rows[0].get(key, lo)["shape"]
        out.append(f"{key} (node = median GPU):  {half_up(median)} TFLOPS @ {shape} (GPU0's shape)")
        if len(measured) > 1:
            out.append(f"{key} slowest / spread:     {half_up(lo['tflops'])} TFLOPS (GPU{lo_gpu}) / "
                       f"{100 * (hi - lo['tflops']) / hi:.1f}% across {len(measured)} GPUs")
    measured = [r for r in rows if "MSMF" in r]

    if idle := [str(gpu) for gpu, r in enumerate(rows) if r["idle_ours"] >= 10]:
        out.append(f"\nwarning: GPU(s) {','.join(idle)} saw idle siblings during their MSMF measurement - the "
                   "matmul on a sibling\n         must have stopped; see the note in their logs")
    if outside:
        gpus = ",".join(map(str, sorted(outside)))
        out.append(f"\nnote: driver GPU(s) {gpus} share the board but are left out of this run (CUDA_VISIBLE_DEVICES"
                   " / NUM_GPUS)\n      and sat idle, so MSMF reads higher than with the whole node computing")
    out += msmf_warnings(rows)
    if len(measured) < len(rows):
        out.append(f"\nwarning: only {len(measured)} of {len(rows)} GPUs produced an MSMF result - see the per-GPU "
                   "logs")
    return "\n".join(out)


def main():
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with mamf-finder.py's when piped to a file
    args = sys.argv[1:]
    if args[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    for a in args:
        if a.split("=")[0] in SET_PER_RUN:
            die(f"{a} is set per GPU by this script; drop it")

    num_gpus = gpu_count()
    out_dir = Path(os.environ.get("OUT_DIR") or HERE / "results" / f"all-gpus-{datetime.now():%Y%m%d-%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if num_gpus == 1:
        print(f"mamf-finder.py on the only GPU, the same as running it on its own; logs: {out_dir}")
    else:
        print(f"mamf-finder.py on {num_gpus} GPUs, one at a time, the others running a continuous matmul; "
              f"logs: {out_dir}")
        others = "GPU1" if num_gpus == 2 else f"GPUs 1-{num_gpus - 1}"
        print(f"GPU0: full search while {others} run{'s' if num_gpus == 2 else ''} a continuous matmul")
    measure(0, args, num_gpus, out_dir)
    gpu0 = parse_log(out_dir / "gpu0.txt")
    if "MSMF" not in gpu0:
        die("GPU0 produced no MSMF result, so there is no shape to measure on the other GPUs")
    shapes = list(dict.fromkeys(gpu0[key]["shape"] for key in ("MAMF", "MSMF") if key in gpu0))
    shapes_file = out_dir / "shapes.txt"
    shapes_file.write_text("".join(f"{s}\n" for s in shapes))
    pinned = [*without_shape_selection(args), "--shapes_file", str(shapes_file)]

    failed = []
    if num_gpus > 1 and not interrupted:
        print(f"\nOther GPUs: GPU0's shapes {', '.join(shapes)} pinned, each in turn while all the others run the "
              "matmul")
        for gpu in range(1, num_gpus):
            if interrupted:
                break
            print(f"\r\033[K  GPU{gpu} ({gpu}/{num_gpus - 1}) ...", end="", flush=True)
            if measure(gpu, pinned, num_gpus, out_dir) and not interrupted:
                failed.append(gpu)
        print("\r\033[K", end="")

    for gpu in failed:
        tail = (out_dir / f"gpu{gpu}.err").read_text(errors="replace").splitlines()[-5:]
        print(f"\nGPU{gpu} exited with an error; last lines of {out_dir}/gpu{gpu}.err:", *tail, sep="\n",
              file=sys.stderr)

    summary = summarize([parse_log(out_dir / f"gpu{i}.txt") for i in range(num_gpus)], physical_gpus(num_gpus))
    print(summary)
    (out_dir / "summary.txt").write_text(summary + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
