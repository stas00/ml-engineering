#!/usr/bin/env python

"""

This is the Maximum Achievable / Sustainable Matmul FLOPS Finder
(MAMF + MSMF).

One run of `--search auto` reports two numbers from the same shape search:

- **MAMF** — Maximum *Achievable* Matmul FLOPS: the boost-clock burst a short
  kernel can catch (validated against the SM boost clock).
- **MSMF** — Maximum *Sustainable* Matmul FLOPS: what the chip holds once
  saturated near TDP (matches sustained training throughput).

For a quick grid sweep use:

python mamf-finder.py --m_range 0 20480 256 --n 4096 --k 4096 --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt

But that is usually an insufficient range. For the default 1-shot auto search,
discussion, and important nuances see:
https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks#maximum-achievable-matmul-flops-finder

Results table:
https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#maximum-achievable-matmul-flops-comparison-table

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
import math
import numpy as np
import os
import platform
import re
import shlex
import signal
import sys
import threading
import time
import torch
from packaging import version
from warnings import warn

# important: when changing how the benchmark measures things bump up its version, so that the old
# reports could be differentiated from the new ones. v3: dual MAMF (boost-validated) + MSMF
# (power-saturated) headlines from a single `--search auto` run. v4: reproducibility hardening -
# thermal soak, per-iteration synchronous-clock MAMF bursts, saturated-clock/spread boost filters,
# fat-shape forcing, lock-in validation gate, and a busy-sibling-GPU warning.
benchmark_version = 4

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

    @property
    def name(self):
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

# Shared GEMM setup for benchmark_mm (saturated timing) and measure_boost_burst (per-iter MAMF).
#
# l2_cache: written between iterations to emulate cache reset. On AMD this is really L3/LLC —
# 256MiB is the highest across recent accelerators so far
# (https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#caches).
# C_rand: re-copied into C each iter so the write actually happens (else the rerun is a no-op and
# draws no power — invalid emulation of a real use case).
def prepare_gemm(m, n, k, dtype, device):
    """Allocate operands and return (op, l2_cache, C, C_rand, flos). `op()` writes into C."""
    l2_cache = torch.empty(int(256 * 2**20 / 4), dtype=torch.int, device=device)
    C = torch.empty(m, n, dtype=dtype, device=device).contiguous()
    C_rand = torch.randn(m, n, device=device).to(dtype=dtype).contiguous()

    fp8_dtypes = [torch.float8_e4m3fn, torch.float8_e4m3fnuz]
    if dtype in fp8_dtypes:
        if version.parse(torch.__version__) < version.parse("2.5"):
            raise ValueError("float8 dtypes require torch>=2.5")
        if dtype == torch.float8_e4m3fn and arch.name == "rocm":
            raise ValueError("ROCm doesn't support float8_e4m3fn, use --dtype float8_e4m3fnuz instead")
        A = torch.randn(m, k, dtype=torch.float32, device=device).contiguous().to(dtype)
        B = torch.randn(n, k, dtype=torch.float32, device=device).contiguous().t().to(dtype)
        scale = torch.tensor([1.0]).to(device)
        # must not move `out=C` as `C = ...` — Gaudi needs it this way
        def op():
            torch._scaled_mm(A, B, scale, scale, out=C)
    else:
        A = torch.randn(m, k, dtype=dtype, device=device).contiguous()
        B = torch.randn(n, k, dtype=dtype, device=device).contiguous().t()
        def op():
            torch.mm(A, B, out=C)
    return op, l2_cache, C, C_rand, 2 * m * n * k


def benchmark_mm(m, n, k, dtype, device, num_iterations, num_warmup_iterations, telem=None, idle_before_s=0.0):
    """Saturated matmul timing. Optional idle_before_s is for rare cool-start callers;
    MAMF boost bursts use measure_boost_burst() (per-iter synchronous clock) instead."""
    op, l2_cache, C, C_rand, flos = prepare_gemm(m, n, k, dtype, device)
    total_iterations = num_iterations + num_warmup_iterations
    start_events = [arch.event(enable_timing=True) for _ in range(total_iterations)]
    end_events = [arch.event(enable_timing=True) for _ in range(total_iterations)]

    if idle_before_s and idle_before_s > 0:
        arch.synchronize()
        time.sleep(idle_before_s)

    # sample power/clock in a background thread while the timed loop runs so each shape is
    # self-validating (high TFLOPS at low power = unsaturated boost, not MSMF)
    global _last_telem
    _last_telem = {}
    have_telem = telem_ok(telem)
    _pw, _clk, _stop, _th = [], [], threading.Event(), None
    if have_telem:
        def _sample_loop():
            while not _stop.is_set():
                p, c = telem.power(), telem.clock()
                if p is not None: _pw.append(p)
                if c is not None: _clk.append(c)
                _stop.wait(0.02)
        _th = threading.Thread(target=_sample_loop, daemon=True)
        _th.start()
    try:
        for i in range(total_iterations):
            with torch.no_grad():
                l2_cache.zero_()
                C.copy_(C_rand)
                start_events[i].record()
                op()
                end_events[i].record()
        arch.synchronize()
        times = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
        times = times[num_warmup_iterations:]
    finally:
        if _th is not None:
            _stop.set()
            _th.join()
            _last_telem = dict(
                power     = float(np.mean(_pw)) if _pw else None,
                clock_min = float(np.min(_clk)) if _clk else None,
                clock_mean= float(np.mean(_clk)) if _clk else None,
                clock_max = float(np.max(_clk)) if _clk else None,
            )

    mean_tflops = flos / (np.mean(times) / 1000 * 10**12)
    median_tflops = flos / (np.median(times) / 1000 * 10**12)
    max_tflops = flos / (np.amin(times) / 1000 * 10**12)
    return mean_tflops, median_tflops, max_tflops


# Rigorous MAMF (achievable / boost) burst.
#
# The background sampler in benchmark_mm() reads power/clock every ~20ms across the WHOLE loop, so a
# short (~2-4ms) boost burst may catch 0-1 samples and the peak sample need not coincide with the
# fastest ("winning") iteration. That makes "was the winner at boost?" unanswerable. Here we instead
# time each iteration in isolation and bracket it with a SYNCHRONOUS clock+power read on BOTH sides,
# so the winning iteration carries its OWN clock. The clock paired with an iteration is
# min(clk_before, clk_after): a conservative floor - the kernel ran strictly between the two reads
# (a sub-ms window, far shorter than a DVFS transition), so it saw at least that clock. A reader can
# thus trust that a published MAMF was produced at the reported boost clock, not sampled at some
# unrelated moment.
def measure_boost_burst(m, n, k, dtype, device, iters, telem=None, idle_before_s=0.0):
    """Return a list of (tflops, clock_MHz_or_None, power_W_or_None) - one entry per iteration."""
    op, l2_cache, C, C_rand, flos = prepare_gemm(m, n, k, dtype, device)
    have_telem = telem_ok(telem)
    def _clk(): return telem.clock() if have_telem else None
    def _pow(): return telem.power() if have_telem else None

    # short idle after allocate lets the SM clock climb back to boost
    if idle_before_s and idle_before_s > 0:
        arch.synchronize()
        time.sleep(idle_before_s)

    out = []
    for _ in range(iters):
        l2_cache.zero_()
        C.copy_(C_rand)
        arch.synchronize()
        c0 = _clk()
        s = arch.event(enable_timing=True); e = arch.event(enable_timing=True)
        with torch.no_grad():
            s.record(); op(); e.record()
        arch.synchronize()
        c1, p1 = _clk(), _pow()
        t = s.elapsed_time(e) / 1000.0  # ms -> s
        tf = flos / (t * 1e12) if t > 0 else 0.0
        clk = min(c0, c1) if (c0 is not None and c1 is not None) else (c0 if c0 is not None else c1)
        out.append((tf, clk, p1))
    return out


### Auto-search helpers (MAMF + MSMF) ###
#
# Instead of brute-forcing a 3D grid of MxNxK shapes, `--search auto` constructs a small set of
# shapes the accelerator should run at/near peak on, using the three rules from "The Case for
# Co-Designing Model Architectures with Hardware" (https://arxiv.org/abs/2401.14489):
#   1. Tensor-core alignment: M, N, K are multiples of `128 bytes / dtype_size` elements.
#   2. Tile quantization:     the MxN output divides evenly into the kernel's tiles (128x256 here).
#   3. Wave quantization:     the number of output tiles is a multiple of the SM count, so the
#                             final wave is full (see `wave_efficiency`).
# K only sets the arithmetic intensity (how compute-bound the GEMM is); it never appears in the
# tile/wave math, so it is pinned and coarsely swept rather than gridded. See benchmarks/README.md.
# After scouting, auto confirms two headlines: MAMF (boost-clock burst) and MSMF (saturated).

def get_sm_count():
    """Compute-unit count for wave packing.

    CUDA/ROCm: `torch.cuda.get_device_properties(0).multi_processor_count` (SMs / CUs).
    HPU/XPU/MPS: returns None — `auto` then skips wave-aware candidates and relies on the
    coarse M×N plane + tight local grid. UNTESTED: whether AMD CU count + the NVIDIA-ish
    128x256 tile assumption still predicts peak shapes on MI300X/MI355X; first-boot should
    compare auto vs a small grid and, if needed, override tile size per arch (see mamf.md).
    """
    try:
        return torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        return None

def dtype_element_size(dtype):
    """Size in bytes of one element of `dtype` (bf16->2, fp8->1, fp32->4)."""
    try:
        return torch.empty(0, dtype=dtype).element_size()
    except Exception:
        return max(torch.finfo(dtype).bits // 8, 1)

def wave_efficiency(m, n, sms, tile_m=128, tile_n=256):
    """Fraction of the scheduled waves that do useful work for an m x n output on `sms` SMs; 1.0
    is a perfectly packed tail wave. blocks = ceil(m/tile_m)*ceil(n/tile_n) run in ceil(blocks/sms)
    waves, and a partial tail wave still costs a full wave, so efficiency = blocks/(waves*sms)."""
    blocks = math.ceil(m / tile_m) * math.ceil(n / tile_n)
    return blocks / (math.ceil(blocks / sms) * sms)


def wave_mn_layouts(sms, max_size, tile_m=128, tile_n=256, waves=range(1, 17), min_dim=1024):
    """Per-wave (M,N) layouts that fill an integer number of SM waves.

    For each wave count returns `(square, layouts)` where `square` is the most-square legal
    (M,N) and `layouts` is the set {square, widest, tallest, transpose-of-square-if-legal}.
    Empty dict if `sms` is None/0.
    """
    out = {}
    if not sms:
        return out
    for w in waves:
        blocks = w * sms
        pairs = []
        for p in range(1, blocks + 1):
            if blocks % p:
                continue
            mm, nn = tile_m * p, tile_n * (blocks // p)
            if min_dim <= mm <= max_size and min_dim <= nn <= max_size:
                pairs.append((mm, nn))
        if not pairs:
            continue
        sq = min(pairs, key=lambda mn: abs(math.log(mn[0] / mn[1])))
        layouts = {sq, max(pairs, key=lambda mn: mn[1]), max(pairs, key=lambda mn: mn[0])}
        # Always include the transpose when wave-legal: abs(log(M/N)) float-ties can pick
        # 2816x1536 over 1536x2816 (H200 peak family), and cuBLAS is not transpose-symmetric.
        if (sq[1], sq[0]) in pairs:
            layouts.add((sq[1], sq[0]))
        out[w] = (sq, layouts)
    return out


def trimmed_median(xs):
    """Median after dropping the single lowest+highest when >=5 samples (kills a lone spike
    or throttle dip). Otherwise plain median. Shrinks run-to-run spread toward ~0.5% noise."""
    xs = sorted(xs)
    if len(xs) >= 5:
        xs = xs[1:-1]
    return float(np.median(xs))


def shape_str(shape):
    """`MxNxK` for a (M,N,K) tuple."""
    return f"{shape[0]}x{shape[1]}x{shape[2]}"


def format_headline(r):
    """One-line MAMF/MSMF headline: `TFLOPS @ MxNxK  WWW MHz` (or `n/a`)."""
    if not r:
        return "n/a"
    extra = ""
    if r.get("power") is not None:
        extra += f"  {r['power']:.0f}W"
    if r.get("clock") is not None:
        extra += f" {r['clock']:.0f}MHz"
    return f"{r['tflops']:.1f} TFLOPS @ {shape_str(r['shape'])} (MxNxK){extra}"


# Populated by benchmark_mm() when a Telemetry sampler is passed. Declared here (before MSMF
# helpers) so confirm/lock-in can read it without forward-ref confusion.
_last_telem = {}


def telem_ok(telem):
    return telem is not None and getattr(telem, "available", False)


def is_suspect(power, ref_power, ratio=0.9):
    """True if `power` is meaningfully below the saturated reference (a boost / idle-gap burst)."""
    if power is None or ref_power is None or ref_power <= 0:
        return False
    return power < ratio * ref_power


def power_rank_score(tflops, power, pmax):
    """Scout ranking score: discount TFLOPS from shapes that drew much less power than the busiest scout."""
    if pmax and power:
        return tflops * min(1.0, power / pmax)
    return tflops


def spread_pct(xs):
    """(max-min)/median as a percent; 0 if empty."""
    return (max(xs) - min(xs)) / np.median(xs) * 100 if xs else 0.0


def median_or_none(xs):
    return float(np.median(xs)) if xs else None


def measure_saturated_reps(shape, n_reps, args, dtype, device, telem, *,
                           warmup_passes=None, collect_means=None):
    """Throwaway warmup passes + `n_reps` saturated benchmarks.

    Returns (means, powers, clocks). Optionally appends each mean into `collect_means`
    (used for the run-wide geometric/arithmetic means in the finish report).
    """
    m, n, k = shape
    n_it, n_wu = args.num_iterations, args.num_warmup_iterations
    if warmup_passes is None:
        warmup_passes = args.confirm_warmup_passes
    for _ in range(max(0, warmup_passes)):
        benchmark_mm(m, n, k, dtype, device, n_it, n_wu, telem=telem)
    means, powers, clocks = [], [], []
    for _ in range(n_reps):
        a, _b, _c = benchmark_mm(m, n, k, dtype, device, n_it, n_wu, telem=telem)
        means.append(a)
        if collect_means is not None:
            collect_means.append(a)
        if _last_telem.get("power") is not None:
            powers.append(_last_telem["power"])
        if _last_telem.get("clock_mean") is not None:
            clocks.append(_last_telem["clock_mean"])
    return means, powers, clocks


def gemm_fits(M, N, K, elem):
    """True if A/B/C(+C_rand) for this shape fit in ~90% of free CUDA VRAM (or unknown)."""
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return True
    need = (M * K + K * N + 2 * M * N) * elem + 256 * 2**20  # A, B, C, C_rand + l2 buffer
    return need < free * 0.9


def top_shapes(measured, n, prefer_mn=None, prefer_slots=0):
    """Top-N scout shapes by power-aware score (unsaturated boosts are discounted).

    If `prefer_mn` is a set of (M,N) and prefer_slots>0, reserve that many confirm slots for the
    best shapes whose (M,N) is in the set. Stops a noisy plane/mid-K family from crowding the
    wave-perfect basin out of the confirm set (the H200 1-shot miss mode).
    """
    pmax = max((p for _, p, _ in measured if p is not None), default=None)
    ranked = {}
    for tf, p, shp in measured:
        s = power_rank_score(tf, p, pmax)
        if shp not in ranked or s > ranked[shp]:
            ranked[shp] = s
    ordered = [shp for shp, _ in sorted(ranked.items(), key=lambda kv: kv[1], reverse=True)]
    n = max(1, n)
    if not prefer_mn or prefer_slots <= 0:
        return ordered[:n]
    prefer_slots = min(prefer_slots, n)
    preferred = [s for s in ordered if (s[0], s[1]) in prefer_mn][:prefer_slots]
    rest = [s for s in ordered if s not in preferred]
    return (preferred + rest)[:n]


def build_msmf_confirm_set(measured, scout_meta, seen, square_by_wave, args):
    """MSMF confirm set: forced low-wave + fat scouts, then power-ranked fillers.

    Returns (shapes, n_confirm). See call-site comments in auto_search for why forcing matters.
    """
    def best_k_for_mn(mm, nn, *, raw=False):
        """Best measured K for this (M,N). raw=True → max scout TFLOPS (for forced shapes:
        high-K often reads SUSPECT/low-power during a noisy scout, and power-rank would wrongly
        lock onto Kmin; confirm re-measures sustainably)."""
        pmax = max((p for _, p, _ in measured if p is not None), default=None)
        best_s, best_shp = -1.0, None
        for tf, p, shp in measured:
            if shp[0] == mm and shp[1] == nn:
                s = tf if raw else power_rank_score(tf, p, pmax)
                if s > best_s:
                    best_s, best_shp = s, shp
        return best_shp

    forced = []
    for w in range(1, 5):
        mn = square_by_wave.get(w)
        if not mn:
            continue
        # both orientations when the transpose was scouted (wave-legal); cuBLAS is not
        # transpose-symmetric.
        orients = [mn]
        if (mn[1], mn[0]) != mn:
            orients.append((mn[1], mn[0]))
        for mm, nn in orients:
            shp = best_k_for_mn(mm, nn, raw=True)
            if shp and shp not in forced:
                forced.append(shp)
    # Force the FATTEST scouts (largest min(M,N,K), then volume). A 20-iter scout is too short to
    # saturate, so everything boosts during scouting - but a shape large in ALL dims saturates to
    # TDP under the 100-iter confirm and anchors the saturated-clock reference. Without fat forcing
    # a confirm set can be all small/skinny (boosting) shapes (seen on fp8).
    fat_forced = sorted(scout_meta.keys(),
                        key=lambda s: (min(s), s[0] * s[1] * s[2]), reverse=True)[:args.confirm_fat_forced]
    for shp in fat_forced:
        if shp not in forced:
            forced.append(shp)
    # Never drop forced shapes — expand budget so basin-diverse power-rank fillers still get slots.
    # Fillers are POWER-RANKED (not raw TFLOPS): tall-skinny boosters belong to MAMF, not MSMF.
    n_confirm = max(args.confirm_top, len(forced) + 4)
    rest = [s for s in top_shapes(measured, n_confirm + len(forced), prefer_mn=seen,
                                  prefer_slots=max(2, n_confirm // 2))
            if s not in forced]
    shapes = (forced + rest)[:n_confirm]
    if forced:
        print(f"[auto] forced confirms (low-wave squares + fattest scouts): {forced}")
    return shapes, n_confirm


def thermal_soak(device, telem, max_size, soak_s):
    """Drive the chip to thermal steady state before MSMF confirm.

    Biggest reproducibility lever: a cold/cooler GPU boosts and reads high, so without soaking
    the MSMF headline depends on how warm the card happened to be. Soak in bf16 (heat is heat)
    on a big square; stop early once the SM clock stops dropping.
    """
    if not soak_s or soak_s <= 0:
        return
    print(f"\n[auto] thermal soak (<= {soak_s}s) to reach steady-state clock before MSMF confirm ...")
    s = min(8192, max_size)
    sa = torch.randn(s, s, dtype=torch.bfloat16, device=device)
    sb = torch.randn(s, s, dtype=torch.bfloat16, device=device)
    sc = torch.empty(s, s, dtype=torch.bfloat16, device=device)
    t_end = time.time() + soak_s
    prev_clk, stable_ticks = None, 0
    while time.time() < t_end:
        for _ in range(100):
            torch.mm(sa, sb, out=sc)
        arch.synchronize()
        if telem_ok(telem):
            clk = telem.clock()
            if clk is not None and prev_clk is not None and clk >= prev_clk - 5:
                stable_ticks += 1
                if stable_ticks >= 3:  # clock stopped dropping across ~3 windows -> settled
                    break
            else:
                stable_ticks = 0
            prev_clk = clk
    del sa, sb, sc
    if telem_ok(telem) and telem.clock() is not None:
        print(f"[auto] soak done, SM clock settled ~{telem.clock():.0f}MHz")


def select_and_lock_msmf(msmf_results, args, dtype, device, telem, reps):
    """Filter confirm results (SUSPECT / boosting / jittery) and lock-in the MSMF headline.

    Returns the chosen result dict (or None). Lock-in re-measures the winner with extra reps and
    re-applies the boost+spread gates — a shape can look saturated in the short confirm then drift
    toward boost over the longer re-measure (seen on H200).
    """
    ref_p = max((r["power"] for r in msmf_results if r["power"] is not None), default=None)
    exclude_ok = telem is not None and telem.validated

    def _suspect(r):
        return exclude_ok and is_suspect(r["power"], ref_p, args.suspect_power_ratio)

    sat_pool = [r for r in msmf_results if not _suspect(r)] or msmf_results
    # Saturated-clock reference = lowest clock among the most power-saturated shapes. A shape
    # running materially above it is still boosting (near-TDP but with clock headroom).
    hi_p = [r for r in sat_pool if r["power"] is not None and ref_p and r["power"] >= 0.97 * ref_p
            and r["clock"] is not None]
    sat_clock = min((r["clock"] for r in hi_p), default=None)

    def _boosting(r, clk=None):
        c = r["clock"] if clk is None else clk
        return (exclude_ok and sat_clock is not None and c is not None
                and c > sat_clock * args.msmf_clock_ratio)

    stable_pool = ([r for r in sat_pool if r["spread"] <= args.msmf_max_spread and not _boosting(r)]
                   or [r for r in sat_pool if r["spread"] <= args.msmf_max_spread]
                   or [r for r in sat_pool if not _boosting(r)] or sat_pool)
    prelim = max(stable_pool, key=lambda r: r["tflops"]) if stable_pool else None
    for r in sat_pool:
        if not (prelim and r["tflops"] > prelim["tflops"]):
            continue
        if _boosting(r):
            print(f"[auto] MSMF ignored {shape_str(r['shape'])} ({r['tflops']:.1f} TFLOPS @ "
                  f"{r['clock']:.0f}MHz vs {sat_clock:.0f}MHz saturated): still boosting, not sustainable "
                  f"(trends toward MAMF)")
        elif r["spread"] > args.msmf_max_spread:
            print(f"[auto] MSMF ignored {shape_str(r['shape'])} ({r['tflops']:.1f} TFLOPS, "
                  f"spread={r['spread']:.1f}% > {args.msmf_max_spread:.0f}%): too jittery to reproduce")
    for r in msmf_results:
        if _suspect(r) and prelim and r["tflops"] > prelim["tflops"]:
            print(f"[auto] MSMF excluded {shape_str(r['shape'])} ({r['tflops']:.1f} TFLOPS @ {r['power']:.0f}W): "
                  f"unsaturated clock-boost, not sustainable (counts toward MAMF)")

    ordered = sorted(stable_pool, key=lambda r: r["tflops"], reverse=True)
    if not (ordered and args.msmf_lock_reps > reps):
        return prelim

    tries = ordered[:max(1, args.msmf_lock_tries)]
    for idx, cand in enumerate(tries):
        shp = cand["shape"]
        print(f"\n[auto] MSMF lock-in: re-measuring {shape_str(shp)} x{args.msmf_lock_reps} ...")
        lm, lp, lc = measure_saturated_reps(shp, args.msmf_lock_reps, args, dtype, device, telem)
        lock_tf = trimmed_median(lm)
        lock_spread = spread_pct(lm) if lm else cand["spread"]
        lock_clk = median_or_none(lc) if lc else cand.get("clock")
        lock_pw = median_or_none(lp) if lp else cand.get("power")
        lock_boosting = _boosting(cand, clk=lock_clk)
        spread_ok = lock_spread <= args.msmf_max_spread
        if (spread_ok and not lock_boosting) or idx == len(tries) - 1:
            cand["tflops"] = lock_tf
            cand["spread"] = lock_spread
            if lock_pw is not None: cand["power"] = lock_pw
            if lock_clk is not None: cand["clock"] = lock_clk
            tags = []
            if not spread_ok: tags.append("still jittery")
            if lock_boosting: tags.append("still boosting")
            tag = f"  (best available; {', '.join(tags)})" if tags else ""
            print(f"[auto] MSMF lock-in: {lock_tf:.1f} TFLOPS  spread={lock_spread:.1f}%"
                  f"{f'  {lock_clk:.0f}MHz' if lock_clk is not None else ''}{tag}  "
                  f"(runs: {', '.join(f'{x:.1f}' for x in lm)})")
            return cand
        why = []
        if not spread_ok:
            why.append(f"spread {lock_spread:.1f}% > {args.msmf_max_spread:.0f}%")
        if lock_boosting:
            why.append(f"clock {lock_clk:.0f}MHz > {sat_clock:.0f}*{args.msmf_clock_ratio:.2f} "
                       f"saturated floor")
        print(f"[auto] MSMF lock-in rejected {shape_str(shp)}: {'; '.join(why)} "
              f"over {args.msmf_lock_reps} reps -> trying next candidate")
    return prelim


def confirm_msmf(shapes, args, dtype, device, telem, reps, all_mean_tflops):
    """Thermal soak + per-shape MSMF confirm measurements.

    Returns a list of result dicts (shape/tflops/power/clock/spread). Caller runs
    `select_and_lock_msmf` to pick and lock-in the headline.
    """
    thermal_soak(device, telem, args.max_size, args.msmf_soak_s)
    red = "trimmed-median" if reps >= 5 else "median"
    print(f"\n[auto] MSMF (sustainable) confirm: {len(shapes)} shapes, {args.num_iterations} iters x {reps} reps, "
          f"{red} of per-shape-warmed means ...")
    msmf_results = []
    for shp in shapes:
        # thermal pre-warmup: throwaway saturated passes so the first TIMED rep is at
        # steady-state clock (a freshly-switched shape otherwise reads cold on rep 1).
        means, powers, clocks = measure_saturated_reps(
            shp, reps, args, dtype, device, telem, collect_means=all_mean_tflops)
        pmed = median_or_none(powers)
        cmed = median_or_none(clocks)
        spread = spread_pct(means)
        msmf_results.append(dict(shape=shp, tflops=trimmed_median(means), power=pmed, clock=cmed, spread=spread))
        ptxt = f"  {pmed:.0f}W" if pmed is not None else ""
        print(f"[auto] MSMF {shape_str(shp)}: {red}-of-{reps} mean={msmf_results[-1]['tflops']:.1f} "
              f"TFLOPS{ptxt}  spread={spread:.1f}%  (runs: {', '.join(f'{x:.1f}' for x in means)})")
    return msmf_results


def select_mamf(mamf_results, boost_clk):
    """Pick the MAMF headline, preferring boost-validated readings."""
    boost_pool = [r for r in mamf_results if r["boost"]] or mamf_results
    mamf = max(boost_pool, key=lambda r: r["tflops"]) if boost_pool else None
    if mamf and not mamf["boost"]:
        print(f"[auto] WARNING: no MAMF candidate reached the boost clock (~{boost_clk:.0f}MHz); "
              f"headline {mamf['tflops']:.1f} TFLOPS is a base/throttled-clock reading, not a true boost burst")
    return mamf


def mamf_candidates(scout_meta, boost_min, n):
    """Boost-clock scout shapes ranked by peak TFLOPS (fall back to all if no clocks)."""
    at_boost = [(shp, mt["mx"]) for shp, mt in scout_meta.items()
                if mt.get("clock_max") is not None and mt["clock_max"] >= boost_min]
    pool_meta = at_boost or [(shp, mt["mx"]) for shp, mt in scout_meta.items()]
    return [shp for shp, _ in sorted(pool_meta, key=lambda kv: kv[1], reverse=True)][:n]


def confirm_mamf(cands, args, dtype, device, telem, reps, boost_clk, all_mean_tflops):
    """Boost-burst confirm for MAMF candidates; returns the chosen result dict (or None).

    Each candidate gets `reps` short bursts preceded by idle so the SM clock recovers to boost.
    Only iterations whose bracketed clock reached boost count toward the headline.
    """
    boost_min = args.boost_clock_ratio * boost_clk if boost_clk else 0.0
    burst_iters = max(1, args.mamf_burst_iters)
    idle_s = max(0.0, args.mamf_idle_s)
    print(f"\n[auto] MAMF (achievable) confirm: {len(cands)} boost-clock shapes "
          f"(boost≈{boost_clk:.0f}MHz, need ≥{boost_min:.0f}MHz), {burst_iters} iters x {reps} reps, "
          f"{idle_s*1000:.0f}ms idle, no warmup, peak iteration ...")
    mamf_results = []
    for shp in cands:
        # rigorous path: time each iteration in isolation, bracketed by synchronous clock reads,
        # `reps` bursts each preceded by a short idle so the clock recovers to boost. Every
        # (tflops, clk) pair is thus self-validating - no reliance on a loosely-timed background
        # sampler for the achievable headline.
        iters_all = []  # (tflops, clk, power) across all reps
        for _ in range(reps):
            b = measure_boost_burst(shp[0], shp[1], shp[2], dtype, device,
                                    burst_iters, telem=telem, idle_before_s=idle_s)
            iters_all += b
            if b:
                all_mean_tflops.append(float(np.mean([t for t, _, _ in b])))
        # only iterations whose bracketed clock reached boost count toward the achievable headline
        at_boost = [x for x in iters_all if boost_min and x[1] is not None and x[1] >= boost_min]
        valid = at_boost or iters_all
        peak, cpk, ppk = max(valid, key=lambda t: t[0]) if valid else (0.0, None, None)
        boost_pw = [p for _, _, p in at_boost if p is not None]
        pmed = median_or_none(boost_pw) if boost_pw else ppk
        is_boost = cpk is not None and boost_min and cpk >= boost_min
        mamf_results.append(dict(shape=shp, tflops=float(peak), power=pmed, clock=cpk, boost=is_boost))
        ptxt = f"  {pmed:.0f}W" if pmed is not None else ""
        ctxt = f" {cpk:.0f}MHz" if cpk is not None else ""
        flag = "" if is_boost else " (base-clock, not boost)"
        top_pk = sorted((t for t, _, _ in iters_all), reverse=True)[:reps]
        print(f"[auto] MAMF {shape_str(shp)}: peak={peak:.1f} TFLOPS{ptxt}{ctxt}{flag}  "
              f"(top peaks: {', '.join(f'{x:.1f}' for x in top_pk)})")
    return select_mamf(mamf_results, boost_clk)


def resolve_dim(vals, rng):
    """Grid mode: explicit list, or np.arange from [start, stop, step] (start=0 → step)."""
    if vals is not None:
        return vals
    start, stop, step = rng
    if start == 0:  # can't have a 0 dimension
        start = step
    return np.arange(start, stop, step)


### Power / clock telemetry (optional, best-effort, per vendor) ###
#
# Used for (a) a self-validating power check per shape and (b) adaptive warmup. Only fast in-process
# libraries are used (NVML / amdsmi / pyhlml), so a read is ~1us and can be sampled per iteration.
# If the vendor library isn't installed the feature degrades silently (available == False).
#   NVIDIA : pip install nvidia-ml-py    (pynvml)   - VALIDATED (SUSPECT exclusion trusted)
#   AMD    : amdsmi (ships with ROCm)               - UNTESTED: sample+report only until first-boot
#   Gaudi  : pip install habana-pyhlml   (pyhlml)   - UNTESTED: sample+report only until first-boot
# XPU/MPS have no fast in-process telemetry wired (would need a slow `xpu-smi` subprocess), so they
# fall through to unavailable and the benchmark still runs, just without power/clock reporting.
# See mamf.md "Untested vendors" for API assumptions and the first-boot checklist.

# Backends whose power readings have been validated with mamf_spike_probe.py against a known
# boost-vs-saturated gap. Others may still *sample* (for logging) but must not *exclude*.
VALIDATED_BACKENDS = frozenset({"nvml"})


class Telemetry:
    """Best-effort in-process power/clock sampler (NVML / amdsmi / pyhlml)."""

    def __init__(self, arch, index=0):
        self.backend = None
        self._m = self._h = self._clk_arg = None
        name = getattr(arch, "name", "") if arch is not None else ""
        try:
            if name == "cuda":
                import pynvml as m
                m.nvmlInit()
                self._m = m
                self._h = m.nvmlDeviceGetHandleByIndex(index)
                self._clk_arg = m.NVML_CLOCK_SM
                self.backend = "nvml"
            elif name == "rocm":
                # UNTESTED on real hardware — see mamf.md "Untested vendors".
                import amdsmi as m
                m.amdsmi_init()
                self._m = m
                self._h = m.amdsmi_get_processor_handles()[index]
                self._clk_arg = m.AmdSmiClkType.GFX
                self.backend = "amdsmi"
            elif name == "hpu":
                # UNTESTED on real hardware — see mamf.md "Untested vendors".
                import pyhlml as m
                m.hlmlInit()
                self._m = m
                self._h = m.hlmlDeviceGetHandleByIndex(index)
                self.backend = "hlml"
        except Exception:
            self.backend = None

    @property
    def available(self):
        return self.backend is not None

    @property
    def validated(self):
        """If False, sample+report power but do NOT use it to exclude shapes from the headline."""
        return self.backend in VALIDATED_BACKENDS

    def power(self):  # Watts, or None
        try:
            if self.backend == "nvml":
                return self._m.nvmlDeviceGetPowerUsage(self._h) / 1000.0  # mW -> W
            if self.backend == "hlml":
                # ASSUMED: HLML mirrors NVML and returns milliwatts. Confirm on first Gaudi box.
                return self._m.hlmlDeviceGetPowerUsage(self._h) / 1000.0
            if self.backend == "amdsmi":
                # ASSUMED: values are already Watts (strings "N/A" possible). Confirm on first MI box.
                info = self._m.amdsmi_get_power_info(self._h)
                w = info.get("current_socket_power")
                if w in (None, "N/A"):
                    w = info.get("average_socket_power")
                return float(w) if w not in (None, "N/A") else None
        except Exception:
            return None
        return None

    def clock(self):  # SM/GFX clock in MHz, or None
        try:
            if self.backend == "nvml":
                return float(self._m.nvmlDeviceGetClockInfo(self._h, self._clk_arg))
            if self.backend == "hlml":
                # ASSUMED: clock type 0 is the compute clock. Confirm on first Gaudi box.
                return float(self._m.hlmlDeviceGetClockInfo(self._h, 0))
            if self.backend == "amdsmi":
                return float(self._m.amdsmi_get_clock_info(self._h, self._clk_arg)["clk"])
        except Exception:
            return None
        return None

    def siblings_busy(self, self_index=0, util_pct=10, mem_mb=4096):
        """Return [(idx, util%, mem_MiB)] for OTHER physical GPUs that look actively COMPUTING.

        Sibling GPUs on the same board share its power/cooling budget, so running one while others
        are busy drags the device-under-test's *sustained* clock around run-to-run and quietly ruins
        single-run MSMF reproducibility. Keyed on utilization (real power/heat) - a small resident
        CUDA context (idle, 0% util) doesn't perturb the clock, so the memory threshold is high and
        only flags a genuinely loaded neighbor. Only wired for NVML (the validated backend)."""
        out = []
        if self.backend != "nvml":
            return out
        try:
            count = self._m.nvmlDeviceGetCount()
        except Exception:
            return out
        for i in range(count):
            if i == self_index:
                continue
            try:
                h = self._m.nvmlDeviceGetHandleByIndex(i)
                util = self._m.nvmlDeviceGetUtilizationRates(h).gpu
                mem = self._m.nvmlDeviceGetMemoryInfo(h).used / 2**20
            except Exception:
                continue
            if util >= util_pct or mem >= mem_mb:
                out.append((i, int(util), int(mem)))
        return out

    def device_name(self):
        try:
            if self.backend == "nvml":
                name = self._m.nvmlDeviceGetName(self._h)
                return name.decode() if isinstance(name, bytes) else name
            if self.backend == "amdsmi":
                info = self._m.amdsmi_get_gpu_asic_info(self._h)
                return info.get("market_name") or info.get("vendor_id") or "AMD GPU"
            if self.backend == "hlml":
                return f"Gaudi:{self._h}"
        except Exception:
            return None
        return None


class FakeTelemetry(Telemetry):
    """Scripted (power_W, clock_MHz) sequence for offline SUSPECT / ranking tests. No GPU.

    Each `power()` call advances one step; `clock()` returns the clock paired with the last
    `power()` reading (mirrors how the finder sampler calls power then clock each tick).
    """

    def __init__(self, samples=None, loop=True):
        self.backend = "fake"
        self._samples = list(samples or [(1000.0, 1300.0)])
        self._i = 0
        self._loop = loop
        self._last = self._samples[0] if self._samples else (None, None)

    @property
    def validated(self):
        return True  # tests exercise the validated exclusion path

    def power(self):
        if not self._samples:
            return None
        if self._i >= len(self._samples):
            if not self._loop:
                self._last = self._samples[-1]
                return self._last[0]
            self._i = 0
        self._last = self._samples[self._i]
        self._i += 1
        return self._last[0]

    def clock(self):
        return self._last[1] if self._last else None

    def sample(self):
        """Return (power, clock) advancing one step — preferred in tests."""
        return self.power(), self.clock()

def setup_checks():
    if arch.name == "rocm":
        if int(os.environ.get("PYTORCH_TUNABLEOP_ENABLED", "0")) == 0:
            warn("AMD GPUs usually require `export PYTORCH_TUNABLEOP_ENABLED=1` to measure the best possible compute, but it hasn't been set. Proceeding as is - expect potentially bad/invalid results.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Shape selection. Required for `--search grid`; optional (ignored) for `--search auto`, which
    # discovers a near-peak shape on its own. Passing any shape argument implies grid mode.
    m_group = parser.add_mutually_exclusive_group()
    m_group.add_argument("--m", nargs="+", type=int, help='The first dimension of the GEMM, enter any number of arguments')
    m_group.add_argument("--m_range", nargs='+', type=int, help="The first dimension of the GEMM, [start,stop,step]")

    n_group = parser.add_mutually_exclusive_group()
    n_group.add_argument("--n", nargs="*", type=int, help='The last dimension of the GEMM, enter any number of arguments')
    n_group.add_argument("--n_range", nargs='+', type=int, help="The last dimension of the GEMM, [start,stop,step]")

    k_group = parser.add_mutually_exclusive_group()
    k_group.add_argument("--k", nargs="*", type=int, help='The shared (reduction) dimension of the GEMM, enter any number of arguments')
    k_group.add_argument("--k_range", nargs='+', type=int, help="The shared (reduction) dimension of the GEMM, [start,stop,step]")

    parser.add_argument("--search", choices=["auto", "grid"], default="auto",
                        help="auto (default): lean directed search (wave@{Kmin,Kmax} -> plane@Kmin -> tight grid -> confirm); "
                             "grid: brute-force sweep over --m/--n/--k[_range]. Passing any shape argument implies grid.")
    parser.add_argument("--num_iterations", type=int, default=100, help='The number of iterations used to benchmark each GEMM')
    parser.add_argument("--num_warmup_iterations", type=int, default=50, help='The number of warmup iterations')
    parser.add_argument("--scout_num_iterations", type=int, default=20, help='auto: iterations per shape while scouting (best shapes are re-measured with --num_iterations)')
    parser.add_argument("--scout_num_warmup_iterations", type=int, default=8, help='auto: warmup iterations per shape while scouting (GPU is globally warm, so ranking needs few)')
    parser.add_argument("--confirm_top", type=int, default=10, help='auto: how many of the best scouted (power-ranked) shapes to re-measure with the full iteration count (higher = more reliable 1-shot peak, still seconds)')
    parser.add_argument("--confirm_fat_forced", type=int, default=3, help='auto: also force this many FATTEST scouts (largest min(M,N,K), then largest volume) into the MSMF confirm set - a shape large in every dim reliably saturates to TDP under the confirm, so it anchors the saturated-clock reference and guarantees a real sustainable candidate even when the rest of the set is small/skinny')
    parser.add_argument("--mamf_burst_iters", type=int, default=20, help='auto: iterations per rep for the MAMF (achievable) confirm - kept SHORT so the boost burst is not re-saturated away; the peak iteration across reps is the MAMF')
    parser.add_argument("--mamf_idle_s", type=float, default=0.25, help='auto: idle time before each MAMF burst so the SM clock recovers to boost (a short real kernel enjoys this); 0 to disable')
    parser.add_argument("--boost_clock_ratio", type=float, default=0.97, help='auto: a MAMF reading counts as a real boost-clock burst only if its peak SM clock >= this * the highest clock seen in the run; below that it is a throttled/base-clock reading and is flagged')
    parser.add_argument("--confirm_reps", type=int, default=5, help='auto: repeat each confirmed shape this many times and report the trimmed-MEDIAN mean - defeats power-cap clock-jitter spikes. Default 5 enables the trimmed median (drop min+max); use 3 for a faster but noisier headline')
    parser.add_argument("--msmf_max_spread", type=float, default=3.0, help='auto: a sustainable (MSMF) shape whose reps swing more than this %% cannot be reproduced by a reader, so it is kept out of the headline (still reported); only falls back to jittery shapes if none measured stably')
    parser.add_argument("--msmf_clock_ratio", type=float, default=1.04, help='auto: a MSMF shape whose saturated SM clock exceeds the run saturated-clock floor (the clock of the most power-saturated shape) by more than this factor is still boosting (near-TDP power but clock headroom, i.e. a tall-skinny layout) - its number is elevated/irreproducible and is kept out of the headline (counts toward MAMF)')
    parser.add_argument("--msmf_lock_reps", type=int, default=9, help='auto: re-measure the MSMF candidate this many times as a stability GATE - publish the trimmed-median only if its spread is within --msmf_max_spread, else move to the next candidate. A tighter, reproducible headline. Set <= confirm_reps to skip the lock-in pass')
    parser.add_argument("--msmf_lock_tries", type=int, default=4, help='auto: how many top MSMF candidates the lock-in gate may walk through (high-TFLOPS first) before accepting the best-available one, if none pass the spread gate')
    parser.add_argument("--msmf_soak_s", type=float, default=20.0, help='auto: seconds to drive the GPU to thermal steady-state (hot, clock settled at the saturated floor) before the MSMF confirm, so the sustainable number does not depend on how warm the card happened to be. Stops early once the clock stops dropping. 0 to disable')
    parser.add_argument("--confirm_warmup_passes", type=int, default=1, help='auto: throwaway saturated passes run before the timed MSMF reps of each shape (and before the lock-in), so a freshly-switched shape starts at steady-state clock instead of reading cold/boosted on the first rep (false jitter). 0 to disable')
    parser.add_argument("--refine_grid", default=True, action=argparse.BooleanOptionalAction, help='auto: run a tight exhaustive local grid around the top scout seeds before confirm (--no-refine_grid to skip)')
    parser.add_argument("--refine_seeds", type=int, default=4, help='auto: how many top scout seeds to center the tight local grid on')
    parser.add_argument("--refine_radius_mn", type=int, default=2, help='auto: tight-grid half-width along M and N, in steps of 256')
    parser.add_argument("--refine_radius_k", type=int, default=2, help='auto: tight-grid half-width along K, in steps of 1024')
    parser.add_argument("--max_size", type=int, default=20480, help='auto: largest M/N/K dimension to consider')
    parser.add_argument("--warmup", choices=["adaptive", "fixed"], default="adaptive",
                        help="adaptive (default): warm up until matmul throughput plateaus (works on any accelerator, stops as soon as it's warm); fixed: a flat 30s")
    parser.add_argument("--telemetry", choices=["on", "off"], default="on",
                        help="sample power/SM-clock (NVML/amdsmi/pyhlml, ~1us/read) to report and validate each measurement; off to skip")
    parser.add_argument("--suspect_power_ratio", type=float, default=0.9,
                        help="a contender drawing less than this fraction of the running-max power is SUSPECT "
                             "(unsaturated boost). Exclusion from the headline only runs on VALIDATED backends "
                             "(currently NVIDIA nvml); others sample+report but do not exclude. See mamf.md.")
    parser.add_argument("--cuda_device", type=int, default=0, help="The cuda device to run the benchmark on")
    parser.add_argument("--output_file", type=str, default=f"{file_dir}/results/mm.out")
    parser.add_argument("--notes", type=str, default="", help="benchmark-specific notes to add to the output_file's header")
    parser.add_argument("--verbose", default=True, action=argparse.BooleanOptionalAction, help='log to stdout besides output_file?')
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type to use for the benchmark (e.g., float32, float16, bfloat16, float8_e4m3fn, torch.float8_e4m3fnuz)")
    args = parser.parse_args()

    dtype = get_torch_dtype(args.dtype)
    device = arch.device

    setup_checks()

    # telemetry: sample the *physical* device torch is using (CUDA_VISIBLE_DEVICES[0] if set)
    _vis = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
    try:
        telem_index = int(_vis.split(",")[0]) if _vis else args.cuda_device
    except (ValueError, AttributeError):
        telem_index = args.cuda_device
    telem = Telemetry(arch, telem_index) if args.telemetry == "on" else None
    power_ref = {"max": 0.0} # running max mean-power, for the per-shape validity check

    # Any explicit shape argument means the user wants a specific sweep -> grid mode.
    shape_args_given = any(x is not None for x in (args.m, args.m_range, args.n, args.n_range, args.k, args.k_range))
    mode = "grid" if (args.search == "grid" or shape_args_given) else "auto"

    m = n = k = None
    if mode == "grid":
        missing = [name for name, val, rng in (
            ("m", args.m, args.m_range), ("n", args.n, args.n_range), ("k", args.k, args.k_range))
            if val is None and rng is None]
        if missing:
            parser.error(f"--search grid requires shapes for: {', '.join(missing)} (use --{{dim}} or --{{dim}}_range)")

        m, n, k = args.m, args.n, args.k
        range_info = (
            f"m={args.m_range if m is None else args.m} | "
            f"n={args.n_range if n is None else args.n} | "
            f"k={args.k_range if k is None else args.k}"
        )
        m = resolve_dim(m, args.m_range)
        n = resolve_dim(n, args.n_range)
        k = resolve_dim(k, args.k_range)
        warmup_shape = (int(m[0]), int(n[0]), int(k[0]))
    else:
        range_info = f"auto-search (SMs={get_sm_count()}, dtype={args.dtype}, max_size={args.max_size})"
        warmup_shape = (4096, 4096, 4096)

    sys.stdout = Tee(args.output_file, args.verbose)
    print_benchmark_header(dtype, device, args.notes + f"\n- search mode: {mode}")

    # Reproducibility guard: a published single-run number must be reproducible on the same GPU/setup.
    # The #1 wrecker is a busy sibling GPU sharing this board's power/cooling budget - it silently
    # lowers and destabilizes the sustained (MSMF) clock. Warn loudly rather than publish a bad number.
    if telem_ok(telem):
        busy = telem.siblings_busy(self_index=telem_index)
        if busy:
            blurb = ", ".join(f"GPU{i}({u}% util, {mm}MiB)" for i, u, mm in busy)
            print(f"\n*** REPRODUCIBILITY WARNING: {len(busy)} sibling GPU(s) active: {blurb}.\n"
                  f"    They share board power/cooling with GPU{telem_index}, so MSMF (saturated) will read\n"
                  f"    low and vary run-to-run. For a reproducible single-run number, measure with every\n"
                  f"    other GPU idle (e.g. CUDA_VISIBLE_DEVICES={telem_index} and nothing else running).\n")

    best_tflops = dict(max=0, median=0, mean=0)
    best_config = dict(max="", median="", mean="")
    # auto mode reports two headlines from the same search:
    #   mamf = Maximum ACHIEVABLE  Matmul FLOPS - the boost burst a short kernel can catch
    #   msmf = Maximum SUSTAINABLE Matmul FLOPS - what the chip holds once saturated at ~TDP
    headline = dict(mamf=None, msmf=None)
    num_shapes = 0
    all_mean_tflops = []
    measured = [] # (mean_tflops, power_W_or_None, (M, N, K)) for every shape tried, for the auto confirm phase
    # per-shape scout metadata (peak tflops + peak clock) so the MAMF phase can pick shapes that ran at
    # the boost clock and verify a headline came from boost, not the throttled/base clock.
    scout_meta = {} # (M,N,K) -> dict(mean, mx, power, clock_max)
    boost_ref = {"clk": 0.0} # highest SM clock seen anywhere this run == the effective boost ceiling
    start_time = time.time()

    def measure(M, N, K, num_iter, num_warmup, label="", idle_before_s=0.0):
        """Benchmark one shape, track the running best, print a progress line, return mean TFLOPS."""
        global num_shapes
        num_shapes += 1
        M, N, K = int(M), int(N), int(K)
        mean_tflops, median_tflops, max_tflops = benchmark_mm(M, N, K, dtype, device, num_iter, num_warmup,
                                                              telem=telem, idle_before_s=idle_before_s)
        all_mean_tflops.append(mean_tflops)
        measured.append((mean_tflops, _last_telem.get("power"), (M, N, K)))
        cmax = _last_telem.get("clock_max")
        if cmax is not None:
            boost_ref["clk"] = max(boost_ref["clk"], cmax)
        prev = scout_meta.get((M, N, K))
        # keep the best (peak-tflops) scout reading per shape
        if prev is None or max_tflops > prev["mx"]:
            scout_meta[(M, N, K)] = dict(mean=mean_tflops, mx=max_tflops,
                                         power=_last_telem.get("power"), clock_max=cmax)

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

        # validity: a *contender* shape whose mean power sags well below the running-max power was
        # not truly saturated (a clock-boost burst) -> flag SUSPECT so a lucky reading isn't trusted.
        # Exclusion from the headline (confirm phase) only happens when telem.validated is True.
        tinfo = ""
        if _last_telem.get("power") is not None:
            p = _last_telem["power"]
            # Prefer peak clock in the scout line — that's what MAMF validation keys off.
            ck = _last_telem.get("clock_max") or _last_telem.get("clock_min")
            ref = power_ref["max"]; power_ref["max"] = max(ref, p)
            contender = best_tflops["mean"] > 0 and mean_tflops >= 0.9 * best_tflops["mean"]
            suspect = contender and is_suspect(p, ref, args.suspect_power_ratio)
            tinfo = f" | {p:4.0f}W" + (f" {ck:4.0f}MHz" if ck is not None else "") + (" SUSPECT" if suspect else "")

        tag = f"{label:>7} " if label else ""
        end = "\n" if label else "\r" # auto prints one line per phase step; grid overwrites in place
        print(f"{num_shapes:>6} | {tag}{mean_tflops:6.1f}(mean) {median_tflops:6.1f}(median) {max_tflops:6.1f}(max) @ {cur_config:<20} | best: {best_tflops['mean']:6.1f}(mean) {best_tflops['median']:6.1f}(median) {best_tflops['max']:6.1f}(max) TFLOPS{tinfo}", end=end)
        return mean_tflops

    def finish():
        all_tried_shapes_geometric_mean_tflops  = np.exp(np.log(all_mean_tflops).mean()) if all_mean_tflops else 0
        all_tried_shapes_arithmetic_mean_tflops = np.mean(all_mean_tflops) if all_mean_tflops else 0

        time_delta = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
        print("", end="\033[K")

        if headline.get("mamf") or headline.get("msmf"):
            outcomes = (
                f"MAMF (max achievable,  boost burst): {format_headline(headline.get('mamf'))}\n"
                f"MSMF (max sustainable, saturated):   {format_headline(headline.get('msmf'))}"
            )
        else:
            outcomes = (
                f"mean:   {best_tflops['mean']:.1f} TFLOPS @ {best_config['mean']}\n"
                f"median: {best_tflops['median']:.1f} TFLOPS @ {best_config['median']}\n"
                f"max:    {best_tflops['max']:.1f} TFLOPS @ {best_config['max']}"
            )
        print(f"""
Tried {num_shapes} shapes => the best outcomes were:
{outcomes}

Across {num_shapes} shapes in range: {range_info} in this run:
arithmetic mean: {all_tried_shapes_arithmetic_mean_tflops:.1f} TFLOPS
geometric mean:  {all_tried_shapes_geometric_mean_tflops:.1f} TFLOPS
""")
        print(f"Legend: TFLOPS = 10**12 FLOPS")
        print(f"Elapsed time: {time_str}")

    def auto_search():
        """Lean directed search that reports both MAMF (boost) and MSMF (saturated).

        Ablation (offline replay vs exhaustive H200/B200 grids) showed that coordinate
        descent, re-descent, hill-climb and line-search add probes but no unique reach.
        The keep-set that still matches the grid within 0.02% offline is:

          1. wave-quantization (M,N) @ a short K set (every wave candidate, not top-N only)
          2. coarse M×N plane @ Kmin                             - many-waves / min-K basin
          3. tight local grid around the top scout seeds         - endgame polish
          4. MSMF confirm: saturated, per-shape-warmed, trimmed-median, SUSPECT-filtered
          5. MAMF confirm: boost-clock shapes, short idle+burst, peak iteration, clock-validated
        """
        sms = get_sm_count()
        elem = dtype_element_size(dtype)
        align = max(128 // elem, 1)   # tensor-core element alignment (bf16->64, fp8->128, fp32->32)
        base = 256                    # M/N step: a multiple of `align` and of the 256-wide tile
        if base % align:
            base = ((base // align) + 1) * align
        scout_i, scout_w = args.scout_num_iterations, args.scout_num_warmup_iterations
        max_size = args.max_size
        k_min = 1024
        k_max = max_size - (max_size % 1024) or max_size

        print(f"[auto] SMs={sms} dtype={args.dtype} elem={elem}B base={base} max_size={max_size}")

        memo = {}
        def smeasure(M, N, K, label):
            key = (int(M), int(N), int(K))
            if key not in memo:
                memo[key] = measure(key[0], key[1], key[2], scout_i, scout_w, label)
            return memo[key]

        def discover():
            """Phases 1–3: wave / plane / refine scouting. Returns (seen_mn, square_by_wave)."""
            # Phase 1: wave-quantization-aware (M,N) at a short K set (not just the extremes).
            # Measuring *every* wave (M,N) at several Ks is what gets H200's wave-perfect mid/high-K
            # winners (e.g. 1536x2816x20480) into the confirm set.
            wave_ks = sorted({v for v in (k_min, 8192, 12288, 14336, 16384, k_max) if v <= max_size})
            seen = set()
            square_by_wave = {}  # w -> most-square (M,N); forced into MSMF confirm later
            n_wave = 0
            for w, (sq, layouts) in wave_mn_layouts(sms, max_size).items():
                square_by_wave[w] = sq
                for (mm, nn) in layouts:
                    if (mm, nn) in seen:
                        continue
                    seen.add((mm, nn))
                    for kk in wave_ks:
                        if gemm_fits(mm, nn, kk, elem):
                            smeasure(mm, nn, kk, "wave")
                            n_wave += 1
            print(f"[auto] wave: {len(seen)} (M,N) @ K={wave_ks} -> {n_wave} scouts")

            # Phase 2: coarse M×N plane at Kmin - seeds the many-waves / min-K basin that wave
            # candidates (which bias toward high-AI layouts) can miss.
            plane = [v for v in (2048, 4096, 6144, 8192, 10752, 12288, 14336, 16384, 18432, max_size)
                     if v <= max_size]
            n_plane = 0
            for mm in plane:
                for nn in plane:
                    if gemm_fits(mm, nn, k_min, elem):
                        smeasure(mm, nn, k_min, "plane")
                        n_plane += 1
            print(f"[auto] plane: {len(plane)}x{len(plane)} @ K={k_min} -> {n_plane} scouts")

            # Phase 3: tight local grid around the top scout seeds (endgame polish). Walks a
            # ±r_mn × ±r_mn × ±r_k neighborhood at native lattice resolution (256 / 1024) so an
            # off-axis peak next to a coarse seed is not stepped over. Seed set is
            # basin-diverse (half reserved for wave (M,N)s) so both peaks get polished.
            if args.refine_grid:
                seeds = top_shapes(measured, args.refine_seeds, prefer_mn=seen,
                                   prefer_slots=max(1, args.refine_seeds // 2))
                r_mn, r_k = args.refine_radius_mn, args.refine_radius_k
                print(f"[auto] tight grid (±{r_mn} MN @ {base}, ±{r_k} K @ 1024) around {len(seeds)} seed(s): {seeds}")
                seen_g = set()
                for (sm, sn, sk) in seeds:
                    for dm in range(-r_mn, r_mn + 1):
                        for dn in range(-r_mn, r_mn + 1):
                            for dk in range(-r_k, r_k + 1):
                                mm, nn, kk = sm + dm * base, sn + dn * base, sk + dk * 1024
                                if min(mm, nn, kk) < base or max(mm, nn, kk) > max_size:
                                    continue
                                key = (mm, nn, kk)
                                if key in seen_g:
                                    continue
                                seen_g.add(key)
                                if gemm_fits(mm, nn, kk, elem):
                                    smeasure(mm, nn, kk, "grid")
            return seen, square_by_wave

        def confirm(seen, square_by_wave):
            """Phases 4–5: MSMF (saturated) then MAMF (boost burst) from the same scouts."""
            #   MSMF (sustainable): the number the chip HOLDS once saturated (~TDP). Per-shape warmup
            #     burns off the cold-boost transient, trimmed-median of the steady mean, then drop
            #     shapes that ran below saturation power (clock-boost bursts) via the SUSPECT filter.
            #   MAMF (achievable):  the boost BURST a short real kernel can catch. Global GPU warmup
            #     only - NO per-shape warmup, so the boosted opening iterations are kept - and we take
            #     the peak iteration. No saturation filter (the whole point is the unsaturated boost).
            top, n_confirm = build_msmf_confirm_set(measured, scout_meta, seen, square_by_wave, args)
            reps = max(1, args.confirm_reps)

            msmf_results = confirm_msmf(top, args, dtype, device, telem, reps, all_mean_tflops)
            msmf = select_and_lock_msmf(msmf_results, args, dtype, device, telem, reps)

            # MAMF: candidates = scouts that ran at (near) the run's boost clock, ranked by peak TFLOPS.
            # measure = idle + short burst keeping opening iters; validate = peak at boost clock.
            boost_clk = boost_ref["clk"]
            boost_min = args.boost_clock_ratio * boost_clk if boost_clk else 0.0
            mamf_cands = mamf_candidates(scout_meta, boost_min, n_confirm)
            mamf = confirm_mamf(mamf_cands, args, dtype, device, telem, reps, boost_clk, all_mean_tflops)

            headline["msmf"], headline["mamf"] = msmf, mamf
            # Keep legacy best_* populated for interrupt / grid-compat: MSMF is the primary mean/median;
            # max tracks the MAMF peak when available.
            if msmf:
                cfg = f"{shape_str(msmf['shape'])} (MxNxK)"
                best_tflops.update(mean=msmf["tflops"], median=msmf["tflops"],
                                   max=(mamf["tflops"] if mamf else msmf["tflops"]))
                best_config.update(mean=cfg, median=cfg,
                                   max=(f"{shape_str(mamf['shape'])} (MxNxK)" if mamf else cfg))
            if telem_ok(telem) and not telem.validated:
                print(f"[auto] note: telemetry backend '{telem.backend}' is UNTESTED — power is reported but "
                      f"SUSPECT shapes are NOT excluded, so MSMF may be inflated by a boost burst. Run "
                      f"mamf_spike_probe.py on first access, then add '{telem.backend}' to VALIDATED_BACKENDS "
                      f"in mamf-finder.py if the boost/sat gap holds.")

        seen, square_by_wave = discover()
        confirm(seen, square_by_wave)

    # this is useful for when one wants to interrupt the run - and still report the best outcome so far
    def sigkill_handler(signum, frame):
         finish()
         sys.exit(1)

    signal.signal(signal.SIGINT, sigkill_handler)

    # XXX: the transpose version seemed to work better for MI300X

    # Warm up before measuring: a cold accelerator boosts its clock and over-reports, so run the GPU
    # to steady state first. `adaptive` (default) keys off a *measured characteristic* - the matmul
    # throughput plateau - so it works on any accelerator (power-capped or thermally boosting) and
    # stops as soon as it's warm. The old flat 30s is available as `--warmup fixed`.
    #
    # These two bounds are internal guardrails, not tuning dials, so they're not exposed on the CLI:
    #   MIN - a chip can read stable-but-hot in the first chunks (low CoV, low drift) right after a
    #         boost; the floor forces it to sit long enough to actually thermally settle before we
    #         can declare convergence, avoiding a false "warm" at boosted clock.
    #   MAX - a plain hang guard so an accelerator/telemetry that never plateaus can't spin forever.
    WARMUP_MIN_SECONDS, WARMUP_MAX_SECONDS = 2.0, 45.0
    if telem_ok(telem):
        print(f"telemetry: {telem.backend} (power/clock sampling on"
              f"{'' if telem.validated else '; UNTESTED — report only, no SUSPECT exclusion'})")

    def warmup_adaptive(shape, min_s, max_s, chunk=25, window=4, cov_thr=0.02, drift_thr=0.01):
        # run saturated chunks of matmuls and watch the throughput; converged once a rolling window
        # is both stable (low CoV) and no longer drifting vs the previous window.
        M0, N0, K0 = shape
        hist = []
        t0 = time.monotonic()
        conv = False
        while time.monotonic() - t0 < max_s:
            tf, _, _ = benchmark_mm(M0, N0, K0, dtype, device, chunk, 0)
            hist.append(tf)
            if time.monotonic() - t0 >= min_s and len(hist) >= 2 * window:
                recent, prev = np.array(hist[-window:]), np.array(hist[-2*window:-window])
                cov = recent.std() / recent.mean() if recent.mean() else 1.0
                drift = abs(recent.mean() - prev.mean()) / prev.mean() if prev.mean() else 1.0
                if cov < cov_thr and drift < drift_thr:
                    conv = True
                    break
        el = time.monotonic() - t0
        extra = ""
        if telem_ok(telem):
            c, p = telem.clock(), telem.power()
            if c is not None and p is not None:
                extra = f", clk={c:.0f}MHz pow={p:.0f}W"
        print(f"adaptive warmup: {'throughput plateaued' if conv else 'hit time cap'} after {el:.1f}s / {len(hist)*chunk} iters{extra}")

    if args.warmup == "adaptive":
        print("Warming up (adaptive: until matmul throughput plateaus) ...", flush=True)
        warmup_adaptive(warmup_shape, WARMUP_MIN_SECONDS, WARMUP_MAX_SECONDS)
    else:
        accelerator_warmup_seconds = 30
        end_time = time.monotonic() + accelerator_warmup_seconds
        print(f"Warming up the accelerator for {accelerator_warmup_seconds} secs ... ", end="", flush=True)
        while time.monotonic() < end_time:
            _ = benchmark_mm(warmup_shape[0], warmup_shape[1], warmup_shape[2], dtype, device, args.num_iterations, args.num_warmup_iterations)
        print("accelerator warmup finished")

    if mode == "grid":
        # loop through all sizes to benchmark
        for M in m:
            for N in n:
                for K in k:
                    measure(M, N, K, args.num_iterations, args.num_warmup_iterations)
    else:
        auto_search()

    finish()
