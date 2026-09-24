#!/usr/bin/env python

"""

This is the Maximum Achievable / Sustainable Matmul FLOPS Finder
(MAMF + MSMF).

Both search modes report the SAME two numbers from the shapes they measure; they
differ only in HOW the candidate shapes are chosen:

- **MAMF** — Maximum *Achievable* Matmul FLOPS: the boost-clock burst a short
  kernel can catch (validated against the SM boost clock).
- **MSMF** — Maximum *Sustainable* Matmul FLOPS: what the chip holds once
  saturated near TDP (matches sustained training throughput). For picking a real
  model's shapes this is the number that matters.

- `--search auto` (default): derive near-peak shapes from hardware heuristics and
  report the best MAMF/MSMF the GPU can do anywhere. Great for a spec-sheet
  headline, not tied to any particular model.
- `--search grid`: sweep the M/N/K range YOU give and report the best MAMF/MSMF
  within it. This is the practical case — find the best (and most sustainable)
  shape for a model you're actually running:

python mamf-finder.py --m_range 0 20480 256 --n 4096 --k 4096 --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt

For the auto search, discussion, and important nuances see:
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
# fat-shape forcing, lock-in validation gate, and a busy-sibling-GPU warning. v5: both confirms
# share one candidate set (MAMF no longer pre-filters to scout-time boost clocks); MSMF only
# rejects a high clock when power is also below TDP; print each headline shape in both regimes.
# v6: restore MAMF recall with raw-peak + per-wave candidates, cheap boost-regime screening, then
# run both full confirms over the union of the independently ranked MAMF/MSMF candidates. v7:
# add the low-K boost plane and raw-peak refine seeds needed to cover non-wave MAMF basins. v8:
# the MAMF burst is queued and synchronized once (per-iteration syncs were charging launch latency
# and the idle DVFS re-ramp to the kernel, costing 1.6-19% and re-ranking candidates); clocks are
# attributed per iteration from a sampler timeline instead of per-iteration bracketed reads.
benchmark_version = 8

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
    # Best-effort in-process power/clock telemetry, overridden per vendor below. Defaults are no-ops so
    # a vendor without a fast telemetry library wired up degrades silently (Telemetry.available == False)
    # and the benchmark still runs, just without power/clock reporting. See the Telemetry sampler.
    telemetry_backend   = None    # short id reported in logs, e.g. "nvml" / "amdsmi" / "hlml"
    telemetry_validated = False   # True only where power-based SUSPECT exclusion is spike-probe validated

    # --- auto-search geometry ---
    # `--search auto` derives near-peak GEMM shapes from tile + wave quantization, which needs (a) the
    # compute-unit count for wave packing (compute_unit_count) and (b) a representative kernel tile
    # (gemm_tile_hint). These predict *peak* shapes only where geometry_validated=True; an arch that
    # leaves it False makes `--auto` refuse to run (see main()) because the shapes it would pick are
    # unverified guesses - the user runs an explicit grid instead. See mamf.md "Untested vendors".
    geometry_validated = False
    # Representative (tile_m, tile_n) of the vendor GEMM kernel, used ONLY to seed wave-quantized
    # candidates.
    #
    # What the value implies: the search assumes the kernel emits tile_m x tile_n output tiles, so it
    # builds candidate M x N shapes as multiples of (tile_m, tile_n) that also fill a whole number of
    # waves across compute_unit_count() CUs (blocks = ceil(M/tile_m)*ceil(N/tile_n); see
    # wave_efficiency). So (128, 256) implies "aim for M multiples of 128 and N multiples of 256 that
    # pack the CUs evenly." It sets the *granularity of the guess*, not a hard constraint.
    #
    # Why an approximate value is fine: it is a coarse hint, not measured hardware truth - the real
    # tile a BLAS library picks varies with arch/dtype/shape/version (cuBLASLt can be queried per-shape
    # via CUBLASLT_ALGO_CONFIG_TILE_ID, but that needs a ctypes shim PyTorch doesn't expose). The
    # finder *measures* actual FLOPS on every candidate, so a wrong hint only shifts which shapes are
    # tried (recall), never a reported number. And auto snaps M/N to a base=256 step anyway, so any
    # real tile dividing 256 (64/128/256) is already honored regardless of this value.
    #
    # None here means the arch models no waves. See mamf.md "Untested vendors".
    gemm_tile_hint = None

    def compute_unit_count(self):
        """SMs (NVIDIA) / CUs (AMD) for wave packing; None on archs that don't model waves."""
        return None

    def __init__(self):
        self.arch = "unknown"

    def __repr__(self):
        return self.arch

    @property
    def name(self):
        return self.arch

    # --- telemetry hooks ---
    # An Arch opts into telemetry by setting telemetry_backend to a non-None id (see NVIDIAArch /
    # AMDArch / HPUArch). Once it does, it MUST implement telemetry_init + the readers below: the
    # base raises NotImplementedError (naming the class + missing method) so a half-wired backend
    # fails loudly instead of silently reporting "no data". Archs that leave telemetry_backend = None
    # (XPUArch / MPSArch) opt out and Telemetry never calls these. siblings_busy is the one
    # exception: "no busy siblings" is a legitimate default (only NVIDIAArch enumerates today), so it
    # stays a real no-op rather than a required override.
    def _telemetry_required(self, method):
        raise NotImplementedError(
            f"{type(self).__name__} sets telemetry_backend={self.telemetry_backend!r} but does not "
            f"implement {method}(); implement it, or set telemetry_backend=None to opt out.")

    def telemetry_init(self, index):
        """Acquire and return an opaque per-device handle."""
        self._telemetry_required("telemetry_init")

    def read_power(self, handle):        # Watts
        self._telemetry_required("read_power")

    def read_clock(self, handle):        # SM/GFX clock in MHz
        self._telemetry_required("read_clock")

    def read_device_name(self, handle):  # vendor device name
        self._telemetry_required("read_device_name")

    def siblings_busy(self, handle, self_index=0, util_pct=10, mem_mb=4096):
        """OTHER physical accelerators that look actively computing. Optional even for telemetry
        backends: the safe default is "none reported" (only NVIDIAArch enumerates siblings), so this
        is a real no-op, not a required override."""
        return []

class CudaLikeArch(Arch):
    """ Shared timing/device plumbing for CUDA and ROCm - both are torch device 'cuda'.
    NVIDIAArch and AMDArch below add the vendor-specific bits (compute_info + telemetry). """
    @property
    def device(self):
        return torch.device('cuda:0')

    @property
    def device_info(self):
        return torch.cuda.get_device_properties(device)

    def event(self, enable_timing=True):
        return torch.cuda.Event(enable_timing)

    def synchronize(self):
        torch.cuda.synchronize()

    def compute_unit_count(self):
        # NVIDIA: SM count. ROCm: torch reports the CU count through the same field.
        return torch.cuda.get_device_properties(0).multi_processor_count

class NVIDIAArch(CudaLikeArch):
    """ NVIDIA GPUs (CUDA). Telemetry via NVML - pip install nvidia-ml-py. """
    telemetry_backend   = "nvml"
    telemetry_validated = True    # NVML power validated with mamf_spike_probe.py
    geometry_validated  = True    # wave/tile auto-search matched exhaustive H200/B200 grids
    gemm_tile_hint      = (128, 256)  # representative cuBLAS tile the auto rules were derived against

    def __init__(self):
        self.arch = "cuda"

    @property
    def compute_info(self):
        return f"cuda={torch.version.cuda}"

    def telemetry_init(self, index):
        import pynvml as m
        m.nvmlInit()
        self._nvml = m
        self._clk_arg = m.NVML_CLOCK_SM
        return m.nvmlDeviceGetHandleByIndex(index)

    def read_power(self, handle):
        return self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W

    def read_clock(self, handle):
        return float(self._nvml.nvmlDeviceGetClockInfo(handle, self._clk_arg))

    def read_device_name(self, handle):
        name = self._nvml.nvmlDeviceGetName(handle)
        return name.decode() if isinstance(name, bytes) else name

    def siblings_busy(self, handle, self_index=0, util_pct=10, mem_mb=4096):
        """Return [(idx, util%, mem_MiB)] for OTHER physical GPUs that look actively COMPUTING.

        Sibling GPUs on the same board share its power/cooling budget, so running one while others
        are busy drags the device-under-test's *sustained* clock around run-to-run and quietly ruins
        single-run MSMF reproducibility. Keyed on utilization (real power/heat) - a small resident
        CUDA context (idle, 0% util) doesn't perturb the clock, so the memory threshold is high and
        only flags a genuinely loaded neighbor."""
        m = self._nvml
        out = []
        try:
            count = m.nvmlDeviceGetCount()
        except Exception:
            return out
        for i in range(count):
            if i == self_index:
                continue
            try:
                h = m.nvmlDeviceGetHandleByIndex(i)
                util = m.nvmlDeviceGetUtilizationRates(h).gpu
                mem = m.nvmlDeviceGetMemoryInfo(h).used / 2**20
            except Exception:
                continue
            if util >= util_pct or mem >= mem_mb:
                out.append((i, int(util), int(mem)))
        return out

class AMDArch(CudaLikeArch):
    """ AMD GPUs (ROCm). Telemetry via amdsmi (ships with ROCm).

    UNTESTED ON HARDWARE: the amdsmi calls below were verified against the AMD SMI Python API docs
    (ROCm 6.2+) but have never been run on a real MI box, so telemetry_validated stays False (power
    is reported but not used to exclude shapes). See mamf.md "Untested vendors" for the first-boot
    checklist. If a call is wrong, Telemetry catches it and degrades to unavailable - the benchmark
    still runs, just without power/clock.
    """
    telemetry_backend = "amdsmi"   # telemetry_validated stays False until spike-probed on real hardware
    # geometry_validated stays False (inherited): CU-count-as-SM wave packing and the tile below are
    # UNTESTED on MI300X/MI355X, so `--auto` refuses to run on ROCm until validated. hipBLASLt tiles
    # differ from cuBLAS; this (128,256) is a carried-over placeholder and is INERT until
    # geometry_validated flips. First-boot: compare auto vs a small grid, set the real tile hint, then
    # flip geometry_validated = True. See mamf.md "Untested vendors".
    gemm_tile_hint = (128, 256)

    def __init__(self):
        self.arch = "rocm"

    @property
    def compute_info(self):
        return f"hip={torch.version.hip}, cuda={torch.version.cuda}"

    def telemetry_init(self, index):
        # UNTESTED on hardware; API verified against AMD SMI Python docs (ROCm 6.2+).
        import amdsmi as m
        m.amdsmi_init()
        self._amdsmi = m
        self._clk_arg = m.AmdSmiClkType.GFX   # docs-confirmed enum member (graphics/compute clock)
        return m.amdsmi_get_processor_handles()[index]

    def read_power(self, handle):
        # UNTESTED on hardware. Docs confirm amdsmi_get_power_info() returns Watts:
        # current_socket_power (MI300+), with average_socket_power as the fallback (Navi / MI200 and
        # earlier). Unsupported fields may come back as "N/A" or UINT32_MAX (0xFFFFFFFF).
        info = self._amdsmi.amdsmi_get_power_info(handle)
        INVALID = (None, "N/A")
        # UINT32_MAX guard: some builds report an unsupported field as 0xFFFFFFFF instead of "N/A".
        # Enable this wider set once confirmed on a real MI box (replaces the line above):
        # INVALID = (None, "N/A", 0xFFFFFFFF)
        w = info.get("current_socket_power")
        if w in INVALID:
            w = info.get("average_socket_power")
        return float(w) if w not in INVALID else None

    def read_clock(self, handle):
        # UNTESTED on hardware. ["clk"] is the current-ROCm key for the GFX clock. Confirm on the
        # first MI box.
        return float(self._amdsmi.amdsmi_get_clock_info(handle, self._clk_arg)["clk"])

    def read_device_name(self, handle):
        # UNTESTED on hardware. amdsmi_get_gpu_asic_info() exposes market_name per the docs.
        info = self._amdsmi.amdsmi_get_gpu_asic_info(handle)
        return info.get("market_name") or info.get("vendor_id") or "AMD GPU"

class HPUArch(Arch):
    """ Intel Gaudi*. Telemetry via pyhlml (pip install habana-pyhlml).

    UNTESTED ON HARDWARE: the pyhlml calls below were verified against the Habana pyhlml API docs
    but have never been run on a real Gaudi box, so telemetry_validated stays False (power is
    reported but not used to exclude shapes). See mamf.md "Untested vendors". A wrong call is caught
    by Telemetry and degrades to unavailable - the benchmark still runs, just without power/clock.
    """
    telemetry_backend = "hlml"   # telemetry_validated stays False until spike-probed on real hardware

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

    def telemetry_init(self, index):
        # UNTESTED on hardware; API verified against Habana pyhlml docs.
        import pyhlml as m
        m.hlmlInit()
        self._hlml = m
        return m.hlmlDeviceGetHandleByIndex(index)

    def read_power(self, handle):
        # UNTESTED on hardware. Docs confirm hlmlDeviceGetPowerUsage() returns milliwatts (like
        # NVML), so /1000 -> Watts. Confirm on the first Gaudi box.
        return self._hlml.hlmlDeviceGetPowerUsage(handle) / 1000.0

    def read_clock(self, handle):
        # UNTESTED on hardware. clock_type 0 == HLML_CLOCK_SOC, which the docs list as the only
        # clock domain supported on Gaudi (IC/MME/TPC are Goya-only). Confirm on the first Gaudi box.
        return float(self._hlml.hlmlDeviceGetClockInfo(handle, 0))

    def read_device_name(self, handle):
        # UNTESTED on hardware. pyhlml has no documented market-name call, so report the handle.
        return f"Gaudi:{handle}"

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
    returns: an Arch subclass instance for the active accelerator
    """
    # cuda / rocm (both are torch device 'cuda'; split by HIP for vendor-accurate telemetry)
    if torch.cuda.is_available():
        return AMDArch() if torch.version.hip is not None else NVIDIAArch()

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
# The burst is QUEUED, not run one-synchronized-iteration-at-a-time. Synchronizing before recording
# each start event leaves the GPU idle at the moment the event is taken, which folds kernel-launch
# latency (~6-8us) and the DVFS re-ramp out of idle into the measured window. Measured on B200/H200/
# B300 that costs 1.6-19% depending on shape - and because the penalty scales with kernel footprint
# it silently RE-RANKS candidates, biasing MAMF toward small shapes. See
# results/mamf-20260910-h200-b200-b300-torch214/timing-artifact-findings.md and
# mamf_launch_gap_probe.py.
#
# So: idle first (the SM clock climbs back to boost - that is the regime MAMF reports), then queue
# the whole burst and synchronize once, exactly like benchmark_mm(). Per-iteration clock attribution
# is preserved WITHOUT per-iteration syncs: a background sampler timestamps (clock, power) on the
# host clock while the burst runs, and each iteration's GPU-time window is projected onto that host
# timeline by anchoring the post-sync host timestamp to the burst's last end event. Each iteration
# is then paired with the MINIMUM clock sampled inside its own window - the same conservative floor
# the bracketed reads gave, so a published MAMF is still certified to have run at the reported clock.
def measure_boost_burst(m, n, k, dtype, device, iters, telem=None, idle_before_s=0.0):
    """Return a list of (tflops, clock_MHz_or_None, power_W_or_None) - one entry per iteration."""
    op, l2_cache, C, C_rand, flos = prepare_gemm(m, n, k, dtype, device)
    have_telem = telem_ok(telem)

    # short idle after allocate lets the SM clock climb back to boost
    arch.synchronize()
    if idle_before_s and idle_before_s > 0:
        time.sleep(idle_before_s)

    samples = []  # (host_monotonic, clock, power)
    stop = threading.Event()
    th = None
    if have_telem:
        def _sample_loop():
            while not stop.is_set():
                p, c = telem.power(), telem.clock()
                samples.append((time.monotonic(), c, p))
                stop.wait(0.0005)
        th = threading.Thread(target=_sample_loop, daemon=True)
        th.start()

    burst_start = arch.event(enable_timing=True)
    starts = [arch.event(enable_timing=True) for _ in range(iters)]
    ends = [arch.event(enable_timing=True) for _ in range(iters)]
    try:
        burst_start.record()
        for i in range(iters):
            with torch.no_grad():
                l2_cache.zero_()
                C.copy_(C_rand)
                starts[i].record()
                op()
                ends[i].record()
        arch.synchronize()
        host_end = time.monotonic()
    finally:
        if th is not None:
            stop.set()
            th.join()

    # anchor: host_end corresponds to the GPU-time offset of the final end event
    total_ms = burst_start.elapsed_time(ends[-1])

    def _window(lo_ms, hi_ms):
        """(min clock, mean power) among samples inside this iteration's host-time window."""
        if not samples:
            return None, None
        lo = host_end - (total_ms - lo_ms) / 1000.0
        hi = host_end - (total_ms - hi_ms) / 1000.0
        inside = [(c, p) for (ht, c, p) in samples if lo <= ht <= hi and c is not None]
        if not inside:  # window shorter than the sampling interval - use the nearest reading
            mid = (lo + hi) / 2.0
            nearest = min(samples, key=lambda s: abs(s[0] - mid))
            inside = [(nearest[1], nearest[2])] if nearest[1] is not None else []
        if not inside:
            return None, None
        pw = [p for _, p in inside if p is not None]
        return min(c for c, _ in inside), (float(np.mean(pw)) if pw else None)

    out = []
    for i in range(iters):
        t_ms = starts[i].elapsed_time(ends[i])
        tf = flos / (t_ms / 1000.0 * 1e12) if t_ms > 0 else 0.0
        clk, pw = _window(burst_start.elapsed_time(starts[i]), burst_start.elapsed_time(ends[i]))
        out.append((tf, clk, pw))
    return out


### Auto-search helpers (MAMF + MSMF) ###
#
# Instead of brute-forcing a 3D grid of MxNxK shapes, `--search auto` constructs a small set of
# shapes the accelerator should run at/near peak on, using the three rules from "The Case for
# Co-Designing Model Architectures with Hardware" (https://arxiv.org/abs/2401.14489):
#   1. Tensor-core alignment: M, N, K are multiples of `128 bytes / dtype_size` elements.
#   2. Tile quantization:     the MxN output divides evenly into the kernel's tile (Arch.gemm_tile_hint,
#                             (128,256) on NVIDIA - a coarse hint, not a queried per-kernel tile).
#   3. Wave quantization:     the number of output tiles is a multiple of the compute-unit count
#                             (Arch.compute_unit_count()), so the final wave is full (see `wave_efficiency`).
# K only sets the arithmetic intensity (how compute-bound the GEMM is); it never appears in the
# tile/wave math, so it is pinned and coarsely swept rather than gridded. See benchmarks/README.md.
# After scouting, auto confirms two headlines: MAMF (boost-clock burst) and MSMF (saturated).
#
# The compute-unit count and tile hint that feed these rules are per-Arch (compute_unit_count() /
# gemm_tile_hint) and only *validated* to predict peak shapes where Arch.geometry_validated=True
# (NVIDIA today). main() refuses `--auto` on any other arch - see the geometry gate there.

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


def fmt_runs(xs):
    """Comma-joined 1-decimal list for confirm/lock-in logs."""
    return ", ".join(f"{x:.1f}" for x in xs)


def apply_headlines(headline, best_tflops, best_config, msmf, mamf):
    """Store MAMF/MSMF in `headline` and mirror into legacy best_* for interrupt fallback."""
    headline["msmf"], headline["mamf"] = msmf, mamf
    if not msmf:
        return
    cfg = f"{shape_str(msmf['shape'])} (MxNxK)"
    best_tflops.update(mean=msmf["tflops"], median=msmf["tflops"],
                       max=(mamf["tflops"] if mamf else msmf["tflops"]))
    best_config.update(mean=cfg, median=cfg,
                       max=(f"{shape_str(mamf['shape'])} (MxNxK)" if mamf else cfg))


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


def top_shapes_by_peak(scout_meta, n):
    """Top scout shapes by observed peak, without MSMF's power discount."""
    return [shp for shp, _ in sorted(
        scout_meta.items(), key=lambda kv: kv[1]["mx"], reverse=True)[:max(0, n)]]


def build_mamf_recall_pool(scout_meta, wave_layouts_by_wave, args):
    """Bounded boost-screen pool: raw leaders plus top-K scouts per wave (M,N) layout.

    A saturated scout mean cannot predict an idle+burst winner. Keeping several K choices for
    every wave layout prevents a noisy 20-iteration scout from dropping a disconnected boost
    basin (the v5 H200/B200 regression) before it is measured in the MAMF regime.
    """
    raw = top_shapes_by_peak(scout_meta, args.mamf_raw_forced)
    wave = []
    per_layout = max(1, args.mamf_wave_k)
    for w in sorted(wave_layouts_by_wave):
        for mn in sorted(wave_layouts_by_wave[w]):
            ranked = sorted(
                ((shp, mt["mx"]) for shp, mt in scout_meta.items() if shp[:2] == mn),
                key=lambda kv: kv[1], reverse=True)
            wave.extend(shp for shp, _ in ranked[:per_layout])
    pool = list(dict.fromkeys(raw + wave))
    provenance = {s: [] for s in pool}
    for s in raw:
        provenance[s].append("raw")
    for s in wave:
        provenance[s].append("wave")
    return pool, provenance


def screen_mamf_candidates(pool, args, dtype, device, telem, boost_clk):
    """Cheap MAMF-regime screen; return the strongest boost-validated candidates."""
    if not pool:
        return []
    boost_min = args.boost_clock_ratio * boost_clk if boost_clk else 0.0
    scored = []
    print(f"\n[confirm] MAMF recall screen: {len(pool)} shapes, "
          f"{args.mamf_screen_iters} iters, {args.mamf_screen_idle_s*1000:.0f}ms idle ...")
    for shp in pool:
        burst = measure_boost_burst(
            shp[0], shp[1], shp[2], dtype, device, max(1, args.mamf_screen_iters),
            telem=telem, idle_before_s=max(0.0, args.mamf_screen_idle_s))
        at_boost = [x for x in burst if not boost_min or (x[1] is not None and x[1] >= boost_min)]
        valid = at_boost or burst
        if valid:
            peak, clk, power = max(valid, key=lambda x: x[0])
            scored.append((float(peak), shp, clk, power, bool(at_boost)))
    scored.sort(reverse=True)
    chosen = [shp for _, shp, _, _, _ in scored[:max(1, args.mamf_confirm_top)]]
    preview = ", ".join(f"{shape_str(s)}={tf:.1f}" for tf, s, _, _, _ in
                        scored[:min(8, len(scored))])
    print(f"[confirm] MAMF recall screen leaders: {preview}")
    return chosen


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
        print(f"[confirm] forced confirms (low-wave squares + fattest scouts): {forced}")
    return shapes, n_confirm


def thermal_soak(device, telem, max_size, soak_s):
    """Drive the chip to thermal steady state before MSMF confirm.

    Biggest reproducibility lever: a cold/cooler GPU boosts and reads high, so without soaking
    the MSMF headline depends on how warm the card happened to be. Soak in bf16 (heat is heat)
    on a big square; stop early once the SM clock stops dropping.
    """
    if not soak_s or soak_s <= 0:
        return
    print(f"\n[confirm] thermal soak (<= {soak_s}s) to reach steady-state clock before MSMF confirm ...")
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
        print(f"[confirm] soak done, SM clock settled ~{telem.clock():.0f}MHz")


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
    hi_p = [r for r in sat_pool if r["power"] is not None and ref_p
            and r["power"] >= args.msmf_sat_power_ratio * ref_p and r["clock"] is not None]
    sat_clock = min((r["clock"] for r in hi_p), default=None)
    if exclude_ok and sat_clock is None:
        # Common with a narrow grid that has no fat/high-K shape: everything still floats above
        # the true saturated floor, so there is no honest low-clock reference to filter against.
        print("[confirm] WARNING: no saturated-clock anchor in the confirm set (no near-TDP shape with "
              "a usable clock). MSMF filters are weakened — widen the range (esp. larger min(M,N,K) / "
              "higher K) so a truly sustainable shape can pin the floor.")

    def _saturated(p):
        return ref_p is not None and p is not None and p >= args.msmf_sat_power_ratio * ref_p

    def _boosting(r, clk=None, pw=None):
        c = r["clock"] if clk is None else clk
        if not (exclude_ok and sat_clock is not None and c is not None
                and c > sat_clock * args.msmf_clock_ratio):
            return False
        # A high clock alone does not prove the shape is still riding a boost transient: a less
        # dense layout can sit pinned at TDP *and* hold a higher clock, and then its number is
        # genuinely sustainable (measured on B300, where the clock-only gate rejected the true
        # MSMF winner). Only reject when the power draw is also short of saturation.
        return not _saturated(r["power"] if pw is None else pw)

    stable_pool = ([r for r in sat_pool if r["spread"] <= args.msmf_max_spread and not _boosting(r)]
                   or [r for r in sat_pool if r["spread"] <= args.msmf_max_spread]
                   or [r for r in sat_pool if not _boosting(r)] or sat_pool)
    prelim = max(stable_pool, key=lambda r: r["tflops"]) if stable_pool else None
    for r in sat_pool:
        if not (prelim and r["tflops"] > prelim["tflops"]):
            continue
        if _boosting(r):
            print(f"[confirm] MSMF ignored {shape_str(r['shape'])} ({r['tflops']:.1f} TFLOPS @ "
                  f"{r['clock']:.0f}MHz vs {sat_clock:.0f}MHz saturated): still boosting, not sustainable "
                  f"(trends toward MAMF)")
        elif r["spread"] > args.msmf_max_spread:
            print(f"[confirm] MSMF ignored {shape_str(r['shape'])} ({r['tflops']:.1f} TFLOPS, "
                  f"spread={r['spread']:.1f}% > {args.msmf_max_spread:.0f}%): too jittery to reproduce")
    for r in msmf_results:
        if _suspect(r) and prelim and r["tflops"] > prelim["tflops"]:
            print(f"[confirm] MSMF excluded {shape_str(r['shape'])} ({r['tflops']:.1f} TFLOPS @ {r['power']:.0f}W): "
                  f"unsaturated clock-boost, not sustainable (counts toward MAMF)")

    ordered = sorted(stable_pool, key=lambda r: r["tflops"], reverse=True)
    if not (ordered and args.msmf_lock_reps > reps):
        return prelim

    tries = ordered[:max(1, args.msmf_lock_tries)]
    for idx, cand in enumerate(tries):
        shp = cand["shape"]
        print(f"\n[confirm] MSMF lock-in: re-measuring {shape_str(shp)} x{args.msmf_lock_reps} ...")
        lm, lp, lc = measure_saturated_reps(shp, args.msmf_lock_reps, args, dtype, device, telem)
        lock_tf = trimmed_median(lm)
        lock_spread = spread_pct(lm) if lm else cand["spread"]
        lock_clk = median_or_none(lc) if lc else cand.get("clock")
        lock_pw = median_or_none(lp) if lp else cand.get("power")
        lock_boosting = _boosting(cand, clk=lock_clk, pw=lock_pw)
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
            print(f"[confirm] MSMF lock-in: {lock_tf:.1f} TFLOPS  spread={lock_spread:.1f}%"
                  f"{f'  {lock_clk:.0f}MHz' if lock_clk is not None else ''}{tag}  "
                  f"(runs: {fmt_runs(lm)})")
            if tags:
                print("[confirm] WARNING: MSMF headline is a best-available fallback (not lock-in clean). "
                      "For a reproducible sustainable number, widen the shape range so a fat/high-K "
                      "shape can saturate and pass the spread+clock gates — common with a narrow grid.")
            return cand
        why = []
        if not spread_ok:
            why.append(f"spread {lock_spread:.1f}% > {args.msmf_max_spread:.0f}%")
        if lock_boosting:
            why.append(f"clock {lock_clk:.0f}MHz > {sat_clock:.0f}*{args.msmf_clock_ratio:.2f} "
                       f"saturated floor")
        print(f"[confirm] MSMF lock-in rejected {shape_str(shp)}: {'; '.join(why)} "
              f"over {args.msmf_lock_reps} reps -> trying next candidate")
    return prelim


def confirm_msmf(shapes, args, dtype, device, telem, reps, all_mean_tflops):
    """Thermal soak + per-shape MSMF confirm measurements.

    Returns a list of result dicts (shape/tflops/power/clock/spread). Caller runs
    `select_and_lock_msmf` to pick and lock-in the headline.
    """
    thermal_soak(device, telem, args.max_size, args.msmf_soak_s)
    red = "trimmed-median" if reps >= 5 else "median"
    print(f"\n[confirm] MSMF (sustainable) confirm: {len(shapes)} shapes, {args.num_iterations} iters x {reps} reps, "
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
        print(f"[confirm] MSMF {shape_str(shp)}: {red}-of-{reps} mean={msmf_results[-1]['tflops']:.1f} "
              f"TFLOPS{ptxt}  spread={spread:.1f}%  (runs: {fmt_runs(means)})")
    return msmf_results


def select_mamf(mamf_results, boost_clk):
    """Pick the MAMF headline, preferring boost-validated readings."""
    boost_pool = [r for r in mamf_results if r["boost"]] or mamf_results
    mamf = max(boost_pool, key=lambda r: r["tflops"]) if boost_pool else None
    if mamf and not mamf["boost"]:
        print(f"[confirm] WARNING: no MAMF candidate reached the boost clock (~{boost_clk:.0f}MHz); "
              f"headline {mamf['tflops']:.1f} TFLOPS is a base/throttled-clock reading, not a true boost burst")
    return mamf


def confirm_mamf(cands, args, dtype, device, telem, reps, boost_clk, all_mean_tflops):
    """Boost-burst confirm; returns one result dict per candidate (caller picks the headline).

    Each candidate gets `reps` short bursts preceded by idle so the SM clock recovers to boost.
    Only iterations whose bracketed clock reached boost count toward the headline. Fat/saturated
    scouts are included: idle+burst recovers boost even when the scout itself ran at the floor.
    """
    boost_min = args.boost_clock_ratio * boost_clk if boost_clk else 0.0
    burst_iters = max(1, args.mamf_burst_iters)
    idle_s = max(0.0, args.mamf_idle_s)
    print(f"\n[confirm] MAMF (achievable) confirm: {len(cands)} shapes (same set as MSMF; "
          f"boost≈{boost_clk:.0f}MHz, need ≥{boost_min:.0f}MHz), {burst_iters} iters x {reps} reps, "
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
        print(f"[confirm] MAMF {shape_str(shp)}: peak={peak:.1f} TFLOPS{ptxt}{ctxt}{flag}  "
              f"(top peaks: {fmt_runs(top_pk)})")
    return mamf_results


def _burst_one(shp, args, dtype, device, telem, reps, boost_clk, all_mean_tflops):
    """One-shape MAMF burst (used to fill a missing cross-check cell)."""
    results = confirm_mamf([shp], args, dtype, device, telem, reps, boost_clk, all_mean_tflops)
    return results[0] if results else None


def print_regime_crosscheck(msmf, mamf, msmf_results, mamf_results, args, dtype, device, telem,
                            reps, boost_clk, all_mean_tflops):
    """Print each headline shape in both regimes so the boost→saturated penalty is readable."""
    msmf_map = {r["shape"]: r for r in msmf_results}
    mamf_map = {r["shape"]: r for r in mamf_results}
    shapes = []
    for r in (mamf, msmf):
        if r and r["shape"] not in shapes:
            shapes.append(r["shape"])
    if not shapes:
        return

    def cell(r):
        if not r:
            return "—"
        bits = [f"{r['tflops']:.1f}"]
        if r.get("power") is not None:
            bits.append(f"{r['power']:.0f}W")
        if r.get("clock") is not None:
            bits.append(f"{r['clock']:.0f}MHz")
        return " ".join(bits)

    print("\n[confirm] same-shape cross-check (each headline shape in both regimes):")
    print(f"  {'shape':<22}  {'MAMF (boost)':<32}  {'MSMF (saturated)':<32}  MSMF/MAMF")
    for shp in shapes:
        if shp not in mamf_map:
            filled = _burst_one(shp, args, dtype, device, telem, reps, boost_clk, all_mean_tflops)
            if filled:
                mamf_map[shp] = filled
        if shp not in msmf_map:
            means, powers, clocks = measure_saturated_reps(
                shp, reps, args, dtype, device, telem, collect_means=all_mean_tflops)
            msmf_map[shp] = dict(shape=shp, tflops=trimmed_median(means),
                                 power=median_or_none(powers), clock=median_or_none(clocks),
                                 spread=spread_pct(means))
        m, s = mamf_map.get(shp), msmf_map.get(shp)
        ratio = f"{100.0 * s['tflops'] / m['tflops']:.0f}%" if m and s and m["tflops"] else "—"
        roles = []
        if mamf and shp == mamf["shape"]:
            roles.append("MAMF*")
        if msmf and shp == msmf["shape"]:
            roles.append("MSMF*")
        tag = f"  ({', '.join(roles)})" if roles else ""
        print(f"  {shape_str(shp):<22}  {cell(m):<32}  {cell(s):<32}  {ratio}{tag}")


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
# The vendor-specific reads live on the Arch subclasses (NVIDIAArch / AMDArch / HPUArch) as
# telemetry_init / read_power / read_clock / read_device_name / siblings_busy, exactly like event()
# and synchronize(). Telemetry below is a thin, vendor-agnostic sampler: it holds the per-device
# handle and the failure/validated state and delegates every read to `arch`. Add a vendor by
# overriding those hooks on its Arch subclass - nothing here changes.
# See mamf.md "Untested vendors" for API assumptions and the first-boot checklist.

# Backends whose power readings have been validated with mamf_spike_probe.py against a known
# boost-vs-saturated gap. Derived from the Arch subclasses so there is a single source of truth: a
# backend graduates by flipping telemetry_validated = True on its Arch subclass. Others may still
# *sample* (for logging) but must not *exclude* shapes from the headline.
VALIDATED_BACKENDS = frozenset(
    a.telemetry_backend for a in (NVIDIAArch, AMDArch, HPUArch)
    if a.telemetry_backend is not None and a.telemetry_validated
)


class Telemetry:
    """Vendor-agnostic in-process power/clock sampler.

    Holds a per-device handle acquired from the active Arch and delegates every vendor-specific read
    to it - all vendor knowledge lives on the Arch subclasses. Degrades silently to unavailable when
    the vendor library is missing or a device handle can't be acquired.
    """

    def __init__(self, arch, index=0):
        self.arch = arch
        self._h = None
        # Only archs that opted into telemetry (telemetry_backend set) are asked for a handle; the
        # rest (XPU/MPS/unknown) degrade to unavailable without ever touching the hooks. A missing
        # vendor lib or an un-acquirable handle degrades silently, but a NotImplementedError (backend
        # declared yet a hook unimplemented) is a wiring bug and propagates loudly.
        if arch is not None and arch.telemetry_backend is not None:
            try:
                self._h = arch.telemetry_init(index)
            except NotImplementedError:
                raise
            except Exception:
                self._h = None
        self.backend = arch.telemetry_backend if self._h is not None else None

    @property
    def available(self):
        return self._h is not None

    @property
    def validated(self):
        """If False, sample+report power but do NOT use it to exclude shapes from the headline."""
        return self._h is not None and self.arch.telemetry_validated

    # NotImplementedError propagates (a declared backend forgot a hook); other errors degrade to None.
    def power(self):  # Watts, or None
        if self._h is None:
            return None
        try:
            return self.arch.read_power(self._h)
        except NotImplementedError:
            raise
        except Exception:
            return None

    def clock(self):  # SM/GFX clock in MHz, or None
        if self._h is None:
            return None
        try:
            return self.arch.read_clock(self._h)
        except NotImplementedError:
            raise
        except Exception:
            return None

    def siblings_busy(self, self_index=0, util_pct=10, mem_mb=4096):
        """OTHER physical accelerators that look actively COMPUTING - see Arch.siblings_busy and its
        NVIDIAArch override. A loaded neighbor on the same board drags the device-under-test's
        sustained clock and ruins single-run MSMF reproducibility."""
        if self._h is None:
            return []
        try:
            return self.arch.siblings_busy(self._h, self_index=self_index,
                                           util_pct=util_pct, mem_mb=mem_mb)
        except Exception:
            return []

    def device_name(self):
        if self._h is None:
            return None
        try:
            return self.arch.read_device_name(self._h)
        except Exception:
            return None


class FakeTelemetry(Telemetry):
    """Scripted (power_W, clock_MHz) sequence for offline SUSPECT / ranking tests. No GPU.

    Each `power()` call advances one step; `clock()` returns the clock paired with the last
    `power()` reading (mirrors how the finder sampler calls power then clock each tick).
    """

    def __init__(self, samples=None, loop=True):
        self.arch = None
        self._h = None
        self.backend = "fake"
        self._samples = list(samples or [(1000.0, 1300.0)])
        self._i = 0
        self._loop = loop
        self._last = self._samples[0] if self._samples else (None, None)

    @property
    def available(self):
        return True

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
    parser.add_argument("--shapes_file", type=str,
                        help="grid: exact M,N,K tuples, one per line (whitespace, comma, or MxNxK); "
                             "avoids taking the Cartesian product of independent dimension lists")

    parser.add_argument("--search", choices=["auto", "grid"], default="auto",
                        help="Both modes report MAMF (boost) + MSMF (sustainable) via the same confirm phase. "
                             "auto (default): lean directed search over heuristic shapes (wave@{Kmin,Kmax} -> "
                             "plane@Kmin -> tight grid -> confirm) - the best the GPU can do anywhere. "
                             "grid: sweep the --m/--n/--k[_range] YOU give, then confirm - the best shape in your "
                             "range for a real model. Passing any shape argument implies grid.")
    parser.add_argument("--scout_only", action="store_true",
                        help="skip MAMF/MSMF confirm after scouting; intended for partitioned "
                             "exhaustive-oracle generation whose candidates are confirmed later")
    parser.add_argument("--num_iterations", type=int, default=100, help='The number of iterations used to benchmark each GEMM')
    parser.add_argument("--num_warmup_iterations", type=int, default=50, help='The number of warmup iterations')
    parser.add_argument("--scout_num_iterations", type=int, default=20, help='scout-phase iterations per shape (auto heuristics and grid sweep); winners are re-measured with --num_iterations in confirm')
    parser.add_argument("--scout_num_warmup_iterations", type=int, default=8, help='scout-phase warmup iterations per shape (GPU is globally warm, so ranking needs few)')
    parser.add_argument("--confirm_top", type=int, default=10, help='how many of the best scouted (power-ranked) shapes to re-measure with the full iteration count (higher = more reliable 1-shot peak, still seconds)')
    parser.add_argument("--confirm_fat_forced", type=int, default=3, help='also force this many FATTEST scouts (largest min(M,N,K), then largest volume) into the MSMF confirm set - a shape large in every dim reliably saturates to TDP under the confirm, so it anchors the saturated-clock reference and guarantees a real sustainable candidate even when the rest of the set is small/skinny')
    parser.add_argument("--mamf_confirm_top", type=int, default=12, help='how many winners from the cheap boost-regime recall screen to promote into the shared full confirm set')
    parser.add_argument("--mamf_raw_forced", type=int, default=8, help='how many global scout-peak leaders to force into the MAMF recall pool in addition to per-wave candidates')
    parser.add_argument("--mamf_wave_k", type=int, default=3, help='auto: how many K variants per wave (M,N) layout enter the cheap MAMF recall screen; protects boost winners whose saturated scout ranking is noisy')
    parser.add_argument("--mamf_screen_iters", type=int, default=2, help='isolated iterations per candidate in the cheap MAMF recall screen before the full repeated confirm')
    parser.add_argument("--mamf_screen_idle_s", type=float, default=0.05, help='idle before each cheap MAMF recall-screen burst; enough to expose boost candidates without paying the full confirm idle')
    parser.add_argument("--mamf_burst_iters", type=int, default=20, help='iterations per rep for the MAMF (achievable) confirm - kept SHORT so the boost burst is not re-saturated away; the peak iteration across reps is the MAMF')
    parser.add_argument("--mamf_idle_s", type=float, default=0.25, help='idle time before each MAMF burst so the SM clock recovers to boost (a short real kernel enjoys this); 0 to disable')
    parser.add_argument("--boost_clock_ratio", type=float, default=0.97, help='a MAMF reading counts as a real boost-clock burst only if its peak SM clock >= this * the highest clock seen in the run; below that it is a throttled/base-clock reading and is flagged')
    parser.add_argument("--confirm_reps", type=int, default=5, help='repeat each confirmed shape this many times and report the trimmed-MEDIAN mean - defeats power-cap clock-jitter spikes. Default 5 enables the trimmed median (drop min+max); use 3 for a faster but noisier headline')
    parser.add_argument("--msmf_max_spread", type=float, default=3.0, help='a sustainable (MSMF) shape whose reps swing more than this %% cannot be reproduced by a reader, so it is kept out of the headline (still reported); only falls back to jittery shapes if none measured stably')
    parser.add_argument("--msmf_clock_ratio", type=float, default=1.04, help='a MSMF shape whose saturated SM clock exceeds the run saturated-clock floor (the clock of the most power-saturated shape) by more than this factor is treated as still boosting ONLY if its power is also below --msmf_sat_power_ratio * max power; a sparse layout pinned at TDP with a higher clock is kept')
    parser.add_argument("--msmf_sat_power_ratio", type=float, default=0.97, help='a MSMF shape drawing at least this fraction of the run max power counts as power-saturated: it pins the saturated-clock floor, and it is exempt from the --msmf_clock_ratio boost rejection (a sparse layout can hold a high clock while still pinned at TDP, and that number is real)')
    parser.add_argument("--msmf_lock_reps", type=int, default=9, help='re-measure the MSMF candidate this many times as a stability GATE - publish the trimmed-median only if its spread is within --msmf_max_spread, else move to the next candidate. A tighter, reproducible headline. Set <= confirm_reps to skip the lock-in pass')
    parser.add_argument("--msmf_lock_tries", type=int, default=4, help='how many top MSMF candidates the lock-in gate may walk through (high-TFLOPS first) before accepting the best-available one, if none pass the spread gate')
    parser.add_argument("--msmf_soak_s", type=float, default=20.0, help='seconds to drive the GPU to thermal steady-state (hot, clock settled at the saturated floor) before the MSMF confirm, so the sustainable number does not depend on how warm the card happened to be. Stops early once the clock stops dropping. 0 to disable')
    parser.add_argument("--confirm_warmup_passes", type=int, default=1, help='throwaway saturated passes run before the timed MSMF reps of each shape (and before the lock-in), so a freshly-switched shape starts at steady-state clock instead of reading cold/boosted on the first rep (false jitter). 0 to disable')
    parser.add_argument("--refine_grid", default=True, action=argparse.BooleanOptionalAction, help='auto: run a tight exhaustive local grid around the top scout seeds before confirm (--no-refine_grid to skip)')
    parser.add_argument("--refine_seeds", type=int, default=4, help='auto: how many top scout seeds to center the tight local grid on')
    parser.add_argument("--refine_radius_mn", type=int, default=4, help='auto: tight-grid half-width along M and N, in steps of 256; 4 bridges adjacent coarse-plane cells for non-wave MAMF basins')
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
    shape_args_given = args.shapes_file is not None or any(
        x is not None for x in (args.m, args.m_range, args.n, args.n_range, args.k, args.k_range))
    mode = "grid" if (args.search == "grid" or shape_args_given) else "auto"

    # Auto-search geometry (compute_unit_count + gemm_tile_hint) is validated to predict near-peak
    # shapes only on NVIDIA. On every other arch it is unverified guesswork, so refuse `--auto` and
    # point the user at an explicit search space instead of silently publishing shaky numbers.
    if mode == "auto" and not arch.geometry_validated:
        parser.error(
            f"--search auto is validated only on NVIDIA; its geometry (compute_unit_count + "
            f"gemm_tile_hint) is UNTESTED on {arch.name!r}, so the shapes it would pick are unverified "
            f"guesses rather than near-peak. Either:\n"
            f"  (a) run an explicit search space WITHOUT auto: --search grid with --m/--n/--k "
            f"(or --m_range/--n_range/--k_range) or --shapes_file; or\n"
            f"  (b) validate the {arch.name} geometry on real hardware (compare auto vs a small grid, "
            f"set the real gemm_tile_hint) and set geometry_validated = True on its Arch subclass. "
            f"See mamf.md 'Untested vendors'.")

    m = n = k = None
    explicit_shapes = None
    if mode == "grid":
        if args.shapes_file:
            explicit_shapes = []
            for lineno, line in enumerate(Path(args.shapes_file).read_text().splitlines(), 1):
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                fields = re.split(r"[\s,xX]+", line)
                if len(fields) != 3:
                    parser.error(f"{args.shapes_file}:{lineno}: expected M,N,K, got {line!r}")
                explicit_shapes.append(tuple(map(int, fields)))
            if not explicit_shapes:
                parser.error(f"--shapes_file {args.shapes_file!r} contains no shapes")
            range_info = f"exact shapes from {args.shapes_file} ({len(explicit_shapes)} tuples)"
            warmup_shape = explicit_shapes[0]
        else:
            missing = [name for name, val, rng in (
                ("m", args.m, args.m_range), ("n", args.n, args.n_range), ("k", args.k, args.k_range))
                if val is None and rng is None]
            if missing:
                parser.error(f"--search grid requires shapes for: {', '.join(missing)} "
                             "(use --{dim}, --{dim}_range, or --shapes_file)")

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
        range_info = f"auto-search (CUs={arch.compute_unit_count()}, dtype={args.dtype}, max_size={args.max_size})"
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
    # both modes report two headlines from the shapes they measure:
    #   mamf = Maximum ACHIEVABLE  Matmul FLOPS - the boost burst a short kernel can catch
    #   msmf = Maximum SUSTAINABLE Matmul FLOPS - what the chip holds once saturated at ~TDP
    # auto picks candidate shapes from heuristics; grid picks them from the user-supplied range.
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

    def confirm_phase(seen=None, square_by_wave=None, wave_layouts_by_wave=None):
        """MSMF (saturated) then MAMF (boost burst) confirm — shared by auto and grid.

        Both modes populate `measured`/`scout_meta` first (auto via heuristic scouts, grid via
        its user-supplied sweep); this then ranks candidates and produces the two headlines:
          MSMF (sustainable): the number the chip HOLDS once saturated (~TDP). Per-shape warmup
            burns off the cold-boost transient, trimmed-median of the steady mean, then drop
            shapes that ran below saturation power (clock-boost bursts) via the SUSPECT filter.
          MAMF (achievable):  the boost BURST a short real kernel can catch. Short idle+burst,
            opening iterations kept, peak iteration, clock-validated. No saturation filter.
            The full confirms share the union of independently ranked MAMF and MSMF candidates:
            a fat shape that saturated while scouting still recovers boost after idle, while a
            low-power wave candidate cannot be crowded out by MSMF's power-aware ranking.
        The wave metadata is populated by auto; grid uses raw scout leaders from its supplied range.
        """
        seen = seen or set()
        square_by_wave = square_by_wave or {}
        wave_layouts_by_wave = wave_layouts_by_wave or {}
        msmf_cands, _n_confirm = build_msmf_confirm_set(
            measured, scout_meta, seen, square_by_wave, args)
        reps = max(1, args.confirm_reps)
        boost_clk = boost_ref["clk"]

        recall_pool, provenance = build_mamf_recall_pool(
            scout_meta, wave_layouts_by_wave, args)
        screened = screen_mamf_candidates(
            recall_pool, args, dtype, device, telem, boost_clk)
        raw_forced = top_shapes_by_peak(scout_meta, args.mamf_raw_forced)
        mamf_cands = list(dict.fromkeys(raw_forced + screened))
        top = list(dict.fromkeys(msmf_cands + mamf_cands))
        print(f"[confirm] shared full-confirm union: {len(top)} shapes "
              f"(MSMF={len(msmf_cands)}, MAMF={len(mamf_cands)}, overlap="
              f"{len(set(msmf_cands) & set(mamf_cands))})")
        if mamf_cands:
            detail = ", ".join(
                f"{shape_str(s)}[{'+'.join(provenance.get(s, ['screen']))}]"
                for s in mamf_cands)
            print(f"[confirm] MAMF promoted candidates: {detail}")

        msmf_results = confirm_msmf(top, args, dtype, device, telem, reps, all_mean_tflops)
        msmf = select_and_lock_msmf(msmf_results, args, dtype, device, telem, reps)

        mamf_results = confirm_mamf(top, args, dtype, device, telem, reps, boost_clk, all_mean_tflops)
        mamf = select_mamf(mamf_results, boost_clk)
        print_regime_crosscheck(msmf, mamf, msmf_results, mamf_results, args, dtype, device, telem,
                                reps, boost_clk, all_mean_tflops)

        apply_headlines(headline, best_tflops, best_config, msmf, mamf)
        if telem_ok(telem) and not telem.validated:
            print(f"[confirm] note: telemetry backend '{telem.backend}' is UNTESTED — power is reported but "
                  f"SUSPECT shapes are NOT excluded, so MSMF may be inflated by a boost burst. Run "
                  f"mamf_spike_probe.py on first access, then flip telemetry_validated = True on its Arch "
                  f"subclass in mamf-finder.py if the boost/sat gap holds.")

    def auto_search():
        """Lean directed search that reports both MAMF (boost) and MSMF (saturated).

        Ablation (offline replay vs exhaustive H200/B200 grids) showed that coordinate
        descent, re-descent, hill-climb and line-search add probes but no unique reach.
        The keep-set that still matches the grid within 0.02% offline is:

          1. wave-quantization (M,N) @ a short K set (every wave candidate, not top-N only)
          2. coarse M×N plane @ Kmin                             - many-waves / min-K basin
          3. tight local grid around the top scout seeds         - endgame polish
          4. MSMF confirm: saturated, per-shape-warmed, trimmed-median, SUSPECT-filtered
          5. MAMF recall screen: raw leaders + several K choices per wave-layout basin
          6. both full confirms over their candidate union (same shapes, different regimes)
        """
        # geometry_validated is guaranteed True here (main() refuses --auto otherwise), so both hooks
        # return usable values.
        sms = arch.compute_unit_count()
        tile_m, tile_n = arch.gemm_tile_hint
        elem = dtype_element_size(dtype)
        align = max(128 // elem, 1)   # tensor-core element alignment (bf16->64, fp8->128, fp32->32)
        base = 256                    # M/N step: a multiple of `align` and of the 256-wide tile
        if base % align:
            base = ((base // align) + 1) * align
        scout_i, scout_w = args.scout_num_iterations, args.scout_num_warmup_iterations
        max_size = args.max_size
        k_min = 1024
        k_max = max_size - (max_size % 1024) or max_size

        print(f"[auto] CUs={sms} tile={tile_m}x{tile_n} dtype={args.dtype} elem={elem}B base={base} max_size={max_size}")

        memo = {}
        def smeasure(M, N, K, label):
            key = (int(M), int(N), int(K))
            if key not in memo:
                memo[key] = measure(key[0], key[1], key[2], scout_i, scout_w, label)
            return memo[key]

        def discover():
            """Phases 1–3: wave / plane / refine scouting plus wave provenance for confirm."""
            # Phase 1: wave-quantization-aware (M,N) at a short K set (not just the extremes).
            # Measuring *every* wave (M,N) at several Ks is what gets H200's wave-perfect mid/high-K
            # winners (e.g. 1536x2816x20480) into the confirm set.
            wave_ks = sorted({v for v in (k_min, 8192, 12288, 14336, 16384, k_max) if v <= max_size})
            seen = set()
            square_by_wave = {}  # w -> most-square (M,N); forced into MSMF confirm later
            wave_layouts_by_wave = {}
            n_wave = 0
            for w, (sq, layouts) in wave_mn_layouts(sms, max_size, tile_m, tile_n).items():
                square_by_wave[w] = sq
                wave_layouts_by_wave[w] = set(layouts)
                for (mm, nn) in layouts:
                    if (mm, nn) in seen:
                        continue
                    seen.add((mm, nn))
                    for kk in wave_ks:
                        if gemm_fits(mm, nn, kk, elem):
                            smeasure(mm, nn, kk, "wave")
                            n_wave += 1
            print(f"[auto] wave: {len(seen)} (M,N) @ K={wave_ks} -> {n_wave} scouts")

            # Phase 2: coarse M×N planes at Kmin and the low-K boost basin. K=3072 is where the
            # B200 exhaustive oracle found its best non-wave MAMF family; Kmin alone cannot seed it.
            plane = [v for v in (2048, 4096, 6144, 8192, 10752, 12288, 14336, 16384, 18432, max_size)
                     if v <= max_size]
            plane_ks = sorted({v for v in (k_min, 3072) if v <= max_size})
            n_plane = 0
            for kk in plane_ks:
                for mm in plane:
                    for nn in plane:
                        if gemm_fits(mm, nn, kk, elem):
                            smeasure(mm, nn, kk, "plane")
                            n_plane += 1
            print(f"[auto] plane: {len(plane)}x{len(plane)} @ K={plane_ks} -> {n_plane} scouts")

            # Phase 3: tight local grid around the top scout seeds (endgame polish). Walks a
            # ±r_mn × ±r_mn × ±r_k neighborhood at native lattice resolution (256 / 1024) so an
            # off-axis peak next to a coarse seed is not stepped over. Seed set is
            # basin-diverse (half reserved for wave (M,N)s) so both peaks get polished.
            if args.refine_grid:
                n_power = max(1, args.refine_seeds // 2)
                power_seeds = top_shapes(
                    measured, n_power, prefer_mn=seen, prefer_slots=max(1, n_power // 2))
                peak_seeds = top_shapes_by_peak(scout_meta, args.refine_seeds - n_power)
                seeds = list(dict.fromkeys(power_seeds + peak_seeds))
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
            return seen, square_by_wave, wave_layouts_by_wave

        seen, square_by_wave, wave_layouts_by_wave = discover()
        confirm_phase(seen, square_by_wave, wave_layouts_by_wave)

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
        # Sweep every shape in the user's range as short SCOUTS (same budget as auto), then
        # the shared confirm phase re-measures the winners for MAMF + MSMF. Full iters on every
        # grid point would just re-pay the confirm cost without changing the headlines.
        scout_i, scout_w = args.scout_num_iterations, args.scout_num_warmup_iterations
        after = "without confirm" if args.scout_only else "then confirm"
        if explicit_shapes is not None:
            print(f"[grid] sweeping {len(explicit_shapes)} exact shapes "
                  f"(scout {scout_i} iters / {scout_w} warmup), {after} ...")
            for M, N, K in explicit_shapes:
                measure(M, N, K, scout_i, scout_w, label="grid")
        else:
            print(f"[grid] sweeping {len(m)}x{len(n)}x{len(k)} shapes "
                  f"(scout {scout_i} iters / {scout_w} warmup), {after} ...")
            for M in m:
                for N in n:
                    for K in k:
                        measure(M, N, K, scout_i, scout_w, label="grid")
        if not args.scout_only:
            confirm_phase()
    else:
        auto_search()

    finish()
