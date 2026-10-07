#!/usr/bin/env python

"""

This is the Maximum Achievable and Sustainable Matmul FLOPS finder
(MAMF + MSMF).

Both search modes report the SAME two numbers from the shapes they measure; they
differ only in HOW the candidate shapes are chosen:

- **MAMF** — Maximum *Achievable* Matmul FLOPS: the median of a shape's 5 fastest
  iterations that ran at the boost clock, each burst started after an idle gap.
- **MSMF** — Maximum *Sustainable* Matmul FLOPS: the highest throughput a shape
  holds steady over a 2s window under full load (matches sustained training
  throughput). For picking a real model's shapes this is the number that matters.

- `--search auto` (default): derive near-peak shapes from hardware heuristics and
  report the best MAMF/MSMF the GPU can do anywhere. Great for a spec-sheet
  headline, not tied to any particular model.
- `--search grid`: sweep the M/N/K range YOU give and report the best MAMF/MSMF
  within it. This is the practical case — find the best (and most sustainable)
  shape for a model you're actually running:

python mamf-finder.py --m_range 0 20480 256 --n 4096 --k 4096 --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt

For the auto search, discussion, and important nuances see:
https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks#maximum-achievable-and-sustainable-matmul-flops-finder

Results table:
https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#maximum-achievable-and-sustainable-matmul-flops-comparison-table

Credits:
- Parts of this benchmark have been derived from https://github.com/EleutherAI/cookbook/tree/main/benchmarks/sizing
  (highly recommended!)
- Imtiaz Sajwani: HPU porting
- Xiaoyu Zhang https://github.com/BBuf - flexible dtype support
- Oren Leung https://github.com/OrenLeung - flagging the lack of cache/dest-matrix reset and suggesting a fix - also
  proposing geomean
- Ivan Fioravanti https://github.com/ivanfioravanti - MPS support
"""

from pathlib import Path

import argparse
from dataclasses import dataclass, fields
import datetime
from decimal import Decimal, ROUND_HALF_UP
import itertools
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

# important: when changing how the benchmark measures things bump up its version, so that the old reports could be
# differentiated from the new ones. v3: dual MAMF (boost burst, clock-validated) + MSMF (power-saturated) headlines
# from one `--search auto` run; wave/tile-aware auto search; settle clock, lock-in and SUSPECT/boost filters for MSMF;
# queued MAMF burst with per-iteration clocks from a sampler timeline; telemetry required where a backend exists.
# v4: MSMF = steady throughput over a timed window, with a lower-K walk; MAMF = median of the 5 fastest boost
# iterations; MAMF screen retries hot-start bursts and screens K past the K ceiling.
benchmark_version = 4

has_hpu = False
try:
    import habana_frameworks.torch as ht
    if torch.hpu.is_available():
        has_hpu = True
except ModuleNotFoundError:
    pass

file_dir = os.path.abspath(os.path.dirname(__file__))

# --dtype choices -> torch dtype, None when this torch is too old for it. The block-scaled formats aren't torch dtypes,
# so they carry the dtype that old torch lacks: mxfp8 (OCP MX) is float8_e4m3fn data with float8_e8m0fnu scales
# (torch>=2.7); mxfp4 (OCP MX) and nvfp4 (NVIDIA) are float4_e2m1fn_x2 data, two values per byte, with float8_e8m0fnu
# and float8_e4m3fn scales respectively; mxfp4 also needs torch._scaled_mm_v2. The code passes the --dtype name around,
# since these can't be told apart by torch dtype.
SUPPORTED_DTYPES = {
    "bfloat16":        torch.bfloat16,
    "float16":         torch.float16,
    "float32":         torch.float32,
    "float8_e4m3fn":   torch.float8_e4m3fn,
    "float8_e4m3fnuz": torch.float8_e4m3fnuz,
    "mxfp8":           getattr(torch, "float8_e8m0fnu", None),
    "mxfp4":           getattr(torch, "float4_e2m1fn_x2", None) if hasattr(torch, "_scaled_mm_v2") else None,
    "nvfp4":           getattr(torch, "float4_e2m1fn_x2", None),
}
BLOCK_SCALED_DTYPES = ("mxfp8", "mxfp4", "nvfp4")



### Architecture specific helper classes ###

class Arch:
    # In-process power/clock telemetry, overridden per vendor below. An arch with no backend wired up
    # (telemetry_backend = None) runs without power/clock reporting; an arch that declares one refuses to run if it
    # can't be used, unless --telemetry off. See the Telemetry sampler.
    telemetry_backend   = None    # short id reported in logs, e.g. "nvml" / "amdsmi" / "hlml"

    # --- auto-search geometry --- `--search auto` derives near-peak GEMM shapes from tile + wave quantization, which
    # needs (a) the compute-unit count for wave packing (compute_unit_count) and (b) a representative kernel tile
    # (gemm_tile_hint). An arch without them can't run auto (main() points the user at grid); geometry_validated=True
    # marks the archs where auto was checked against an exhaustive grid - the others run it with a note, which names
    # geometry_checked, the partial check done so far, if any.
    geometry_validated = False
    geometry_checked = None
    # Representative (tile_m, tile_n) of the vendor GEMM kernel, used ONLY to seed wave-quantized candidates.
    #
    # What the value implies: the search assumes the kernel emits tile_m x tile_n output tiles, so it builds candidate
    # M x N shapes as multiples of (tile_m, tile_n) that also fill a whole number of waves across compute_unit_count()
    # CUs (blocks = ceil(M/tile_m)*ceil(N/tile_n); see wave_efficiency). So (128, 256) implies "aim for M multiples of
    # 128 and N multiples of 256 that pack the CUs evenly." It sets the *granularity of the guess*, not a hard
    # constraint.
    #
    # Why an approximate value is fine: it is a coarse hint, not measured hardware truth - the real tile a BLAS library
    # picks varies with arch/dtype/shape/version (cuBLASLt can be queried per-shape via CUBLASLT_ALGO_CONFIG_TILE_ID,
    # but that needs a ctypes shim PyTorch doesn't expose). The search *measures* actual FLOPS on every candidate, so a
    # wrong hint only shifts which shapes are tried (recall), never a reported number. And auto snaps M/N to a base=256
    # step anyway, so any real tile dividing 256 (64/128/256) is already honored regardless of this value.
    #
    # None here means the arch models no waves.
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

    def set_device(self, index):
        """Make `index` the current device before anything is allocated; no-op on single-device archs."""
        pass

    def dtype_unsupported(self, dtype_name):
        """Why the current device can't run --dtype `dtype_name`, or None if nothing is known against it."""
        return None

    # --- telemetry hooks --- An Arch opts into telemetry by setting telemetry_backend to a non-None id (see NVIDIAArch
    # / AMDArch / HPUArch). Once it does, it MUST implement telemetry_init + the readers below: the base raises
    # NotImplementedError (naming the class + missing method) so a half-wired backend fails loudly instead of silently
    # reporting "no data". Archs that leave telemetry_backend = None (XPUArch / MPSArch) opt out and Telemetry never
    # calls these. siblings_idle is the one exception: "no idle siblings known" is a legitimate default (only
    # NVIDIAArch enumerates today), so it stays a real no-op rather than a required override.
    def _telemetry_required(self, method):
        raise NotImplementedError(
            f"{type(self).__name__} sets telemetry_backend={self.telemetry_backend!r} but does not implement "
            f"{method}(); implement it, or set telemetry_backend=None to opt out.")

    def telemetry_init(self, index):
        """Acquire and return an opaque per-device handle."""
        self._telemetry_required("telemetry_init")

    def read_power(self, handle, instant=False):
        """Power draw in Watts; `instant` asks for the unaveraged reading where the vendor has one."""
        self._telemetry_required("read_power")

    def read_clock(self, handle):
        """Current SM/GFX clock in MHz."""
        self._telemetry_required("read_clock")

    def read_device_name(self, handle):
        """Vendor device name."""
        self._telemetry_required("read_device_name")

    def read_max_clock(self, handle):
        """Rated max SM/GFX clock in MHz. Optional: the default returns None."""
        return None

    def read_temp(self, handle):
        """GPU die temperature in C. Optional: the default returns None."""
        return None

    def siblings_idle(self, handle, self_index=0, util_pct=10):
        """OTHER same-board accelerators that are idle. Optional even for telemetry backends: the safe default is
        "none reported" (only NVIDIAArch enumerates siblings), so this is a real no-op, not a required override."""
        return []

class CudaLikeArch(Arch):
    """ Shared timing/device plumbing for CUDA and ROCm - both are torch device 'cuda'.
    NVIDIAArch and AMDArch below add the vendor-specific bits (compute_info + telemetry). """
    def set_device(self, index):
        """Pin the device in-process: GPUs running within a VM don't see the GPUs via visible-devices env vars."""
        torch.cuda.set_device(index)

    @property
    def device(self):
        return torch.device(f'cuda:{torch.cuda.current_device()}')

    @property
    def device_info(self):
        return torch.cuda.get_device_properties(device)

    def pci_address(self):
        """domain:bus:device of the device torch uses. The telemetry libraries can number the GPUs in a different order
        than torch and *_VISIBLE_DEVICES do, so the PCI address is what finds the benchmarked GPU's handle."""
        props = torch.cuda.get_device_properties(self.device)
        if not hasattr(props, "pci_bus_id"):
            raise RuntimeError(f"{self.telemetry_backend} telemetry requires torch>=2.8 (to find the benchmarked GPU "
                               f"by its PCI address), found torch=={torch.__version__}")
        return f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}"

    def event(self, enable_timing=True):
        return torch.cuda.Event(enable_timing)

    def synchronize(self):
        torch.cuda.synchronize()

    def compute_unit_count(self):
        """NVIDIA: SM count. ROCm: torch reports the CU count through the same field."""
        return torch.cuda.get_device_properties(0).multi_processor_count

class NVIDIAArch(CudaLikeArch):
    """ NVIDIA GPUs (CUDA). Telemetry via NVML - pip install nvidia-ml-py. """
    telemetry_backend   = "nvml"
    geometry_validated  = True    # wave/tile auto-search matched exhaustive H200/B200 grids
    gemm_tile_hint      = (128, 256)  # representative cuBLAS tile the auto rules were derived against

    def __init__(self):
        self.arch = "cuda"

    @property
    def compute_info(self):
        return f"cuda={torch.version.cuda}"

    def dtype_unsupported(self, dtype_name):
        cc = torch.cuda.get_device_capability()
        if dtype_name in BLOCK_SCALED_DTYPES and cc < (10, 0):
            return f"{torch.cuda.get_device_name()} is compute capability {cc[0]}.{cc[1]}"
        return None

    def telemetry_init(self, index):
        import pynvml as m
        m.nvmlInit()
        self._nvml = m
        self._clk_arg = m.NVML_CLOCK_SM
        return m.nvmlDeviceGetHandleByPciBusId(self.pci_address().encode())

    def read_power(self, handle, instant=False):
        """nvmlDeviceGetPowerUsage averages over the last 1s on Ampere (except GA100) and newer. The instantaneous
        field costs ~170us per read against ~6us, so only the MSMF window sampler, which reads every 20ms, asks for it.
        """
        if instant:
            fv = self._nvml.nvmlDeviceGetFieldValues(handle, [186])[0]  # NVML_FI_DEV_POWER_INSTANT
            if fv.nvmlReturn == 0:
                return fv.value.uiVal / 1000.0
        return self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W

    def read_clock(self, handle):
        return float(self._nvml.nvmlDeviceGetClockInfo(handle, self._clk_arg))

    def read_max_clock(self, handle):
        return float(self._nvml.nvmlDeviceGetMaxClockInfo(handle, self._clk_arg))

    def read_temp(self, handle):
        return float(self._nvml.nvmlDeviceGetTemperature(handle, self._nvml.NVML_TEMPERATURE_GPU))

    def read_device_name(self, handle):
        name = self._nvml.nvmlDeviceGetName(handle)
        return name.decode() if isinstance(name, bytes) else name

    def siblings_idle(self, handle, self_index=0, util_pct=10):
        """Return [(idx, util%)] for OTHER same-board GPUs that are not computing.

        Sibling GPUs on the same board share a power/cooling budget. In real work they all compute at once, so an MSMF
        measured while they idle gets headroom a full node never has and is only a single-GPU upper bound. Keyed on
        utilization (real power/heat) - a resident CUDA context with 0% util draws no power, whatever its memory.

        Only a shared baseboard (HGX/SXM, NVLink-connected) couples GPUs this way. A GPU without active NVLink is a
        standalone PCIe card with its own power delivery, and a different model is a different card, so neither counts
        as a sibling."""
        m = self._nvml
        out = []
        if not self._has_active_nvlink(handle):
            return out
        try:
            count = m.nvmlDeviceGetCount()
            my_name = self.read_device_name(handle)
            my_index = m.nvmlDeviceGetIndex(handle)
        except Exception:
            return out
        for i in range(count):
            if i == my_index:
                continue
            try:
                h = m.nvmlDeviceGetHandleByIndex(i)
                if self.read_device_name(h) != my_name:
                    continue
                util = m.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:
                continue
            if util < util_pct:
                out.append((i, int(util)))
        return out

    def _has_active_nvlink(self, handle):
        m = self._nvml
        for link in range(getattr(m, "NVML_NVLINK_MAX_LINKS", 18)):
            try:
                if m.nvmlDeviceGetNvLinkState(handle, link) == m.NVML_FEATURE_ENABLED:
                    return True
            except Exception:
                continue
        return False

class AMDArch(CudaLikeArch):
    """ AMD GPUs (ROCm). Telemetry via amdsmi (ships with ROCm).

    The amdsmi power/clock/name reads below were confirmed on MI300X (ROCm 10.0, SR-IOV VF, SPX). If a call is wrong on
    another MI part, Telemetry catches it and degrades to unavailable - the benchmark still runs, just without
    power/clock.
    """
    telemetry_backend = "amdsmi"
    # geometry_validated stays False (inherited), so auto runs with a note. On one MI300X (BF16) CU-count wave packing
    # with this (128,256) placeholder tile matched a 16,000-shape grid to 0.1%; flip to True once FP8 and a second box
    # agree.
    geometry_checked = "one MI300X in BF16, where it came within 0.1% of a 16,000-shape grid"
    gemm_tile_hint = (128, 256)

    def __init__(self):
        self.arch = "rocm"

    @property
    def compute_info(self):
        return f"hip={torch.version.hip}, cuda={torch.version.cuda}"

    def dtype_unsupported(self, dtype_name):
        gfx = torch.cuda.get_device_properties(self.device).gcnArchName
        if dtype_name in BLOCK_SCALED_DTYPES and "gfx950" not in gfx:
            return f"{torch.cuda.get_device_name()} is {gfx}"
        return None

    def telemetry_init(self, index):
        """Find the amdsmi handle by the PCI address of the device torch is using.
        amdsmi lists GPUs in PCI address order, while HIP (torch and HIP_VISIBLE_DEVICES) numbers them in KFD topology order.
        The two orders can differ on some nodes, so indexing amdsmi by the torch index can sample the wrong GPU."""
        import amdsmi as m
        m.amdsmi_init()
        self._amdsmi = m
        self._clk_arg = m.AmdSmiClkType.GFX   # docs-confirmed enum member (graphics/compute clock)
        props = torch.cuda.get_device_properties(self.device)
        bdf = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}"
        for handle in m.amdsmi_get_processor_handles():
            if m.amdsmi_get_gpu_device_bdf(handle).lower().startswith(bdf):
                return handle
        raise RuntimeError(f"no amdsmi GPU at PCI address {bdf}")

    def read_power(self, handle, instant=False):
        """amdsmi_get_power_info() returns Watts (confirmed on MI300X): current_socket_power (MI300+), with
        average_socket_power as the fallback (Navi / MI200 and earlier). Unsupported fields may come back as "N/A" or
        UINT32_MAX (0xFFFFFFFF)."""
        info = self._amdsmi.amdsmi_get_power_info(handle)
        INVALID = (None, "N/A")
        # UINT32_MAX guard: some builds report an unsupported field as 0xFFFFFFFF instead of "N/A". Enable this wider
        # set once confirmed on a real MI box (replaces the line above):
        # INVALID = (None, "N/A", 0xFFFFFFFF)
        w = info.get("current_socket_power")
        if w in INVALID:
            w = info.get("average_socket_power")
        return float(w) if w not in INVALID else None

    def read_clock(self, handle):
        """["clk"] is the GFX clock in MHz (confirmed on MI300X, ROCm 10.0)."""
        return float(self._amdsmi.amdsmi_get_clock_info(handle, self._clk_arg)["clk"])

    def read_temp(self, handle):
        """UNTESTED on hardware: the hotspot (junction) temperature in C, per the amdsmi docs."""
        m = self._amdsmi
        return float(m.amdsmi_get_temp_metric(handle, m.AmdSmiTemperatureType.HOTSPOT,
                                              m.AmdSmiTemperatureMetric.CURRENT))

    def read_device_name(self, handle):
        """amdsmi_get_gpu_asic_info() exposes market_name per the docs."""
        info = self._amdsmi.amdsmi_get_gpu_asic_info(handle)
        return info.get("market_name") or info.get("vendor_id") or "AMD GPU"

class HPUArch(Arch):
    """ Intel Gaudi*. Telemetry via pyhlml (pip install habana-pyhlml).

    UNTESTED ON HARDWARE: the pyhlml calls below were verified against the Habana pyhlml API docs but have never been
    run on a real Gaudi box. A wrong call is caught by Telemetry and degrades to unavailable - the benchmark still
    runs, just without power/clock.
    """
    telemetry_backend = "hlml"

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
        """UNTESTED on hardware; API verified against Habana pyhlml docs."""
        import pyhlml as m
        m.hlmlInit()
        self._hlml = m
        return m.hlmlDeviceGetHandleByIndex(index)

    def read_power(self, handle, instant=False):
        """UNTESTED on hardware. Docs confirm hlmlDeviceGetPowerUsage() returns milliwatts (like NVML), so /1000 ->
        Watts. Confirm on the first Gaudi box."""
        return self._hlml.hlmlDeviceGetPowerUsage(handle) / 1000.0

    def read_clock(self, handle):
        """UNTESTED on hardware. clock_type 0 == HLML_CLOCK_SOC, which the docs list as the only clock domain
        supported on Gaudi (IC/MME/TPC are Goya-only). Confirm on the first Gaudi box."""
        return float(self._hlml.hlmlDeviceGetClockInfo(handle, 0))

    def read_device_name(self, handle):
        """UNTESTED on hardware. pyhlml has no documented market-name call, so report the handle."""
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

# one scout row per shape: TFLOPS mean/median/max, best mean so far, then W and MHz with telemetry
SCOUT_HEADER = f"{'#':>6}  {'MxNxK':<18} {'mean':>6} {'median':>6} {'max':>6} {'best':>6}"


class Tee(object):
    def __init__(self, filename, verbose):
        Path(filename).resolve().parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filename, "w")
        self.verbose = verbose
        self.after_cr = False  # the last write was a progress row that the next write overwrites
        if self.verbose:
            self.stdout = sys.stdout

    def write(self, message):

        if self.verbose:
            # progress rows end in `\r` and vary in width, so clear the old row before overwriting it
            if self.after_cr and message and not message.startswith("\033[K"):
                self.stdout.write("\033[K")
            self.stdout.write(message)
        if message:
            self.after_cr = message.endswith("\r")
        # replace `\r` and `033\[K` which are nice in the console, but we don't want those in the log file
        message = re.sub(r"(\r|\033\[K)", "\n", message)
        self.file.write(message)

    def status(self, message):
        """Console-only in-place progress line: overwritten by the next write, never logged."""
        if not self.verbose:
            return
        if self.after_cr:
            self.stdout.write("\033[K")
        self.stdout.write(message + "\r")
        self.stdout.flush()
        self.after_cr = True

    def flush(self):
        self.file.flush()
        if self.verbose:
            self.stdout.flush()



def status(message):
    """Show a transient progress line on the console (no-op when stdout isn't the Tee)."""
    if isinstance(sys.stdout, Tee):
        sys.stdout.status(message)


def detail(message):
    """Log-only line: per-shape tables and diagnostics that would clutter the console."""
    if isinstance(sys.stdout, Tee):
        sys.stdout.file.write(message + "\n")
    else:
        print(message)


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

{"-" * 80}""")

# Shared GEMM setup for benchmark_mm (saturated timing) and measure_boost_burst (per-iter MAMF).
#
# l2_cache: written between iterations to emulate cache reset. On AMD this is really L3/LLC — 256MiB is the highest
# across recent accelerators so far (https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#caches).
# C_rand: re-copied into C each iter so the write actually happens (else the rerun is a no-op and draws no power —
# invalid emulation of a real use case).
def prepare_gemm(m, n, k, dtype, device):
    """Allocate operands for --dtype `dtype` (a SUPPORTED_DTYPES name) and return (op, l2_cache, C, C_rand, flos).
    `op()` writes into C."""
    l2_cache = torch.empty(int(256 * 2**20 / 4), dtype=torch.int, device=device)
    out_dtype = SUPPORTED_DTYPES[dtype]

    if dtype in BLOCK_SCALED_DTYPES:
        # One scale per block of 32 elements along K (OCP MX, float8_e8m0fnu) or 16 (nvfp4, float8_e4m3fn). The scales
        # are all 1.0, so the swizzled layout cuBLAS/hipBLASLt read holds the same values as any other and only the
        # size matters: rows padded to 128 and K-blocks to 4 (oneDNN on XPU wants it unpadded).
        out_dtype = torch.bfloat16
        if dtype == "mxfp8":
            A = torch.randn(m, k, dtype=torch.float32, device=device).contiguous().to(torch.float8_e4m3fn)
            B = torch.randn(n, k, dtype=torch.float32, device=device).contiguous().t().to(torch.float8_e4m3fn)
        else:
            # fp4 packs two e2m1 values per byte along K, and torch can't cast to it. e2m1 has no NaN/Inf, so random
            # bytes are random fp4 data.
            if k % 32:
                raise ValueError(f"--dtype {dtype} needs K to be a multiple of 32, got {k}")
            A = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device=device).view(torch.float4_e2m1fn_x2)
            B = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device).view(torch.float4_e2m1fn_x2).t()
        block, scale_dtype = (16, torch.float8_e4m3fn) if dtype == "nvfp4" else (32, torch.float8_e8m0fnu)
        k_blocks = -(-k // block)
        def block_scales(rows):
            numel = rows * k_blocks if arch.name == "xpu" else -(-rows // 128) * 128 * -(-k_blocks // 4) * 4
            return torch.full((numel,), 1.0, dtype=scale_dtype, device=device)
        scale_a, scale_b = block_scales(m), block_scales(n)
        if dtype == "mxfp4":
            # torch._scaled_mm takes float8_e8m0fnu block scales only with fp8 data, so mxfp4 needs the
            # explicit-recipe op behind torch.nn.functional.scaled_mm, which has no `out=`
            F = torch.nn.functional
            recipe = [F.ScalingType.BlockWise1x32.value]
            swizzle = [(F.SwizzleType.NO_SWIZZLE if arch.name == "xpu" else F.SwizzleType.SWIZZLE_32_4_4).value]
            def op():
                torch._scaled_mm_v2(A, B, [scale_a], recipe, swizzle, [scale_b], recipe, swizzle, None, out_dtype,
                                    out=C)
        else:
            def op():
                torch._scaled_mm(A, B, scale_a, scale_b, out_dtype=out_dtype, out=C)
    elif dtype in ("float8_e4m3fn", "float8_e4m3fnuz"):
        if version.parse(torch.__version__) < version.parse("2.5"):
            raise ValueError("float8 dtypes require torch>=2.5")
        if dtype == "float8_e4m3fn" and arch.name == "rocm":
            raise ValueError("ROCm doesn't support float8_e4m3fn, use --dtype float8_e4m3fnuz instead")
        A = torch.randn(m, k, dtype=torch.float32, device=device).contiguous().to(out_dtype)
        B = torch.randn(n, k, dtype=torch.float32, device=device).contiguous().t().to(out_dtype)
        scale = torch.tensor([1.0]).to(device)
        # must not move `out=C` as `C = ...` — Gaudi needs it this way
        def op():
            torch._scaled_mm(A, B, scale, scale, out=C)
    else:
        A = torch.randn(m, k, dtype=out_dtype, device=device).contiguous()
        B = torch.randn(n, k, dtype=out_dtype, device=device).contiguous().t()
        def op():
            torch.mm(A, B, out=C)
    C = torch.empty(m, n, dtype=out_dtype, device=device).contiguous()
    C_rand = torch.randn(m, n, device=device).to(dtype=out_dtype).contiguous()
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

    # sample power/clock in a background thread while the timed loop runs so each shape is self-validating (high TFLOPS
    # at low power = unsaturated boost, not MSMF)
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
# The burst is QUEUED, not run one-synchronized-iteration-at-a-time. Synchronizing before recording each start event
# leaves the GPU idle at the moment the event is taken, which folds kernel-launch latency (~6-8us) and the DVFS re-ramp
# out of idle into the measured window. Measured on B200/H200/B300 that costs 1.6-19% depending on shape - and because
# the penalty scales with kernel footprint it silently RE-RANKS candidates, biasing MAMF toward small shapes.
#
# So: idle first (the SM clock climbs back to boost - that is the regime MAMF reports), then queue the whole burst and
# synchronize once, exactly like benchmark_mm(). Per-iteration clock attribution is preserved WITHOUT per-iteration
# syncs: a background sampler timestamps (clock, power) on the host clock while the burst runs, and each iteration's
# GPU-time window is projected onto that host timeline by anchoring the post-sync host timestamp to the burst's last
# end event. Each iteration is then paired with the MINIMUM clock sampled inside its own window - the same conservative
# floor the bracketed reads gave, so a published MAMF is still certified to have run at the reported clock.
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
# Instead of brute-forcing a 3D grid of MxNxK shapes, `--search auto` constructs a small set of shapes the accelerator
# should run at/near peak on, using the three rules from "The Case for Co-Designing Model Architectures with Hardware"
# (https://arxiv.org/abs/2401.14489):
#   1. Tensor-core alignment: M, N, K are multiples of `128 bytes / dtype_size` elements.
#   2. Tile quantization:     the MxN output divides evenly into the kernel's tile (Arch.gemm_tile_hint,
#                             (128,256) on NVIDIA - a coarse hint, not a queried per-kernel tile).
#   3. Wave quantization:     the number of output tiles is a multiple of the compute-unit count
#                             (Arch.compute_unit_count()), so the final wave is full (see `wave_efficiency`).
# K only sets the arithmetic intensity (how compute-bound the GEMM is); it never appears in the tile/wave math, so it
# is pinned and coarsely swept rather than gridded. See benchmarks/README.md. After scouting, auto confirms two
# headlines: MAMF (boost-clock burst) and MSMF (saturated).
#
# The compute-unit count and tile hint that feed these rules are per-Arch (compute_unit_count() / gemm_tile_hint) and
# only *validated* to predict peak shapes where Arch.geometry_validated=True (NVIDIA today). Elsewhere main() runs auto
# with a note, or refuses it where the arch has no compute-unit count / tile hint - see the geometry check there.

def dtype_element_size(dtype):
    """Size in bytes of one element of --dtype `dtype` (bf16->2, fp8->1, fp4->0.5, fp32->4)."""
    if dtype in ("mxfp4", "nvfp4"):
        return 0.5
    if dtype == "mxfp8":
        return 1
    return torch.empty(0, dtype=SUPPORTED_DTYPES[dtype]).element_size()

def wave_efficiency(m, n, sms, tile_m=128, tile_n=256):
    """Fraction of the scheduled waves that do useful work for an m x n output on `sms` SMs; 1.0
    is a perfectly packed tail wave. blocks = ceil(m/tile_m)*ceil(n/tile_n) run in ceil(blocks/sms)
    waves, and a partial tail wave still costs a full wave, so efficiency = blocks/(waves*sms)."""
    blocks = math.ceil(m / tile_m) * math.ceil(n / tile_n)
    return blocks / (math.ceil(blocks / sms) * sms)


def wave_mn_layouts(sms, max_size, tile_m=128, tile_n=256, waves=range(1, 17), min_dim=1024):
    """Per-wave (M,N) layouts that fill an integer number of SM waves.

    For each wave count returns `(square, layouts)` where `square` is the most-square legal (M,N) and `layouts` is the
    set {square, widest, tallest, transpose-of-square-if-legal}. Empty dict if `sms` is None/0.
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
        # Always include the transpose when wave-legal: abs(log(M/N)) float-ties can pick 2816x1536 over 1536x2816
        # (H200 peak family), and cuBLAS is not transpose-symmetric.
        if (sq[1], sq[0]) in pairs:
            layouts.add((sq[1], sq[0]))
        out[w] = (sq, layouts)
    return out


def shape_str(shape):
    """`MxNxK` for a (M,N,K) tuple."""
    return f"{shape[0]}x{shape[1]}x{shape[2]}"


def round_tflops(x):
    """Nearest integer TFLOPS. A half rounds away from zero, so 754.5 is 755."""
    return int(Decimal(f"{x:.6f}").quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def format_headline(r):
    """One-line MAMF/MSMF headline: `TFLOPS @ MxNxK  WWW MHz C` (or `n/a`)."""
    if not r:
        return "n/a"
    extra = ""
    if r.get("power") is not None:
        extra += f"  {r['power']:.0f}W"
    if r.get("clock") is not None:
        extra += f" {r['clock']:.0f}MHz"
    if r.get("temp") is not None:
        extra += f" {r['temp']:.0f}C"
    return f"{round_tflops(r['tflops'])} TFLOPS @ {shape_str(r['shape'])} (MxNxK){extra}"


def phase_result(row, label):
    """Re-print a phase's winning row so it stays under the column header in place of the progress row."""
    print(f"{row}  <- {label}")


def fmt_runs(xs):
    """Space-joined 1-decimal list for confirm rows."""
    return " ".join(f"{x:.1f}" for x in xs)


def fmt_opt(x, width, fmt=".0f"):
    """Right-aligned value, or '-' when telemetry didn't provide it, so table columns stay aligned."""
    return f"{x:{width}{fmt}}" if x is not None else f"{'-':>{width}}"


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


# Populated by benchmark_mm() when a Telemetry sampler is passed; the scouts read it.
_last_telem = {}


TELEMETRY_PACKAGE_NAMES = {"nvml": "nvidia-ml-py", "amdsmi": "amdsmi", "hlml": "habana-pyhlml"}
TELEMETRY_PACKAGES = {"nvml": "pip install nvidia-ml-py (not the deprecated `pynvml` package)",
                      "amdsmi": "install amdsmi (ships with ROCm: pip install /opt/rocm/share/amd_smi)",
                      "hlml": "pip install habana-pyhlml"}


def telemetry_install_hint():
    pkg = TELEMETRY_PACKAGES.get(arch.telemetry_backend)
    return f" To fix: {pkg}" if pkg else ""


def telem_ok(telem):
    return telem is not None and getattr(telem, "available", False)


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


def measure_window(shape, dtype, device, telem, window_s, warm_s, n_sub=8):
    """Back-to-back iterations: `warm_s` untimed, then `window_s` timed in `n_sub` consecutive sub-windows.

    Each iteration does the same work as benchmark_mm() and only op() is timed, so the TFLOPS are comparable. Returns
    dict(tflops over the whole window, subs=per-sub-window TFLOPS, drop=% the second half is slower than the first,
    spread, power/clock = means of the samples taken during the timed part).
    """
    m, n, k = shape
    op, l2_cache, C, C_rand, flos = prepare_gemm(m, n, k, dtype, device)

    def run(count, starts=None, ends=None):
        with torch.no_grad():
            for i in range(count):
                l2_cache.zero_()
                C.copy_(C_rand)
                if starts is not None:
                    starts[i].record()
                op()
                if ends is not None:
                    ends[i].record()

    s, e = arch.event(enable_timing=True), arch.event(enable_timing=True)
    run(2)
    s.record()
    run(5)
    e.record()
    arch.synchronize()
    t_it = max(s.elapsed_time(e) / 5 / 1000, 1e-6)
    run(max(1, int(warm_s / t_it)))
    n_it = max(n_sub, int(window_s / t_it))
    starts = [arch.event(enable_timing=True) for _ in range(n_it)]
    ends = [arch.event(enable_timing=True) for _ in range(n_it)]
    arch.synchronize()
    pw, clk, tmp, stop, th = [], [], [], threading.Event(), None
    if telem_ok(telem):
        def _sample_loop():
            while not stop.is_set():
                p, c, t = telem.power(instant=True), telem.clock(), telem.temp()
                if p is not None: pw.append(p)
                if c is not None: clk.append(c)
                if t is not None: tmp.append(t)
                stop.wait(0.02)
        th = threading.Thread(target=_sample_loop, daemon=True)
        th.start()
    try:
        run(n_it, starts, ends)
        arch.synchronize()
    finally:
        if th is not None:
            stop.set()
            th.join()
    times = np.array([a.elapsed_time(b) for a, b in zip(starts, ends)])
    subs = [flos * len(c) / (c.sum() / 1000) / 10**12 for c in np.array_split(times, n_sub)]
    half = n_sub // 2
    drop = (1 - np.mean(subs[half:]) / np.mean(subs[:half])) * 100
    return dict(shape=shape, tflops=flos * n_it / (times.sum() / 1000) / 10**12, subs=subs, drop=float(drop),
                spread=spread_pct(subs), power=float(np.mean(pw)) if pw else None,
                clock=float(np.mean(clk)) if clk else None, temp=float(np.mean(tmp)) if tmp else None)


def msmf_k_walk(full, ranked, steady, fmt, args, dtype, device, telem, walk_mn=()):
    """Walk K down from the fastest steady (M,N) layouts and the `walk_mn` ones, measuring each step over a full
    window.

    At the power cap a shorter K can sustain a higher clock than the K the boost-regime search favoured, so the confirm
    set may hold the right layout at the wrong K: H200 bf16's 2816x1536 holds 727 TFLOPS at K=20480 and 749 at K=6144,
    where the clock reaches its ceiling just as the power reaches the cap. A walk starts at the seed's K, or at max_size
    if the seed's K is past it: the K-edge screen adds such shapes for MAMF, and past that point a longer K only
    sustains less. Steps are K/8 rounded down to a power of two; a walk ends after msmf_kwalk_steps steps or once a
    step is more than msmf_kwalk_stop % below the best seen in it.

    The fastest layouts at long K can sit within noise of each other, so which ones are walked is partly luck. The
    `walk_mn` layouts, the most-square 1-wave ones, are walked regardless: on H200 fp8 six layouts held 1276-1291
    TFLOPS at long K, 1536x2816 read lowest of them yet sustains 1337 at K=10240, and walks from the 4 fastest missed
    it.
    """
    if args.tune.msmf_kwalk_seeds <= 0:
        return []
    seeds, seen, by_shape = [], {r["shape"] for r in full}, {r["shape"]: r for r in full}
    for r in sorted((r for r in full if steady(r)), key=lambda r: r["tflops"], reverse=True):
        if r["shape"][:2] not in [s["shape"][:2] for s in seeds]:
            seeds.append(r)
        if len(seeds) >= args.tune.msmf_kwalk_seeds:
            break
    for mn in walk_mn:
        if mn in [s["shape"][:2] for s in seeds]:
            continue
        cands = [r for r in full if r["shape"][:2] == mn] or [r for r in ranked if r["shape"][:2] == mn]
        if cands:
            seeds.append(max(cands, key=lambda r: r["tflops"]))
    out = []
    for seed in seeds:
        m, n, k = seed["shape"]
        k0 = min(k, args.tune.max_size - args.tune.max_size % 1024 or args.tune.max_size)
        step = max(256, 2 ** int(np.log2(k0 / 8)))
        best = by_shape.get((m, n, k0), {}).get("tflops", 0.0)
        for i in range(0 if k0 < k else 1, args.tune.msmf_kwalk_steps + 1):
            shp = (m, n, k0 - i * step)
            if shp[2] < step or shp in seen:
                continue
            seen.add(shp)
            r = measure_window(shp, dtype, device, telem, args.tune.msmf_window_s, args.tune.msmf_warm_s)
            r["row"] = fmt(r, "  k-walk" + ("" if steady(r) else ", not steady"))
            detail(r["row"])
            out.append(r)
            best = max(best, r["tflops"])
            if r["tflops"] < best * (1 - args.tune.msmf_kwalk_stop / 100):
                break
    return out


def confirm_msmf(shapes, args, dtype, device, telem, all_mean_tflops, walk_mn=()):
    """MSMF: the highest throughput a shape holds, steady, over a timed window under full load.

    After settle clock every confirm shape is ranked on a short window (0.2s warm + msmf_rank_s timed), and the
    msmf_top fastest get the full window: msmf_warm_s untimed, then msmf_window_s timed in 8 sub-windows. A shape is
    steady if its second half is at most msmf_max_drop % slower than its first and its sub-windows spread at most
    msmf_max_spread %; the window's TFLOPS is total FLOPs over total kernel time. The K walk then extends the
    msmf_kwalk_seeds fastest steady layouts and the `walk_mn` ones downward in K. The fastest shape that was steady,
    or faster than every steady one (a shape can still be settling in its first window), is re-measured over
    msmf_lock_s and published if steady there, else the next one is tried. Returns (results for every shape, chosen
    result or None).
    """
    print(f"\nMSMF (sustainable) confirm: {len(shapes)} shapes ...")
    detail(f"  rank: {args.tune.msmf_rank_s}s each; top {args.tune.msmf_top}: {args.tune.msmf_warm_s}s warm + "
           f"{args.tune.msmf_window_s}s timed; lock-in {args.tune.msmf_lock_s}s")
    settle_clock(device, telem, args.tune.max_size, args.tune.msmf_settle_clock_s)
    fmt = lambda r, tag="": (f"  {shape_str(r['shape']):<18} {r['tflops']:6.1f} {fmt_opt(r['power'], 5)} "
                             f"{fmt_opt(r['clock'], 5)} {fmt_opt(r['temp'], 3)} drop {r['drop']:5.1f}% "
                             f"spread {r['spread']:4.1f}%{tag}")
    print(f"  {'MxNxK':<18} {'TFLOPS':>6} {'W':>5} {'MHz':>5} {'C':>3}")
    ranked = []
    for shp in shapes:
        r = measure_window(shp, dtype, device, telem, args.tune.msmf_rank_s, 0.2, n_sub=2)
        r["row"], r["rank"] = fmt(r, "  rank"), True
        status(r["row"])
        detail(r["row"])
        ranked.append(r)
    ranked.sort(key=lambda r: r["tflops"], reverse=True)
    steady = lambda r: r["drop"] <= args.tune.msmf_max_drop and r["spread"] <= args.tune.msmf_max_spread
    full = []
    for r0 in ranked[:max(1, args.tune.msmf_top)]:
        r = measure_window(r0["shape"], dtype, device, telem, args.tune.msmf_window_s, args.tune.msmf_warm_s)
        r["row"] = fmt(r, "" if steady(r) else "  not steady")
        print(r["row"], end="\r", flush=True)
        full.append(r)
    full += msmf_k_walk(full, ranked, steady, fmt, args, dtype, device, telem, walk_mn)
    all_mean_tflops.extend(r["tflops"] for r in full)
    results = {r["shape"]: r for r in ranked}
    results.update({r["shape"]: r for r in full})
    floor = max((r["tflops"] for r in full if steady(r)), default=0.0)
    order = sorted([r for r in full if steady(r) or r["tflops"] > floor], key=lambda r: r["tflops"], reverse=True)
    for cand in order[:max(1, args.tune.msmf_lock_tries)]:
        lock = measure_window(cand["shape"], dtype, device, telem, args.tune.msmf_lock_s, args.tune.msmf_warm_s)
        ok = steady(lock)
        lock["row"] = fmt(lock, "  locked in" if ok else "  lock-in not steady")
        detail(lock["row"])
        if ok:
            results[cand["shape"]] = lock
            return list(results.values()), lock
    order = [r for r in order if steady(r)]
    if order:
        return list(results.values()), order[0]
    print("WARNING: no confirm shape held a steady rate over its window; MSMF is the fastest unsteady one")
    return list(results.values()), max(full, key=lambda r: r["tflops"]) if full else None


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

    If `prefer_mn` is a set of (M,N) and prefer_slots>0, reserve that many confirm slots for the best shapes whose
    (M,N) is in the set. Stops a noisy plane/mid-K family from crowding the wave-perfect basin out of the confirm set
    (the H200 1-shot miss mode).
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

    A saturated scout mean cannot predict an idle+burst winner. Keeping several K choices for every wave layout
    prevents a noisy 20-iteration scout from dropping a disconnected boost basin (seen on H200 and B200) before
    it is measured in the MAMF regime.
    """
    raw = top_shapes_by_peak(scout_meta, args.tune.mamf_raw_forced)
    wave = []
    per_layout = max(1, args.tune.mamf_wave_k)
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


def boost_reference(boost_clk, iters):
    """Highest SM clock seen so far: scouts run saturated, so the bursts' own clocks usually set it."""
    clks = [c for _, c, _ in iters if c is not None]
    return max([boost_clk or 0.0] + clks)


def screen_burst(shp, args, dtype, device, telem, boost_min):
    """A screening burst. The GPU can still be hot from scouting - H200's first screened shapes burst at 1635MHz
    against a 1980MHz boost - so a burst that never reaches boost_min is retried once after the confirm's idle."""
    iters = max(1, args.tune.mamf_screen_iters)
    burst = measure_boost_burst(*shp, dtype, device, iters, telem=telem,
                                idle_before_s=max(0.0, args.tune.mamf_screen_idle_s))
    if boost_min and not any(c is not None and c >= boost_min for _, c, _ in burst):
        burst += measure_boost_burst(*shp, dtype, device, iters, telem=telem,
                                     idle_before_s=max(0.0, args.tune.mamf_idle_s))
    return burst


def screen_k_edge(scored, args, dtype, device, telem, boost_min):
    """Screen K past the search's K ceiling for the layouts still rising at that ceiling.

    Some layouts keep gaining past max_size: H200 bf16's 2816x1536 bursts 828 TFLOPS at K=20480 and 842 at K=32768.
    Scouts can't see it - they run hot, and a long-K call throttles more, so they read a falling curve - but a burst
    after an idle gap compares K values fairly. Walks K up per layout while the peak still rises.
    """
    if args.tune.k_edge_layouts <= 0 or not scored:
        return []
    k_ceil = max(shp[2] for _, shp, *_ in scored)
    at_ceil, below = {}, {}
    for peak, shp, *_ in scored:
        if shp[2] == k_ceil:
            at_ceil[shp[:2]] = (peak, shp)
        else:
            below[shp[:2]] = max(below.get(shp[:2], 0.0), peak)
    # still climbing into the ceiling, steepest first: a flat layout's long K just ties it
    edge = sorted(((peak / below[mn] - 1, peak, shp) for mn, (peak, shp) in at_ceil.items()
                   if below.get(mn) and peak > below[mn]), reverse=True)[:args.tune.k_edge_layouts]
    elem = dtype_element_size(dtype)
    out = []
    for _, last, (mm, nn, k0) in edge:
        for f in (1.5, 2):
            kk = int(k0 * f) // 1024 * 1024
            if kk > args.tune.k_edge_max or not gemm_fits(mm, nn, kk, elem):
                break
            burst = screen_burst((mm, nn, kk), args, dtype, device, telem, boost_min)
            at_boost = [x for x in burst if not boost_min or (x[1] is not None and x[1] >= boost_min)]
            valid = at_boost or burst
            if not valid:
                break
            peak, clk, power = max(valid, key=lambda x: x[0])
            out.append((float(peak), (mm, nn, kk), clk, power, bool(at_boost)))
            if peak <= last:
                break
            last = peak
    detail(f"  k-edge: {len(edge)} layouts rising at K={k_ceil}, {len(out)} shapes screened past it: " +
           ", ".join(f"{shape_str(s)}={tf:.1f}" for tf, s, *_ in out))
    return out


def screen_mamf_candidates(pool, args, dtype, device, telem, boost_clk):
    """Cheap MAMF-regime screen; return the strongest boost-validated candidates."""
    if not pool:
        return []
    scored = []
    print(f"\nMAMF recall screen: {len(pool)} shapes ...")
    detail(f"  {args.tune.mamf_screen_iters} iters, {args.tune.mamf_screen_idle_s*1000:.0f}ms idle")
    print(f"  {'#':>4}  {'MxNxK':<18} {'peak':>6} {'MHz':>5}")
    bursts = []
    row = lambda i, shp, tf, ck: f"  {i:>4}  {shape_str(shp):<18} {tf:6.1f} {fmt_opt(ck, 5)}"
    retry_min = args.tune.boost_clock_ratio * boost_clk if boost_clk else 0.0
    for i, shp in enumerate(pool, 1):
        burst = screen_burst(shp, args, dtype, device, telem, retry_min)
        bursts.append((shp, burst))
        tf, ck, _ = max(burst, key=lambda x: x[0]) if burst else (0.0, None, None)
        print(row(i, shp, tf, ck), end="\r", flush=True)
    retried = [shape_str(s) for s, b in bursts if len(b) > max(1, args.tune.mamf_screen_iters)]
    detail(f"  {len(retried)} shapes re-burst below boost clock: {', '.join(retried[:12])}")
    boost_clk = boost_reference(boost_clk, [x for _, b in bursts for x in b])
    boost_min = args.tune.boost_clock_ratio * boost_clk if boost_clk else 0.0
    for shp, burst in bursts:
        at_boost = [x for x in burst if not boost_min or (x[1] is not None and x[1] >= boost_min)]
        valid = at_boost or burst
        if valid:
            peak, clk, power = max(valid, key=lambda x: x[0])
            scored.append((float(peak), shp, clk, power, bool(at_boost)))
    scored.sort(reverse=True)
    scored = sorted(scored + screen_k_edge(scored, args, dtype, device, telem, boost_min), reverse=True)
    if scored:
        peak, shp, clk, _, _ = scored[0]
        phase_result(row(pool.index(shp) + 1 if shp in pool else "k", shp, peak, clk), "best peak")
    chosen = [shp for _, shp, _, _, _ in scored[:max(1, args.tune.mamf_confirm_top)]]
    preview = ", ".join(f"{shape_str(s)}={tf:.1f}" for tf, s, _, _, _ in
                        scored[:min(8, len(scored))])
    detail(f"  leaders: {preview}")
    return chosen


def build_msmf_confirm_set(measured, scout_meta, seen, square_by_wave, args):
    """MSMF confirm set: forced low-wave + fat scouts, then power-ranked fillers.

    Returns (shapes, n_confirm). See call-site comments in auto_search for why forcing matters.
    """
    def best_k_for_mn(mm, nn, *, raw=False):
        """Best measured K for this (M,N). raw=True → max scout TFLOPS (for forced shapes:
        high-K often reads low-power during a noisy scout, and power-rank would wrongly
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
        # both orientations when the transpose was scouted (wave-legal); cuBLAS is not transpose-symmetric.
        orients = [mn]
        if (mn[1], mn[0]) != mn:
            orients.append((mn[1], mn[0]))
        for mm, nn in orients:
            shp = best_k_for_mn(mm, nn, raw=True)
            if shp and shp not in forced:
                forced.append(shp)
    # Force the FATTEST scouts (largest min(M,N,K), then volume). A 20-iter scout is too short to saturate, so
    # everything boosts during scouting - but a shape large in ALL dims saturates to TDP over the MSMF window. Without
    # fat forcing a confirm set can be all small/skinny (boosting) shapes (seen on fp8).
    fat_forced = sorted(scout_meta.keys(),
                        key=lambda s: (min(s), s[0] * s[1] * s[2]), reverse=True)[:args.tune.confirm_fat_forced]
    for shp in fat_forced:
        if shp not in forced:
            forced.append(shp)
    # Never drop forced shapes — expand budget so basin-diverse power-rank fillers still get slots. Fillers are
    # POWER-RANKED (not raw TFLOPS): tall-skinny boosters belong to MAMF, not MSMF.
    n_confirm = max(args.tune.confirm_top, len(forced) + 4)
    rest = [s for s in top_shapes(measured, n_confirm + len(forced), prefer_mn=seen,
                                  prefer_slots=max(2, n_confirm // 2))
            if s not in forced]
    shapes = (forced + rest)[:n_confirm]
    if forced:
        detail(f"  forced (low-wave squares + fattest scouts): {', '.join(map(shape_str, forced))}")
    return shapes, n_confirm


def settle_clock(device, telem, max_size, settle_s):
    """Run the chip at full load until its SM clock stops dropping, before MSMF confirm.

    Biggest reproducibility lever: an idle GPU boosts and reads high, so without this the MSMF headline depends on
    what the card was doing before the run. Runs bf16 on a big square; the temperature is not waited for, as below
    the throttle point it barely moves the clock.
    """
    if not settle_s or settle_s <= 0:
        return
    detail(f"  settle clock: full load until the SM clock stops dropping (<= {settle_s}s) ...")
    s = min(8192, max_size)
    sa = torch.randn(s, s, dtype=torch.bfloat16, device=device)
    sb = torch.randn(s, s, dtype=torch.bfloat16, device=device)
    sc = torch.empty(s, s, dtype=torch.bfloat16, device=device)
    t_end = time.time() + settle_s
    prev_clk, stable_ticks = None, 0
    t0 = time.time()
    while time.time() < t_end:
        for _ in range(100):
            torch.mm(sa, sb, out=sc)
        arch.synchronize()
        clk = telem.clock() if telem_ok(telem) else None
        status(f"  settle clock {time.time() - t0:4.1f}s" + (f"  {clk:.0f}MHz" if clk is not None else ""))
        if telem_ok(telem):
            if clk is not None and prev_clk is not None and clk >= prev_clk - 5:
                stable_ticks += 1
                if stable_ticks >= 3:  # clock stopped dropping across ~3 windows -> settled
                    break
            else:
                stable_ticks = 0
            prev_clk = clk
    del sa, sb, sc
    if telem_ok(telem) and telem.clock() is not None:
        detail(f"  settled at ~{telem.clock():.0f}MHz")


def select_mamf(mamf_results, boost_clk):
    """Pick the MAMF headline, preferring boost-validated readings."""
    boost_pool = [r for r in mamf_results if r["boost"]] or mamf_results
    mamf = max(boost_pool, key=lambda r: r["tflops"]) if boost_pool else None
    if mamf and mamf.get("row"):
        phase_result(mamf["row"], "MAMF")
    if mamf and not boost_clk:
        print(f"WARNING: no SM clock readings, so MAMF {round_tflops(mamf['tflops'])} TFLOPS is not boost-clock "
              "validated")
    elif mamf and not mamf["boost"]:
        print(f"WARNING: no MAMF candidate reached the boost clock (~{boost_clk:.0f}MHz); headline "
              f"{round_tflops(mamf['tflops'])} TFLOPS is a base/throttled-clock reading, not a true boost burst")
    return mamf


def confirm_mamf(cands, args, dtype, device, telem, reps, boost_clk, all_mean_tflops, quiet=False):
    """Boost-burst confirm; returns one result dict per candidate (caller picks the headline).

    MAMF: the median of the 5 fastest boost-clock iterations of a shape. Each candidate gets `reps` bursts of
    mamf_burst_iters queued iterations, each burst after mamf_idle_s of idle so the SM clock recovers to boost (see
    measure_boost_burst). An iteration counts as boost if the lowest clock sampled during it is at least
    boost_clock_ratio x the highest clock seen this run. Fat/saturated scouts are included: idle+burst recovers boost
    even when the scout itself ran at the floor.
    """
    burst_iters = max(1, args.tune.mamf_burst_iters)
    idle_s = max(0.0, args.tune.mamf_idle_s)
    say = detail if quiet else print
    say(f"\nMAMF (achievable) confirm: {len(cands)} shapes ...")
    detail(f"  same set as MSMF, {burst_iters} iters x {reps} reps, {idle_s*1000:.0f}ms idle, no warmup, TFLOPS = "
           "median of the 5 fastest boost iterations")
    def evaluate(shp, iters_all, ref):
        boost_min = args.tune.boost_clock_ratio * ref if ref else 0.0
        at_boost = [x for x in iters_all if boost_min and x[1] is not None and x[1] >= boost_min]
        valid = at_boost or iters_all
        peak, cpk, ppk = max(valid, key=lambda t: t[0]) if valid else (0.0, None, None)
        boost_pw = [p for _, _, p in at_boost if p is not None]
        pmed = median_or_none(boost_pw) if boost_pw else ppk
        is_boost = bool(cpk is not None and boost_min and cpk >= boost_min)
        flag = "" if is_boost or not ref else "  base-clock, not boost"
        top_pk = sorted((t for t, _, _ in iters_all), reverse=True)[:5]
        top5 = float(np.median(sorted((t for t, _, _ in valid), reverse=True)[:5])) if valid else 0.0
        row = (f"  {shape_str(shp):<18} {top5:6.1f} {fmt_opt(pmed, 5)} {fmt_opt(cpk, 5)}  {fmt_runs(top_pk)}{flag}")
        return dict(shape=shp, tflops=top5, power=pmed, clock=cpk, boost=is_boost, boost_ref=ref, row=row)

    say(f"  {'MxNxK':<18} {'TFLOPS':>6} {'W':>5} {'MHz':>5}  top peaks")
    measured = []  # (shape, [(tflops, clk, power) across all reps])
    for shp in cands:
        iters_all = []
        for _ in range(reps):
            b = measure_boost_burst(shp[0], shp[1], shp[2], dtype, device,
                                    burst_iters, telem=telem, idle_before_s=idle_s)
            iters_all += b
            if b:
                all_mean_tflops.append(float(np.mean([t for t, _, _ in b])))
        measured.append((shp, iters_all))
        # the row is judged against the highest clock seen so far; the final pick below re-judges every shape against
        # the run-wide reference
        boost_clk = boost_reference(boost_clk, iters_all)
        row = evaluate(shp, iters_all, boost_clk)["row"]
        if quiet:
            detail(row)
        else:
            print(row, end="\r", flush=True)
    boost_min = args.tune.boost_clock_ratio * boost_clk if boost_clk else 0.0
    detail(f"  boost reference: " + (
        f"{boost_clk:.0f}MHz (highest clock seen this run), need ≥{boost_min:.0f}MHz" if boost_clk
        else "unknown (no clock readings)"))
    return [evaluate(shp, iters_all, boost_clk) for shp, iters_all in measured]


def _burst_one(shp, args, dtype, device, telem, reps, boost_clk, all_mean_tflops):
    """One-shape MAMF burst (used to fill a missing cross-check cell)."""
    results = confirm_mamf([shp], args, dtype, device, telem, reps, boost_clk, all_mean_tflops, quiet=True)
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

    detail("\nSame-shape cross-check (each headline shape in both regimes):")
    detail(f"  {'shape':<22}  {'MAMF (boost)':<32}  {'MSMF (saturated)':<32}  MSMF/MAMF")
    for shp in shapes:
        if shp not in mamf_map:
            filled = _burst_one(shp, args, dtype, device, telem, reps, boost_clk, all_mean_tflops)
            if filled:
                mamf_map[shp] = filled
        if shp not in msmf_map or msmf_map[shp].get("rank"):
            msmf_map[shp] = measure_window(shp, dtype, device, telem, args.tune.msmf_window_s, args.tune.msmf_warm_s)
            all_mean_tflops.append(msmf_map[shp]["tflops"])
        m, s = mamf_map.get(shp), msmf_map.get(shp)
        ratio = f"{100.0 * s['tflops'] / m['tflops']:.0f}%" if m and s and m["tflops"] else "—"
        roles = []
        if mamf and shp == mamf["shape"]:
            roles.append("MAMF*")
        if msmf and shp == msmf["shape"]:
            roles.append("MSMF*")
        tag = f"  ({', '.join(roles)})" if roles else ""
        detail(f"  {shape_str(shp):<22}  {cell(m):<32}  {cell(s):<32}  {ratio}{tag}")


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
# Used for (a) the SM clock that certifies a MAMF iteration ran at boost, (b) power-aware scout ranking, (c) adaptive
# warmup and (d) the power/clock reported next to each number. MSMF itself is judged on throughput alone. Only fast
# in-process libraries are used (NVML / amdsmi / pyhlml), so a read is ~1us and can be sampled per iteration. If the
# vendor library is missing or a read fails, main() refuses to run (override: --telemetry off).
#   NVIDIA : pip install nvidia-ml-py    (pynvml)
#   AMD    : amdsmi (ships with ROCm)               - reads confirmed on MI300X
#   Gaudi  : pip install habana-pyhlml   (pyhlml)   - UNTESTED until first-boot
# XPU/MPS have no fast in-process telemetry wired (would need a slow `xpu-smi` subprocess), so they fall through to
# unavailable and the benchmark still runs, just without power/clock reporting. The vendor-specific reads live on the
# Arch subclasses (NVIDIAArch / AMDArch / HPUArch) as telemetry_init / read_power / read_clock / read_device_name /
# siblings_idle, exactly like event() and synchronize(). Telemetry below is a thin, vendor-agnostic sampler: it holds
# the per-device handle and the failure state and delegates every read to `arch`. Add a vendor by overriding those
# hooks on its Arch subclass - nothing here changes.


class Telemetry:
    """Vendor-agnostic in-process power/clock sampler.

    Holds a per-device handle acquired from the active Arch and delegates every vendor-specific read to it - all vendor
    knowledge lives on the Arch subclasses. Degrades silently to unavailable when the vendor library is missing or a
    device handle can't be acquired.
    """

    def __init__(self, arch, index=0):
        self.arch = arch
        self.index = index
        self._h = None
        self.missing_package = False
        self.error = None if arch is not None and arch.telemetry_backend is not None \
            else f"no telemetry backend for {arch}"
        # Only archs that opted into telemetry (telemetry_backend set) are asked for a handle; the rest
        # (XPU/MPS/unknown) degrade to unavailable without ever touching the hooks. A missing vendor lib or an
        # un-acquirable handle degrades silently, but a NotImplementedError (backend declared yet a hook unimplemented)
        # is a wiring bug and propagates loudly.
        if arch is not None and arch.telemetry_backend is not None:
            try:
                self._h = arch.telemetry_init(index)
            except NotImplementedError:
                raise
            except ImportError:
                self._h = None
                # not the ImportError text: nvidia-ml-py's module is named `pynvml`, which reads like advice to install
                # the deprecated `pynvml` PyPI package
                pkg = TELEMETRY_PACKAGE_NAMES.get(arch.telemetry_backend, arch.telemetry_backend)
                self.error = f"Python package `{pkg}` is not installed"
                self.missing_package = True
            except Exception as e:
                self._h = None
                self.error = f"{type(e).__name__}: {e}"
        self.backend = arch.telemetry_backend if self._h is not None else None

    def probe(self):
        """{read: None if it works, else the error string} for power/clock/max_clock."""
        out = {}
        for name, fn in (("power", self.arch.read_power), ("clock", self.arch.read_clock),
                         ("max_clock", self.arch.read_max_clock)):
            try:
                v = fn(self._h)
                out[name] = None if v is not None else "not provided"
            except Exception as e:
                out[name] = f"{type(e).__name__}: {e}"
        return out

    def max_clock(self):
        """Rated max SM/GFX clock in MHz, or None."""
        if self._h is None:
            return None
        try:
            return self.arch.read_max_clock(self._h)
        except Exception:
            return None

    @property
    def available(self):
        return self._h is not None

    # NotImplementedError propagates (a declared backend forgot a hook); other errors degrade to None.
    def power(self, instant=False):
        """Power draw in Watts, or None."""
        if self._h is None:
            return None
        try:
            return self.arch.read_power(self._h, instant=instant)
        except NotImplementedError:
            raise
        except Exception:
            return None

    def clock(self):
        """Current SM/GFX clock in MHz, or None."""
        if self._h is None:
            return None
        try:
            return self.arch.read_clock(self._h)
        except NotImplementedError:
            raise
        except Exception:
            return None

    def temp(self):
        """GPU die temperature in C, or None."""
        if self._h is None:
            return None
        try:
            return self.arch.read_temp(self._h)
        except NotImplementedError:
            raise
        except Exception:
            return None

    def siblings_idle(self, util_pct=10):
        """OTHER same-board accelerators that are idle - see Arch.siblings_idle and its NVIDIAArch override."""
        if self._h is None:
            return []
        try:
            return self.arch.siblings_idle(self._h, self_index=self.index, util_pct=util_pct)
        except Exception:
            return []

    def watch_siblings(self, period_s=0.5, util_pct=10):
        """Start sampling siblings_idle every `period_s` in a daemon thread; the returned stop() joins it and returns
        {idx: fraction of samples idle}. A single snapshot is not enough: a sibling's work can come and go during the
        measurement, so it can be busy for most of the MSMF measurement and idle at the instant of a check."""
        stop, idle, samples = threading.Event(), {}, [0]

        def loop():
            while not stop.wait(period_s):
                samples[0] += 1
                for i, _ in self.siblings_idle(util_pct=util_pct):
                    idle[i] = idle.get(i, 0) + 1

        th = threading.Thread(target=loop, daemon=True)
        th.start()

        def finish():
            stop.set()
            th.join()
            return {i: n / samples[0] for i, n in sorted(idle.items())} if samples[0] else {}
        return finish

    def device_name(self):
        if self._h is None:
            return None
        try:
            return self.arch.read_device_name(self._h)
        except Exception:
            return None


class FakeTelemetry(Telemetry):
    """Scripted (power_W, clock_MHz) sequence for offline ranking tests. No GPU.

    Each `power()` call advances one step; `clock()` returns the clock paired with the last `power()` reading (mirrors
    how the Telemetry sampler calls power then clock each tick).
    """

    def __init__(self, samples=None, loop=True):
        self.arch = None
        self._h = None
        self.error = None
        self.backend = "fake"
        self._samples = list(samples or [(1000.0, 1300.0)])
        self._i = 0
        self._loop = loop
        self._last = self._samples[0] if self._samples else (None, None)

    @property
    def available(self):
        return True

    def power(self, instant=False):
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

    def probe(self):
        return {"power": None, "clock": None, "max_clock": "not provided"}

    def max_clock(self):
        return None

    def sample(self):
        """Return (power, clock) advancing one step — preferred in tests."""
        return self.power(), self.clock()

def setup_checks():
    if arch.name == "rocm":
        if int(os.environ.get("PYTORCH_TUNABLEOP_ENABLED", "0")) == 0:
            warn("AMD GPUs usually require `export PYTORCH_TUNABLEOP_ENABLED=1` to measure the best possible compute, "
                 "but it hasn't been set. Proceeding as is - expect potentially bad/invalid results.")


def dtype_checks(dtype_name, device):
    """Exit if the current device can't run --dtype `dtype_name`: a known torch/arch limit, else a tiny matmul, since
    the vendor libraries don't support every dtype everywhere and failing here beats failing mid-search."""
    if SUPPORTED_DTYPES[dtype_name] is None:
        reason = f"torch {torch.__version__} is too old"
    else:
        reason = arch.dtype_unsupported(dtype_name)
    if reason is None:
        try:
            prepare_gemm(128, 128, 128, dtype_name, device)[0]()
            arch.synchronize()
        except Exception as e:
            reason = str(e).splitlines()[0].rstrip(".")
    if reason:
        hint = {
            "mxfp8": " mxfp8 needs hardware MX support (NVIDIA Blackwell or AMD MI355X) and a recent PyTorch.",
            "mxfp4": " mxfp4 needs hardware MX support (NVIDIA Blackwell or AMD MI355X) and a recent PyTorch.",
            "nvfp4": " nvfp4 needs NVIDIA Blackwell and a recent PyTorch.",
        }.get(dtype_name, "")
        sys.exit(f"error: --dtype {dtype_name} doesn't run on this device: {reason}.{hint}")


def telemetry_setup(telemetry, cuda_device):
    """Return a Telemetry for the device torch uses, or None with --telemetry off. The SM clock validates MAMF (boost
    check), so where a backend exists but doesn't work, running would publish an unvalidated headline - exit
    instead."""
    if telemetry == "off":
        return None
    # sample the *physical* device: CUDA_VISIBLE_DEVICES[--cuda_device] if set
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
    try:
        index = int(visible.split(",")[cuda_device]) if visible else cuda_device
    except (ValueError, IndexError):
        index = cuda_device
    telem = Telemetry(arch, index)
    if arch.telemetry_backend is None:
        return telem
    if not telem.available:
        problem = telem.error
    else:
        broken = {k: v for k, v in telem.probe().items() if v and k != "max_clock"}
        problem = "; ".join(f"{k} read failed ({v})" for k, v in broken.items())
    if problem:
        hint = telemetry_install_hint() if telem.missing_package else ""
        sys.exit(f"error: telemetry is required to validate MAMF but is unavailable: {problem}."
                 f"{hint}\nOr pass --telemetry off to run anyway with an unvalidated MAMF.")
    return telem


def search_setup(args):
    """Return (mode, grid_shapes, range_info, warmup_shape) for the requested search, grid_shapes being None in auto
    mode. Exit if the request can't be searched."""
    # any explicit shape argument means the user wants a specific sweep -> grid mode
    shape_args_given = args.shapes_file is not None or any(
        x is not None for x in (args.m, args.m_range, args.n, args.n_range, args.k, args.k_range))
    if args.search == "grid" or shape_args_given:
        if args.shapes_file:
            shapes = []
            for lineno, line in enumerate(Path(args.shapes_file).read_text().splitlines(), 1):
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                fields = re.split(r"[\s,xX]+", line)
                if len(fields) != 3:
                    sys.exit(f"error: {args.shapes_file}:{lineno}: expected M,N,K, got {line!r}")
                shapes.append(tuple(map(int, fields)))
            range_info = f"exact shapes from {args.shapes_file} ({len(shapes)} shapes)"
        else:
            dims = (("m", args.m, args.m_range), ("n", args.n, args.n_range), ("k", args.k, args.k_range))
            missing = [name for name, vals, rng in dims if vals is None and rng is None]
            if missing:
                sys.exit(f"error: --search grid requires shapes for: {', '.join(missing)} (use --m/--n/--k, "
                         "--m_range/--n_range/--k_range, or --shapes_file)")
            shapes = list(itertools.product(*(resolve_dim(vals, rng) for _, vals, rng in dims)))
            range_info = " | ".join(f"{name}={','.join(map(str, vals)) if vals is not None else f'range{tuple(rng)}'}"
                                    for name, vals, rng in dims)
        if not shapes:
            sys.exit(f"error: no shapes to search in {range_info}")
        if args.dtype in ("mxfp4", "nvfp4"):
            bad_k = sorted({k for _, _, k in shapes if k % 32})
            if bad_k:
                sys.exit(f"error: --dtype {args.dtype} needs K to be a multiple of 32, "
                         f"got K={','.join(map(str, bad_k))}")
        return "grid", shapes, range_info, tuple(map(int, shapes[0]))

    # auto search derives its candidate shapes from the compute-unit layout (compute_unit_count + gemm_tile_hint)
    if arch.gemm_tile_hint is None or not arch.compute_unit_count():
        sys.exit(f"error: --search auto derives its shapes from the GPU's compute-unit layout, which mamf-finder.py "
                 f"doesn't model for {arch.name!r}; search an explicit range instead with --m/--n/--k (or "
                 "--m_range/--n_range/--k_range) or --shapes_file")
    if not arch.geometry_validated and arch.geometry_checked:
        print(f"note: --search auto was checked against an exhaustive grid only on {arch.geometry_checked}; on "
              "other GPUs and dtypes compare it with a small --search grid before trusting the shapes it picks")
    elif not arch.geometry_validated:
        print(f"note: --search auto has not been checked against an exhaustive grid on {arch.name!r}; compare it with "
              "a small --search grid before trusting the shapes it picks")
    range_info = f"auto-search (CUs={arch.compute_unit_count()}, dtype={args.dtype}, max_size={args.tune.max_size})"
    return "auto", None, range_info, (4096, 4096, 4096)


# PyTorch TunableOp (ROCm, PYTORCH_TUNABLEOP_ENABLED=1) benchmarks the candidate kernels of every GEMM shape the first
# time it sees it - ~2 min per shape on MI300X / ROCm 10 - so letting it tune during a search of thousands of shapes
# would take days. Instead the search runs with tuning paused (unseen shapes get the default kernel, which is enough to
# rank them) and only the confirm shortlist is tuned, before any of it is timed.
def tunableop_setup(confirm_max):
    """If TunableOp is on, pause tuning for the search and return True."""
    tunable = getattr(torch.cuda, "tunable", None)
    if tunable is None or not tunable.is_enabled():
        return False
    tunable.tuning_enable(False)
    print(f"TunableOp: on - the search runs with tuning paused; up to {confirm_max} confirm shapes get tuned before "
          "they are measured")
    return True


def tunableop_shortlist(msmf_cands, mamf_cands, max_shapes):
    """Up to max_shapes shapes, alternating the two best-first candidate lists so each keeps its leaders."""
    out = []
    for pair in itertools.zip_longest(msmf_cands, mamf_cands):
        for s in pair:
            if s is not None and s not in out:
                out.append(s)
    return out[:max_shapes]


def tunableop_tune(shapes, dtype, device):
    """Tune each shape once, then pause tuning again: the tuned kernels stay in use, while any other shape (e.g. the
    settle clock's) runs the default kernel instead of stalling on a fresh tune."""
    print(f"TunableOp: tuning {len(shapes)} confirm shapes (~2 min each on MI300X / ROCm 10)")
    torch.cuda.tunable.tuning_enable(True)
    try:
        for i, shape in enumerate(shapes, 1):
            t0 = time.time()
            op = prepare_gemm(*shape, dtype, device)[0]
            op()
            arch.synchronize()
            print(f"  {i}/{len(shapes)} {shape_str(shape)} tuned in {time.time() - t0:.0f}s", flush=True)
    finally:
        torch.cuda.tunable.tuning_enable(False)



@dataclass
class Tuning:
    """Knobs of the search and confirm algorithm. The defaults are what the published numbers were measured with, so
    leave them alone unless you are working on the algorithm itself. Override one with `--tune NAME=VALUE`."""

    # search
    max_size: int = 20480               # auto: largest M/N/K to consider
    warmup: str = "adaptive"            # adaptive: until throughput plateaus; fixed: a flat 30s
    scout_num_iterations: int = 20      # timed iterations per shape while scouting
    scout_num_warmup_iterations: int = 8
    refine_grid: bool = True            # auto: scan a tight local grid around the best scouted shapes...
    refine_seeds: int = 4               # ...centered on this many of them
    refine_radius_mn: int = 4           # ...this many 256-steps wide along M and N
    refine_radius_k: int = 2            # ...and 1024-steps along K
    scout_only: bool = False            # stop after scouting (for building an exhaustive reference grid in parts)

    # confirm
    confirm_top: int = 10               # best scouted shapes re-measured for MSMF
    confirm_fat_forced: int = 3         # largest-in-every-dimension shapes always added: they reach the power cap
    confirm_reps: int = 10              # MAMF bursts per confirmed shape
    tunableop_confirm_max: int = 8      # with TunableOp on: shapes tuned and confirmed (~2 min each on MI300X)

    # MAMF (boost burst)
    mamf_confirm_top: int = 12          # screen winners added to the confirm set
    mamf_raw_forced: int = 8            # scouting peak leaders always screened
    mamf_wave_k: int = 3                # auto: K values screened per wave-packed M,N layout
    mamf_screen_iters: int = 2          # iterations per screened shape
    mamf_screen_idle_s: float = 0.05    # idle seconds before each screened burst
    k_edge_layouts: int = 8             # auto: layouts still rising at the K ceiling whose K is screened past it...
    k_edge_max: int = 65536             # ...at 1.5x and 2x the ceiling, up to this K
    mamf_burst_iters: int = 20          # iterations per confirm burst; short, so it ends before the clock drops
    mamf_idle_s: float = 0.25           # idle seconds before each confirm burst, so the clock recovers to boost
    boost_clock_ratio: float = 0.97     # an iteration is boost if its SM clock is at least this x the run's highest

    # MSMF (sustained window)
    msmf_settle_clock_s: float = 20.0   # max seconds of full load to let the SM clock settle before MSMF; 0 to skip
    msmf_rank_s: float = 1.0            # timed seconds per confirm shape to rank them...
    msmf_top: int = 10                  # ...then this many get the full window
    msmf_warm_s: float = 1.0            # untimed seconds before each full window
    msmf_window_s: float = 2.0          # timed seconds per full window, split into 8 sub-windows
    msmf_max_drop: float = 1.0          # a window whose second half is this % slower than its first isn't steady
    msmf_max_spread: float = 10.0       # sub-window sanity limit; at the power cap 0.25s sub-windows jitter ~4%
    msmf_kwalk_seeds: int = 4           # auto: walk K down from this many fastest steady (M,N) layouts (0 = off)
    msmf_kwalk_waves: int = 1           # auto: also walk the most-square layouts of up to this many waves, both ways
    msmf_kwalk_steps: int = 10          # at most this many K steps per walk
    msmf_kwalk_stop: float = 1.0        # end a walk once a step is this % below the best seen in it
    msmf_lock_s: float = 4.0            # timed seconds re-measuring the leader
    msmf_lock_tries: int = 4            # leaders tried before accepting the best available

    @classmethod
    def from_overrides(cls, pairs):
        """Build from `NAME=VALUE` strings; raise ValueError naming the bad one."""
        tune, types = cls(), {f.name: f.type for f in fields(cls)}
        for pair in pairs:
            name, sep, value = pair.partition("=")
            if not sep or name not in types:
                raise ValueError(f"--tune {pair!r}: expected NAME=VALUE, NAME one of: {', '.join(types)}")
            if types[name] is bool:
                if value.lower() not in ("true", "false", "1", "0"):
                    raise ValueError(f"--tune {pair!r}: expected true or false")
                setattr(tune, name, value.lower() in ("true", "1"))
            else:
                try:
                    setattr(tune, name, types[name](value))
                except ValueError:
                    raise ValueError(f"--tune {pair!r}: expected a{'n' * (types[name] is int)} {types[name].__name__}")
        if tune.warmup not in ("adaptive", "fixed"):
            raise ValueError(f"--tune warmup={tune.warmup!r}: expected adaptive or fixed")
        return tune


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Append "(default: X)" only where there is a default worth showing; keep the description's paragraphs."""
    def _get_help_string(self, action):
        if action.default in (None, "", []):
            return action.help
        return super()._get_help_string(action)

    def _fill_text(self, text, width, indent):
        return "\n\n".join(super(_HelpFormatter, self)._fill_text(p, width, indent) for p in text.split("\n\n"))


def parse_args():
    """Return the parsed args; the search/confirm knobs are in `args.tune` (see Tuning)."""
    parser = argparse.ArgumentParser(
        formatter_class=_HelpFormatter,
        description="Find the maximum achievable (MAMF, boost burst) and maximum sustainable (MSMF, power-saturated) "
                    "matmul TFLOPS of one accelerator.\n\n**TLDR**: On NVIDIA and AMD GPUs start by running it "
                    "without any arguments: the default --search auto finds the best shapes on its own. Other "
                    "accelerators need --search grid with a --m/--n/--k range.")

    what = parser.add_argument_group("what to measure")
    what.add_argument("--dtype", default="bfloat16",
                      choices=SUPPORTED_DTYPES, metavar="{" + ", ".join(SUPPORTED_DTYPES) + "}",
                      help="float8_e4m3fn is NVIDIA's fp8 and float8_e4m3fnuz AMD MI300's; mxfp8 and mxfp4 need "
                           "NVIDIA Blackwell or AMD MI355X; nvfp4 needs NVIDIA Blackwell")
    what.add_argument("--search", choices=["auto", "grid"], default="auto",
                      help="auto: find the best shape anywhere; grid: the best shape in the --m/--n/--k range you "
                           "give. Any shape argument implies grid")
    for dim, desc in (("m", "first dimension"), ("n", "last dimension"), ("k", "shared (reduction) dimension")):
        g = what.add_mutually_exclusive_group()
        g.add_argument(f"--{dim}", nargs="+", type=int, help=f"grid: the GEMM's {desc}, one or more values")
        g.add_argument(f"--{dim}_range", nargs="+", type=int, metavar="N",
                       help=f"grid: the GEMM's {desc} as START STOP [STEP]")
    what.add_argument("--shapes_file", type=str,
                      help="grid: exact M,N,K shapes, one per line (MxNxK, commas or spaces), instead of the "
                           "product of --m/--n/--k")

    run = parser.add_argument_group("how to run")
    run.add_argument("--cuda_device", type=int, default=0, help="index of the device to measure")
    run.add_argument("--telemetry", choices=["on", "off"], default="on",
                     help="sample power and SM clock to validate both headlines; off runs without validation")
    run.add_argument("--tune", action="append", default=[], metavar="NAME=VALUE",
                     help="override a search/confirm knob, repeatable; the knobs and their defaults are in the "
                          "Tuning class of this script")

    out = parser.add_argument_group("output")
    out.add_argument("--output_file", type=str, default=f"{file_dir}/results/mm.out", help="log file")
    out.add_argument("--notes", type=str, default="", help="text to add to the log's header")
    out.add_argument("--verbose", default=True, action=argparse.BooleanOptionalAction,
                     help="also print the log to the console")

    args = parser.parse_args()
    try:
        args.tune = Tuning.from_overrides(args.tune)
    except ValueError as e:
        parser.error(str(e))
    return args


if __name__ == '__main__':
    args = parse_args()

    arch.set_device(args.cuda_device)
    device = arch.device
    setup_checks()
    dtype_checks(args.dtype, device)
    dtype = args.dtype
    tunableop = tunableop_setup(args.tune.tunableop_confirm_max)
    telem = telemetry_setup(args.telemetry, args.cuda_device)
    mode, grid_shapes, range_info, warmup_shape = search_setup(args)
    if mode == "grid":  # grid reports the best shape within the user's range, so nothing measures K outside it
        args.tune.k_edge_layouts = args.tune.msmf_kwalk_seeds = 0

    sys.stdout = Tee(args.output_file, args.verbose)
    print_benchmark_header(dtype, device, args.notes + f"\n- search mode: {mode}")

    best_tflops = dict(max=0, median=0, mean=0)
    best_config = dict(max="", median="", mean="")
    # both modes report two headlines from the shapes they measure:
    #   mamf = Maximum ACHIEVABLE  Matmul FLOPS - the boost burst a short kernel can catch
    #   msmf = Maximum SUSTAINABLE Matmul FLOPS - what the chip holds once saturated at ~TDP
    # auto picks candidate shapes from heuristics; grid picks them from the user-supplied range.
    headline = dict(mamf=None, msmf=None, msmf_idle_siblings={})
    num_shapes = 0
    all_mean_tflops = []
    measured = [] # (mean_tflops, power_W_or_None, (M, N, K)) for every shape tried, for the auto confirm phase
    # per-shape scout metadata (peak tflops + peak clock) so the MAMF phase can pick shapes that ran at the boost clock
    # and verify a headline came from boost, not the throttled/base clock.
    scout_meta = {} # (M,N,K) -> dict(mean, mx, power, clock_max)
    boost_ref = {"clk": 0.0} # highest SM clock seen anywhere this run == the effective boost ceiling
    scout_rows = {} # "MxNxK" -> its progress row, to re-print the winner when scouting ends
    scout_winner_shown = []
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

        tinfo = ""
        if _last_telem.get("power") is not None:
            p = _last_telem["power"]
            # Prefer peak clock in the scout line — that's what MAMF validation keys off.
            ck = _last_telem.get("clock_max") or _last_telem.get("clock_min")
            tinfo = f" {p:5.0f}" + (f" {ck:5.0f}" if ck is not None else "")

        # the column header is re-printed only when another message interrupted the progress rows
        if not getattr(sys.stdout, "after_cr", False):
            print(SCOUT_HEADER + (f" {'W':>5} {'MHz':>5}" if telem_ok(telem) else ""))
        row = (f"{num_shapes:>6}  {cur_config:<18} {mean_tflops:6.1f} {median_tflops:6.1f} {max_tflops:6.1f} "
               f"{best_tflops['mean']:6.1f}{tinfo}")
        scout_rows[cur_config] = row
        print(row, end="\r", flush=True)
        return mean_tflops

    def show_scout_winner():
        """Leave the best-mean scout row under the header once scouting ends."""
        shp = best_config["mean"].removesuffix(" (MxNxK)")
        if shp in scout_rows and not scout_winner_shown:
            scout_winner_shown.append(True)
            phase_result(scout_rows[shp], "best mean")

    def finish():
        show_scout_winner()
        all_tried_shapes_geometric_mean_tflops  = np.exp(np.log(all_mean_tflops).mean()) if all_mean_tflops else 0
        all_tried_shapes_arithmetic_mean_tflops = np.mean(all_mean_tflops) if all_mean_tflops else 0

        time_delta = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
        print("", end="\033[K")

        if headline.get("mamf") or headline.get("msmf"):
            outcomes = (
                f"MAMF (max achievable,  boost burst): {format_headline(headline.get('mamf'))}\nMSMF (max "
                f"sustainable, saturated):   {format_headline(headline.get('msmf'))}"
            )
            # a sibling idle for under 10% of the window is start/finish skew between concurrent copies, not idleness
            idle = {i: f for i, f in headline["msmf_idle_siblings"].items() if f >= 0.1}
            if headline.get("msmf") and idle:
                gpus = ", ".join(f"GPU{i} {f:.0%}" for i, f in idle.items())
                outcomes += (f"\nnote: {len(idle)} same-board sibling GPU(s) sat idle during the MSMF measurement, so "
                             f"MSMF reads high.\n      Share of the time each was idle: {gpus}\n      With all "
                             "siblings idle it is a single-GPU upper bound; for full-node MSMF use "
                             "mamf-finder-all-gpus.py.")
        else:
            outcomes = (
                f"mean:   {best_tflops['mean']:.1f} TFLOPS @ {best_config['mean']}\nmedian: "
                f"{best_tflops['median']:.1f} TFLOPS @ {best_config['median']}\nmax:    {best_tflops['max']:.1f} "
                f"TFLOPS @ {best_config['max']}"
            )
        print(f"""
{"-" * 80}

** Results:

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

        Both modes populate `measured`/`scout_meta` first (auto via heuristic scouts, grid via its user-supplied
        sweep); this then ranks candidates and produces the two headlines:
          MSMF (sustainable): the highest throughput a shape holds steady over a timed window under full load (see
            confirm_msmf).
          MAMF (achievable): the median of a shape's 5 fastest boost-clock iterations, each burst after an idle gap
            (see confirm_mamf).
        Both confirms measure the union of independently ranked MAMF and MSMF candidates: a fat shape that saturated
        while scouting still recovers boost after idle, while a low-power wave candidate cannot be crowded out by
        MSMF's power-aware ranking. The wave metadata is populated by auto; grid uses raw scout leaders from its range.
        """
        seen = seen or set()
        square_by_wave = square_by_wave or {}
        wave_layouts_by_wave = wave_layouts_by_wave or {}
        show_scout_winner()
        detail("\nConfirming the best candidates (MSMF, then MAMF) ...")
        msmf_cands, _n_confirm = build_msmf_confirm_set(
            measured, scout_meta, seen, square_by_wave, args)
        reps = max(1, args.tune.confirm_reps)
        boost_clk = boost_ref["clk"]

        recall_pool, provenance = build_mamf_recall_pool(
            scout_meta, wave_layouts_by_wave, args)
        screened = screen_mamf_candidates(
            recall_pool, args, dtype, device, telem, boost_clk)
        raw_forced = top_shapes_by_peak(scout_meta, args.tune.mamf_raw_forced)
        mamf_cands = list(dict.fromkeys(raw_forced + screened))
        top = list(dict.fromkeys(msmf_cands + mamf_cands))
        detail(f"  full confirm set: {len(top)} shapes (MSMF={len(msmf_cands)}, MAMF={len(mamf_cands)}, "
               f"overlap={len(set(msmf_cands) & set(mamf_cands))})")
        if mamf_cands:
            detail("  MAMF candidates: " + ", ".join(
                f"{shape_str(s)}[{'+'.join(provenance.get(s, ['screen']))}]" for s in mamf_cands))
        if tunableop:
            top = tunableop_shortlist(msmf_cands, mamf_cands, args.tune.tunableop_confirm_max)
            tunableop_tune(top, dtype, device)

        stop_watch = telem.watch_siblings() if telem_ok(telem) else None
        walk_mn = [mn for w in range(1, args.tune.msmf_kwalk_waves + 1) if (sq := square_by_wave.get(w))
                   for mn in dict.fromkeys([sq, sq[::-1]])]
        msmf_results, msmf = confirm_msmf(top, args, dtype, device, telem, all_mean_tflops, walk_mn)
        if stop_watch:
            headline["msmf_idle_siblings"] = stop_watch()
        if msmf and msmf.get("row"):
            phase_result(msmf["row"], "MSMF")

        mamf_results = confirm_mamf(top, args, dtype, device, telem, reps, boost_clk, all_mean_tflops)
        boost_clk = max([boost_clk] + [r["boost_ref"] for r in mamf_results])
        mamf = select_mamf(mamf_results, boost_clk)
        rated = telem.max_clock() if telem_ok(telem) else None
        if rated and boost_clk and boost_clk < args.tune.boost_clock_ratio * rated:
            print(f"note: highest SM clock this run {boost_clk:.0f}MHz is below the rated max {rated:.0f}MHz - clocks "
                  "locked, power-capped or thermally limited?")
        print_regime_crosscheck(msmf, mamf, msmf_results, mamf_results, args, dtype, device, telem,
                                reps, boost_clk, all_mean_tflops)

        apply_headlines(headline, best_tflops, best_config, msmf, mamf)

    def auto_search():
        """Lean directed search that reports both MAMF (boost) and MSMF (saturated).

        Ablation (offline replay vs exhaustive H200/B200 grids) showed that coordinate descent, re-descent, hill-climb
        and line-search add probes but no unique reach. The keep-set that still matches the grid within 0.02% offline
        is:

          1. wave-quantization (M,N) @ a short K set (every wave candidate, not top-N only)
          2. coarse M×N plane @ Kmin                             - many-waves / min-K basin
          3. tight local grid around the top scout seeds         - endgame polish
          4. MAMF recall screen: raw leaders + several K choices per wave-layout basin, plus K past the K ceiling for
             the layouts still rising at it
          5. both confirms over their candidate union (same shapes, different regimes); MSMF walks K down from its
             fastest steady layout
        """
        sms = arch.compute_unit_count()  # main() checked that both geometry hooks return usable values
        tile_m, tile_n = arch.gemm_tile_hint
        elem = dtype_element_size(dtype)
        align = max(int(128 // elem), 1)  # tensor-core element alignment (bf16->64, fp8->128, fp4->256, fp32->32)
        base = 256                    # M/N step: a multiple of `align` and of the 256-wide tile
        if base % align:
            base = ((base // align) + 1) * align
        scout_i, scout_w = args.tune.scout_num_iterations, args.tune.scout_num_warmup_iterations
        max_size = args.tune.max_size
        k_min = 1024
        k_max = max_size - (max_size % 1024) or max_size

        print(f"Auto search: CUs={sms} tile={tile_m}x{tile_n} dtype={args.dtype} elem={elem}B base={base} "
              f"max_size={max_size}")

        memo = {}
        def smeasure(M, N, K, label):
            key = (int(M), int(N), int(K))
            if key not in memo:
                memo[key] = measure(key[0], key[1], key[2], scout_i, scout_w, label)
            return memo[key]

        def discover():
            """Phases 1–3: wave / plane / refine scouting plus wave provenance for confirm."""
            # Phase 1: wave-quantization-aware (M,N) at a short K set (not just the extremes). Measuring *every* wave
            # (M,N) at several Ks is what gets H200's wave-perfect mid/high-K winners (e.g. 1536x2816x20480) into the
            # confirm set, and B200's low-K ones (e.g. 3072x18944x2048), which burst fastest at K=2048-3072.
            wave_ks = sorted({v for v in (k_min, 2048, 3072, 4096, 8192, 12288, 14336, 16384, k_max) if v <= max_size})
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
            detail(f"  wave: {len(seen)} MxN @ K={','.join(map(str, wave_ks))} -> {n_wave} scouts")

            # Phase 2: coarse M×N planes at Kmin and the low-K boost basin. K=3072 is where the B200 exhaustive oracle
            # found its best non-wave MAMF family; Kmin alone cannot seed it.
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
            detail(f"  plane: {len(plane)}x{len(plane)} MxN @ K={','.join(map(str, plane_ks))} -> {n_plane} scouts")

            # Phase 3: tight local grid around the top scout seeds (endgame polish). Walks a ±r_mn × ±r_mn × ±r_k
            # neighborhood at native lattice resolution (256 / 1024) so an off-axis peak next to a coarse seed is not
            # stepped over. Seed set is basin-diverse (half reserved for wave (M,N)s) so both peaks get polished.
            if args.tune.refine_grid:
                n_power = max(1, args.tune.refine_seeds // 2)
                power_seeds = top_shapes(
                    measured, n_power, prefer_mn=seen, prefer_slots=max(1, n_power // 2))
                peak_seeds = top_shapes_by_peak(scout_meta, args.tune.refine_seeds - n_power)
                seeds = list(dict.fromkeys(power_seeds + peak_seeds))
                r_mn, r_k = args.tune.refine_radius_mn, args.tune.refine_radius_k
                detail(f"  tight grid (±{r_mn} MN @ {base}, ±{r_k} K @ 1024) around {len(seeds)} seed(s): "
                       f"{', '.join(map(shape_str, seeds))}")
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
         signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl-C would re-enter finish() mid-report
         finish()
         sys.exit(1)

    signal.signal(signal.SIGINT, sigkill_handler)

    # XXX: the transpose version seemed to work better for MI300X

    # Warm up before measuring: a cold accelerator boosts its clock and over-reports, so run the GPU to steady state
    # first. `adaptive` (default) keys off a *measured characteristic* - the matmul throughput plateau - so it works on
    # any accelerator (power-capped or thermally boosting) and stops as soon as it's warm. The old flat 30s is
    # available as `--tune warmup=fixed`.
    #
    # These two bounds are internal guardrails, not tuning dials, so they're not exposed on the CLI:
    #   MIN - a chip can read stable-but-hot in the first chunks (low CoV, low drift) right after a
    #         boost; the floor forces it to sit long enough to actually thermally settle before we
    #         can declare convergence, avoiding a false "warm" at boosted clock.
    #   MAX - a plain hang guard so an accelerator/telemetry that never plateaus can't spin forever.
    WARMUP_MIN_SECONDS, WARMUP_MAX_SECONDS = 2.0, 45.0
    if telem_ok(telem):
        print(f"telemetry: {telem.backend} (power/clock sampling on)")
        rated = telem.max_clock()
        if rated:
            print(f"  rated max SM clock {rated:.0f} MHz")
    elif args.telemetry == "off":
        print("*** telemetry: off - MAMF is not boost-validated")

    def warmup_adaptive(shape, min_s, max_s, chunk=25, window=4, cov_thr=0.02, drift_thr=0.01):
        """Run saturated chunks of matmuls and watch the throughput; converged once a rolling window is both stable
        (low CoV) and no longer drifting vs the previous window."""
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
        print(f"adaptive warmup: {'throughput plateaued' if conv else 'hit time cap'} after {el:.1f}s / "
              f"{len(hist)*chunk} iters{extra}")

    if args.tune.warmup == "adaptive":
        print("Warming up (adaptive: until matmul throughput plateaus) ...", flush=True)
        warmup_adaptive(warmup_shape, WARMUP_MIN_SECONDS, WARMUP_MAX_SECONDS)
    else:
        accelerator_warmup_seconds = 30
        end_time = time.monotonic() + accelerator_warmup_seconds
        print(f"Warming up the accelerator for {accelerator_warmup_seconds} secs ... ", end="", flush=True)
        while time.monotonic() < end_time:
            _ = benchmark_mm(warmup_shape[0], warmup_shape[1], warmup_shape[2], dtype, device, 100, 50)
        print("accelerator warmup finished")

    if mode == "grid":
        # Sweep every shape in the user's range as short SCOUTS (same budget as auto), then the shared confirm phase
        # re-measures the winners for MAMF + MSMF. Full iters on every grid point would just re-pay the confirm cost
        # without changing the headlines.
        scout_i, scout_w = args.tune.scout_num_iterations, args.tune.scout_num_warmup_iterations
        after = "without confirm" if args.tune.scout_only else "then confirm"
        print(f"Grid search: sweeping {len(grid_shapes)} shapes (scout {scout_i} iters / {scout_w} warmup), "
              f"{after} ...")
        for M, N, K in grid_shapes:
            measure(M, N, K, scout_i, scout_w, label="grid")
        if not args.tune.scout_only:
            confirm_phase()
    else:
        auto_search()

    finish()
