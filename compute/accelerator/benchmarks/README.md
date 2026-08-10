# Accelerator Benchmarks

## Maximum Achievable Matmul FLOPS Finder

Maximum Achievable Matmul FLOPS (MAMF) Benchmark: [mamf-finder.py](./mamf-finder.py) was derived from research found in [The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489) paper.

For a detailed discussion and the numbers for various accelerators see [Maximum Achievable FLOPS](../README.md#maximum-achievable-flops).

While some accelerator manufacturers publish the theoretical TFLOPS these usually can't be reached. As a result of this when we try to optimize our software we have no realistic performance bar to compare ourselves to. The Model FLOPS Utilization (MFU) metric measures TFLOPS achieved against theoretical TFLOPS. Usually when one scores around 50% MFU it's considered a win. But this gives us no indication how far are we from the real achievable throughput.

This benchmark scans various large shapes of matmul and reports the highest achievable TFLOPS it registered. As transformers training and partially inference workloads are dominated by large matmul operations it's safe to use the best matmul TFLOPS one can measure on each accelerator as a rough estimation that this is the Maximum Achievable Matmul FLOPS (MAMF). Now instead of the previously used MFU, one can use Model Achievable Matmul FLOPS Utilization (MAMFU).

Therefore now you can compare the TFLOPS you measured for your training or inference against a realistic number. As you will now be much closer to 100% it'll be much easier to know when to stop optimizing.

Currently supported high end architectures:
- NVIDIA: V100, A100, H100, ...
- AMD: MI250, MI300X, MI325X, ...
- Intel Gaudi2/3

Important notes:
- if you can find a better and more efficient way to detect the best matmul TFLOPS by approaching each new accelerator as a black box, please kindly send a PR with the improvement including the generated log file.
- also if you know that this benchmark should be run under special conditions to show the best results, such as some kernel settings or similar, please submit a PR to add such special instructions. For example, for AMD MI300X I'm being told disabling the numa_balancing is supposed to help.
- since a big part of the overhead comes from HBM IO, if you're using a fused kernel with 2 or more matmuls, whose results don't leave the accelerator's registers, the performance will be definitely faster than what this benchmark reports.
- It also helps to sample your accelerator's actual clock speed. If your accelerator is running at a slower clock than the one used in the spec, there is no chance you can get the theoretical TFLOPS (see [How To Calculate Theoretical TFLOPS](../README.md#how-to-calculate-theoretical-tflops)).

### Architecture specific notes:

Follow the special setup instructions before running the benchmark to achieve the best results:

**MI300x, MI325X, etc.**:

1. Turn numa_balancing off for better performance:
```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```
2. Enable PyTorch TunableOp:
```bash
export PYTORCH_TUNABLEOP_ENABLED=1
```
This will make the first iteration very slow, while it's searching for the best GEMM algorithm in the BLAS libraries for each `matmul` shape it encounters, but subsequent operations are likely to be significantly faster than the baseline. See [Accelerating models on ROCm using PyTorch TunableOp](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html) (requires `torch>=2.3`) [doc](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md).

**Intel dGPUs (A770, A750, B580, etc.)**
- Follow Intel Extension for PyTorch [installation steps](https://pytorch-extension.intel.com/installation?platform=gpu)

**AMD / Gaudi telemetry (power / SUSPECT):** NVIDIA NVML is the only *validated*
backend for excluding unsaturated boost bursts from the headline. `amdsmi` and
`pyhlml` are wired and will *report* power/clock when installed, but do not
exclude `SUSPECT` shapes until validated on real iron (`VALIDATED_BACKENDS` in
`mamf-finder.py`). Adaptive warmup and median-of-reps confirm still work without
vendor telemetry (`--telemetry off`). On first access, run
[`mamf_spike_probe.py`](mamf_spike_probe.py) on a known-good shape: if the
idle-gap phase shows much higher TFLOPS at much lower power than the continuous
phase, the boost pathology holds and that backend can be promoted.

### Examples of usage

`K` is the reduction dimension: `(MxK)*(KxN)=(MxN)`. Default dtype is `bfloat16`
(`--dtype` accepts any `torch` dtype, e.g. `float8_e4m3fn`, `float16`, `float32`).
Default iterations are 50 warmup + 100 measured per shape (`--num_warmup_iterations`,
`--num_iterations`).

#### 1. Auto search (default) — find the best shape on this accelerator

```bash
./mamf-finder.py --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt
# equivalent: ./mamf-finder.py --search auto ...
```

Runs in under a minute (typically ~40–60 s including adaptive warmup, a thermal
soak, and both confirms). It prints **two** headline numbers and this is what
produced the
[MAMF & MSMF table](../README.md#maximum-achievable-matmul-flops-comparison-table):

- **MAMF** (Maximum *Achievable* Matmul FLOPS) — the boost-clock burst ceiling.
- **MSMF** (Maximum *Sustainable* Matmul FLOPS) — the power-saturated, sustained
  rate that matches real training throughput.

How `--search auto` works:

1. **Wave candidates @ a short K set** — enumerate `(M,N)` shapes that fill an
   integer number of full waves of thread-block tiles across the SMs (most-square /
   widest / tallest per wave count 1..16), and measure *every* such `(M,N)` at
   several Ks (min / mid / max). Covers the high-arithmetic-intensity /
   wave-perfect basin; measuring all wave candidates (not only a ranked top-N)
   is what keeps mid/high-K winners in the confirm set.
2. **Coarse M×N plane @ `Kmin`** — a cheap grid at the smallest K. Seeds the
   many-waves / min-K basin that wave candidates can miss.
3. **Tight local grid** around the top scout seeds (`±2` steps of 256 on M/N,
   `±2` steps of 1024 on K) — endgame polish for off-axis peaks.
4. **Thermal soak** — before any sustainable measurement, drive the chip to
   steady state (hot, SM clock settled at the saturated floor), stopping early
   once the clock stops dropping (`--msmf_soak_s`). This is the single biggest
   reproducibility lever: a cold/cooler card boosts and reads high, so without
   soaking the MSMF headline would depend on how warm the card happened to be.
5. **MSMF confirm** (sustainable): re-measure the top shapes with the full
   iteration count and per-shape warmup, repeated (`--confirm_reps`, default 5 →
   trimmed median), with a short thermal pre-warmup per shape so a freshly
   switched shape doesn't read cold on its first rep. The confirm set always
   includes (a) the best scouted K for each most-square low-wave shape (w=1..4)
   in **both** `(M,N)` orientations — e.g. H200's `1536×2816` family — and (b)
   the **fattest** scouts (largest smaller-dimension), which reliably saturate to
   TDP and anchor the saturated-clock reference. A shape is kept out of the MSMF
   headline (still reported) if it is **not saturated** — either drawing well
   below peak power (`SUSPECT`) or running materially **above the saturated-clock
   floor** (`--msmf_clock_ratio`, i.e. a tall-skinny layout that keeps clock
   headroom) — or if it measures **too jittery to reproduce** (`--msmf_max_spread`).
   Finally the winner is **lock-in validated** (`--msmf_lock_reps`): re-measured
   with extra reps and published only if that longer run is itself within the
   spread tolerance, else the next candidate is tried. All exclusions apply only
   on validated telemetry backends (currently NVIDIA NVML); untested backends
   report power/clock but do not exclude.
6. **MAMF confirm** (achievable): take the scouts that ran at (near) the run's
   **boost** clock — the low-power shapes MSMF rejects — idle briefly so the
   clock recovers, then measure a SHORT burst. Each iteration is timed in
   isolation and **bracketed by a synchronous clock read on both sides**, so the
   peak ("winning") iteration carries its *own* measured clock (`min` of the two
   brackets) rather than a clock sampled at some unrelated moment. The peak is
   reported only if that bracketed clock reached boost (`--boost_clock_ratio`),
   so a throttled/base-clock reading can't be published as MAMF.

Scout measurements are memoized, so overlapping candidates cost nothing.

**Reproducibility — the whole point of a single-run number.** A published figure
must be reproducible on the same GPU with the same setup, so:

- **Measure with every other GPU idle.** Sibling GPUs share the board's
  power/cooling budget; a busy neighbor silently lowers and destabilizes the
  sustained clock. The finder **warns** at startup if a sibling is active — use
  `CUDA_VISIBLE_DEVICES=<id>` and run one benchmark at a time.
- **The headline value is the *mean* of the winning shape** (not a lucky max),
  measured at a known clock, both printed in the headline (`… 985W 1360MHz`).
- **To reproduce a published number, re-run its exact shape in grid mode** — e.g.
  `--m 9472 --n 6144 --k 12288` — not the search. Auto's job is to *find* the
  shape; grid *reproduces* it (the sustained mean is stable to well under 1%).
  Re-running `auto` re-searches and may land on a different (equivalent) shape.

With this protocol the MSMF headline is reproducible to ~1–2% run-to-run and MAMF
to ~1–2% (boost-locked); most of the residual is genuine saturated-clock jitter.

**What we measured: alone vs all-8 concurrent.**

Same protocol on both chips — five `--search auto` runs on GPU0 with siblings
idle, then five rounds with the same script on all 8 GPUs at once:

*B200 bf16 (torch 2.13.0+cu130):*

| Setup | MSMF mean | MSMF range | MSMF clock | MAMF mean |
| :---- | --------: | ---------: | ---------: | --------: |
| GPU0 alone (n=5) | 1455.7 | 1.1% | ~1330–1460 MHz | 1732.6 |
| GPU0 while all 8 busy (n=5) | 1438.7 (−1.2%) | 1.8% | ~1316–1332 MHz | 1757.6 |
| All 8 GPUs × 5 rounds (n=40) | 1426.4 (−2.0%) | **10.4%** | 1228–1410 MHz | 1751.2 |

*H200 bf16 (torch 2.14.0.dev20260810+cu130):*

| Setup | MSMF mean | MSMF range | MSMF clock | MAMF mean |
| :---- | --------: | ---------: | ---------: | --------: |
| GPU0 alone (n=5) | 702.9 | 1.4% | ~1440–1465 MHz | 828.0 |
| GPU0 while all 8 busy (n=5) | 701.3 (−0.2%) | 2.9% | ~1440–1525 MHz | 825.8 |
| All 8 GPUs × 5 rounds (n=40) | 698.6 (−0.6%) | 4.5% | 1405–1527 MHz | 823.9 |

Lesson: sibling load barely moves **MAMF** (boost bursts still hit the boost
clock — they draw only ~150–300 W, so board power/cooling still has headroom).
It *does* move **MSMF**, and the severity is board-dependent: on B200 (1000 W
TDP) concurrent siblings pulled GPU0's saturated clock down ~50 MHz (−1.2% MSMF)
and opened a **~10% MSMF spread across the 8 GPUs**; on H200 (700 W TDP) the
same protocol was much milder (−0.2% on GPU0, ~4.5% across the board). So a
published MSMF number is only reproducible if measured on an otherwise-idle
node; a full-node training run will see something closer to the concurrent
distribution, and how far that sits below the alone headline depends on the
board's shared power/cooling budget. That is why the table numbers are
alone-GPU measurements, and why the finder warns when siblings are busy.

Useful knobs: `--dtype`, `--max_size`, `--confirm_top` (default 10),
`--confirm_reps`, `--confirm_fat_forced`, `--msmf_soak_s`, `--msmf_max_spread`,
`--msmf_clock_ratio`, `--msmf_lock_reps`, `--mamf_burst_iters`, `--mamf_idle_s`,
`--boost_clock_ratio`, `--no-refine_grid`, `--telemetry off`.

#### Reading the output

Each measured shape prints something like:

```
  42 | 1483.9(mean) 1484.9(median) 1502.0(max) @ 9472x1024x18432 | best: ... TFLOPS |  875W 1417MHz
```

The trailing `| 875W 1417MHz` is live NVML (or amdsmi/hlml) sampling during the
timed loop. Contenders that draw ≪ the run's peak power get a `SUSPECT` tag and
are dropped from the **MSMF** (sustainable) headline on validated backends — a
low-power boost burst is not sustainable — while those same boost readings feed
the **MAMF** (achievable) headline (see
[MAMF & MSMF](../README.md#maximum-achievable-matmul-flops-comparison-table)).

#### What makes a fast shape

Peak GEMM shapes are not mysterious; they follow the paper's recipe
([arXiv:2401.14489](https://arxiv.org/abs/2401.14489)):

1. **Tensor-core alignment** — M, N, K multiples of `128 B / sizeof(dtype)`
   elements (bf16→64, fp8→128).
2. **Tile quantization** — output tiles cleanly into the kernel's tile
   (canonical efficient tile is 128×256), so M,N multiples of 256 are safe.
3. **Wave quantization** — thread-block count ≈ an exact multiple of the SM
   count, so there is no partial tail wave. Winners typically have
   `wave_eff ≥ ~0.99`.
4. **K only sets arithmetic intensity** — it does not enter the tile/wave math.
   Larger K → more compute-bound. Empirically there are *two* high-TFLOPS basins:
   high-AI (large K) and many-waves (min-K, huge M×N) — which is why `auto`
   covers both.

For model-design work (picking shapes your layers will emit), also see
[Vector and matrix size divisibility](../../../training/performance/README.md#vector-and-matrix-size-divisibility).

#### 2. Grid search — constrained shape ranges

Use `--search grid` (or just pass any `--m`/`--n`/`--k`/`_*_range` argument — that
implies grid) when you care about a **specific subspace**: e.g. the shapes your
new model will actually emit, a single training shape, or an accelerator-specific
band you want to map exhaustively.

```bash
# shapes your model will use (example: M in 2k..8k, N=K=4096)
./mamf-finder.py --m_range 2048 8193 256 --n 4096 --k 4096 --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt

# one exact shape
./mamf-finder.py --m 1024 --n 1024 --k 1024 --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt

# fp8 sweep over a coarse lattice
./mamf-finder.py --m_range 0 20480 1024 --n_range 0 20480 1024 --k_range 0 20480 1024 \
    --dtype float8_e4m3fn --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt
```

You can Ctrl-C a long grid run and still get the best result so far. Finer steps
(512 / 256 instead of 1024) cost 8× / 64× wall time. For which shapes tend to
peak on a given accelerator, see
[Vector and matrix size divisibility](../../../training/performance/README.md#vector-and-matrix-size-divisibility).

Architecture-specific setup (MI300X `numa_balancing` / TunableOp, Intel dGPU
install) is under [Architecture specific notes](#architecture-specific-notes)
above.


### Results

The measurements that I have gathered so far can be found at [Maximum Achievable Matmul FLOPS comparison table](../README.md#maximum-achievable-matmul-flops-comparison-table). When I had access to a particular accelerator I run the benchmarks myself, when I didn't it was the kind contributors who invested their time to get these numbers. So I'm very grateful to [those](../../../contributors.md).




## How to benchmark accelerators

### CUDA benchmakrs

There are a few excellent detailed write ups on how to perform CUDA benchmarks:

1. [How to Accurately Time CUDA Kernels in PyTorch](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch)
2. [How to Benchmark Code on CUDA Devices?](https://salykova.github.io/sgemm-gpu#2-how-to-benchmark-code-on-cuda-devices) - this one is different from (1) in that it suggests to set both GPU and Memory clocks, whereas (1) only locks the GPU clock.

You can see these instructions applied in [mamf-finder.py](./mamf-finder.py) (other than clock locking)

### Input data affects measured performance

Kernel time is not a function of tensor shapes alone. The *values* flowing through a GEMM change how hard the chip works, and that feeds back into the clock.

Dynamic power is roughly proportional to clock frequency times the number of transistors that flip. High-entropy or "unpredictable" bit patterns flip more transistors per cycle than sparse, sorted, or all-zero patterns, so the same-shape matmul draws more power. Once draw approaches the configured power (or thermal / current) limit, NVIDIA's GPU Boost and AMD's equivalent step the SM clock down to stay inside the envelope - the mechanism already described under [Power consumption](../README.md#power-consumption), [Cooling](../README.md#cooling) and the chip's [V-F-T curve](../README.md#silicon-lottery). So less "predictable" inputs can make an identical kernel run slower, without any change in the code or the shapes. Horace He's [Strangely, Matrix Multiplications on GPUs Run Faster When Given "Predictable" Data](https://www.thonking.ai/p/strangely-matrix-multiplications) demonstrates the effect by comparing all-zeros against random inputs; [Input-Dependent Power Usage in GPUs](https://ar5iv.labs.arxiv.org/html/2409.18324) measures GEMM power swinging by up to ~38-40% across input patterns (entropy, sparsity, bit similarity, Hamming weight). A secondary, smaller path is denormals / subnormals: some ops take a slower route for them, and the reduced normal range of bf16/fp16 makes them more likely - see NVIDIA's [Flush Denormals with Confidence](https://developer.nvidia.com/blog/cuda-pro-tip-flush-denormals-confidence/).

Practical consequences for a benchmark:

1. **Seed the RNG that builds the inputs** (`torch.manual_seed(...)` before the tensors are created). Without a fixed seed, run-to-run input variation adds a small, uncontrolled jitter on top of measurement noise, even when the *distribution* is unchanged.
2. **Use a realistic distribution**, not all zeros or other pathological patterns. All-zeros under-reports power and over-reports throughput - exactly the benchmarking trap Horace's post is about. Uniform-random token IDs (or whatever your real workload feeds) keep the measured power / clock regime close to production.
3. **Expect a small effect when the distribution is held fixed**, not zero. Seeding removes one noise source; it does not make two differently-valued tensors of the same shape run at identical clocks.

This is distinct from [numerical reproducibility](../../../training/reproducibility/README.md), which forces the *same results* via deterministic algorithms. Here the goal is the *same timing conditions* - same data, same power draw, same clock.
