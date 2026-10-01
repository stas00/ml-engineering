# Accelerator Benchmarks

## How to benchmark accelerators

### CUDA benchmarks

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


## Maximum Achievable Matmul FLOPS finder

Maximum Achievable Matmul FLOPS (MAMF) Benchmark: [mamf-finder.py](./mamf-finder.py) was derived from research found in [The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489) paper.

For a detailed discussion and the numbers for various accelerators see [Maximum Achievable FLOPS](../README.md#maximum-achievable-flops).

While some accelerator manufacturers publish the theoretical TFLOPS these usually can't be reached. As a result of this when we try to optimize our software we have no realistic performance bar to compare ourselves to. The Model FLOPS Utilization (MFU) metric measures TFLOPS achieved against theoretical TFLOPS. Usually when one scores around 50% MFU it's considered a win. But this gives us no indication how far are we from the real achievable throughput.

This benchmark scans various large shapes of matmul and reports both the highest *achievable* (MAMF — boost burst) and the highest *sustainable* TFLOPS (MSMF — saturated near TDP). As ML workloads are dominated by large matmul operations, MSMF is a realistic bar for sustained training throughput, while MAMF is the short-burst ceiling. Now instead of the previously used MFU, which is not achievable, one can use Model Achievable Matmul FLOPS Utilization (MAMFU) or Model Sustainable Matmul FLOPS Utilization (MSMFU) and actually be able to optimize the workloads to be closer to 100%, in particular with MSFM, since achievable is still not sustainable. Which is very important in order to know when to stop optimizing.

Supported accelerators:
- NVIDIA: Volta, Ampere, Hopper, Blackwell, Rubin
- AMD: MI250, and higher
- Intel Gaudi2/3
- Intel GPUs (A770, A750, B580, etc.)
- Apple silicon (MPS)

`--search auto` derives its shapes from the GPU's compute-unit layout, which `mamf-finder.py` models only for NVIDIA and AMD; on the others give it a range with `--search grid`. Power/clock telemetry, which validates both headlines, exists for NVIDIA, AMD and Gaudi (see the telemetry note below). [`mamf-finder-all-gpus.py`](mamf-finder-all-gpus.py) runs on NVIDIA and AMD only.

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
This will make the first iteration very slow, while it's searching for the best GEMM algorithm in the BLAS libraries for each `matmul` shape it encounters, but subsequent operations are likely to be significantly faster than the baseline. See [Accelerating models on ROCm using PyTorch TunableOp](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html) [doc](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md). On ROCm 10, tuning a new shape takes ~2 minutes, which would turn a ~2,000-shape search into days, so `mamf-finder.py` pauses tuning while it searches (the default kernel is within ~1–2% on good shapes, close enough to rank them) and tunes only the shapes it confirms, before timing them. `--tunableop_confirm_max` (default 8) caps how many shapes that is; a BF16 `--search auto` run on MI300X then needs about 5 minutes of search plus ~16 minutes of tuning.

On ROCm, `--search auto` prints a note that its wave/tile geometry is only validated on NVIDIA, although on one MI300X it came within 0.1% of a 16,000-shape BF16 grid. A newer stack can move the winning shape rather than the peak: `10240x15360x8192` dropped from 663.6 (ROCm 6.3, torch 2.5.1) to 625.5 (ROCm 10, torch 2.12) while `12288x9728x8192` held ~660 on both, so re-search after a stack upgrade instead of re-timing the old winner. Pick the GPU with `--cuda_device N`, not `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`: on some virtualized MI300X hosts, hiding GPU0 left torch with no usable device.

**Intel dGPUs (A770, A750, B580, etc.)**
- Follow Intel Extension for PyTorch [installation steps](https://pytorch-extension.intel.com/installation?platform=gpu)

**AMD / Gaudi telemetry (power / SUSPECT):** NVIDIA NVML is the only *validated* backend for excluding unsaturated boost bursts from the headline. `amdsmi` and `pyhlml` are wired and will *report* power/clock when installed, but do not exclude `SUSPECT` shapes until validated on real iron (`VALIDATED_BACKENDS` in `mamf-finder.py`). On NVIDIA, AMD and Gaudi `mamf-finder.py` refuses to start if the telemetry package is missing (`nvidia-ml-py`, `amdsmi`, `habana-pyhlml`) or can't read power/clock; `--telemetry off` runs anyway, with MAMF not boost-validated and no SUSPECT filtering. On MI300X `amdsmi` reads sane Watts/MHz per GPU, but the card hits its 750 W cap within one BF16 kernel and bursts after an idle start *slower*, not faster. So there is no boost burst to filter out, `amdsmi` stays report-only, and AMD MAMF is less reliable than MSMF.

### Examples of usage

`K` is the reduction dimension: `(MxK)*(KxN)=(MxN)`. Default dtype is `bfloat16` (`--dtype` accepts any `torch` dtype, e.g. `float8_e4m3fn`, `float16`, `float32`). Default iterations are 50 warmup + 100 measured per shape (`--num_warmup_iterations`, `--num_iterations`).

#### 1. Auto search (default) — best the GPU can do anywhere

```bash
./mamf-finder.py --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt
# equivalent: ./mamf-finder.py --search auto --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt
```

Finds near-peak shapes via hardware heuristics and reports **two** headlines. Great for a spec-sheet number; not tied to any particular model. This is what produced the [MAMF & MSMF table](../README.md#maximum-achievable-matmul-flops-comparison-table). On H200, B200 and B300, three fast-search repeats stayed within 0.43% of a 128,000-shape exhaustive scout and beat it on half the GPU/dtype pairs.

- **MAMF** (Maximum *Achievable* Matmul FLOPS) — the boost-clock burst ceiling.
- **MSMF** (Maximum *Sustainable* Matmul FLOPS) — the power-saturated, sustained rate that matches real training throughput. **For picking shapes a real model will use, MSMF is the number that matters.**

On NVIDIA it typically runs in 1–3 minutes including adaptive warmup, the recall screen, thermal soak, and both confirms. On AMD with TunableOp, tuning the confirm shapes adds ~2 minutes per shape, for example it took 20 minutes to run on MI300X (see [Architecture specific notes](#architecture-specific-notes)).

How `--search auto` chooses candidate shapes (then both modes share the same confirm phase below):

1. **Wave candidates @ a short K set** — enumerate `(M,N)` shapes that fill an integer number of full waves of thread-block tiles across the SMs (most-square / widest / tallest per wave count 1..16), and measure *every* such `(M,N)` at several Ks (min / mid / max). Covers the high-arithmetic-intensity / wave-perfect basin; measuring all wave candidates (not only a ranked top-N) is what keeps mid/high-K winners in the confirm set.
2. **Coarse M×N planes @ `Kmin` and K=3072** — cheap grids at the smallest K and the low-K boost basin. The latter is required for B200's non-wave BF16 MAMF family found by exhaustive search.
3. **Tight local grid** around both power-ranked and raw-peak scout seeds (`±4` steps of 256 on M/N, `±2` steps of 1024 on K) — endgame polish for off-axis peaks without letting MSMF-oriented ranking hide a MAMF basin.
4. **MAMF recall screen** — cheaply measure raw scout leaders plus the top several K choices from every wave `(M,N)` layout using the actual idle+burst regime. Promote its leaders into a dedicated MAMF shortlist.

Shared confirm phase (identical for `--search auto` and `--search grid`):

5. **Candidate union + thermal soak** — combine the independently ranked MAMF and MSMF shortlists, then drive the chip to steady state before any sustainable measurement steady state (hot, SM clock settled at the saturated floor), stopping early once the clock stops dropping (`--msmf_soak_s`). This is the single biggest reproducibility lever: a cold/cooler card boosts and reads high, so without soaking the MSMF headline would depend on how warm the card happened to be.
6. **MSMF confirm** (sustainable): re-measure every shape in the union with the full iteration count and per-shape warmup, repeated (`--confirm_reps`, default 5 → trimmed median), with a short thermal pre-warmup per shape so a freshly switched shape doesn't read cold on its first rep. The confirm set always includes (a) the best scouted K for each most-square low-wave shape (w=1..4) in **both** `(M,N)` orientations — e.g. H200's `1536×2816` family — and (b) the **fattest** scouts (largest `min(M,N,K)`, then volume), which reliably saturate to TDP and anchor the saturated-clock reference. A shape is kept out of the MSMF headline (still reported) if it is **not saturated** — either drawing well below peak power (`SUSPECT`) or running materially **above the saturated-clock floor** (`--msmf_clock_ratio`) *while also* drawing below `--msmf_sat_power_ratio` of max power. A sparse layout pinned at TDP can hold a higher clock than a fat one; that number is still sustainable and is kept. Jittery shapes (`--msmf_max_spread`) are kept out unless nothing else is stable. Finally the winner is **lock-in validated** (`--msmf_lock_reps`): re-measured with extra reps and published only if that longer run is itself within the spread tolerance, else the next candidate is tried. All exclusions apply only on validated telemetry backends (currently NVIDIA NVML); untested backends report power/clock but do not exclude.
7. **MAMF confirm** (achievable): re-measure the **same union** as MSMF — not MSMF's candidate list and not a scout-clock pre-filter. A fat shape that saturates while scouting still recovers boost after a short idle (measured on B300). Idle, then a SHORT burst that is **queued and synchronized once**. Synchronizing before each start event (v5–v7) left the GPU idle at the moment the event was taken, which charged kernel-launch latency and the idle DVFS re-ramp to the timed kernel — 1.6–19% depending on shape, and worst for large footprints, so it re-ranked candidates. Per-iteration clocks are instead recovered by projecting each iteration's GPU-time window onto a background sampler's host timeline and taking the minimum clock inside it. The peak is reported only if that clock reached boost (`--boost_clock_ratio`). After both headlines are picked, `mamf-finder.py` prints a **same-shape cross-check**: each winning shape measured in both regimes, so the boost→saturated penalty is comparable on the same GEMM.

Each shape is scouted only once, so a candidate that several phases propose costs nothing extra.

**Reproducibility — the whole point of a single-run number.** A published figure must be reproducible on the same GPU with the same setup, so:

- **Measure while the other GPUs compute.** A real workload runs every GPU of the node at once, and they share the board's power/cooling budget — measured with idle siblings, MSMF gets headroom a full node never has. [`mamf-finder-all-gpus.py`](mamf-finder-all-gpus.py) runs a continuous matmul on every GPU it is not measuring, which pins it at its power limit, while it measures: first the full search on GPU0, which gives MAMF, MSMF and the MSMF shape, then that shape pinned on each other GPU in turn while all the rest, GPU0 included, run the matmul. The node MSMF is the slowest GPU, since synchronous training runs at its pace; the summary also gives the median and spread across GPUs. GPU0's search gets all your arguments and the pinned runs get them minus the shape selection; the script shows GPU0's console, writes each GPU's log to its own file, and runs everything with the python you start it with, so use the one whose torch you want measured. On 8 H200s the default `--search auto` took about 4.5 minutes and gave a node MSMF of 727 TFLOPS against a median of 752. A single GPU's MSMF moved by up to 5% between runs there, so rerun before calling one GPU slow.
- **Single GPU: run `mamf-finder.py` on its own.** With idle siblings MAMF is a best-case boost burst and MSMF is a single-GPU upper bound. `mamf-finder.py` samples its same-board siblings throughout the MSMF measurement and, if any sat idle, says so under the results with each idle GPU's share of the time.
- **The headline value is the *mean* of the winning shape** (not a lucky max), measured at a known clock, both printed in the headline (`… 985W 1360MHz`).
- **To reproduce a published number, re-run its exact shape in grid mode** — e.g. `--m 9472 --n 6144 --k 12288`. Auto's job is to *find* a near-peak shape anywhere; grid *reproduces* (or searches within) a known range and still runs the same MAMF/MSMF confirm. Re-running `auto` re-searches and may land on a different (equivalent) shape.

On a single GPU with idle siblings the MSMF headline is reproducible to ~1–2% run-to-run and MAMF to ~1–2% (boost-locked); most of the residual is genuine saturated-clock jitter. With every other GPU computing at the same time, MSMF varies more between GPUs, as measured below.

**What we measured: alone vs all-8 concurrent.**

Same protocol on both chips — five `--search auto` runs on GPU0 with siblings idle, then five rounds with the same script on all 8 GPUs at once:

*B200 bf16 (torch 2.13.0+cu130):*

| Setup                        | MSMF mean      | MSMF range | MSMF clock     | MAMF mean |
| :--------------------------- | -------------: | ---------: | -------------: | --------: |
| GPU0 alone (n=5)             |         1455.7 |       1.1% | ~1330–1460 MHz |    1732.6 |
| GPU0 while all 8 busy (n=5)  | 1438.7 (−1.2%) |       1.8% | ~1316–1332 MHz |    1757.6 |
| All 8 GPUs × 5 rounds (n=40) | 1426.4 (−2.0%) |  **10.4%** |  1228–1410 MHz |    1751.2 |

*H200 bf16 (torch 2.14.0.dev20260810+cu130):*

| Setup                        | MSMF mean     | MSMF range | MSMF clock     | MAMF mean |
| :--------------------------- | ------------: | ---------: | -------------: | --------: |
| GPU0 alone (n=5)             |         702.9 |       1.4% | ~1440–1465 MHz |     828.0 |
| GPU0 while all 8 busy (n=5)  | 701.3 (−0.2%) |       2.9% | ~1440–1525 MHz |     825.8 |
| All 8 GPUs × 5 rounds (n=40) | 698.6 (−0.6%) |       4.5% |  1405–1527 MHz |     823.9 |

Lesson: the other GPUs computing at the same time barely moves **MAMF** (boost bursts still hit the boost clock — they draw only ~150–300 W, so board power/cooling still has headroom). It *does* move **MSMF**, and the severity is board-dependent: on B200 (1000 W TDP) concurrent siblings pulled GPU0's saturated clock down ~50 MHz (−1.2% MSMF) and opened a **~10% MSMF spread across the 8 GPUs**; on H200 (700 W TDP) the same protocol was much milder (−0.2% on GPU0, ~4.5% across the board). So the alone-GPU MSMF overstates what a full node sustains: a full-node training run sees the concurrent distribution, and since it waits on its slowest GPU, the bottom of that range is what it actually gets. How far that sits below the alone number depends on the board's shared power/cooling budget. In the [results table](../README.md#maximum-achievable-matmul-flops-comparison-table), `Sib` = `yes` is the `mamf-finder-all-gpus.py` measurement, every other GPU running a continuous matmul (MSMF is the slowest GPU); `Sib` = `no` is still one GPU with siblings idle, a single-GPU upper bound.

Useful knobs: `--dtype`, `--max_size`, `--confirm_top` (default 10), `--confirm_reps`, `--confirm_fat_forced`, `--msmf_soak_s`, `--msmf_max_spread`, `--msmf_clock_ratio`, `--msmf_sat_power_ratio`, `--msmf_lock_reps`, `--mamf_burst_iters`, `--mamf_idle_s`, `--boost_clock_ratio`, `--no-refine_grid`, `--telemetry off`.

#### Reading the output

The console shows each phase as a title, a column header and one row that is updated in place as shapes are measured; when the phase ends, its winning row stays on screen:

```
Grid search: sweeping 4x1x1 shapes (scout 20 iters / 8 warmup), then confirm ...
     #  MxNxK                mean median    max   best     W   MHz
     4  4096x4096x4096      779.5  779.3  784.2  779.5   590  1980  <- best mean

MAMF recall screen: 4 shapes ...
     #  MxNxK                peak   MHz
     1  4096x4096x4096      777.7  1980  <- best peak

MSMF (sustainable) confirm: 4 shapes ...
  MxNxK              TFLOPS     W   MHz spread  runs
  4096x4096x4096      743.0   679  1792   1.6%  749.6 743.2 740.4 738.7 740.3 737.6 743.0 743.3 746.2  <- MSMF

MAMF (achievable) confirm: 4 shapes ...
  MxNxK              TFLOPS     W   MHz  top peaks
  4096x4096x4096      780.5   332  1980  780.5 780.1 779.3 779.1 778.2  <- MAMF

--------------------------------------------------------------------------------

** Results:

Tried 4 shapes => the best outcomes were:
MAMF (max achievable,  boost burst): 781 TFLOPS @ 4096x4096x4096 (MxNxK)  332W 1980MHz
MSMF (max sustainable, saturated):   743 TFLOPS @ 4096x4096x4096 (MxNxK)  679W 1792MHz
```

The MSMF row shows the lock-in re-measurement (9 runs), which is what the headline uses.

`W` and `MHz` are live NVML (or amdsmi/hlml) samples taken during the timed loop. On a short MAMF burst NVML's power sample can lag the kernel: on A100 PCIe a ~80 ms burst after an idle reads near-idle watts, so that `W` is not the burst's draw. The MAMF and MSMF headlines are integer TFLOPS; a half rounds up. Every row, plus the details behind each decision (confirm candidates, boost reference, shapes excluded from MSMF and why, lock-in, same-shape cross-check), is written to the `--output_file` log. Shapes that ran well below the run's peak power are excluded from the **MSMF** (sustainable) headline on validated backends — a low-power boost burst is not sustainable — while boost readings feed the **MAMF** (achievable) headline (see [MAMF & MSMF](../README.md#maximum-achievable-matmul-flops-comparison-table)).

#### What makes a fast shape

Peak GEMM shapes are not mysterious; they follow the paper's recipe ([arXiv:2401.14489](https://arxiv.org/abs/2401.14489)):

1. **Tensor-core alignment** — M, N, K multiples of `128 B / sizeof(dtype)` elements (bf16→64, fp8→128).
2. **Tile quantization** — output tiles cleanly into the kernel's tile (canonical efficient tile is 128×256), so M,N multiples of 256 are safe.
3. **Wave quantization** — thread-block count ≈ an exact multiple of the SM count, so there is no partial tail wave. Winners typically have `wave_eff ≥ ~0.99`.
4. **K only sets arithmetic intensity** — it does not enter the tile/wave math. Larger K → more compute-bound. Empirically there are *two* high-TFLOPS basins: high-AI (large K) and many-waves (min-K, huge M×N) — which is why `auto` covers both.

For model-design work (picking shapes your layers will emit), also see [Vector and matrix size divisibility](../../../training/performance/README.md#vector-and-matrix-size-divisibility).

#### 2. Grid search — best shape in *your* range (the practical case)

Use `--search grid` (or just pass any `--m`/`--n`/`--k`/`_*_range` argument — that implies grid) when you care about a **specific subspace**: the shapes your new model will actually emit, a single training shape, or an accelerator-specific band you want to map exhaustively.

Grid reports the **same MAMF + MSMF headlines** as auto. The only difference is how candidates are chosen: auto derives them from hardware heuristics; grid sweeps the M/N/K range *you* give, then runs the shared confirm phase. For model work, **MSMF in your range** is the useful answer — not the GPU's absolute peak.

```bash
# shapes your model will use (example: M in 2k..8k, N=K=4096)
./mamf-finder.py --m_range 2048 8193 256 --n 4096 --k 4096 --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt

# one exact shape (reproduce a published headline)
./mamf-finder.py --m 1024 --n 1024 --k 1024 --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt

# fp8 sweep over a coarse lattice
./mamf-finder.py --m_range 0 20480 1024 --n_range 0 20480 1024 --k_range 0 20480 1024 \
    --dtype float8_e4m3fn --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt
```

**Give the range enough room to saturate.** MSMF needs at least one fat/high-K shape that can pin the saturated-clock floor. A narrow/small grid (e.g. only `K=4096` squares) can leave every candidate still boosting or jittery; `mamf-finder.py` then warns and falls back to a "best available" MSMF. Widen M/N/K (especially `min(M,N,K)` and K) so a truly sustainable shape is in the set.

You can Ctrl-C a long grid run and still get the best result so far. Finer steps (512 / 256 instead of 1024) cost 8× / 64× wall time. For which shapes tend to peak on a given accelerator, see [Vector and matrix size divisibility](../../../training/performance/README.md#vector-and-matrix-size-divisibility).

Architecture-specific setup (MI300X `numa_balancing` / TunableOp, Intel dGPU install) is under [Architecture specific notes](#architecture-specific-notes) above.


### Results

The measurements that I have gathered so far can be found at [Maximum Achievable Matmul FLOPS comparison table](../README.md#maximum-achievable-matmul-flops-comparison-table). When I had access to a particular accelerator I run the benchmarks myself, when I didn't it was the kind contributors who invested their time to get these numbers. So I'm very grateful to [those](../../../contributors.md).
