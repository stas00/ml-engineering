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


<a id="maximum-achievable-matmul-flops-finder"></a>

## Maximum Achievable and Sustainable Matmul FLOPS finder

Maximum Achievable Matmul FLOPS (MAMF) Benchmark: [mamf-finder.py](./mamf-finder.py) was derived from research found in [The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489) paper.

For a detailed discussion and the numbers for various accelerators see [Maximum Achievable and Sustainable FLOPS](../README.md#maximum-achievable-and-sustainable-flops).

While some accelerator manufacturers publish the theoretical TFLOPS these usually can't be reached. As a result of this when we try to optimize our software we have no realistic performance bar to compare ourselves to. The Model FLOPS Utilization (MFU) metric measures TFLOPS achieved against theoretical TFLOPS. Usually when one scores around 50% MFU it's considered a win. But this gives us no indication how far are we from the real achievable throughput.

This benchmark scans various large shapes of matmul and reports both the highest *achievable* (MAMF — boost burst) and the highest *sustainable* TFLOPS (MSMF — held steady under full load). As ML workloads are dominated by large matmul operations, MSMF is a realistic bar for sustained training throughput, while MAMF is the short-burst ceiling. Now instead of the previously used MFU, which is not achievable, one can use Model Achievable Matmul FLOPS Utilization (MAMFU) or Model Sustainable Matmul FLOPS Utilization (MSMFU) and actually be able to optimize the workloads to be closer to 100%, in particular with MSFM, since achievable is still not sustainable. Which is very important in order to know when to stop optimizing.

Supported accelerators:
- NVIDIA: Volta, Ampere, Hopper, Blackwell, Rubin
- AMD: MI250, and higher
- Intel Gaudi2/3
- Intel GPUs (A770, A750, B580, etc.)
- Apple silicon (MPS)

`--search auto` (default) derives its shapes from the GPU's compute-unit layout, which `mamf-finder.py` models only for NVIDIA and AMD; on the others give it a range with `--search grid`. Power/clock telemetry, whose SM clock certifies that a MAMF iteration ran at boost, exists for NVIDIA, AMD and Gaudi (see the telemetry note below). [`mamf-finder-all-gpus.py`](mamf-finder-all-gpus.py) runs on NVIDIA and AMD only.

Important notes:
- if you can find a better and more efficient way to detect the best matmul TFLOPS by approaching each new accelerator as a black box, please kindly send a PR with the improvement including the generated log file.
- also if you know that this benchmark should be run under special conditions to show the best results, such as some kernel settings or similar, please submit a PR to add such special instructions. For example, for AMD MI300X I'm being told disabling the numa_balancing is supposed to help.
- since a big part of the overhead comes from HBM IO, if you're using a fused kernel with 2 or more matmuls, whose results don't leave the accelerator's registers, the performance will be definitely faster than what this benchmark reports.
- It also helps to sample your accelerator's actual clock speed. If your accelerator is running at a slower clock than the one used in the spec, there is no chance you can get the theoretical TFLOPS (see [How To Calculate Theoretical TFLOPS](../README.md#how-to-calculate-theoretical-tflops)).

### How MAMF and MSMF are defined

Both numbers are measured on the same candidate shapes with the same per-iteration work: zero a 256MB buffer to flush the L2 cache, rewrite the output matrix, then run the matmul. Only the matmul is timed, with device events, and TFLOPS is `2*M*N*K` divided by that time.

**MAMF** (Maximum *Achievable* Matmul FLOPS) is the rate a short burst reaches at the boost clock. Each shape gets 10 bursts of 20 back-to-back iterations, and each burst starts after 0.25s of idle so the SM clock climbs back to boost. A background thread samples the SM clock every ~0.5ms, and each iteration is paired with the lowest clock sampled while it ran. An iteration counts as boost if that clock is at least 0.97x the highest clock seen anywhere in the run. A shape's MAMF is the median of its 5 fastest boost iterations, so a single lucky iteration can't set it; on H200 and B200 the median of 5 and the single fastest iteration differ by at most 0.7%. The MAMF headline is the highest shape MAMF among the shapes that reached boost.

**MSMF** (Maximum *Sustainable* Matmul FLOPS) is the highest rate a shape holds without falling, over a timed window on a GPU that has run at full load long enough for its clock to settle:

1. **Settle clock.** Up to 20s of 8192x8192x8192 BF16 matmuls, stopping early once the SM clock stops dropping. An idle GPU boosts and reads high, and this brings it down to the clock its power limit allows under load. It doesn't wait for the temperature to settle: on H200 one GPU went on warming from 63 to 72°C over the minutes of measurement that followed, but below the throttle point warming costs under 1% on most GPUs (see [Cooling](../README.md#cooling)).
2. **Rank.** Every candidate runs 0.2s untimed, then 1s timed.
3. **Window.** The 10 fastest get the full window: 1s untimed, then 2s timed, split into 8 consecutive sub-windows. The window's TFLOPS is total FLOPs over total matmul time. A shape is steady if its last 4 sub-windows are at most 1% slower than its first 4. The sub-windows may also spread by at most 10%: at the power cap a 0.25s sub-window jitters by up to ~4% while the window's mean holds.
4. **K walk** (`--search auto` only, explained below). Starting from the 4 fastest steady layouts and the most-square 1-wave layouts, K is stepped down, each step measured over the same window.
5. **Lock-in.** The fastest shape that was steady, or that was faster than every steady one, is re-measured over a 4s window and published if it is steady there; otherwise the next fastest is tried, up to 4 of them. If none holds steady over the 4s window, the fastest shape that was steady over its 2s window is published. If no shape is steady at all, `mamf-finder.py` warns and publishes the fastest unsteady one.

The window is a time, not an iteration count, because a fixed count of 100 iterations lasts 13ms on a fast B200 shape and about 1s on a slow one, and a GPU keeps some of its boost for a fraction of a second after a shape starts. On one H200 a 0.3s window read `2816x1536x20480` at 745–752 TFLOPS, its 4s lock-in window at 728–735, and a 10s run at 736–739. Power doesn't enter the definition: a shape whose throughput doesn't fall is sustainable, whatever lets it hold that rate. The `W` and `MHz` printed next to MSMF are the mean instantaneous power and SM clock over the timed window; on NVIDIA the instantaneous power comes from NVML's `NVML_FI_DEV_POWER_INSTANT` field, since its default power reading is a 1s average.

**The K walk.** At the power cap, the K that a short burst favors is not the K that sustains best. A shorter K lets the same `(M,N)` hold a higher clock at the same power, up to where the clock reaches its ceiling: on H200 BF16 `2816x1536` sustains 727 TFLOPS at K=20480, 741 at K=10240, 749 at K=6144 (1879 MHz, 685 W), and drops to 722 at K=4096, where the clock sits at its 1980 MHz ceiling and power falls below the 700 W cap. The search's candidates sit at long K, so the walk takes the 4 fastest steady `(M,N)` layouts and steps each one's K down by K/8 rounded down to a power of two (2048 from K=20480), for up to 10 steps, stopping once a step is more than 1% below the best in that walk. A walk never starts above the search's largest K (20480): the shapes past it are there for MAMF, and a longer K only sustains less. Several layouts, because two can look alike at long K and part at short K: on H200 FP8, `1280x16896` and `1536x2816` both sustain about 1290–1305 TFLOPS at K=16384–20480, but only `1536x2816` climbs to 1335–1372 at K=10240, and walks from one layout found it in 5 of 8 runs, against 10 of 11 with two. Four, because three can tie: in one more H200 FP8 run, three layouts sat within 0.2% of each other at long K and `1536x2816` was not among the two walked, so MSMF came out 1281 instead of about 1330. Even four can miss: in one of 32 H200 FP8 runs, six layouts sat at 1276–1291 TFLOPS at long K, `1536x2816` read the lowest of them and was not walked, and MSMF came out 1289 instead of about 1337. So the most-square 1-wave layouts (`msmf_kwalk_waves`=1; one wave is one thread-block tile per SM, see [What makes a fast shape](#what-makes-a-fast-shape)), `1536x2816` among them on H200, are walked in both orientations no matter how they rank; in 32 more H200 FP8 runs none missed, and the extra walks cost about 10s on H200 and nothing measurable on B200. On an A100 PCIe, walking 4 layouts instead of 2 plus the 1-wave ones took a run from 5:13–5:32 to 6:06, with the same MAMF and MSMF shapes. In the 11th of the two-layout runs, a 0.3s ranking window read `1536x2816` 2.5% below its usual rate, which left it out of the 6 shapes that then got a full window, so neither walk started from it and MSMF came out 4% low. That is why the ranking window is 1s and the 10 fastest get a full window: with 0.3s and the top 6, 2 of 32 more H200 FP8 runs missed the same way, and with 1s and the top 10 none of 32 did, at the cost of about a minute per run.

**How close it gets.** On an H200 and a B200, five `--search auto` runs each gave MSMF of 743–746 and 1421–1428 TFLOPS, and MAMF of 843–850 and 1682–1705. Each published MSMF shape, run back to back for 10s in two passes, sustained within 0.6% of its MSMF; the best 10s rate among them was 750 on the H200 and 1433 on the B200.

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
This will make the first iteration very slow, while it's searching for the best GEMM algorithm in the BLAS libraries for each `matmul` shape it encounters, but subsequent operations are likely to be significantly faster than the baseline. See [Accelerating models on ROCm using PyTorch TunableOp](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html) [doc](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md). On ROCm 10, tuning a new shape takes ~2 minutes, which would turn a ~2,000-shape search into days, so `mamf-finder.py` pauses tuning while it searches (the default kernel is within ~1–2% on good shapes, close enough to rank them) and tunes only the shapes it confirms, before timing them. `--tune tunableop_confirm_max=N` (default 8) caps how many shapes that is; a BF16 `--search auto` run on MI300X then needs about 5 minutes of search plus ~16 minutes of tuning.

On ROCm, `--search auto` prints a note that it was checked against an exhaustive grid only on one MI300X in BF16, where it came within 0.1% of a 16,000-shape grid. A newer stack can move the winning shape rather than the peak: `10240x15360x8192` dropped from ~664 (ROCm 6.3, torch 2.5.1) to ~626 (ROCm 10, torch 2.12) while `12288x9728x8192` held ~660 on both, so re-search after a stack upgrade instead of re-timing the old winner. Pick the GPU with `--cuda-device N`, not `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`: on some virtualized MI300X hosts, hiding GPU0 left torch with no usable device.

**Intel dGPUs (A770, A750, B580, etc.)**
- Follow Intel Extension for PyTorch [installation steps](https://pytorch-extension.intel.com/installation?platform=gpu)

**Telemetry on NVIDIA, AMD and Gaudi:** the SM clock certifies that a MAMF iteration ran at boost, power ranks the scouted shapes, and both are printed next to each number; MSMF is judged on throughput alone. `mamf-finder.py` refuses to start if the telemetry package is missing (`nvidia-ml-py`, `amdsmi`, `habana-pyhlml` correspondingly) or can't read power/clock; `--telemetry off` runs anyway, with MAMF not boost-validated. The `amdsmi` reads are confirmed on MI300X and as of this writing the `pyhlml` reads are untested on hardware. On MI300X the card hits its 750 W cap within one BF16 kernel and bursts after an idle start *slower*, not faster, so on this particular gpu MAMF is less reliable than MSMF.

### Examples of usage

`K` is the reduction dimension: `(MxK)*(KxN)=(MxN)`. Default dtype is `bfloat16`; `--dtype` also accepts `float16`, `float32`, `float8_e4m3fn` (the fp8 of NVIDIA and AMD MI350X/MI355X), `float8_e4m3fnuz` (the fp8 of AMD MI300X/MI325X), and three block-scaled formats with a `bfloat16` output: `mxfp8` (`float8_e4m3fn` operands with one `float8_e8m0fnu` scale per 32 elements), `mxfp4` (fp4 `e2m1` operands with one `float8_e8m0fnu` scale per 32 elements) and `nvfp4` (fp4 `e2m1` operands with one `float8_e4m3fn` scale per 16 elements). `mxfp8` and `mxfp4` need hardware MX support such as NVIDIA Blackwell or AMD MI350X/MI355X; `nvfp4` needs NVIDIA Blackwell, and both fp4 formats need `K` to be a multiple of 32. Before searching, `mamf-finder.py` runs a tiny matmul in the requested dtype and exits with the vendor library's error if the device or the software stack can't run it - and for an fp8 dtype, it names the other fp8 variant if that one runs. How each shape is timed is in [How MAMF and MSMF are defined](#how-mamf-and-msmf-are-defined).

#### 1. Auto search (default) — best the GPU can do anywhere

```bash
./mamf-finder.py --output-file=$(date +'%Y-%m-%d-%H:%M:%S').txt
# equivalent: ./mamf-finder.py --search auto --output-file=$(date +'%Y-%m-%d-%H:%M:%S').txt
```

Finds near-peak shapes via hardware heuristics and reports **two** headlines. Great for a spec-sheet number; not tied to any particular model. This is what produced the [MAMF & MSMF table](../README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table). On H200 and B200, in BF16 and FP8, every `--search auto` run came within 0.6% of a 128,000-shape exhaustive scout followed by the same confirm, and most runs beat it, by up to 3.6%. A run took about 4 minutes on one H200 or B200, against 17–29 minutes on 8 GPUs for the scout.

- **MAMF** (Maximum *Achievable* Matmul FLOPS) — the boost-clock burst ceiling.
- **MSMF** (Maximum *Sustainable* Matmul FLOPS) — the steady rate under full load, which matches real training throughput. **For picking shapes a real model will use, MSMF is the number that matters.**

Both are defined in [How MAMF and MSMF are defined](#how-mamf-and-msmf-are-defined). How long a run takes depends on the GPU: about 4 minutes on an H200, 4.5 on a B200 and 6 on an A100 PCIe, including adaptive warmup, the recall screen, the settle-clock step, and both confirms. On AMD with TunableOp, tuning the confirm shapes adds ~2 minutes per shape, for example it took 20 minutes to run on MI300X (see [Architecture specific notes](#architecture-specific-notes)).

How `--search auto` chooses candidate shapes (then both modes share the same confirm phase below):

1. **Wave candidates @ a short K set** — enumerate `(M,N)` shapes that fill an integer number of full waves of thread-block tiles across the SMs (most-square / widest / tallest per wave count 1..16), and measure *every* such `(M,N)` at K = 1024, 2048, 3072, 4096, 8192, 12288, 14336, 16384 and 20480. Covers the wave-perfect basin at both ends of K; measuring all wave candidates (not only a ranked top-N) at all those Ks is what keeps both H200's high-K winners (`1536x2816x20480`) and B200's low-K ones (`3072x18944x2048`) in the confirm set. Without K=2048–4096, B200 runs reached the best burst family only when a step-3 seed happened to land near it, and their MAMF varied by 1–2% on the same GPU.
2. **Coarse M×N planes @ `Kmin` and K=3072** — cheap grids at the smallest K and the low-K boost basin. The latter is required for B200's non-wave BF16 MAMF family found by exhaustive search.
3. **Tight local grid** around both power-ranked and raw-peak scout seeds (`±4` steps of 256 on M/N, `±2` steps of 1024 on K) — endgame polish for off-axis peaks without letting MSMF-oriented ranking hide a MAMF basin.
4. **MAMF recall screen** — cheaply measure raw scout leaders plus the top several K choices from every wave `(M,N)` layout using the actual idle+burst regime (2 iterations after 50ms idle), and promote its leaders into a dedicated MAMF shortlist. The GPU is still hot from scouting, so a burst that never reaches the boost clock is retried after the confirm's 0.25s idle: on H200 the first screened shapes otherwise burst at 1635 MHz against a 1980 MHz boost. Then the screen looks past the search's largest K (`max_size`, 20480): for up to 8 layouts (`k_edge_layouts`) whose burst at that K beats every shorter K they were screened at, steepest gain first, K at 1.5x and 2x is screened too (up to `k_edge_max`=65536), stopping once the burst stops rising. H200 BF16's `2816x1536` bursts 828 TFLOPS at K=20480 and 842 at K=32768. The scouts can't show this: they run hot, and a long-K call throttles more, so they read a falling curve.

Shared confirm phase (identical for `--search auto` and `--search grid`, except that grid measures only shapes in its range, so it skips the K-edge screen and the MSMF K walk):

5. **Candidate union + settle clock** — combine the independently ranked MAMF and MSMF shortlists, then run the chip at full load until its clock stops dropping, before any sustainable measurement (`msmf_settle_clock_s`). Without it an idle card boosts and reads high, so MSMF would depend on what the card was doing before the run.
6. **MSMF confirm** (sustainable): the ranking, windows, K walk and lock-in of [How MAMF and MSMF are defined](#how-mamf-and-msmf-are-defined), over every shape in the union. The union always includes (a) the best scouted K for each most-square low-wave shape (w=1..4) in **both** `(M,N)` orientations — e.g. H200's `1536×2816` family — and (b) the **fattest** scouts (largest `min(M,N,K)`, then volume), which reliably reach the power cap.
7. **MAMF confirm** (achievable): re-measure the **same union** as MSMF — not MSMF's candidate list and not a scout-clock pre-filter. A fat shape that saturates while scouting still recovers boost after a short idle (measured on B300). Each burst is **queued and synchronized once**: synchronizing before each start event would leave the GPU idle at the moment the event is taken, charging kernel-launch latency and the idle DVFS re-ramp to the timed kernel — 1.6–19% depending on shape, worst for large footprints, which re-ranks candidates. Per-iteration clocks are recovered by projecting each iteration's GPU-time window onto the background sampler's host timeline and taking the minimum clock inside it. After both headlines are picked, `mamf-finder.py` prints a **same-shape cross-check**: each winning shape measured in both regimes, so the boost→saturated penalty is comparable on the same GEMM.

Each shape is scouted only once, so a candidate that several phases propose costs nothing extra.

**Reproducibility — the whole point of a single-run number.** A published figure must be reproducible on the same GPU with the same setup, so:

- **Measure while the other GPUs compute.** A real workload runs every GPU of the node at once, and they share the board's power/cooling budget — measured with idle siblings, MSMF gets headroom a full node never has. [`mamf-finder-all-gpus.py`](mamf-finder-all-gpus.py) runs a continuous matmul on every GPU it is not measuring, which pins it at its power limit, while it measures: first the full search on GPU0, which gives MAMF, MSMF and their shapes, then those shapes pinned on each other GPU in turn while all the rest, GPU0 included, run the matmul. The node MAMF and MSMF are the median GPU: the GPUs of one node differ by a few percent from chip to chip, and the median describes the GPU model rather than the luck of one chip, while one broken card barely moves it. The summary also gives the slowest GPU, which synchronous training runs at, and the spread. GPU0's search gets all your arguments and the pinned runs get them minus the shape selection; the script shows GPU0's console, writes each GPU's log to its own file, and runs everything with the python you start it with, so use the one whose torch you want measured. On 8 H200s the default `--search auto` took about 9 minutes and gave a node MSMF of 755 TFLOPS, the slowest GPU 745; on 8 B200s it took about 9 minutes too and gave 1439, the slowest GPU 1421. A single GPU's MSMF moved by up to 1.3% between runs there, while the GPUs of one node differed by up to 4%, so rerun before calling one GPU slow.
- **Single GPU: run `mamf-finder.py` on its own.** With idle siblings MAMF is a best-case boost burst and MSMF is a single-GPU upper bound. `mamf-finder.py` samples its same-board siblings throughout the MSMF measurement and, if any sat idle, says so under the results with each idle GPU's share of the time.
- **Neither headline is a lucky maximum.** MSMF is the mean over a 4s window that held steady, and MAMF is the median of the 5 fastest boost iterations; the power and clock they ran at are printed in the headline (`… 985W 1360MHz`).
- **To reproduce a published number, re-run its exact shape in grid mode** — e.g. `--m 9472 --n 6144 --k 12288`. Auto's job is to *find* a near-peak shape anywhere; grid *reproduces* (or searches within) a known range and still runs the same MAMF/MSMF confirm. Re-running `auto` re-searches and may land on a different (equivalent) shape.

How much a result moves between runs, shown on two GPUs as examples rather than measured for every GPU: with the default settings, over 4 runs on every GPU of an 8x H200 node (FP8) and an 8x B200 node (BF16), all 8 measured at once, a GPU's MSMF moved by 0.5–0.6% at the median and 2.2% at most, its MAMF by 0.3–0.9% and 1.5%, and none of the 64 runs missed the best shape. The 2.2% was one H200 run whose 4s lock-in read 1.6% below the same shape's window in the K walk, as the GPU warmed from 63 to 72°C. The GPUs of one node differ by more than a GPU does between runs, as measured below.

**What was measured: alone vs all-8 concurrent.**

Same protocol on both chips — five `--search auto` runs on GPU0 with siblings idle, then five rounds with the same script on all 8 GPUs at once:

*B200 bf16 (torch 2.14.0+cu130):*

| Setup                        | MSMF mean      | MSMF range | MSMF clock    | MAMF mean |
| :--------------------------- | -------------: | ---------: | ------------: | --------: |
| GPU0 alone (n=5)             |         1423.8 |       0.5% | 1396–1411 MHz |    1697.6 |
| GPU0 while all 8 busy (n=5)  | 1417.2 (−0.5%) |       0.5% | 1387–1396 MHz |    1688.8 |
| All 8 GPUs × 5 rounds (n=40) | 1437.5 (+1.0%) |   **4.9%** | 1368–1527 MHz |    1746.7 |

*H200 bf16 (torch 2.14.0+cu130):*

| Setup                        | MSMF mean     | MSMF range | MSMF clock    | MAMF mean |
| :--------------------------- | ------------: | ---------: | ------------: | --------: |
| GPU0 alone (n=5)             |         744.8 |       0.4% | 1835–1873 MHz |     848.4 |
| GPU0 while all 8 busy (n=5)  | 743.4 (−0.2%) |       1.1% | 1850–1872 MHz |     847.2 |
| All 8 GPUs × 5 rounds (n=40) | 755.5 (+1.4%) |       3.4% | 1850–1938 MHz |     852.1 |

Lesson: on these two nodes, the other GPUs computing at the same time barely moved a GPU's own numbers. GPU0's MSMF fell 0.5% on B200 and 0.2% on H200, and its MAMF stayed within run-to-run noise, since boost bursts draw only ~150–330 W. What sets the GPUs apart is the chips themselves: across the 40 runs MSMF spans **4.9%** on the B200 node and 3.4% on the H200 node, and each GPU stays in its own part of that range from round to round (on B200, GPU0 averaged 1417 and GPU5 1472). GPU0 happened to be the slowest chip on both nodes, which is why the 8-GPU mean sits above it. So a single GPU's MSMF describes that chip, not the GPU model, and the published node figure is the median GPU. A full-node synchronous training run still waits on its slowest GPU, so the bottom of the 8-GPU range is what it gets. How much the concurrency itself costs depends on the board's shared power and cooling budget, so measure the whole node. In the [results table](../README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table), `Sib` = `yes` is the `mamf-finder-all-gpus.py` measurement, every other GPU running a continuous matmul, with MAMF and MSMF the median GPU; `Sib` = `no` is one GPU with its siblings idle.

`mamf-finder.py -h` lists the options. The algorithm's own knobs, the names in backticks in the steps above (`confirm_reps`, `msmf_settle_clock_s`, ...), live in the `Tuning` class near the end of the script with their defaults and a line on what each does. To try another value pass e.g. `--tune msmf_settle_clock_s=30 --tune confirm_reps=7`.

#### Reading the output

The console shows each phase as a title, a column header and one row that is updated in place as shapes are measured; when the phase ends, its winning row stays on screen. On an H200, `mamf-finder.py --dtype bfloat16 --m-range 1024 4097 1024 --n 1536 2816 --k 4096 8192 20480` prints:

```
Grid search: sweeping 24 shapes (scout 20 iters / 8 warmup), then confirm ...
     #  MxNxK                mean median    max   best     W   MHz
    18  3072x2816x20480     828.4  828.6  831.6  828.4   506  1980  <- best mean

MAMF recall screen: 8 shapes ...
     #  MxNxK                peak   MHz
     1  3072x2816x20480     837.7  1965  <- best peak

MSMF (sustainable) confirm: 10 shapes ...
  MxNxK              TFLOPS     W   MHz
  3072x2816x8192      722.5   681  1612 drop   0.7% spread  4.2%  locked in  <- MSMF

MAMF (achievable) confirm: 10 shapes ...
  MxNxK              TFLOPS     W   MHz  top peaks
  3072x2816x20480     842.3   172  1980  842.8 842.3 842.3 841.9 841.2  <- MAMF

--------------------------------------------------------------------------------

** Results:

Tried 24 shapes => the best outcomes were:
MAMF (max achievable,  boost burst): 842 TFLOPS @ 3072x2816x20480 (MxNxK)  172W 1980MHz
MSMF (max sustainable, saturated):   722 TFLOPS @ 3072x2816x8192 (MxNxK)  681W 1612MHz
note: 7 same-board sibling GPU(s) sat idle during the MSMF measurement, so MSMF reads high.
      Share of the time each was idle: GPU1 100%, GPU2 100%, GPU3 98%, GPU4 94%, GPU5 94%, GPU6 94%, GPU7 94%
      With all siblings idle it is a single-GPU upper bound; for full-node MSMF use mamf-finder-all-gpus.py.
```

The MSMF row is the 4s lock-in window, which is what the headline uses: `drop` is how much slower its second half ran than its first, and `spread` how far its 8 sub-windows differ. The MAMF row's `top peaks` are the shape's 5 fastest boost iterations, whose median is the headline. The two headlines landed on different K here: K=20480 bursts fastest, while K=8192 holds a higher clock at the power cap. `--search auto` on the same GPU found MSMF 745 at `1536x2816x6144`, a K this grid doesn't include.

`W`, `MHz` and, in the MSMF rows, `C` (the GPU temperature) are live NVML (or amdsmi/hlml) samples taken while the shape ran. On a short MAMF burst NVML's power sample can lag the kernel: on A100 PCIe a ~80 ms burst after an idle reads near-idle watts, so that `W` is not the burst's draw. The MAMF and MSMF headlines are integer TFLOPS; a half rounds up. Every row, plus the details behind each decision (the MSMF ranking, K walk and lock-in rows, the boost reference, the same-shape cross-check), is written to the `--output-file` log (see [MAMF & MSMF](../README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table) for how the two numbers are used).

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
./mamf-finder.py --m-range 2048 8193 256 --n 4096 --k 4096 --output-file=$(date +'%Y-%m-%d-%H:%M:%S').txt

# one exact shape (reproduce a published headline)
./mamf-finder.py --m 1024 --n 1024 --k 1024 --output-file=$(date +'%Y-%m-%d-%H:%M:%S').txt

# fp8 sweep over a coarse lattice
./mamf-finder.py --m-range 0 20480 1024 --n-range 0 20480 1024 --k-range 0 20480 1024 \
    --dtype float8_e4m3fn --output-file=$(date +'%Y-%m-%d-%H:%M:%S').txt
```

**Give the range enough room to saturate.** MSMF needs at least one shape that holds a steady rate over its window. A narrow/small grid can leave every candidate unsteady; `mamf-finder.py` then warns and publishes the fastest unsteady one. Widen M/N/K (especially `min(M,N,K)`) so a shape that reaches the power cap is in the set. Grid doesn't run the K walk, so the best sustainable K is found only if your K range includes it.

You can Ctrl-C a long grid run and still get the best result so far. Finer steps (512 / 256 instead of 1024) cost 8× / 64× wall time. For which shapes tend to peak on a given accelerator, see [Vector and matrix size divisibility](../../../training/performance/README.md#vector-and-matrix-size-divisibility).

Architecture-specific setup (MI300X `numa_balancing` / TunableOp, Intel dGPU install) is under [Architecture specific notes](#architecture-specific-notes) above.


### Results

The measurements that I have gathered so far can be found at [Maximum Achievable and Sustainable Matmul FLOPS comparison table](../README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table). When I had access to a particular accelerator I run the benchmarks myself, when I didn't it was the kind contributors who invested their time to get these numbers. So I'm very grateful to [those](../../../contributors.md).

### How to report MAMF/MSMF entries

On a node with more than one GPU, measure with [`mamf-finder-all-gpus.py`](mamf-finder-all-gpus.py) rather than running `mamf-finder.py` on its own. In training every GPU of the node computes at once and they share the board's power and cooling budget, so a GPU measured while its siblings sit idle gets headroom a busy node never has, and its MSMF is only a single-GPU upper bound. The GPUs of one node also differ from chip to chip - by up to 4% in MSMF on the 8x H200 and 8x B200 nodes measured above - so one GPU's numbers describe that chip rather than the GPU model. The wrapper measures every GPU while all the others run a continuous matmul and reports the median GPU, which is what the table's `Siblings` = `yes` rows are, along with the slowest GPU and the spread. It also warns about GPUs whose MSMF window wasn't steady, ran 15°C or more hotter than the coolest GPU, or drew under 90% of the highest power: they pull the median down because of the node's cooling rather than the GPU model. With a single GPU it runs `mamf-finder.py` on its own, which gives a `Siblings` = `no` row. Pass it the same arguments you'd give `mamf-finder.py`:

```bash
python mamf-finder-all-gpus.py                        # bf16
python mamf-finder-all-gpus.py --dtype float8_e4m3fn  # fp8
```

On 8 H200s or 8 B200s the default bf16 run takes about 9 minutes.

If you get a better MAMF or MSMF than the [table](../README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table) shows, e.g. with a more recent `torch`, CUDA or ROCm, BLAS library or driver, or you measured an accelerator or a dtype the table doesn't have yet, please contribute it via a [PR](https://github.com/stas00/ml-engineering/pulls) that updates the table or an [Issue](https://github.com/stas00/ml-engineering/issues). Share the full log files of the run: the wrapper's output directory (`results/all-gpus-<timestamp>/` next to the script, or `OUT_DIR`) with `summary.txt` and every GPU's `gpu<N>.txt`, or the `--output-file` log of a single-GPU run. The logs record the software stack, the shapes, the power and the clocks, which is what makes the entry checkable and reproducible. Also mention any system settings the logs don't show, such as a disabled `numa_balancing`. A re-run of the same GPU can move its numbers by up to 2.2%, so a gain smaller than that is noise rather than an improvement.

## Straggler finder

A straggler is a GPU that works correctly but slower than its peers. It throws no Xid errors, its health counters read clean, and nothing crashes, yet in any job that runs its GPUs in lockstep - training, or inference with tensor, expert or pipeline parallelism - every GPU waits for the slowest one at each collective, so the whole job runs at the slow GPU's pace.

[gpu-straggler-finder.py](gpu-straggler-finder.py) finds them in one run. It runs the same training step, by default a Llama-3.1-8B one - its 32 decoder layers, forward and backward, on one 8192-token sequence - on every GPU at once, with no communication between them, then a matmul and two memory-bandwidth workloads, compares each GPU with its node's median GPU and each node with the median node in the same 2-second windows, and flags every GPU that is slower than its peers, or than itself the rest of the time, in its slowest stretches of any of them, with the UUID, serial and PCI bus id to report it by. [How one straggler was found](#how-one-straggler-was-found) walks through the case it was written for: a B200 GPU that neither `mamf-finder.py` nor DCGM flagged, yet it made the compute part of every step of a 32-GPU job 3-4% slower. It is one of the compute passes of [How to evaluate the cluster](../../../insights/how-to-choose-cloud-provider.md#how-to-evaluate-the-cluster), which runs it on every GPU of every node of a trial allocation.

```bash
python gpu-straggler-finder.py  # all the GPUs of one node
torchrun --nnodes 4 --nproc_per_node 8 --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:29500 gpu-straggler-finder.py
```

Each GPU is judged by its 90th percentile window, the one only a tenth of its windows are slower than. A GPU is flagged as slow when that window in any workload is an outlier among all the GPUs - a [modified z-score](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm) above 3.5 - and more than 1% slower than its node's median GPU, and as unsteady when that window is more than 1.25% slower than its own median window and an outlier too. A node is flagged as slow when its median GPU's 90th percentile window is more than 3% slower than the median node's.

Run it on GPUs that have been idle for a minute or two: it times the step for 60 seconds from its 4th step, since a straggler can be slow only some of the time - the one in [How one straggler was found](#how-one-straggler-was-found) mostly in the first 20-25 seconds after the load started. The whole run takes about 3 minutes. A GPU that is slow only some of the time can pass a single run, so if the first run flags anything, run it 3 more times with `--repeat 3`, 2 minutes idle apart, which took 12 minutes on 4 nodes, and take a GPU flagged in all 4 runs to be a straggler. On 1 or 2 nodes there are too few nodes to tell which one is slow, so give `--baseline` the `--json` files of earlier runs on known-good nodes to compare them with. On a GPU with less memory than a B200 lower `--layers` or `--seqlen`, and `--help` lists the other settings.

The finder is currently NVIDIA-only, but it can easily be extended to other vendors. Only its telemetry is vendor-specific: the timings use `torch.accelerator`, while the clock, power and temperature columns and the UUID and serial come from NVIDIA's NVML library (`pip install nvidia-ml-py`). Supporting another vendor, e.g. AMD through its `amdsmi` library, takes one subclass of the script's `Telemetry` class, as `NVIDIATelemetry` is for NVML. Without telemetry the script reports only the timings, which is all that is needed to find a straggler, but it hasn't been run on another vendor's accelerators yet.

Here is its report on 4 nodes of 8x B200, one of them the node of the straggler in [How one straggler was found](#how-one-straggler-was-found), on GPUs that had been idle for a minute:

```
measuring 32 GPUs at once: the training step for 60s, then a matmul, a memory copy and a memory stream for 30s each - about 2m50s in all

setup: allocating and filling the buffers ...
setup: allocating and filling the buffers: done in 12.8s
gathering the results ...
gathering the results: done in 0.4s

32 GPUs (NVIDIA B200), 4 node(s)
step: 32x decoder layer h=4096 ffn=14336 heads=32/8, 1x8192 tokens, bf16, forward+backward, 100ms idle after each step - median 404.85ms, fastest 399.34ms, slowest 411.96ms, spread 3.2%, median absolute deviation 2.22ms
matmul: 8192x8192x8192 bf16 back to back for 30s - median 1315 TFLOPS, slowest 1264, fastest 1343, spread 6.2%
copy: a 1024MiB cudaMemcpy within each GPU's memory for 30s - median 6444 GB/s read+written, slowest 6398, fastest 6494, spread 1.5%
stream: c = a + b on 1024MiB buffers, on the SMs, for 30s - median 6966 GB/s read+written, slowest 6920, fastest 7006, spread 1.2%
each node's median GPU vs the median node's in the same 2s window, + is slower: the median window / the 90th percentile window

node                   step        matmul          copy        stream
cluster-a-3     +0.6/ +0.9%   +1.3/ +1.4%   +0.2/ +0.2%   +0.2/ +0.2%
cluster-a-0     +0.1/ +0.2%   +0.2/ +0.3%   -0.1/ -0.1%   +0.0/ +0.0%
cluster-a-1     -0.2/ -0.1%   -0.8/ -0.6%   +0.1/ +0.1%   -0.0/ +0.0%
cluster-a-2     -0.3/ -0.1%   -0.2/ -0.1%   -0.1/ -0.1%   -0.1/ -0.1%

each workload in 2s windows, each GPU's time vs its node's median GPU's in the same window, + is slower: the median window / the 90th percentile window

node           GPU   step ms          step        matmul          copy        stream  step MHz  matmul MHz  power W  max C  power-capped  other limits
cluster-a-3      3    411.96   +1.0/ +3.4%   +3.2/ +3.4%   -0.3/ -0.3%   -0.1/ -0.1%      1560        1065      889     61          100%    <-- straggler: step unsteady
cluster-a-1      4    407.85   +1.1/ +1.4%   +2.2/ +2.7%   +0.5/ +0.6%   +0.5/ +0.5%      1567        1102      894     66          100%
cluster-a-2      2    406.95   +0.9/ +1.3%   +0.3/ +0.8%   +0.6/ +0.6%   +0.5/ +0.5%      1582        1192      894     69          100%
cluster-a-2      6    405.86   +0.7/ +1.1%   +0.3/ +0.5%   +0.6/ +0.6%   +0.5/ +0.5%      1582        1170      895     71          100%
cluster-a-0      3    407.33   +0.6/ +1.1%   +1.0/ +1.3%   -0.1/ -0.1%   -0.0/ -0.0%      1575        1140      890     60          100%
cluster-a-2      0    405.97   +0.7/ +1.1%   -0.3/ -0.1%   +0.6/ +0.6%   +0.5/ +0.5%      1590        1162      892     70          100%
cluster-a-1      6    406.56   +0.6/ +1.0%   +0.9/ +1.2%   +0.6/ +0.6%   +0.6/ +0.7%      1590        1185      896     69          100%
cluster-a-0      0    406.52   +0.4/ +0.9%   +0.7/ +0.9%   +0.7/ +0.7%   +0.6/ +0.6%      1590        1102      895     71          100%
cluster-a-0      1    404.06   -0.2/ +0.1%   +3.1/ +3.2%   -0.5/ -0.5%   -0.5/ -0.5%      1612        1080      885     60          100%
cluster-a-0      4    404.02   -0.2/ -0.0%   -0.8/ -0.3%   +0.1/ +0.3%   +0.0/ +0.0%      1597        1185      895     67          100%
cluster-a-2      5    400.18   -0.8/ -0.2%   -0.4/ -0.1%   -0.4/ -0.4%   -0.4/ -0.4%      1597        1140      888     61          100%
cluster-a-0      5    399.34   -1.3/ -1.0%   -1.1/ -0.4%   -0.9/ -0.8%   -0.5/ -0.5%      1620        1177      890     58          100%
... and 20 more GPUs, step -0.7 to +1.0% in their 90th percentile windows (--unflagged-rows -1 prints them)

1 straggler(s) - report them by UUID and serial, which don't change with the node name or the GPU's index:
  cluster-a-3 GPU 3: unsteady in step: its 90th percentile window 2.3% slower than its median one (z 17.6) - uuid GPU-00000000-1111-2222-3333-444444444444, serial 1650000000001, pci 00000000:63:00.0
a straggler is any of
  slow: a 90th percentile window above a modified z-score of 3.5 among the GPUs measured, and more than 1% slower than its node's median GPU (--outlier, --threshold)
  unsteady: a 90th percentile window more than 1.25% slower than its own median window, above a modified z-score of 3.5 among the GPUs measured (--unsteady)
  slow node: its median GPU's 90th percentile window more than 3% slower than the median node's (--slow-node)

runtimes:
  setup: allocating and filling the buffers     12.8s
  step                                          1m27s
  matmul                                        30.3s
  copy                                          30.0s
  stream                                        30.0s
  gathering the results                          0.4s
  report                                         0.3s
  total                                         3m13s
```

GPU 3 of `cluster-a-3` is flagged as unsteady: its step's 90th percentile window was 3.4% slower than its node's median GPU, while its median window was 1.0% slower, and no other GPU's 90th percentile step window was more than 1.4% slower. Its matmul ran 3.2-3.4% slower too, but so did that of `cluster-a-0` GPU 1, 3.1-3.2%, whose step ran at full speed, so the matmul alone couldn't tell the straggler from a healthy GPU.

On 4 healthy nodes, `cluster-a-0` to `cluster-a-2` with `cluster-a-4` in place of `cluster-a-3`, its report ends instead with:

```
no straggler - per workload, the slowest 90th percentile window, and the largest gap between a GPU's 90th percentile and median windows:
  step: slowest cluster-a-4 GPU 7 +1.9% (z 2.1), least steady cluster-a-0 GPU 5 +0.5% (z 1.5), slowest node cluster-a-0 +0.7%
  matmul: slowest cluster-a-0 GPU 1 +3.4% (z 3.2), least steady cluster-a-4 GPU 3 +0.7% (z 2.4), slowest node cluster-a-0 +0.9%
  copy: slowest cluster-a-0 GPU 0 +0.8% (z 1.7), least steady cluster-a-0 GPU 5 +0.2% (z 9.6), slowest node cluster-a-1 +0.3%
  stream: slowest cluster-a-1 GPU 6 +0.7% (z 1.4), least steady cluster-a-4 GPU 2 +0.0% (z 4.8), slowest node cluster-a-1 +0.1%
a straggler is any of
  slow: a 90th percentile window above a modified z-score of 3.5 among the GPUs measured, and more than 1% slower than its node's median GPU (--outlier, --threshold)
  unsteady: a 90th percentile window more than 1.25% slower than its own median window, above a modified z-score of 3.5 among the GPUs measured (--unsteady)
  slow node: its median GPU's 90th percentile window more than 3% slower than the median node's (--slow-node)
```

[How to find a straggler GPU](../nvidia/debug.md#how-to-find-a-straggler-gpu) shows how to record a flagged NVIDIA GPU and report it to the provider.

On 2 or more nodes it also sorts the nodes by their slowest GPU, the fastest node first. `--hostfile-to-sort` takes a hostfile - the file that a launcher such as `deepspeed` or `mpirun` reads the job's nodes from, one per line - and writes its lines in that order to the same path with `-fast-to-slow` appended. Its first lines are then the fastest nodes, so that a job that needs fewer nodes than were allocated can run on those - here 2 of 4:

```bash
torchrun --nnodes 4 --nproc_per_node 8 --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:29500 gpu-straggler-finder.py --hostfile-to-sort hostfile
head -2 hostfile-fast-to-slow > hostfile-fast
deepspeed --hostfile hostfile-fast train.py
```

A hostfile line is matched to a node by its first word, the node's hostname or one of its IP addresses, as in `10.0.0.1 slots=8`. Here is the list for the 4 nodes of the report above:

```
nodes fastest first, by their pace in a job that runs the GPUs in lockstep: the slowest GPU's step in each 2s window, at the 90th percentile window - a node with a flagged GPU, or flagged itself, after the others:
  cluster-a-2    408.92ms   +0.0%
  cluster-a-1    409.60ms   +0.2%
  cluster-a-0    409.71ms   +0.2%
  cluster-a-3    420.06ms   +2.7%  flagged
```

A job on all 4 nodes would run at `cluster-a-3`'s pace, 2.5% slower than a job on the other 3.

### How one straggler was found

Here is how the straggler in the report above was found on a 4 8x B200 node job, while re-measuring [comm-compute-overlap-bench.py](../../../network/benchmarks/comm-compute-overlap-bench.py). The symptom was that the compute part of every step came out 12-18ms (3-4%) above what the same benchmark had measured on a different 4-node allocation two days earlier - e.g. the `ddp` step at 8192 tokens took ~404ms against ~391ms.

1. Rule out the network. The same benchmark also times the communication alone, and that matched the earlier allocation within 0.5% (~141ms on both). So the network wasn't slower - the GPUs were.

2. Run the same workload on each node alone, all the nodes at once. 3 nodes took ~388-389ms and one took ~404ms, which is the slowdown of the 4-node job - so a single node was setting the pace for all 4.

3. Run the same workload on each GPU alone, all the GPUs at once - 8 separate 1-GPU processes per node, on all the nodes, each with its own rendezvous port:

   ```bash
   for g in 0 1 2 3 4 5 6 7; do
     CUDA_VISIBLE_DEVICES=$g torchrun --nproc_per_node=1 \
       --rdzv_backend c10d --rdzv_endpoint localhost:$((9500+g)) \
       comm-compute-overlap-bench.py --pattern ddp --layers 32 \
       --tokens 8192 --num-iterations 20 \
       >> console-$(hostname)-gpu$g.txt 2>&1 &
   done
   wait
   ```

   The 31 healthy GPUs took ~379-390ms (median ~385ms) and GPU 3 of the slow node took ~404ms - 5.1% above the median and 3.6% above the next slowest GPU. That gap is larger than the whole 2.9% spread of the other 31.

4. Confirm it in a profiler trace of the multi-GPU job. In the [Nsight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`) trace of the slow node, GPU 3 ran the very same kernels longer than the node's other 7 GPUs: the matmuls by 3.5-9%, flash attention by 4.2%, the reductions by 4.7% and the copies by 3.1%. The slowdown is spread over compute-bound and memory-bound kernels alike, which points at the GPU rather than at any one kernel, while its device-to-device memcpys, limited by the HBM bandwidth, ran within 2% of the other GPUs' - so the SMs were slow, not the memory.

Running everything at once matters for two reasons. The GPUs share the node's power, cooling and PCIe switches, and a GPU that is fast when it runs alone can be slow when its neighbors are busy - which is how it is used in training. And measuring 32 GPUs one after another takes 32 times longer while the conditions drift between the runs.

Use a workload that looks like your training step rather than a single matmul. [mamf-finder](#maximum-achievable-and-sustainable-matmul-flops-finder) ran on all 32 GPUs at once on this allocation, and it did rank GPU 3 the slowest of its node, at 1702 TFLOPS against the node's median of 1730 - but GPU 0 and GPU 4 of the same node reached 1703 and 1704 TFLOPS and ran the training step at full speed (~389ms each), and so did a GPU on another node that also reached 1702 TFLOPS (~387ms). GPU 3's maximum sustainable matmul FLOPS (MSMF) of 1419 TFLOPS wasn't even its node's lowest, GPU 0's 1417 was. A 3.5% spread in maximum achievable matmul FLOPS (MAMF) across 8 GPUs is normal (see [Silicon lottery](../README.md#silicon-lottery)), so by the matmul numbers alone GPU 3 looked like any other GPU at the low end of it. Under the sustained matmul it held 1417 MHz at 977 W, in the middle of its siblings' 1400-1452 MHz - its slowdown, described below, showed up only under the mix of a training step: matmuls of different shapes, attention, reductions and copies.

The usual health checks didn't flag it either: PCIe ran at Gen5 x16, the power limit was 1000 W, there were no ECC errors or remapped rows, and no GPU reported a thermal or hardware slowdown. Nor did [DCGM](../nvidia/debug.md#running-diagnostics): `dcgmi diag -r 3` passed every one of its tests on all 8 GPUs of the slow node, GPU 3 included, which took 45 minutes. DCGM checks that each GPU works and clears a minimum, and a GPU that is 5% slower than its siblings still clears it.

What showed the cause was logging the clocks every 200ms while the 8 runs of step 3 were repeated on the slow node:

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,clocks.sm,\
clocks.mem,power.draw,power.draw.instant,temperature.gpu,\
clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,\
clocks_event_reasons.hw_slowdown,\
clocks_event_reasons.sw_thermal_slowdown \
  --format=csv -lms 200 >> smi-$(hostname).csv &
```

GPU 3 was slow again - ~405ms against ~384-390ms. Under the load all 8 GPUs were held back by their 1000 W power cap (`sw_power_cap` was active in 58-72% of the samples on each of them), and none of them throttled for temperature - GPU 3 peaked at 52°C, in the middle of its siblings' 50-59°C. But at the same power GPU 3 ran its SM clock at a median of 1462 MHz, against 1537-1574 MHz on the other 7 - 5% lower, while the memory clock was the same 3996 MHz on all 8. In the finder's run above, which samples the clock every 10ms, GPU 3 ran the step at 1560 MHz against its 7 siblings' 1575-1605 MHz, and the matmul at 1065 MHz, the lowest of the 32 GPUs, while its copy and stream, bound by the HBM bandwidth, ran at its siblings' speed - its report with `--unflagged-rows -1` lists every GPU's clock. None of its functions fail, which is why no diagnostic flags it - but in a synchronous job it slows every other GPU down to its pace.

It was also slow only some of the time. With the same 32-layer step followed by 100ms of idle each time, started on GPUs that had been idle for 2 minutes, GPU 3 was 3-4% slower than its 7 siblings for the first 20-25 seconds and about 1% slower for the rest of the minute, while in the traces and the clock log it stayed about 5% slower throughout. A check that reports a GPU's best or typical speed can miss it - mamf-finder's MAMF is the median of a shape's 5 fastest runs - so time a prolonged run and judge each GPU by its worst stretches.
