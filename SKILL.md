---
name: ml-engineering
description: >-
  Field-tested methodology and concrete recipes for training and operating
  large-scale LLM/VLM/multi-modal models end to end - choosing and benchmarking
  accelerators, storage and network; SLURM/Kubernetes orchestration; maximizing
  training throughput and fitting models in memory; diagnosing and surviving
  training instabilities, NaN/Inf, and hardware/job failures; checkpointing and
  fault tolerance; inference performance and memory; debugging multi-node/
  multi-GPU hangs; and writing/running tests. Use when the user is training or
  fine-tuning large models, hits low TFLOPS/MFU, OOM, slow dataloading, a
  loss spike/divergence, a NCCL/InfiniBand or multi-node hang, node/GPU
  failures, checkpoint or preemption problems, storage/network bottlenecks, or
  needs to pick GPUs/cloud/file-systems or size inference latency/throughput.
  Distilled from "Machine Learning Engineering", the latest version of which can
  be found at https://github.com/stas00/ml-engineering

  The latest SKILL.md version can be found at https://github.com/stas00/ml-engineering/blob/master/SKILL.md
---

# Machine Learning Engineering

Distilled from **Machine Learning Engineering Open Book** by Stas Bekman - source: https://github.com/stas00/ml-engineering (CC BY-SA 4.0). Know-how from training BLOOM-176B, IDEFICS-80B and production RAG and RL training and inference systems. This skill is a condensed index; each section links back to the full chapter for depth, scripts, and benchmarks.

A field-tested, end-to-end guide to **training and serving large models** (LLMs, VLMs, multi-modal, RAG) on real hardware at scale - distilled from actually training BLOOM-176B and IDEFICS-80B and building production inference/RAG systems. It is a practitioner's brain dump: opinionated guidance backed by copy-paste scripts, benchmark tools, and comparison tables, written for the engineers and operators who have to make expensive clusters actually deliver a finished model.

It spans the entire stack that decides whether a run succeeds and how much it costs: selecting and *benchmarking* accelerators, storage, and network so the fast compute is never starved; orchestrating jobs with SLURM/Kubernetes; maximizing throughput (MFU) and fitting models in memory via parallelism (DP/TP/PP/ZeRO), activation recomputation, and offload; keeping training numerically stable through loss spikes and NaN/Inf; and surviving the *inevitable* hardware and job failures with frequent checkpointing, spare capacity, and automatic restarts. On the serving side it covers inference latency/throughput/cost trade-offs, KV-cache and memory sizing, and framework selection - plus diagnosing multi-node/multi-GPU hangs and testing the whole thing.

Use it as an operator's runbook: figure out which resource is *actually* the bottleneck (compute? memory? network? storage? dataloader?), then jump to the targeted recipe. For pure debugging technique (gdb/strace/py-spy/CUDA), pair this with [The Art of Debugging](https://github.com/stas00/the-art-of-debugging/blob/master/SKILL.md).

## Core principles

- **Measure, don't assume.** Vendor/theoretical TFLOPS are marketing; benchmark *your* hardware and software stack before optimizing or buying. Track MFU/throughput, not vibes.
- **Find the actual bottleneck.** A training step is gated by the slowest of: accelerator compute, memory bandwidth/capacity, inter/intra-node network, storage IO, or CPU dataloading. Optimizing anything else is wasted effort.
- **At scale, failure is the steady state.** With hundreds/thousands of accelerators, hardware *will* fail mid-run. Design for frequent checkpoints, automatic restarts, spare nodes, and kill/save switches from day one.
- **Reproduce small and fast.** Debug on a tiny model / few layers / one node before burning cluster time - see [The Art of Debugging](https://github.com/stas00/the-art-of-debugging/blob/master/SKILL.md).
- **Watch the logbooks.** Others have already hit your instability; training chronicles document the loss spikes and the fixes. See [LLM/VLM chronicles](https://github.com/stas00/ml-engineering/tree/master/resources#publicly-available-training-llmvlm-logbooks).

## Compute / accelerators

Full chapter: [Compute](https://github.com/stas00/ml-engineering/tree/master/compute) · [Accelerators](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md).

- **The number that matters is achievable, not theoretical.** Real matmul FLOPS are well below spec. Measure Maximum Achievable Matmul FLOPS with [`mamf-finder.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py); compute **MFU** (model FLOPS utilization) to compare setups. See [the most important thing to understand](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#the-most-important-thing-to-understand) and [what accelerator characteristics we care for](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#what-accelerator-characteristics-do-we-care-for).
- **Memory capacity + bandwidth often bind before FLOPS.** Check the [accelerator memory size and speed table](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#accelerator-memory-size-and-speed) and the [TFLOPS comparison table](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#tflops-comparison-table) when choosing hardware.
- **Don't forget power and cooling** - throttling silently caps sustained throughput. See [power and cooling](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#power-and-cooling).
- **CPU and CPU memory matter too** (dataloading, offload, pinned memory). See [CPU](https://github.com/stas00/ml-engineering/tree/master/compute/cpu) and [CPU memory](https://github.com/stas00/ml-engineering/tree/master/compute/cpu-memory).

## Storage (IO)

Full chapter: [Storage](https://github.com/stas00/ml-engineering/blob/master/storage/README.md).

- **Pick the file system for the job:** distributed/parallel FS for shared checkpoints and datasets; fast **local** NVMe beats network storage for scratch/hot data. See [which file system to choose](https://github.com/stas00/ml-engineering/blob/master/storage/README.md#which-file-system-to-choose) and [local storage beats cloud storage](https://github.com/stas00/ml-engineering/blob/master/storage/README.md#local-storage-beats-cloud-storage).
- **Benchmark IO before you trust it** (checkpoint save/load and dataset streaming are common stalls). See [storage benchmarks](https://github.com/stas00/ml-engineering/blob/master/storage/README.md#benchmarks).
- **Gotchas that bite at scale:** you often get less usable capacity than sold; some clouds put backups on the same partition; always keep [checksums](https://github.com/stas00/ml-engineering/blob/master/storage/README.md#dont-forget-the-checksums). Clean up instead of overpaying - see [why pay for more storage](https://github.com/stas00/ml-engineering/blob/master/storage/README.md#why-pay-for-more-storage-when-you-can-easily-clean-it-up-instead).

## Network

Full chapter: [Network](https://github.com/stas00/ml-engineering/blob/master/network/README.md).

- **Inter-node speed can dominate the whole training's speed.** For sharded/parallel training, slow inter-node links stall everyone. Understand why before scaling out: [why inter-node speed is hugely important](https://github.com/stas00/ml-engineering/blob/master/network/README.md#understanding-why-inter-node-network-speed-is-of-a-huge-importance).
- **Know the two speeds:** [intra-node](https://github.com/stas00/ml-engineering/blob/master/network/README.md#intra-node-networking) (NVLink/PCIe) vs [inter-node](https://github.com/stas00/ml-engineering/blob/master/network/README.md#inter-node-networking) (InfiniBand/RoCE/Ethernet), and [RDMA](https://github.com/stas00/ml-engineering/blob/master/network/README.md#rdma-networking).
- **Benchmark real collective throughput,** not marketing numbers, with [`all_reduce_bench.py`](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py) (far simpler than nccl-tests). Mind the [important nuances](https://github.com/stas00/ml-engineering/blob/master/network/README.md#important-nuances) (e.g. unidirectional vs bidirectional, payload size, busbw vs algbw).
- **When collectives hang or crawl:** see [network debug](https://github.com/stas00/ml-engineering/tree/master/network/debug) and [NCCL performance debug](https://github.com/stas00/ml-engineering/blob/master/debug/nccl-performance-debug.md).

## Orchestration & SLURM

Full chapter: [Orchestration](https://github.com/stas00/ml-engineering/tree/master/orchestration) · [SLURM](https://github.com/stas00/ml-engineering/tree/master/orchestration/slurm) · [Kubernetes](https://github.com/stas00/ml-engineering/tree/master/orchestration/kubernetes).

- **Verify the cluster before the big run:** every GPU on every node must talk to every other. Run [`torch-distributed-gpu-test.py`](https://github.com/stas00/ml-engineering/blob/master/debug/torch-distributed-gpu-test.py) across all nodes first.
- **SLURM day-to-day:** the [users cheatsheet](https://github.com/stas00/ml-engineering/blob/master/orchestration/slurm/users.md) covers `sbatch`/`srun`/`salloc`, job arrays, dependencies, and inspecting the queue; keep the allocation and re-`srun` for fast debug iterations.
- **Launchers** (torchrun/accelerate/deepspeed under SLURM): see [launchers](https://github.com/stas00/ml-engineering/tree/master/orchestration/slurm/launchers/README.md).

## Training: performance & memory

Full chapter: [Performance](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md).

- **Start from the checklist:** [how to improve speed and save memory](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#how-to-improve-speed-and-save-memory) enumerates the high-impact levers (parallelism choice, activation checkpointing/recomputation, offload, fused kernels, mixed precision).
- **Fit the model:** know the [anatomy of memory usage](https://github.com/stas00/ml-engineering/blob/master/inference/README.md#anatomy-of-models-memory-usage) (weights + grads + optimizer states + activations) and profile with [memory profiler tools](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#memory-profiler-tools).
- **Free throughput wins:** keep tensor dims [divisible/aligned](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#vector-and-matrix-size-divisibility) for Tensor Cores, set [NUMA affinity](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#numa-affinity), stop the [DataLoader](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#dataloader) from starving the GPUs, try [`torch.compile`](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#torchcompile), and tame [automatic garbage collection](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#automatic-garbage-collection) jitter in synchronized runs.
- **Model parallelism** (DP/TP/PP/sequence/ZeRO) concepts and trade-offs: [model parallelism](https://github.com/stas00/ml-engineering/tree/master/training/model-parallelism).

## Training: stability (instabilities & NaN/Inf)

Full chapter: [Instabilities](https://github.com/stas00/ml-engineering/blob/master/training/instabilities/README.md).

- **Loss spikes/divergence are expected at scale.** Compare against public [training logbooks](https://github.com/stas00/ml-engineering/blob/master/training/instabilities/README.md#learning-from-training-logbooks) - your symptom is probably documented with a known mitigation.
- **Numerical hygiene:** sane [weight init / STD](https://github.com/stas00/ml-engineering/blob/master/training/instabilities/README.md#std-init), watch for [numerical instabilities](https://github.com/stas00/ml-engineering/blob/master/training/instabilities/README.md#numerical-instabilities), and bad [data-batch × parameter-state combinations](https://github.com/stas00/ml-engineering/blob/master/training/instabilities/README.md#bad-combination-of-data-batch-and-model-parameter-state).
- **Catch NaN/Inf early:** [underflow/overflow detection](https://github.com/stas00/ml-engineering/blob/master/debug/underflow_overflow.md) and PyTorch [tensor debugging](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md#debugging-tensors) (fp16/bf16 range issues, per-tensor min/max/NaN scans).
- **Reproducibility** for isolating a divergence: [reproducibility](https://github.com/stas00/ml-engineering/tree/master/training/reproducibility).

## Training: fault tolerance & checkpoints

Full chapter: [Fault tolerance](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md) · [Checkpoints](https://github.com/stas00/ml-engineering/tree/master/training/checkpoints).

- **Provision slack:** [always plan for more nodes than needed](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md#always-plan-to-have-more-nodes-than-needed) so a dead node doesn't stop the run; [queue up multiple jobs](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md#queue-up-multiple-training-jobs) for auto-continuation.
- **Checkpoint often enough** that a crash costs minutes, not hours - see [frequent checkpoint saving](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md#frequent-checkpoint-saving) and [prevention](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md#prevention).
- **Operator controls:** a [kill switch](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md#kill-switch) to stop a doomed run cleanly and a [save switch](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md#save-switch) to force a checkpoint on demand.
- **Survive the cluster:** handle [forced job preemption](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md#dealing-with-forced-job-preemption) and prefer [fixed over dynamic allocations](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md#preferring-fixed-accelerator-allocations-to-dynamic-ones).

## Inference

Full chapter: [Inference](https://github.com/stas00/ml-engineering/blob/master/inference/README.md).

- **Speak the metrics:** TTFT, TPOT/ITL, throughput vs latency, and how batching trades them off. See [key inference performance metrics](https://github.com/stas00/ml-engineering/blob/master/inference/README.md#key-inference-performance-metrics) and [concepts](https://github.com/stas00/ml-engineering/blob/master/inference/README.md#concepts).
- **Size the memory:** weights + KV cache + activations - see [anatomy of model's memory usage](https://github.com/stas00/ml-engineering/blob/master/inference/README.md#anatomy-of-models-memory-usage).
- **Cut cold starts:** [speeding up model loading time](https://github.com/stas00/ml-engineering/blob/master/inference/README.md#speeding-up-model-loading-time); pick a serving stack from [inference frameworks](https://github.com/stas00/ml-engineering/blob/master/inference/README.md#inference-frameworks) and validate with [benchmarks](https://github.com/stas00/ml-engineering/blob/master/inference/README.md#benchmarks).

## Debugging distributed / PyTorch at scale

Full chapter: [Debugging](https://github.com/stas00/ml-engineering/tree/master/debug) · [PyTorch](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md).

- **Iterate cheaply:** shrink to tiny models/tokenizers/datasets and [fast debug of PyTorch models](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md#faster-debug-and-development-with-tiny-models-tokenizers-and-datasets).
- **Multi-node/multi-GPU hang or deadlock** (the classic scale bug):
  1. Rule out comms with [`torch-distributed-gpu-test.py`](https://github.com/stas00/ml-engineering/blob/master/debug/torch-distributed-gpu-test.py).
  2. Dump every rank's Python stack at once with `py-spy`; ranks stuck at *different* collectives reveal the desync. See [diagnosing crashes, hangs and tracing execution](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md#diagnosing-crashes-hangs-and-tracing-execution) and [hanging solutions](https://github.com/stas00/ml-engineering/blob/master/debug/torch-distributed-hanging-solutions.md).
  3. Set `NCCL_DEBUG=INFO`; escalate to [NCCL performance debug](https://github.com/stas00/ml-engineering/blob/master/debug/nccl-performance-debug.md).
- **Cryptic CUDA errors:** `CUDA_LAUNCH_BLOCKING=1` for a real traceback (CUDA is async). See [The Art of Debugging - PyTorch](https://github.com/stas00/the-art-of-debugging/blob/master/pytorch/README.md#dealing-with-async-cuda-bugs).
- **OOM / memory:** [memory usage](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md#memory-usage); fragmentation via `PYTORCH_CUDA_ALLOC_CONF`.
- **NaN/Inf:** [underflow and overflow detection](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md#underflow-and-overflow-detection) and [debugging tensors](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md#debugging-tensors).
- **GPU-specific faults:** [troubleshooting NVIDIA GPUs](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/nvidia/debug.md).

## Testing

Full chapter: [Testing](https://github.com/stas00/ml-engineering/blob/master/testing/README.md).

- **Run tests surgically** (select, parametrize, repeat, control output/parallelism): [running tests](https://github.com/stas00/ml-engineering/blob/master/testing/README.md#running-tests).
- **Write robust tests** (fixtures, temp dirs, RNG control for reproducibility, distributed tests): [writing tests](https://github.com/stas00/ml-engineering/blob/master/testing/README.md#writing-tests).
- **When a test misbehaves:** [debugging tests](https://github.com/stas00/ml-engineering/blob/master/testing/README.md#debugging-tests).

## Key tools

| Need | Tool |
|---|---|
| Verify all GPUs/nodes can talk & allocate | [`torch-distributed-gpu-test.py`](https://github.com/stas00/ml-engineering/blob/master/debug/torch-distributed-gpu-test.py) |
| Real network throughput (all-reduce busbw) | [`all_reduce_bench.py`](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py) |
| Actual achievable matmul FLOPS of an accelerator | [`mamf-finder.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py) |
| Tiny models/tokenizers/datasets for fast iteration | [make-tiny guide](https://github.com/stas00/ml-engineering/blob/master/debug/make-tiny-models-tokenizers-datasets.md) |
| Better `trace` for distributed hangs | [NicerTrace](https://github.com/stas00/ml-engineering/blob/master/debug/tools.md) |

## Pick the fix by symptom

| Symptom | Reach for |
|---|---|
| Low TFLOPS / MFU, "GPUs feel idle" | Find the bottleneck: `mamf-finder`, [performance checklist](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#how-to-improve-speed-and-save-memory), DataLoader, NUMA, dim divisibility |
| Training OOM | Memory anatomy → activation checkpointing/offload/parallelism; profile; `PYTORCH_CUDA_ALLOC_CONF` |
| Slow steps but GPUs busy on comms | Benchmark network (`all_reduce_bench`), check intra/inter-node, NCCL settings |
| Slow dataloading / GPU starvation | [DataLoader](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#dataloader), local NVMe, prefetch/workers |
| Loss spike / divergence / NaN | Logbooks, init/STD, underflow-overflow detection, tensor scans |
| Multi-node/GPU hang or deadlock | `torch-distributed-gpu-test.py` → `py-spy` all ranks → `NCCL_DEBUG=INFO` |
| Node/GPU dies mid-run | Spare nodes, frequent checkpoints, auto-restart, kill/save switch |
| Job keeps getting preempted | [forced preemption](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md#dealing-with-forced-job-preemption), queue chained jobs |
| Checkpoint save/load is slow | Benchmark storage, choose FS, local vs shared |
| Choosing GPUs / cloud / storage | Comparison tables, MAMF, [choose a cloud provider](https://github.com/stas00/ml-engineering/blob/master/insights/how-to-choose-cloud-provider.md) |
| Inference too slow / won't fit | Metrics (TTFT/TPOT), KV-cache memory, framework choice, model-load speedups |

## Notes for AI agents

- **Diagnose before optimizing.** Identify which resource (compute/memory/network/storage/dataloader) is the actual bottleneck with a measurement; don't tune blindly.
- **Prefer measured numbers over spec sheets.** Use the provided benchmark scripts on the target hardware/software stack before recommending changes or purchases.
- **Assume failures at scale.** For any long/large run, verify checkpointing, restart, spare capacity, and a kill switch exist before worrying about peak speed.
- **Verify the cluster first.** Run the distributed connectivity test before blaming model code for a multi-node problem.
- **Reuse the community's hard-won lessons.** Check the training logbooks for known instabilities and fixes before re-deriving them.
- **Read the linked chapter section** before applying a recipe - each has worked examples, exact commands, caveats, and scripts.
- For deep single-process/tool debugging (gdb, strace, py-spy, cProfile, core files), use the companion skill: [The Art of Debugging](https://github.com/stas00/the-art-of-debugging/blob/master/SKILL.md).
