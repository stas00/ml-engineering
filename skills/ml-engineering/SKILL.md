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
  failures, a straggler GPU slowing the whole job, communication that
  doesn't overlap with compute, checkpoint or preemption problems,
  storage/network bottlenecks, or needs to pick GPUs/cloud/file-systems or
  size inference latency/throughput.
  Distilled from "Machine Learning Engineering", the latest version of which can
  be found at https://github.com/stas00/ml-engineering

  The latest SKILL.md version can be found at https://github.com/stas00/ml-engineering/blob/master/skills/ml-engineering/SKILL.md
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

- **The number that matters is achievable, not theoretical.** Real matmul FLOPS are well below spec. Measure Maximum Achievable Matmul FLOPS with [`mamf-finder.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py); compute **MFU** (model FLOPS utilization) to compare setups. See [the most important thing to understand](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#the-most-important-thing-to-understand) and [what accelerator characteristics we care for](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#what-accelerator-characteristics-do-we-care-for). For a full trial-node loop (compute + network + storage → markdown report) see [How to evaluate the cluster](https://github.com/stas00/ml-engineering/blob/master/insights/how-to-choose-cloud-provider.md#how-to-evaluate-the-cluster) and the [evaluate-cluster skill](https://github.com/stas00/ml-engineering/blob/master/skills/evaluate-cluster/SKILL.md).
- **Find the stragglers.** A GPU that works correctly but runs slower than its peers throws no Xid errors and can pass DCGM and a matmul benchmark, yet any job that runs its GPUs in lockstep runs at the slow GPU's pace. [`gpu-straggler-finder.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/gpu-straggler-finder.py) runs a training step on every GPU of every node at once and flags slow nodes, and slow or unsteady GPUs by UUID and serial, which don't change when node names get recycled. See [Straggler finder](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/README.md#straggler-finder) and [How to find a straggler GPU](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/nvidia/debug.md#how-to-find-a-straggler-gpu).
- **Memory capacity + bandwidth often bind before FLOPS.** Check the [accelerator memory size and speed table](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#accelerator-memory-size-and-speed) and the [TFLOPS comparison table](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#tflops-comparison-table) when choosing hardware.
- **Don't forget power and cooling** - throttling silently caps sustained throughput. See [power consumption](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#power-consumption) and [cooling](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#cooling).
- **CPU and CPU memory matter too** (dataloading, offload, pinned memory). See [CPU](https://github.com/stas00/ml-engineering/tree/master/compute/cpu) and [CPU memory](https://github.com/stas00/ml-engineering/tree/master/compute/cpu-memory).

## Storage (IO)

Full chapter: [Storage](https://github.com/stas00/ml-engineering/blob/master/storage/README.md).

- **Pick the file system for the job:** distributed/parallel FS for shared checkpoints and datasets; fast **local** NVMe beats network storage for scratch/hot data. See [which file system to choose](https://github.com/stas00/ml-engineering/blob/master/storage/README.md#which-file-system-to-choose) and [local storage beats cloud storage](https://github.com/stas00/ml-engineering/blob/master/storage/README.md#local-storage-beats-cloud-storage).
- **Benchmark IO before you trust it** (checkpoint save/load and dataset streaming are common stalls). See [storage benchmarks](https://github.com/stas00/ml-engineering/blob/master/storage/README.md#benchmarks). Pair with the [cluster evaluation](https://github.com/stas00/ml-engineering/blob/master/skills/evaluate-cluster/SKILL.md) loop (`fio-scan` on local NVMe *and* shared FS).
- **Gotchas that bite at scale:** you often get less usable capacity than sold; some clouds put backups on the same partition; always keep [checksums](https://github.com/stas00/ml-engineering/blob/master/storage/README.md#dont-forget-the-checksums). Clean up instead of overpaying - see [why pay for more storage](https://github.com/stas00/ml-engineering/blob/master/storage/README.md#why-pay-for-more-storage-when-you-can-easily-clean-it-up-instead).

## Network

Full chapter: [Network](https://github.com/stas00/ml-engineering/blob/master/network/README.md).

- **Inter-node speed can dominate the whole training's speed.** For sharded/parallel training, slow inter-node links stall everyone. Understand why before scaling out: [why inter-node speed is hugely important](https://github.com/stas00/ml-engineering/blob/master/network/README.md#understanding-why-inter-node-network-speed-is-of-a-huge-importance).
- **Know the two speeds:** [intra-node](https://github.com/stas00/ml-engineering/blob/master/network/README.md#intra-node-networking) (NVLink/PCIe) vs [inter-node](https://github.com/stas00/ml-engineering/blob/master/network/README.md#inter-node-networking) (InfiniBand/RoCE/Ethernet), and [RDMA](https://github.com/stas00/ml-engineering/blob/master/network/README.md#rdma-networking).
- **Benchmark real collective throughput,** not marketing numbers, with [`torch-dist-bench.py`](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/torch-dist-bench.py) (far simpler than nccl-tests). Mind the [important nuances](https://github.com/stas00/ml-engineering/blob/master/network/README.md#important-nuances) (e.g. unidirectional vs bidirectional, payload size, busbw vs algbw). Intra-node plus ≥4-node inter-node is part of [How to evaluate the cluster](https://github.com/stas00/ml-engineering/blob/master/insights/how-to-choose-cloud-provider.md#how-to-evaluate-the-cluster).
- **Size the network to the parallelism:** [how much inter-node bandwidth training needs](https://github.com/stas00/ml-engineering/blob/master/network/README.md#how-much-inter-node-bandwidth-training-needs) gives rough targets (typically at least 400Gbps; ZeRO-3 needs more than TP+PP). What matters is the communication that doesn't overlap with compute: [Exposed Communication by Parallelism Pattern](https://github.com/stas00/ml-engineering/blob/master/network/exposed-communication.md) measures it for DDP, FSDP2, DeepSpeed ZeRO, TP, SP/CP, EP and PP, and [`comm-compute-overlap-bench.py`](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/README.md#comm-compute-overlap-bench) measures it on your hardware without setting up a framework and a model.
- **Collectives steal SMs from compute:** each NCCL collective's kernel takes one SM per CTA away from the overlapping compute, except `all_gather` and `all_to_all` under CTA policy `zero` with symmetric memory, which run on the copy engines. Lower a process group's limit with `max_ctas` or set `cta_policy` in its NCCL config; the `NCCL_MAX_CTAS` and `NCCL_CTA_POLICY` environment variables silently override every group's config. See [CTA policy](https://github.com/stas00/ml-engineering/blob/master/network/README.md#cta-policy) and the other [performance-oriented NCCL environment variables](https://github.com/stas00/ml-engineering/blob/master/network/README.md#performance-oriented-nccl-environment-variables). On bare metal with PCIe switches, [disable Access Control Services](https://github.com/stas00/ml-engineering/blob/master/network/README.md#disable-access-control-services): ACS sends GPU-to-NIC traffic up through the PCIe root complex, which slows NCCL down or hangs it.
- **Benchmarking 4 nodes to plan for 400:** see [extrapolating benchmark results from several nodes to many](https://github.com/stas00/ml-engineering/blob/master/network/README.md#extrapolating-benchmark-results-from-several-nodes-to-many). Libraries that replace part of NCCL's path, and when each one pays off (MSCCL++ for many short collectives, UCCL-collective for congested fabrics, DeepEP for MoE dispatch), are in [other collective libraries](https://github.com/stas00/ml-engineering/blob/master/network/README.md#other-collective-libraries).
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
- **Long sequences:** activation checkpointing still keeps each layer's input on the accelerator, which at 128K tokens can be the largest part of activation memory; [offload the activation checkpoints to CPU memory](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#offloading-the-activation-checkpoints-to-cpu-memory) (`transformers>=5.17`: `gradient_checkpointing_kwargs={"offload": True}`). It lowers the peak only if the peak is in `forward`, and it costs CPU memory and step time.
- **Model parallelism** (DP/TP/PP/sequence/ZeRO) concepts and trade-offs: [model parallelism](https://github.com/stas00/ml-engineering/tree/master/training/model-parallelism), including the sequence-parallel variants (Ulysses, Ulysses-Offload, UPipe, USP, LoongTrain), the [expert-parallel communication libraries](https://github.com/stas00/ml-engineering/blob/master/training/model-parallelism/README.md#expert-parallel-communication-libraries) (DeepEP, Hybrid-EP and others), and the [collectives each parallelism runs](https://github.com/stas00/ml-engineering/blob/master/training/model-parallelism/README.md#parallelism-network-collectives).

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
- **Recover without a checkpoint restart:** with at least 2 data-parallel replicas, the failed replica's state can be restored from a healthy one, losing at most a step. [`torchft`](https://github.com/meta-pytorch/torchft) does this for DDP and HSDP: the healthy groups keep training with a smaller global batch while the failed group restarts and fetches the state from a healthy group. A node that is failing but still works can be moved to a standby node even with a single replica, by capturing and restoring its processes together with their GPU memory (CRIU and `cuda-checkpoint`; research systems so far) while the whole job pauses. See [Multi-Replica-based fault tolerance](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md#multi-replica-based-fault-tolerance).
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
- **Cryptic CUDA errors:** `CUDA_LAUNCH_BLOCKING=1` for a real traceback (CUDA is async). See [The Art of Debugging - PyTorch](https://github.com/stas00/the-art-of-debugging/tree/master/pytorch#dealing-with-async-cuda-bugs).
- **OOM / memory:** [memory usage](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md#memory-usage); fragmentation via `PYTORCH_ALLOC_CONF`.
- **NaN/Inf:** [underflow and overflow detection](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md#underflow-and-overflow-detection) and [debugging tensors](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md#debugging-tensors).
- **Did communication overlap with compute, or is one GPU slow?** Put `nsys profile` in front of each node's `torchrun` to get one timeline per node with every rank's kernels and streams; add `--gpu-metrics-devices=all` for the GPU clocks, which needs elevated permissions. Tracing slows host-bound steps, so take step times from an untraced run. See [NVIDIA Nsight Systems](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md#nvidia-nsight-systems).
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
| Real network throughput (all-reduce busbw) | [`torch-dist-bench.py`](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/torch-dist-bench.py) |
| Actual achievable matmul FLOPS of an accelerator | [`mamf-finder.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py) |
| A GPU or node slower than its peers | [`gpu-straggler-finder.py`](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/gpu-straggler-finder.py) |
| How much communication a parallelism pattern leaves exposed | [`comm-compute-overlap-bench.py`](https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/comm-compute-overlap-bench.py) |
| Tiny models/tokenizers/datasets for fast iteration | [make-tiny guide](https://github.com/stas00/ml-engineering/blob/master/debug/make-tiny-models-tokenizers-datasets.md) |
| Better `trace` for distributed hangs | [NicerTrace](https://github.com/stas00/ml-engineering/blob/master/debug/tools.md) |

## Pick the fix by symptom

| Symptom | Reach for |
|---|---|
| Low TFLOPS / MFU, "GPUs feel idle" | Find the bottleneck: `mamf-finder`, [performance checklist](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#how-to-improve-speed-and-save-memory), DataLoader, NUMA, dim divisibility |
| Training OOM | Memory anatomy → activation checkpointing/offload/parallelism; profile; `PYTORCH_ALLOC_CONF` |
| Slow steps but GPUs busy on comms | Benchmark network (`torch-dist-bench`), check intra/inter-node, NCCL settings; measure the exposed communication (`comm-compute-overlap-bench`, `nsys`) |
| Whole job slower than its GPUs should be, nothing errors | `gpu-straggler-finder.py` on all nodes at once; confirm a flagged GPU with `--repeat 3` (on 1-2 nodes, compare with `--baseline`); leave its node out (`--hostfile-to-sort`), and record and report the GPU by UUID and serial |
| Slow dataloading / GPU starvation | [DataLoader](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#dataloader), local NVMe, prefetch/workers |
| Loss spike / divergence / NaN | Logbooks, init/STD, underflow-overflow detection, tensor scans |
| Multi-node/GPU hang or deadlock | `torch-distributed-gpu-test.py` → `py-spy` all ranks → `NCCL_DEBUG=INFO` |
| Node/GPU dies mid-run | Spare nodes, frequent checkpoints, auto-restart, kill/save switch; with ≥2 data-parallel replicas, [recovery from a healthy replica](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md#multi-replica-based-fault-tolerance) or `torchft` |
| Job keeps getting preempted | [forced preemption](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md#dealing-with-forced-job-preemption), queue chained jobs |
| Checkpoint save/load is slow | Benchmark storage, choose FS, local vs shared |
| Choosing GPUs / cloud / storage | Comparison tables, MAMF, [choose a cloud provider](https://github.com/stas00/ml-engineering/blob/master/insights/how-to-choose-cloud-provider.md) |
| Inference too slow / won't fit | Metrics (TTFT/TPOT), KV-cache memory, framework choice, model-load speedups |

## Notes for AI agents

- **Diagnose before optimizing.** Identify which resource (compute/memory/network/storage/dataloader) is the actual bottleneck with a measurement; don't tune blindly.
- **Prefer measured numbers over spec sheets.** Use the provided benchmark scripts on the target hardware/software stack before recommending changes or purchases.
- **Assume failures at scale.** For any long/large run, verify checkpointing, restart, spare capacity, and a kill switch exist before worrying about peak speed.
- **Verify the cluster first.** Run the distributed connectivity test before blaming model code for a multi-node problem, and the straggler finder before trusting a new allocation's step times.
- **Reuse the community's hard-won lessons.** Check the training logbooks for known instabilities and fixes before re-deriving them.
- **Read the linked chapter section** before applying a recipe - each has worked examples, exact commands, caveats, and scripts.
- For deep single-process/tool debugging (gdb, strace, py-spy, cProfile, core files), use the companion skill: [The Art of Debugging](https://github.com/stas00/the-art-of-debugging/blob/master/SKILL.md).
