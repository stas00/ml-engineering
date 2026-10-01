# CPU

This chapter is short because GPUs dominate ML work - but the CPU's share has been growing, and unlike the GPU it's easy to under-provision without noticing.

Originally Machine Learning workloads didn't use much CPU other than for data processing (text/audio/video).

Around 2025 CPUs started to get more use for RAG workloads which need to perform database queries.

In 2026 the load on CPUs started increasing due to AI tool calling in RL workloads. These workloads may execute a variety of computer programs, which may perform validation of the generated data or code, compilation and execution of generated code, and many other things, and all of it typically running in a sandbox for security reasons. This need sometimes calls for dedicated CPU nodes to prevent stalling GPUs, since the CPU-cores co-located with the GPUs might prove insufficient.

## How many CPU cores do you need for DataLoader work

In a typical training workload a DataLoader is what consumes most of the CPU cores.

Per 1 accelerator you need:

1. 1 CPU core per process that is tied to the accelerator
2. 1 CPU core for each `DataLoader` worker process - and typically you need 2-4 workers.

2 workers is usually plenty for LMs, especially if the data is already preprocessed.

If you need to do dynamic transforms, which is often the case with computer vision models or VLMs, you may need 3-4 and sometimes more workers.

The goal is to be able to pull from the `DataLoader` instantly, and not block the accelerator's compute, which means that you need to pre-process a bunch of samples for the next iteration, while the current iteration is running. In other words your next batch needs to take no longer than a single iteration accelerator compute of the batch of the same size.

Besides preprocessing if you're pulling dynamically from the cloud instead of local storage you also need to make sure that the data is pre-fetched fast enough to feed the workers that feed the accelerator furnace.

Multiply that by the number of accelerators, add a few cores for the Operation system (let's say 4).

If the node has 8 accelerators, and you have `num_workers`, then you need `8*(num_workers+1)+4`. If you're doing NLP, it'd be usually about 2 workers per accelerator, so `8*(2+1)+4` => 28 CPU cores. If you do CV training, and, say, you need 4 workers per accelerator, then it'd be `8(4+1)+4` => 44 CPU cores.

What happens if you have more very active processes than the total number of CPU cores? Some processes will get preempted (put in the queue for when CPU cores become available) and you absolutely want to avoid any context switching.

But modern cloud offerings typically have 50-100+ CPU-cores so usually there is no problem to have enough cores to go around.

See also [Asynchronous DataLoader](../../training/performance/README.md#asynchronous-dataloader).



## CPU needs for sandboxed code execution

To score what a model generated, RL tool-use workloads have to run it - a test suite, a compilation, a package install, a symbolic math check. That runs in a sandbox: an isolated environment, since generated code is untrusted by construction and one sample leaking state into the next corrupts the reward signal. Depending on how strong the isolation has to be, this is a subprocess with resource limits, a container, a user-space kernel like [gVisor](https://gvisor.dev/), or a micro-VM like [Firecracker](https://firecracker-microvm.github.io/).

None of that work touches the accelerator, and there is a lot of it - a batch of hundreds of trajectories each making dozens of tool calls means tens of thousands of short programs per training step.

Two things make it costlier than it looks:

1. Isolation taxes system calls, not arithmetic. gVisor adds no cost to CPU instructions, but routes every system call through its own user-space kernel, so in some situations file- and process-heavy work could slow by multiples - and a test run or a `pip install` is exactly that kind of work ([gVisor performance guide](https://gvisor.dev/docs/architecture_guide/performance/)). Micro-VMs avoid that tax but pay a per-sandbox boot instead, ~125ms for Firecracker.
2. It sits on the accelerators' critical path. A synchronous RL step can't proceed until the whole batch is done, so a CPU shortfall, a straggling sandbox, or a verifier hanging on a pathological output idles every GPU in the job.

Give the sandboxes their own cores, disjoint from the ones the training process and its `DataLoader` workers run on. A child process inherits its parent's allowed cores, so this has to be set on the sandbox itself, with [`os.sched_setaffinity`](../../training/performance/README.md#ossched_setaffinity) or the container runtime's cpuset. The same section shows how the trainer side of that split is done. Cap the concurrency of the expensive phases like dependency installs and test runs, and put a timeout on everything - a generated infinite loop will otherwise hold a core until something kills it. At scale this is what pushes sandboxes onto dedicated CPU nodes.

footnote: A [Nebius agentic-RL writeup](https://nebius.com/blog/posts/in-agentic-rl-faster-tokens-are-not-enough) (2026-09) measured this on a Terminal-Bench workload, GLM-5.2 on 32 B300s. With the number of in-flight trajectories held at 1,024, assigning the sandboxes deterministic non-overlapping CPU sets cut batch-collection time from 332.1s to 303.0s, with no change to the model or the GPU configuration.



## CPU offload

Some frameworks, like [DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/) and [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) can offload some compute work to CPUs without creating a bottleneck. In which case you'd want additional CPU-cores beyond what you already use.



## NUMA affinity

See [NUMA affinity](../../training/performance/README.md#numa-affinity).



## Hyperthreads

[Hyper-Threads](https://en.wikipedia.org/wiki/Hyper-threading) double the CPU cores number, by virtualizing each physical core into 2 virtual ones, allowing 2 threads to use the same CPU core at the same time. Depending on the type of workload this feature may or may not increase the overall performance. Intel, the inventor of this technology, suggests a possible 30% performance increase in some situations.

See also [To enable Hyper-Threads or not](../../orchestration/slurm/performance.md#to-enable-hyper-threads-or-not).
