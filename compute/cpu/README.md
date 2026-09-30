# CPU

This chapter is tiny because GPUs dominate ML work and there is not much to tell about CPUs' use.

Originally Machine learning workloads didn't use much CPU other than for data processing (text/audio/video).

Around 2025 CPUs started to get more use for RAG workloads which need to perform database queries.

In 2026 the load on CPUs started increasing due to AI tool calling in RL workloads. These workloads may execute a variety of computer programs, which may perform validation of the generated data or code, compilation and execution of generated code, and many other things. This need sometimes calls for dedicated CPU nodes to prevent stalling GPUs, since the co-located with GPUs cpu-cores might prove insufficient.

It's also possible CPUs may evolve to become more like GPUs, so there will be more work offloaded to CPUs.

## How many cpu cores do you need for DataLoader work

In a typical training workload a DataLoader is what consumes most of the CPU cores.

Per 1 accelerator you need:

1. 1 cpu core per process that is tied to the accelerator
2. 1 cpu core for each `DataLoader` worker process - and typically you need 2-4 workers.

2 workers is usually plenty for LMs, especially if the data is already preprocessed.

If you need to do dynamic transforms, which is often the case with computer vision models or VLMs, you may need 3-4 and sometimes more workers.

The goal is to be able to pull from the `DataLoader` instantly, and not block the accelerator's compute, which means that you need to pre-process a bunch of samples for the next iteration, while the current iteration is running. In other words your next batch needs to take no longer than a single iteration accelerator compute of the batch of the same size.

Besides preprocessing if you're pulling dynamically from the cloud instead of local storage you also need to make sure that the data is pre-fetched fast enough to feed the workers that feed the accelerator furnace.

Multiply that by the number of accelerators, add a few cores for the Operation system (let's say 4).

If the node has 8 accelerators, and you have `num_workers`, then you need `8*(num_workers+1)+4`. If you're doing NLP, it'd be usually about 2 workers per accelerator, so `8*(2+1)+4` => 28 cpu cores. If you do CV training, and, say, you need 4 workers per accelerator, then it'd be `8(4+1)+4` => 44 cpu cores.

What happens if you have more very active processes than the total number of cpu cores? Some processes will get preempted (put in the queue for when cpu cores become available) and you absolutely want to avoid any context switching.

But modern cloud offerings typically have 50-100+ cpu-cores so usually there is no problem to have enough cores to go around.

See also [Asynchronous DataLoader](../../training/performance/README.md#asynchronous-dataloader).



### CPU offload

Some frameworks, like [DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/) and [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) can offload some compute work to CPUs without creating a bottleneck. In which case you'd want additional cpu-cores beyond what you already use.



## NUMA affinity

See [NUMA affinity](../../training/performance/README.md#numa-affinity).



## Hyperthreads

[Hyper-Threads](https://en.wikipedia.org/wiki/Hyper-threading) double the cpu cores number, by virtualizing each physical core into 2 virtual ones, allowing 2 threads to use the same cpu core at the same time. Depending on the type of workload this feature may or may not increase the overall performance. Intel, the inventor of this technology, suggests a possible 30% performance increase in some situations.

See also [To enable Hyper-Threads or not](../../orchestration/slurm/performance.md#to-enable-hyper-threads-or-not).
