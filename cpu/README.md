# CPU

XXX: This chapter needs a lot more work

## How many cpu cores do you need

Per 1 gpu you need:

1. 1 cpu core per process that is tied to the gpu
2. 1 cpu core for each DataLoader worker process - and you need 2-4 workers.

2 workers is usually plenty for NLP, especially if the data is preprocessed

If you need to do dynamic transforms, which is often the case with computer vision models, you may need 3-4 and sometimes more workers.

The goal is to be able to pull from the DataLoader instantly, and not block the GPU's compute, which means that you need to pre-process a bunch of samples for the next iteration, while the current iteration is running. In other words your next batch needs to take no longer than a single iteration GPU compute of the batch of the same size.

Besides preprocessing if you're pulling dynamically from the cloud instead of local storage you also need to make sure that the data is pre-fetched fast enough to feed the workers that feed the gpu furnace.

Multiply that by the number of GPUs, add a few cores for the Operation system (let's say 4).

If the node has 8 gpus, and you have n_workers, then you need `8*(num_workers+1)+4`. If you're doing NLP, it'd be usually about 2 workers per gpu, so `8*(2+1)+4` => 28 cpu cores. If you do CV training, and, say, you need 4 workers per gpu, then it'd be `8(4+1)+4` => 44 cpu cores.

What happens if you have more very active processes than the total number of cpu cores? Some processes will get preempted (put in the queue for when cpu cores become available) and you absolutely want to avoid any context switching.

But modern cloud offerings typically have 48+ cpu-cores so usually there is no problem to have enough cores to go around.

### CPU offload

Some frameworks, like [Deepspeed](https://www.deepspeed.ai/tutorials/zero-offload/) can offload some compute work to CPU without creating an bottleneck. In which case you'd want additional cpu-cores.


## Hyperthreads

Doubles the cpu cores number

XXX:
