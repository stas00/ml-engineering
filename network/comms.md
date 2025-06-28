# Communication Patterns

The intention of this chapter is not to show code examples and explain APIs for which there are many tutorials, but to have excellent visuals that explain how the various types of communication patterns work.

## Point-to-point communications

Point-to-point communications are the simplest type of communication where there is always a single sender and a single receiver.

For example, [Pipeline Parallelism](../training/model-parallelism#pipeline-parallelism) performs a point-to-point communication where the activations from the current vertical stage is sent to the next stage. So the current gpu performs `send` and the gpu holding the next stage performs `recv`.

PyTorch has `send` and `recv` for blocking, `isend` and `irecv` for non-blocking p2p comms. [more](https://pytorch.org/tutorials/intermediate/dist_tuto.html#id1).


## Collective communications

Collective communications include either multiple senders and a single receiver, a single sender and multiple receivers or multiple senders and multiple receivers.

In the world of PyTorch typically each process is tied to a single accelerator, and thus accelerators perform collective communications via process groups. The same process may belong to multiple process groups.

### Broadcast

![broadcast](images/collective-broadcast-1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

![broadcast](images/collective-broadcast-2.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

PyTorch API example:

`dist.broadcast(tensor, src, group)`: Copies `tensor` from `src` to all other processes. [doc](https://pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast).



### Gather

![gather](images/collective-gather-1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

![gather](images/collective-gather-2.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

PyTorch API example:

`dist.gather(tensor, gather_list, dst, group)`: Copies `tensor` from all processes in `dst`. [doc](https://pytorch.org/docs/stable/distributed.html#torch.distributed.gather)



### All-gather

![all-gather](images/collective-all-gather-1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

![all-gather](images/collective-all-gather-2.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

For example, this collective is used in [ZeRO](../training/model-parallelism#zero-data-parallelism) (Deepspeed and FSDP) to gather the sharded model weights before `forward` and `backward` calls.

PyTorch API example:

`dist.all_gather(tensor_list, tensor, group)`: Copies `tensor` from all processes to `tensor_list`, on all processes. [doc](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather)



### Reduce

![reduce](images/collective-reduce-1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

![reduce](images/collective-reduce-2.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

PyTorch API example:

`dist.reduce(tensor, dst, op, group)`: Applies `op` to every `tensor` and stores the result in `dst`. [doc](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce)

PyTorch supports multiple reduction operations like: `avg`, `sum`, `product`, `min`, `max`, `band`, `bor`, `bxor`, and others - [full list](https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp).



### All-reduce

![all-reduce](images/collective-all-reduce-1.png)

[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)
![all-reduce](images/collective-all-reduce-2.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

For example, this collective is used in [DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) to reduce gradients between all participating ranks.

PyTorch API example:

`dist.all_reduce(tensor, op, group)`: Same as reduce, but the result is stored in all processes. [doc](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)



### Scatter

![scatter](images/collective-scatter-1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

![scatter](images/collective-scatter-2.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

PyTorch API example:

`dist.scatter(tensor, scatter_list, src, group)`: Copies the `i`-th tensor `scatter_list[i]` to the `i`-th process. [doc](https://pytorch.org/docs/stable/distributed.html#torch.distributed.scatter)




### Reduce-Scatter

![reduce-scatter](images/collective-reduce-scatter.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

For example, this collective is used in [ZeRO](../training/model-parallelism#zero-data-parallelism) (Deepspeed and FSDP) to efficiently reduce gradients across all participating ranks. This is 2x more efficient than [all-reduce](#all-reduce).

PyTorch API example:

`reduce_scatter(output, input_list, op, group, async_op)`: Reduces, then scatters a list of tensors to all processes in a group. [doc](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter)




### All-to-all

![all-to-all](images/collective-all-to-all-1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

![all-to-all](images/collective-all-to-all.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

For example, this collective is used in [Deepspeed Sequence Parallelism](../training/model-parallelism#deepspeed-ulysses-sp) for attention computation, and in MoE [Expert Parallelism](../training/model-parallelism#expert-parallelism).


PyTorch API example:

`dist.all_to_all(output_tensor_list, input_tensor_list, group)`: Scatters list of input tensors to all processes in a group and return gathered list of tensors in output list. [doc](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all)



## Algorithms

The collective communications may have a variety of different implementations, and comm libraries like `nccl` may switch between different algorithms depending on internal heuristics, unless overridden by users.

### Ring


#### Broadcast with unidirectional ring

Given:

- N: bytes to broadcast
- B: bandwidth of each link
- k: number of GPUs

A naive broadcast will send `N/B` at each step. The total time to broadcast to `k` GPUs will take: `(k-1)*N/B`

Here is an example of how a ring-based broadcast is performed:

![ring-based broadcast](images/broadcast-ring.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

This algorithm splits `N` into `S` messages

At each step `N/(S*B)` is sent, which is `S` times less than the naive algorithm sends per step.

The total time to broadcast `N` bytes to `k` GPUs will take:

`S*N/(S*B) + (k − 2)*N*/(S*B) = N*(S + k − 2)/(S*B)`

and if split messages are very small so that`S>>k`: `S + k − 2` is `~S` and then the total time is about `N/B`.



#### All-reduce with unidirectional ring

Ring-based `all-reduce` is done similarly to [broadcast](#broadcast-with-unidirectional-ring). The message is split into many small messages and each GPU sends a small message to the next GPU in parallel with other GPUs. `all-reduce` has to perform 2x steps than `broadcast`, because it performs a reduction - so the size of the message needs to be sent twice over the wire.

Moreover, the whole message can be first split into chunks, to make the process even more efficient. Here is the reduction of the first chunk:

![ring-based all-reduce chunk 1](images/all-reduce-ring-chunk1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

Then the next chunk is done, until all smaller messages are reduced:

![ring-based all-reduce chunk 2](images/all-reduce-ring-chunk2.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)


## More guides

Here are some additional guides with good visuals:

- [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/High-performant_DL/Multi_GPU/hpdlmultigpu.html#Communication-Primitives)
