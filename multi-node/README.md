# Multi-node

## Guides

- [emulate-multi-node.md](./emulate-multi-node.md) - instructions on how to emulate a multi-node setup using just a single node - we use the `deepspeed` launcher here.


## Tools

- [printflock.py](./printflock.py) - a tiny library that makes your `print` calls non-interleaved in a multi-gpu environment.

- [multi-gpu-non-interleaved-print.py](./multi-gpu-non-interleaved-print.py) - a `flock`-based wrapper around `print` that prevents messages from getting interleaved when multiple processes print at the same time - which is the case with `torch.distributed` used with multiple-gpus.

- [all_reduce_bench.py](./all_reduce_bench.py) - a tool to benchmark the real network bandwidth while performing all_reduce on a largish amount of data. This is useful to find out what one gets in reality as compared to the promised spec.

- [all_gather_object_vs_all_reduce.py](./all_gather_object_vs_all_reduce.py) - a quick benchmark showing 23x speed up when moving from `all_gather_object` to `all_reduce` when collecting completion status from the process group. e.g. when implementing some sort of all-processes-are-done flag. This technique is usually used for synchronizing gpus when they may complete at different number of iterations - which one needs for inference over multiple DP channels, or when one wants to sync a `StopIteration` event in `DataLoader`. See also [all_gather_object_vs_all_gather.py](./all_gather_object_vs_all_gather.py).
