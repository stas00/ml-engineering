# Networking Benchmarks

- [all_reduce_bench.py](all_reduce_bench.py) - a tool to benchmark the real network bandwidth while performing `all_reduce` on a largish amount of data. This is useful to find out what one gets in reality as compared to the advertised spec.

- [all_gather_object_vs_all_reduce.py](all_gather_object_vs_all_reduce.py) - a quick benchmark showing 23x speed up when moving from `all_gather_object` to `all_reduce` when collecting completion status from the process group. e.g. when implementing some sort of all-processes-are-done flag. This technique is usually used for synchronizing gpus when they may complete at different number of iterations - which one needs for inference over multiple DP channels, or when one wants to sync a `StopIteration` event in `DataLoader`. See also [all_gather_object_vs_all_gather.py](./all_gather_object_vs_all_gather.py).

- [all_reduce_latency_comp.py](all_reduce_latency_comp.py) - exemplifies how 1x 4GB reduction is much faster than 1000x 4MB reduction
