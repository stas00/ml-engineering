# Guides to training models

- [Solutions to hanging problems during training](./torch-distributed-hanging-solutions.md)

- [Debugging pytorch](./pytorch-debug.md) - this document goes into what you can do when you don't get an obvious traceback that you can act on and fix your program when it breaks.

- [Tuning ML software for best performance](./software-performance.md) - tweaking software for the best performance.

- [Tuning ML hardware for best performance](./hardware-performance.md) - choosing and configuring machine learning hardware for best performance.

- [Debugging NCCL issues and performance](./nccl-performance-debug.md) - notes for debugging NCCL-based software and tuning it up for the peak performance

- [Hardware Troubleshooting](./hardware-troubleshooting.md) - what to do when one runs into hardware problems

# Tools for debugging pytorch-based training

- [printflock.py](./printflock.py) - a tiny library that makes your `print` calls non-interleaved in a multi-gpu environment.

- [all_reduce_bench.py](./all_reduce_bench.py) - a tool to benchmark the real network bandwidth while performing all_reduce on a largish amount of data. This is useful to find out what one gets in reality as compared to the promised spec.

- [torch-checkpoint-shrink.py](./torch-checkpoint-shrink.py) - this script fixes checkpoints which for some reason stored tensors with storage larger than their view at the moment of saving. It clones the current view and re-saves them with just the storage of the current view.

- [multi-gpu-non-interleaved-print.py](./multi-gpu-non-interleaved-print.py) - a `flock`-based wrapper around `print` that prevents messages from getting interleaved when multiple processes print at the same time - which is the case with `torch.distributed` used with multiple-gpus.

- [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py) - this a `torch.distributed` diagnostics
  script that checks that all GPUs in the cluster (one or many nodes) can talk to each other and allocate gpu memory.

- [emulate-multi-node.md](./emulate-multi-node.md) - instructions on how to emulate a multi-node setup using just a single node - we use the `deepspeed` launcher here.

- [all_gather_object_vs_all_reduce.py](./all_gather_object_vs_all_reduce.py) - a quick benchmark showing 23x speed up when moving from `all_gather_object` to `all_reduce` when collecting completion status from the process group. e.g. when implementing some sort of all-processes-are-done flag. This technique is usually used for synchronizing gpus when they may complete at different number of iterations - which one needs for inference over multiple DP channels, or when one wants to sync a `StopIteration` event in `DataLoader`. See also [all_gather_object_vs_all_gather.py](./all_gather_object_vs_all_gather.py).
