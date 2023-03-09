# guides to training models 


- [debugging pytorch](./pytorch-debug.md) - this document goes into what you can do when you don't get an obvious traceback that you can act on and fix your program.

- [hardware performance](./hardware-performance.md) - choosing and configuring machine learning hardware for best performance.

- [nccl-performance-debug](./nccl-performance-debug.md) - notes for debugging NCCL-based software and tuning it up for the peak performance


# pytorch tools for training models

- [printflock.py](./printflock.py) - a tiny library that makes your `print` calls non-interleaved in a multi-gpu environment.

- [all_reduce_bench.py](./all_reduce_bench.py) - a tool to benchmark the real network bandwidth while performing all_reduce on a largish amount of data. This is useful to find out what one gets in reality as compared to the promised spec.

- [torch-checkpoint-shrink.py](./torch-checkpoint-shrink.py) - this script fixes checkpoints which for some reason stored tensors with storage larger than their view at the moment of saving. It clones the current view and re-saves them with just the storage of the current view.

- [multi-gpu-non-interleaved-print.py](./multi-gpu-non-interleaved-print.py) - a `flock`-based wrapper around `print` that prevents messages from getting interleaved when multiple processes print at the same time - which is the case with `torch.distributed` used with multiple-gpus.

- [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py) - this a `torch.distributed` diagnostics
  script that checks that all GPUs in the cluster (one or many nodes) can talk to each other and allocate gpu memory. And the accompanying solutions to `torch.distributed` hanging: [torch-distributed-hanging-solutions.md](./torch-distributed-hanging-solutions.md).

- [emulate-multi-node.md](./emulate-multi-node.md) - instructions on how to emulate a multi-node setup using just a single node - we use the `deepspeed` launcher here.

- [all_gather_object_vs_all_reduce.py](./all_gather_object_vs_all_reduce.py) - a quick benchmark showing 23x speed up when moving from `all_gather_object` to `all_reduce` when collecting completion status from the group. e.g. some sort of all-done-flag. This technique is usually used for synchronizing gpus.
