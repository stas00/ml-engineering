# Debugging and Troubleshooting


## Guides

- [Debugging PyTorch programs](./pytorch.md)

- [Diagnosing Hangings and Deadlocks in Multi-Node Multi-GPU Python Programs](./torch-distributed-hanging-solutions.md)

- [Hardware Troubleshooting](./hardware-troubleshooting.md) - what to do when one runs into hardware problems

- [Underflow and Overflow Detection](./underflow_overflow.md) -

- [NCCL Debug and Performance](./nccl-performance-debug.md) - notes for debugging NCCL-based software and tuning it up for the peak performance


## Tools

- [Debug Tools](./tools.md)

- [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py) - this a `torch.distributed` diagnostics
  script that checks that all GPUs in the cluster (one or many nodes) can talk to each other and allocate gpu memory.

- [NicerTrace](./NicerTrace.py) - this is an improved `trace` python module with multiple additional flags added to the constructor and more useful output.
