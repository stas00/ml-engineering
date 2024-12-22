# README - 中文翻译

# 调试与故障排除

## 指南

- [调试PyTorch程序](./pytorch.md)

- [多节点多GPU Python程序挂起和死锁诊断解决方案](./torch-distributed-hanging-solutions.md)

- [网络调试](../network/debug/)

- [NVIDIA GPU故障排除](../compute/accelerator/nvidia/debug.md)

- [下溢和溢出检测](./underflow_overflow.md)

## 工具

- [调试工具](./tools.md)

- [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py) - 这是一个检查集群中所有GPU（一个或多个节点）是否可以相互通信并分配GPU内存的`torch.distributed`诊断脚本。

- [NicerTrace](./NicerTrace.py) - 这是一个改进的带有多个附加标志和更有用输出的`trace` Python模块。