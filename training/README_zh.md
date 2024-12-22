# README - 中文翻译

# 训练

**子章节**:

- [模型并行性](model-parallelism)

- [性能](performance)

- [容错性](fault-tolerance)

- [可重复性](reproducibility)

- [不稳定性](instabilities)

- [检查点](checkpoints)

- [训练超参数和模型初始化](hparams.md)

- [张量精度/数据类型](dtype.md)

- [使用单节点模拟多节点设置](emulate-multi-node.md) - 指示如何仅使用单个节点模拟多节点设置——这里我们使用`deepspeed`启动器。

- [使用微调示例从头重新训练HF hub模型](re-train-hub-models.md)

**工具**:

- [printflock.py](tools/printflock.py) - 一个使你在多GPU环境中`print`调用不交错的小型库。

- [multi-gpu-non-interleaved-print.py](tools/multi-gpu-non-interleaved-print.py) - 一个基于`flock`的`print`包装器，可以防止多个进程同时打印时消息交错——这是使用`torch.distributed`和多个GPU时的情况。