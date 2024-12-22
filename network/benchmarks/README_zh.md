# README - 中文翻译

# 网络基准测试

**工具**:

- [all_reduce_bench.py](all_reduce_bench.py) - 用于在处理大量数据时对执行`all_reduce`期间的实际网络带宽进行基准测试的工具。这有助于了解实际性能与广告规格之间的差异。

- [all_gather_object_vs_all_reduce.py](all_gather_object_vs_all_reduce.py) - 一个快速基准测试，展示了从`all_gather_object`到`all_reduce`在收集进程组中的完成状态时速度提高了23倍。例如，当实现某种所有进程都完成的标志时。此技术通常用于同步GPU，因为它们可能在不同数量的迭代中完成——这对于多DP通道的推理或在`DataLoader`中同步`StopIteration`事件很有用。参见[all_gather_object_vs_all_gather.py](./all_gather_object_vs_all_gather.py)。

- [all_reduce_latency_comp.py](all_reduce_latency_comp.py) - 示例说明1次4GB减少比1000次4MB减少要快得多。



## 关键可重复性要求

一系列成功实验最重要的要求是能够一次又一次地重现实验环境，同时仅改变一两个设置变量。

因此，当你试图弄清楚某个更改是否会提高性能或使其变差时，必须找出如何保持稳定性。

例如，你需要找到一种方法来防止网络使用波动。当我们为[108B预BLOOM实验](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide)进行性能优化时，由于我们处于共享的节点间网络中，相同的设置会根据其他用户使用网络的程度产生不同的吞吐量，这几乎是不可能做到的。在BLOOM-176B期间，我们被分配了一个具有隔离网络的专用SLURM分区，只有我们的流量。在这种环境中进行性能优化非常理想。



## 网络吞吐量

理解您的特定模型大小和框架对网络带宽、吞吐量和延迟的需求至关重要。如果您对网络投入不足，您最终可能会有闲置的GPU，从而浪费了金钱和时间。如果您过度投资于非常快速的网络，但GPU较慢，那么同样会浪费金钱和时间。

如果网络非常慢，训练很可能受到网络限制，许多训练设置的改进都不会提高性能。

注意：[EAI食谱](https://github.com/EleutherAI/cookbook)包含了一系列针对每种集体操作的[通信基准测试](https://github.com/EleutherAI/cookbook/tree/main/benchmarks/communication)，您可以用来快速测量您的节点间或节点内网络的吞吐量。

以下是一个简单的all-reduce基准测试，您可以用来快速测量您的节点间网络的吞吐量：

[all_reduce_bench.py](all_reduce_bench.py)

通常建议至少对4个节点进行基准测试，但当然，如果您已经可以访问训练期间将使用的所有节点，则使用所有节点进行基准测试。

在4个节点上运行：

```bash
GPUS_PER_NODE=8
NNODES=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    all_reduce_bench.py
```

注释：
- 如果不是SLURM环境，则需要调整`MASTER_ADDR`以匹配rank 0的主机名。

在SLURM环境下使用4个节点运行：

```bash
salloc --partition=mypartition --nodes=4 --ntasks-per-node=1 --cpus-per-task=48 --gres=gpu:8 --time=1:00:00 bash
srun --gres=gpu:8 --nodes=4 --tasks-per-node=1 python -u -m torch.distributed.run --nproc_per_node=8 --nnodes 4 --rdzv_endpoint $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):6000 --rdzv_backend c10d all_reduce_bench.py
```

注释：
- 您可能需要调整`--cpus-per-task`和`--partition`参数。
- 您只需执行一次`salloc`，然后可以在同一分配上多次重复`srun`。