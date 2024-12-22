# incoming - 中文翻译

# 需要添加/集成的内容

# PDF书籍笔记

来自Sam的想法：https://github.com/saforem2: https://github.com/stas00/ml-engineering/pull/17#discussion_r1439912709
https://quarto.org/, https://quarto.org/docs/gallery/, https://kevinheavey.github.io/modern-polars/, https://quarto.org/docs/output-formats/pdf-basics.html

# 性能

https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html


# 存储章节

### 存储基准测试：

https://github.com/argonne-lcf/dlio_benchmark


Ross Wightman的建议集成：

- 尽量按工作负载分离存储卷，保持“大量小文件”、高周转率的环境和代码与数据集、检查点等大型存储分离。甚至可以进一步分离数据集和检查点，因为数据集通常是静态的而检查点经常在变化。

- 当数据集位于网络存储上时，就像桶存储一样，它们应由大文件组成并且应以大文件形式读取（顺序地以大块方式读取，而不是内存映射！）。避免在数据集中进行寻址。

- HF数据集这样的设置可能会造成误导，看起来像一个大文件，但通常被内存映射了，I/O读取模式非常疯狂，比作为单个文件读取多出3-4倍的IOPS。
  可以关闭内存映射加载，但这可能导致许多数据集一次性加载到内存中太多数据的问题。对于不同用例有更好的意识，并且在适当的时候使用可迭代流。

- 在某种程度上，桶存储如S3，通过接口限制强制执行了一些合理的存储后端模式。哦，它被挂载为一个文件夹，我可以做任何事情（内存映射文件，写入大量小文件，删除它们等）这就是问题所在。

- 也不能期望将分布式文件系统当作本地磁盘使用。如果你按工作负载分离卷，可能能够更高效地利用总存储量。不要将高周转率的小文件与低周转率的大文件混合在一起。

- 另外，一旦你的数据集对大型分布式网络文件系统友好，它们通常可以从云系统中的桶存储中直接流式传输。因此在这种情况下，最好将它们从网络文件系统中移除。

# 调试

内存泄漏检查

``` 
cuda-memcheck --leak-check full python program.py
```

竞态检测：
``` 
cuda-memcheck --tool racecheck
```
带额外选项：
 --save 保存输出到磁盘
 --print-level 控制输出

``` 
cuda-memcheck --tool racecheck --racecheck-report analysis
```

使用cuda-gdb调试

``` 
cuda-gdb
```

- 集成debug_utils.py


# 模型并行性

这里有一个好的表格，列出了每种类型的并行性的缩放方程。
https://www.cerebras.net/blog/cerebras-sets-record-for-largest-ai-models-ever-trained-on-single-device#summary


# 网络

创建一个新的基准测试部分：

1. nccl-tests
2. `all_reduce_bench.py`
3. https://github.com/microsoft/DeepSpeedExamples/tree/master/benchmarks/communication
4. 类似于nccl-tests，另一个常用的HPC站点基准测试是OSU微基准测试，如osu_lat，osu_bw和osu_bibw。

https://mvapich.cse.ohio-state.edu/benchmarks/

这些是基于MPI的基准测试。 他们可以通过GPUDirect RDMA运行，你可以测量MPI性能，无论是同一节点还是跨节点的GPU之间。

## Infiniband

参考资料：
- [Sys Admin Pocket Survival Guide - InfiniBand](https://tin6150.github.io/psg/infiniband.html)


### 诊断

非InfiniBand特定
- `ifconfig` - 显示当前活动接口的状态
- `ip addr show` - 显示系统上配置的所有链接的地址

显示本地主机的IB设备状态（三种不同的视图）。
- `ibstat`
- `ibstatus`
- `ibv_devinfo`

扫描IB网络：
- `ibnetdiscover` - 扫描拓扑结构
- `ibroute` - 显示单播和多播转发表
- `ibdiagnet` - IB诊断网

检查网络错误：
- `ibcheckerrors` - 检查端口/节点的错误计数器是否在预定义的阈值内
- `ibchecknet` - 对子网上的端口/节点/错误进行检查。

测试IB网络配置：
- `ibcheckport` - 对指定端口进行一些基本测试
- `ibchecknode` - 对指定节点进行一些基本测试
- `ibclearcounters` - 清除InfiniBand子网的端口计数器

其他检查：
- `iblinkinfo`
- `ibcheck`
- `wwibcheck`
- `ibswitch` - 验证IB-QNEM是否安装在机架中
- `ibhosts` - 列出IB网络中的所有主机。
`ibswitches` - 列出所有IB交换机

跟踪：
- `ibping` - 在InfiniBand节点之间进行ping/pong
- `ibsysstat` - 获取远程节点的基本信息（主机名、CPU、内存、利用率）
- `ibswitches` - 扫描网络或使用现有的网络拓扑文件并列出所有交换机
- `ibhosts` - 扫描网络或使用现有的网络拓扑文件并列出所有主机

显示网络拓扑：
- `iblinkinfo -R`

使用`ifconfig`发现`IPoIB`网络，例如，如果你得到`ib0`设备，其`inet addr:100.1.1.102`，你可以连接到它 - 例如`ping 100.1.1.102`

查找控制器：
`lspci | grep Mellanox`

打印驱动程序配置（接口名称来自`ifconfig`）：
`ethtool -i enP49239s1`

### 性能

`perftest` 包包含：
- `ib_send_bw`
- `ib_send_lat`
- `ib_write_bw`
- `ib_write_lat`
- `ib_read_bw`
- `ib_read_lat`
- `ib_atomic_bw`
- `ib_atomic_lat`

示例：`ib_send_bw -a address` - 测试带宽

`qperf` 测量两个节点之间的带宽和延迟（TCP/IP和RDMA传输）

如果网络比预期慢得多，可能需要指定要使用的HCAs (`ibv_devinfo` 获取HCAs)
``` 
export NCCL_IB_HCA=mlx5
```

可能需要在VM上安装IB包：

``` 
sudo apt-get install -y automake dh-make git libcap2 libnuma-dev libtool make pkg-config udev curl librdmacm-dev rdma-core \
    libgfortran5 bison chrpath flex graphviz gfortran tk dpatch quilt swig tcl ibverbs-utils infiniband-diags
sudo sed -i -e 's/# OS.EnableRDMA=y/OS.EnableRDMA=y/g' /etc/waagent.conf
```

- Verbs: 允许命令在功能丰富的IB交换机上执行。


# 测试

- 集成testing_utils.py的功能


# 来自Adam Moody团队在LLNL的建议


- NUMA亲和力

https://github.com/LLNL/mpibind/tree/master/python
mpibind for Python允许在任意Python程序中使用mpibind算法。

- 训练挂起检测工具：

这将扩展：
https://github.com/stas00/ml-engineering/tree/master/fault-tolerance#is-job-hanging-watchdog


Adam的笔记：

https://github.com/LLNL/STAT - 栈追踪分析工具
https://hpc.llnl.gov/software/development-environment-software/stat-stack-trace-analysis-tool

https://github.com/grondo/io-watchdog

可以看到我们如何整合STAT的相关内容：

https://hpc.llnl.gov/software/development-environment-software/stat-stack-trace-analysis-tool

有一些“动作”脚本，当io-watchdog检测到挂起时会执行。页面上没有显示内容，但如果你感兴趣我可以查一下。用户可以创建一个配置文件，如下：

``` 
search /usr/local/tools/io-watchdog/actions
timeout = 20m
actions = STAT, kill
```

这配置io-watchdog假设任务停滞如果20分钟内没有输出（从rank 0开始），然后运行“STAT”来收集栈追踪并运行“kill”来scancel任务。我们还有一些其他的，比如一个向用户发送邮件，通知io-watchdog检测到挂起。然后启动时使用：
``` 
srun --io-watchdog mpi_application
```

SCR的一个快速演示。使用它的Python代码相当简洁。

安装SCR库（C + MPI）
https://scr.readthedocs.io/en/v3.0/users/build.html#cmake

安装scr.py模块：
https://github.com/LLNL/scr/tree/develop/python#installing-the-scr-python-module

示例检查点在Python中：
https://github.com/LLNL/scr/blob/1878de8756c2b51882a7cda7b97b142eae4e3995/python/scr_example.py#L64-L105



  396  dmesg | grep -i 'limited by'
  397  sudo dmesg | grep -i 'limited by'
  398  nvidia-smi nvlink -e


GPU VBIOS版本在研究问题时可能很重要。让我们添加名称和总线ID到查询中，我们得到：

``` 
$ nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version --format=csv

$ nvidia-smi -q | grep "VBIOS Version"
    VBIOS Version                         : 96.00.89.00.01
    [...]
    VBIOS Version                         : 96.00.89.00.01
```

检查NVLink链路的错误计数器

``` 
$ nvidia-smi nvlink -e
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-abcdefab-cdef-abdc-abcd-abababababab)
         Link 0: Replay Errors: 0
         Link 0: Recovery Errors: 0
         Link 0: CRC Errors: 0

         Link 1: Replay Errors: 0
         Link 1: Recovery Errors: 0
         Link 1: CRC Errors: 0

         [...]

         Link 17: Replay Errors: 0
         Link 17: Recovery Errors: 0
         Link 17: CRC Errors: 0
```

另一个有用的命令是：
``` 
$ nvidia-smi nvlink --status
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-abcdefab-cdef-abdc-abcd-abababababab)
         Link 0: 26.562 GB/s
         [...]
         Link 17: 26.562 GB/s
```
这个命令告诉你每个链路的当前速度

运行 `nvidia-smi nvlink -h` 发现更多特性（报告、重置计数器等）。