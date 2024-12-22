# debug - 中文翻译

## 解决NVIDIA GPU问题

## Xid错误

没有任何硬件是完美的，有时由于制造问题或由于磨损（尤其是因为高温暴露），GPU可能会遇到各种硬件问题。许多这些问题会自动纠正，而无需真正了解发生了什么。如果应用程序继续运行，则通常没有需要担心的问题。如果应用程序因硬件问题而崩溃，则重要的是要理解为什么会发生这种情况以及如何应对。

对于仅使用少量GPU的普通用户来说，可能永远不需要了解与GPU相关的硬件问题，但如果你接近大规模的机器学习训练，可能需要使用数百到数千个GPU时，你肯定希望了解不同的硬件问题。

在你的系统日志中，你可能会偶尔看到Xid错误，如：

```
NVRM: Xid (PCI:0000:10:1c): 63, pid=1896, Row Remapper: New row marked for remapping, reset gpu to activate.
```

要获取这些日志，可以使用以下方法之一：
```
sudo grep Xid /var/log/syslog
sudo dmesg -T | grep Xid
```

通常情况下，只要训练没有崩溃，这些错误通常表示硬件会自动纠正的问题。

完整的Xid错误列表及其解释可以在[这里](https://docs.nvidia.com/deploy/xid-errors/index.html)找到。

你可以运行`nvidia-smi -q`并查看是否报告了任何错误计数。例如，在这种情况下Xid 63，你会看到类似的内容：

```
Timestamp                                 : Wed Jun  7 19:32:16 2023
Driver Version                            : 510.73.08
CUDA Version                              : 11.6

Attached GPUs                             : 8
GPU 00000000:10:1C.0
    Product Name                          : NVIDIA A100-SXM4-80GB
    [...]
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 177
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 177
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 1
        Uncorrectable Error               : 0
        Pending                           : Yes
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 639 bank(s)
            High                          : 1 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
[...]
```

我们可以看到Xid 63对应于：

```
ECC页面退役或行重新映射记录事件
```

这可能有三个原因：硬件错误/驱动程序错误/帧缓冲区（FB）损坏。

此错误意味着其中一行内存出现故障，并且在重新启动和/或GPU重置后，A100中的640个备用内存行之一将被用来替换坏行。因此我们在报告中看到只有639个银行（共640个）。

`ECC Errors`报告中的Volatile部分指的是自上次重新启动/GPU重置以来记录的错误。Aggregate部分记录了自GPU首次使用以来的相同错误。

现在，有两种类型的错误——可纠正错误和不可纠正错误。可纠正错误是一个单比特ECC错误（SBE），尽管内存有故障，但驱动程序仍可以恢复正确的值。不可纠正错误是指超过一个比特故障，称为双比特ECC错误（DBE）。通常，如果同一内存地址发生1个DBE或2个SBE错误，驱动程序将整页内存退役。有关详细信息，请参阅[此文档](https://docs.nvidia.com/deploy/dynamic-page-retirement/index.html)。

可纠正错误不会影响应用程序，非可纠正错误会导致应用程序崩溃。包含不可纠正ECC错误的内存页将被黑名单并不可访问，直到GPU重置。

如果有计划退役的页面，你将在`nvidia-smi -q`的输出中看到类似的内容：

```
    Retired pages
        Single Bit ECC             : 2
        Double Bit ECC             : 0
        Pending Page Blacklist    : Yes
```

每个退役的页面都会减少应用程序可用的总内存。但是，总共退役的最大页面数量只有4MB，所以它不会显著减少可用的GPU内存总量。

要更深入地进行GPU调试，请参考[此文档](https://docs.nvidia.com/deploy/gpu-debug-guidelines/index.html)，其中包括一个有用的诊断图，帮助确定何时需要更换GPU。此文档还包含关于Xid 63类似错误的额外信息。

例如，它建议：

> 如果与Xid 94相关联，遇到错误的应用程序需要重新启动。系统上的所有其他应用程序可以继续运行，直到有方便的时间重新启动以激活行重新映射。
> 请参见下文，根据行重新映射失败情况确定何时更换GPU。

如果重新启动后，相同的条件发生在相同的内存地址上，这意味着内存重新映射失败，并将再次发出Xid 64。如果这种情况持续发生，说明你遇到了无法自动纠正的硬件问题，需要更换GPU。

在其他时候，你可能会遇到Xid 63或64，并且应用程序崩溃。这通常会产生额外的Xid错误，但大多数情况下，这意味着错误是不可纠正的（即，这是一个DBE类错误，然后它将是Xid 48）。

如前所述，要重置GPU，你可以简单地重新启动机器，或者运行：

```
nvidia-smi -r -i gpu_id
```

其中`gpu_id`是你想要重置的GPU的顺序号，例如`0`为第一个GPU。如果不使用`-i`，则所有GPU都将被重置。

### 遇到不可纠正的ECC错误

如果你遇到错误：

```
CUDA错误：遇到不可纠正的ECC错误
```

如前一节所述，检查`nvidia-smi -q`的输出中的`ECC Errors`条目将告诉你哪个GPU有问题。但是，如果你需要快速检查以回收节点，只需该节点至少有一个GPU存在问题，你可以这样做：

```
$ nvidia-smi -q | grep -i correctable | grep -v 0
            SRAM Uncorrectable            : 1
            SRAM Uncorrectable            : 5
```

在好的节点上，这应该返回空内容，因为所有计数器都应该是0。但在上面的例子中，我们有一个坏的GPU——有两个条目，因为完整的记录是：

```
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 1
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 5
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
```

第一项是`Volatile`（从上次GPU驱动程序重新加载以来的错误）和第二项是`Aggregate`（GPU整个生命周期的总错误计数）。在这个例子中，我们看到`Volatile`的SRAM Uncorrectable错误计数为1，而寿命计数为5——这意味着这不是GPU第一次遇到这个问题。

这通常对应于Xid 94错误（参见：[Xid错误](#xid-errors)，最有可能没有Xid 48。

要解决这个问题，如前一节所述，重置有问题的GPU：
```
nvidia-smi -r -i gpu_id
```
重新启动机器会有同样的效果。

现在，当涉及到Aggregate SRAM Uncorrectable错误时，如果你的错误超过4个，通常是更换GPU的理由。