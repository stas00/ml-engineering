# admin - 中文翻译

## SLURM管理

## 在多个节点上运行命令

1. 为了避免每次登录到新节点时被提示：
    ```
    Are you sure you want to continue connecting (yes/no/[fingerprint])?
    ```
    可以通过以下方式禁用此检查：
    ``` 
    echo "Host *" >> ~/.ssh/config
    echo "  StrictHostKeyChecking no" >> ~/.ssh/config
    ```
    当然，要确保这对你来说足够安全。我假设你已经在SLURM集群内，并且没有SSH到集群之外的主机。你可以选择不设置这一点，那么你就需要手动批准每个新节点。

2. 安装`pdsh`

现在你可以在多个节点上运行所需的命令了。

例如，我们来运行`date`：

```
$ PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26] date
node-25: Sat Oct 14 02:10:01 UTC 2023
node-21: Sat Oct 14 02:10:02 UTC 2023
node-23: Sat Oct 14 02:10:02 UTC 2023
node-24: Sat Oct 14 02:10:02 UTC 2023
node-26: Sat Oct 14 02:10:02 UTC 2023
```

让我们做些更有用和复杂的事情。我们将终止所有与GPU绑定但未在SLURM作业取消时退出的进程：

首先，这个命令会给我们所有占用GPU的进程ID：

``` 
nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq
```

所以现在我们可以一次性终止所有这些进程：

``` 
PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26]  "nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq | xargs -n1 sudo kill -9"
```

## SLURM设置

显示SLURM设置：

``` 
sudo scontrol show config
```

配置文件位于控制器节点上的`/etc/slurm/slurm.conf`。

一旦更新了`slurm.conf`，可以重新加载配置，运行：
``` 
sudo scontrol reconfigure
```
从控制器节点开始。

## 自动重启

如果节点需要安全重启（例如，如果镜像已更新），可以调整节点列表并运行：

``` 
scontrol reboot ASAP node-[1-64]
```

对于每个非空闲节点，该命令会等待当前任务结束，然后重启节点并将其状态恢复为`idle`。

请注意，你需要在控制器节点的`/etc/slurm/slurm.conf`中设置：
``` 
RebootProgram = "/sbin/reboot"
```
并且如果刚刚向配置文件中添加了此条目，则需要重新配置SLURM守护程序。

## 更改节点的状态

更改由`scontrol update`执行。

示例：

使一个已准备好的节点变为可用状态：
``` 
scontrol update nodename=node-5 state=idle
```

将一个节点从SLURM的资源池中移除：
``` 
scontrol update nodename=node-5 state=drain
```

## 恢复因进程退出缓慢而被禁用的节点

有时进程在作业取消时退出较慢。如果SLURM配置为不会无限期等待，它会自动禁用此类节点。但是，这些节点仍然可以供用户使用。

因此，这里是如何自动化处理的方法。

关键是获取由于“Kill task failed”而被禁用的节点列表，这可以通过以下命令获取：

``` 
sinfo -R | grep "Kill task failed"
```

现在提取并扩展节点列表，检查这些节点是否确实没有用户进程（或先尝试杀死它们），然后解除禁用。

之前你已经学会了如何[在多个节点上运行命令](#run-a-command-on-multiple-nodes)，我们将在此脚本中使用它。

这里是完成所有工作的脚本：[undrain-good-nodes.sh](./undrain-good-nodes.sh)

现在你可以直接运行此脚本，任何基本上准备好服务但目前被禁用的节点将切换到`idle`状态，并可供用户使用。

## 修改作业的时间限制

要设置新的时间限制，例如2天：
``` 
scontrol update JobID=$SLURM_JOB_ID TimeLimit=2-00:00:00
```

要在此前的基础上增加更多时间，例如再增加3小时。
``` 
scontrol update JobID=$SLURM_JOB_ID TimeLimit=+10:00:00
```

## 当SLURM出现问题时

分析SLURM日志文件中的事件日志：
``` 
sudo cat /var/log/slurm/slurmctld.log
```

例如，这可以帮助理解为什么某个节点提前被取消了任务或完全被移除了。