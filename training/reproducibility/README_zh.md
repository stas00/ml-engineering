# README - 中文翻译

## 可重复性

### 在基于随机性的软件中实现确定性

在调试时，始终为所有使用的随机数生成器（RNG）设置一个固定的种子，以便每次重新运行时都能获得相同的数据/代码路径。

尽管有许多不同的系统，但覆盖它们可能会很棘手。以下是一个尝试覆盖几个系统的示例：

```python
import random, torch, numpy as np
def enforce_reproducibility(use_seed=None):
    seed = use_seed if use_seed is not None else random.randint(1, 1000000)
    print(f"使用种子: {seed}")

    random.seed(seed)    # python RNG
    np.random.seed(seed) # numpy RNG

    # pytorch RNGs
    torch.manual_seed(seed)          # cpu + cuda
    torch.cuda.manual_seed_all(seed) # 多GPU - 即使没有GPU也可以调用
    if use_seed: # 较慢的速度！https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    return seed
```

如果你使用这些子系统/框架，还有其他可能的选项：
```python
    torch.npu.manual_seed_all(seed)
    torch.xpu.manual_seed_all(seed)
    tf.random.set_seed(seed)
```

当你反复运行同一段代码以解决某个问题时，在代码开始处设置特定种子：

```python
enforce_reproducibility(42)
```

如上所述，这仅适用于调试，因为它激活了各种有助于确定性的PyTorch标志，但会降低速度，因此在生产环境中不希望这样。

然而，你可以调用以下内容以在生产中使用：

```python
enforce_reproducibility()
```

即不带显式种子。然后它会选择一个随机种子并记录下来！因此如果在生产中出现问题，你现在可以重现该问题出现时相同的RNG设置。而且这次没有性能损失，因为只有在你明确提供种子时才会设置`torch.backends.cudnn`标志。假设它记录如下：

```python
使用种子: 1234
```

那么你只需将代码改为：

```python
enforce_reproducibility(1234)
```

即可获得相同的RNG设置。

正如前几段所述，系统中可能涉及许多其他RNG。例如，如果你想让`DataLoader`中的数据以相同的顺序馈入，则需要[设置其种子](https://pytorch.org/docs/stable/notes/randomness.html#dataloader)。

### 复现软件和系统环境

当发现某些结果的差异（如质量或吞吐量）时，这种方法很有用。

这个想法是记录启动训练（或推理）所使用的环境的关键组件，以便在以后阶段需要完全复现时可以做到这一点。

由于系统和组件种类繁多，不可能规定一种总是有效的方法。因此，让我们讨论一种可能的方案，然后你可以根据你的具体环境进行调整。

这被添加到你的Slurm启动脚本中（或其他任何用于启动训练的方式）——这是一个Bash脚本：

```bash
SAVE_DIR=/tmp # 编辑为实际路径
export REPRO_DIR=$SAVE_DIR/repro/$SLURM_JOB_ID
mkdir -p $REPRO_DIR
# 1. 模块（写入stderr）
module list 2> $REPRO_DIR/modules.txt
# 2. 环境变量
/usr/bin/printenv | sort > $REPRO_DIR/env.txt
# 3. pip（包括开发安装的SHA）
pip freeze > $REPRO_DIR/requirements.txt
# 4. 在conda中安装的git克隆未提交的差异
perl -nle 'm|"file://(.*?/([^/]+))"| && qx[cd $1; if [ ! -z "\$(git diff)" ]; then git diff > \$REPRO_DIR/$2.diff; fi]' $CONDA_PREFIX/lib/python*/site-packages/*.dist-info/direct_url.json
```

如你所见，这个方案用于Slurm环境，因此每个新的训练都会保存特定于Slurm作业的环境。

1. 我们保存加载的`模块`，例如在云集群/HPC设置中，你可能会使用这种方式加载CUDA和cuDNN库。

   如果你不使用`模块`，则删除该项。

2. 我们转储环境变量。这可能至关重要，因为在某些环境中，单个环境变量如`LD_PRELOAD`或`LD_LIBRARY_PATH`可能会对性能产生巨大影响。

3. 然后我们转储conda环境包及其版本——这应该适用于任何虚拟Python环境。

4. 如果你使用`pip install -e .`进行开发安装，它除了知道从git SHA安装的仓库外，对其一无所知。但是问题是，你可能已经本地修改了文件，现在`pip freeze`将错过这些更改。因此这部分将遍历所有未安装到conda环境中的包（我们通过查找`site-packages/*.dist-info/direct_url.json`来找到它们）。