# tools - 中文翻译

# 调试工具

## 与 Git 相关的工具


### 有用的别名

显示当前分支中所有相对于 HEAD 修改过的文件的差异：
```bash
alias brdiff="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); git diff origin/\$def_branch..."
```

忽略空白差异，添加 `--ignore-space-at-eol` 或 `-w`：
```bash
alias brdiff-nows="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); git diff -w origin/\$def_branch..."
```

列出当前分支中相对于 HEAD 添加或修改的所有文件：
```bash
alias brfiles="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); git diff --name-only origin/\$def_branch..."
```

有了这个列表后，我们可以自动打开编辑器来加载这些添加和修改的文件：
```bash
alias bremacs="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); emacs \$(git diff --name-only origin/\$def_branch...) &"
```


### git-bisect

（注：这是从 `the-art-of-debugging/methodology.md` 同步的内容，该文件是真正的源文件）

接下来讨论的方法应该适用于支持二分查找的任何版本控制系统。我们将在此讨论中使用 `git bisect`。

`git bisect` 可以帮助快速找到导致特定问题的提交。

使用场景：假设您使用了 `transformers==4.33.0`，然后您需要一个较新的功能，因此升级到了最新的 `transformers@main`，您的代码崩溃了。在这两个版本之间可能有数百个提交，通过逐一查看所有提交来找到导致崩溃的正确提交会非常困难。以下是如何快速找出导致问题的提交的方法。

脚注：HuggingFace Transformers 实际上在不频繁出现问题方面做得相当好，但由于其复杂性和庞大性，问题仍然会发生，并且一旦报告，问题会很快得到解决。由于它是一个非常流行的机器学习库，因此它是很好的调试用例。

解决方案：在已知的正确提交和错误提交之间进行二分查找，以找到导致问题的提交。

我们将使用两个 shell 终端：A 和 B。终端 A 将用于 `git bisect`，终端 B 用于测试您的软件。虽然您可以只使用一个终端完成任务，但使用两个终端更容易操作。

1. 在终端 A 中克隆 git 仓库并将其安装到您的 Python 环境中（开发模式）：
```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install -e .
```
现在，运行应用程序时会自动使用此克隆中的代码，而不是之前从 PyPi、Conda 或其他地方安装的版本。

同样为了简化，我们假设所有依赖项已经安装完毕。

2. 接下来启动二分查找 - 在终端 A 中运行：

```bash
git bisect start
```

3. 发现最后一个已知正确的提交和第一个已知错误的提交

`git bisect` 需要两个数据点才能正常工作。它需要知道一个较早的已知可用的提交（“好”提交）和一个较晚的已知有问题的提交（“坏”提交）。因此，如果您查看某个分支上的提交序列，它会有两个已知点和许多未知质量的提交：

```bash
...... orig_good ..... .... .... .... ..... orig_bad ....
------------->---------------->----------------> 时间
```

例如，如果您知道 `transformers==4.33.0` 是好的，而 `transformers@main` (`HEAD`) 是坏的，您可以在 [发布页面](https://github.com/huggingface/transformers/releases) 上查找对应于标签 `4.33.0` 的提交。我们发现这是一个 SHA 为 `[5a4f340d]` 的提交。

脚注：通常前 8 个十六进制字符足以作为给定仓库的唯一标识符，但您也可以使用完整的 40 字符字符串。

所以现在我们指定第一个已知的正确提交：
```bash
git bisect good 5a4f340d
```

正如我们所说，我们将使用 `HEAD`（最新提交）作为坏的提交，在这种情况下，我们可以直接使用 `HEAD` 而不需要找到对应的 SHA 字符串：
```bash
git bisect bad HEAD
```

如果已知问题出现在 `4.34.0` 版本，您可以按照上述方法找到其最新的提交并使用该提交代替 `HEAD`。

我们现在准备找出导致问题的提交。

在告诉 `git bisect` 好的和坏的提交之后，它已经切换到了中间的一个提交：

```bash
...... orig_good ..... .... current .... .... ..... orig_bad ........
------------->--------------->---------------->----------------> 时间
```

您可以运行 `git log` 查看它切换到了哪个提交。

并且提醒一下，我们已经使用 `pip install -e .` 安装了这个仓库，因此 Python 环境会即时更新到当前提交的代码版本。

4. 好或坏

下一步是告诉 `git bisect` 当前提交是“好”还是“坏”：

为此，在终端 B 中运行一次您的程序。

然后在终端 A 中运行：
```bash
git bisect bad
```
如果失败，或者：
```bash
git bisect good
```
如果成功。

如果，例如，结果是坏的，`git bisect` 会在内部将最后一个提交标记为新坏的，并再次对半分割提交，切换到一个新的当前提交：
```bash
...... orig_good ..... current .... new_bad .... ..... orig_bad ....
------------->--------------->---------------->----------------> 时间
```

反之，如果结果是好的，那么您将有：
```bash
...... orig_good ..... .... new_good .... current ..... orig_bad ....
------------->--------------->---------------->----------------> 时间
```

5. 重复直到没有更多提交

继续重复第 4 步直到找到有问题的提交。

完成二分查找后，`git bisect` 会告诉您是哪个提交导致了问题。

```bash
...... orig_good ..... .... last_good first_bad .... .. orig_bad ....
------------->--------------->---------------->----------------> 时间
```
如果您遵循了小的提交图示，它将对应于 `first_bad` 提交。

您可以通过访问 `https://github.com/huggingface/transformers/commit/` 并附加提交的 SHA 到该 URL 来访问该提交（例如 `https://github.com/huggingface/transformers/commit/57f44dc4288a3521bd700405ad41e90a4687abc0`），这将链接到产生该提交的 PR。然后，您可以通过跟进该 PR 请求帮助。

如果您编写的程序即使有数千个提交需要搜索也不会花费太长时间，那么您面临的将是 `n` 次二分查找步骤，从 `2**n` 开始。因此，1024 个提交可以在 10 步内被搜索到。

如果您的程序运行非常慢，尝试将其减少到一些小的程序——理想情况下是一个能快速展示问题的小型复现程序。通常，注释掉大量看似与问题无关的代码就足够了。

如果您想查看进度，可以要求它显示剩余待检查的提交范围：
```bash
git bisect visualize --oneline
```

6. 清理

现在将 git 仓库克隆恢复到您开始时的状态（最有可能是 `HEAD`）：
```bash
git bisect reset
```

并在向维护人员报告问题的同时重新安装好版本的库。

有时，问题可能是由有意的向后不兼容的 API 更改引起的，您可能只需要阅读项目的文档来看看发生了什么变化。例如，如果您从 `transformers==2.0.0` 切换到 `transformers==3.0.0`，几乎可以肯定您的代码会崩溃，因为主版本号的变化通常表示引入了主要的 API 更改。

7. 可能的问题及其解决方案：

a. 跳过

如果出于某种原因当前提交无法测试 - 可以跳过它：
```bash
git bisect skip
```
`git bisect` 将继续对剩余的提交进行二分查找。

这在中间某个提交的 API 发生更改，您的程序因完全不同的原因开始失败时非常有用。

您还可以尝试编写一个适应新 API 的程序变体并使用它，但这并不总是容易做到的。

b. 反转顺序

通常 `git` 预期“坏”提交在“好”提交之后。


```bash
...... orig_good ..... .... .... .... ..... orig_bad ....
------------->--------------->---------------->----------------> 时间
```

现在，如果“坏”的修订版本在“好”的修订版本之前，并且您想找到修复先前存在的问题的第一个修订版本 - 您可以反转“好”和“坏”的定义——使用新的状态集可能会更清晰——例如，“修复”和“损坏”。以下是具体操作方法。

```bash
git bisect start --term-new=fixed --term-old=broken
git bisect fixed
git bisect broken 6c94774
```
然后使用：
```bash
git fixed / git broken
```
而不是：
```bash
git good / git bad
```

c. 复杂情况

有时还有其他复杂情况，比如不同修订版本的依赖项不一样，例如一个修订版本可能需要 `numpy=1.25`，而另一个需要 `numpy=1.26`。如果依赖包的版本是向后兼容的，安装新版本应该就可以了。但这并不总是可行。因此，有时在重新测试程序之前必须重新安装正确的依赖项。

有时，当有一系列提交实际上以不同的方式损坏时，您可以要么找到不包括其他损坏范围的 `good...bad` 提交范围，要么像前面解释的那样尝试 `git bisect skip` 其他损坏的提交。