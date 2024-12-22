# README - 中文翻译

## 运行测试

### 运行所有测试

```console
pytest
```

我使用以下别名：

```bash
alias pyt="pytest --disable-warnings --instafail -rA"
```

这告诉pytest：

- 禁用警告
- `--instafail` 在发生时显示失败，而不是在最后
- `-rA` 生成简短的测试摘要信息

它需要安装：
```bash
pip install pytest-instafail
```

### 获取所有测试列表

显示测试套件中的所有测试：

```bash
pytest --collect-only -q
```

显示给定测试文件中的所有测试：

```bash
pytest tests/test_optimization.py --collect-only -q
```

我使用以下别名：

```bash
alias pytc="pytest --disable-warnings --collect-only -q"
```

### 运行特定测试模块

运行单个测试模块：

```bash
pytest tests/utils/test_logging.py
```

### 运行特定测试

如果使用`unittest`，你需要知道包含这些测试的`unittest`类的名称。例如：

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

这里：

- `tests/test_optimization.py` - 包含测试的文件
- `OptimizationTest` - 测试类的名称
- `test_adam_w` - 具体测试函数的名称

如果文件包含多个类，你可以选择运行特定类的所有测试。例如：

```bash
pytest tests/test_optimization.py::OptimizationTest
```

这将运行该类中的所有测试。

如前所述，可以通过运行：

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

来查看`OptimizationTest`类中包含的测试。

可以通过关键字表达式运行测试。

仅运行名称包含`adam`的测试：

```bash
pytest -k adam tests/test_optimization.py
```

逻辑“与”和“或”可以用来指示是否所有关键字都必须匹配，或者只需其中一个匹配。“不”可以用来否定。

仅运行名称不包含`adam`的测试：

```bash
pytest -k "not adam" tests/test_optimization.py
```

你还可以组合两种模式：

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

例如，要运行`test_adafactor`和`test_adam_w`，可以使用：

```bash
pytest -k "test_adafactor or test_adam_w" tests/test_optimization.py
```

注意这里我们使用了“或”，因为我们希望任一关键字匹配即可包括两者。

如果你想只包括同时匹配两个模式的测试，应使用“与”：

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### 仅运行修改过的测试

通过使用[pytest-picked](https://github.com/anapaulagomes/pytest-picked)，你可以运行与未提交文件或当前分支相关的测试（根据Git）。这是一种快速验证你的更改没有破坏任何内容的好方法，因为它不会运行你不碰的文件相关的测试。

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

所有测试将从已修改但尚未提交的文件和文件夹中运行。

### 自动重新运行失败的测试

[pytest-xdist](https://github.com/pytest-dev/pytest-xdist) 提供了一个非常有用的功能：检测所有失败的测试，并等待你修改文件并持续重新运行这些失败的测试，直到它们通过，而你修复它们。因此，你无需在修复后重新启动pytest。这会重复进行，直到所有测试通过后再次进行全面运行。

```bash
pip install pytest-xdist
```

进入模式：`pytest -f` 或 `pytest --looponfail`

通过检查 `looponfailroots` 根目录及其所有内容（递归地）来检测文件变化。
如果此值的默认设置对你不起作用，你可以通过在项目中设置配置选项来更改它：

```ini
[tool:pytest]
looponfailroots = transformers tests
```

或在 `pytest.ini`/`tox.ini` 文件中：

```ini
[pytest]
looponfailroots = transformers tests
```

这将仅查找指定目录中的文件变化，相对于 ini 文件所在的目录。

[pytest-watch](https://github.com/joeyespo/pytest-watch) 是这一功能的另一种实现方式。

### 跳过测试模块

如果你只想运行部分测试模块，可以排除一些模块。例如，要运行除了 `test_modeling_*.py` 测试之外的所有测试：

```bash
pytest $(ls -1 tests/*py | grep -v test_modeling)
```

### 清除状态

CI 构建和在重要性高于速度的情况下应清除缓存：

```bash
pytest --cache-clear tests
```

### 并行运行测试

如前所述，`make test` 通过 `pytest-xdist` 插件并行运行测试（`-n X` 参数，例如 `-n 2` 表示运行 2 个并行任务）。

`pytest-xdist` 的 `--dist=` 选项允许控制测试如何分组。`--dist=loadfile` 将位于同一文件中的测试分配到同一进程。

由于执行顺序不同且不可预测，如果使用 `pytest-xdist` 运行测试套件产生失败（意味着我们有一些未检测到的耦合测试），可以使用 [pytest-replay](https://github.com/ESSS/pytest-replay) 以相同的顺序重播测试，这有助于减少该失败序列到最小。

### 测试顺序和重复

重复测试几次，按顺序、随机或分组，以检测潜在的相互依赖性和状态相关错误（清理）。直接多次重复测试只是好用于检测由深度学习的随机性暴露的问题。

#### 重复测试

- [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder)：

```bash
pip install pytest-flakefinder
```

然后运行每个测试多次（默认为50次）：

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

脚注：此插件不与 `pytest-xdist` 的 `-n` 标志一起工作。

脚注：还有一个插件 `pytest-repeat`，但它不适用于 `unittest`。

#### 随机顺序运行测试

```bash
pip install pytest-random-order
```

重要：`pytest-random-order` 的存在会自动随机化测试，无需配置更改或命令行选项。

如前所述，这允许检测耦合测试 - 其中一个测试的状态会影响另一个的状态。当安装了 `pytest-random-order` 时，它将打印该会话使用的随机种子，例如：

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

因此，如果给定特定序列失败，你可以通过添加精确的种子来重现它，例如：

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

只有当你使用完全相同的测试列表（或根本没有列表）时才会重现确切的顺序。

