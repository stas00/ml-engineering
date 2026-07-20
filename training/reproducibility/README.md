# Reproducibility

## Achieve determinism in randomness based software

When debugging always set a fixed seed for all the used Random Number Generators (RNG) so that you get the same data / code path on each re-run.

On GPU, seeding alone isn't enough for bitwise-reproducible results: cuDNN and cuBLAS are free to pick different (but individually valid) algorithms from run to run for the same op, so a truly deterministic re-run also requires forcing them to always pick the same, deterministic ones - that's what the `cudnn.benchmark`, `use_deterministic_algorithms()` and `CUBLAS_WORKSPACE_CONFIG` settings below take care of.

There are many different systems involved, so it's tricky to cover them all - here is an attempt to cover the most common deterministic knobs:

```python
import os, random, torch, numpy as np
def enforce_reproducibility(use_seed=None):
    seed = use_seed if use_seed is not None else random.randint(1, 1000000)
    print(f"Using seed: {seed}")

    if use_seed is not None: # slower speed! https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
        # CUBLAS_WORKSPACE_CONFIG must be set before any CUDA context gets created, i.e. before
        # the torch.cuda.* calls below - if CUDA is initialized earlier elsewhere it's too late,
        # and silently so, hence the explicit assert instead of letting it fail quietly
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ and torch.cuda.is_initialized():
            raise RuntimeError(
                "CUDA was already initialized before enforce_reproducibility() got to set "
                "CUBLAS_WORKSPACE_CONFIG - move this call earlier, or export "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 before the process starts"
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    random.seed(seed)    # python RNG
    np.random.seed(seed) # numpy RNG

    # pytorch RNGs
    torch.manual_seed(seed)          # cpu + cuda
    torch.cuda.manual_seed_all(seed) # multi-gpu - can be called without gpus

    return seed
```

Notes:

- Comment out whichever RNG seeding call you don't need - e.g. if you don't use `numpy` directly. However, some module you depend on might use it internally without your knowledge, so it's usually best to leave them all in and be safe.

- `CUBLAS_WORKSPACE_CONFIG` plugs a gap `use_deterministic_algorithms(True)` doesn't close on its own: cuBLAS - the library behind `torch.mm`/`mv`/`bmm` and other `matmul`-based ops - can be non-deterministic on CUDA >= 10.2, because it dynamically picks internal workspace memory for routines running in parallel CUDA streams on the same handle, so results can differ from run to run even with the same seed. Without this env var set, `use_deterministic_algorithms(True)` will raise a `RuntimeError` the moment it hits a `matmul` op - that's your signal it needs to be set. It **must be set before the CUDA context is created**, i.e. before the first CUDA call; if set after CUDA has already been initialized elsewhere in the process, it's silently ignored, which is why it's set at the very top of `enforce_reproducibility()` above, ahead of the `torch.cuda.*` calls.

  Since a silently-ignored env var is easy to miss, [`torch.cuda.is_initialized()`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_initialized.html) is used above to `raise` instead, if `CUBLAS_WORKSPACE_CONFIG` isn't set yet and CUDA has already been lazily initialized by some earlier call (e.g. a `.cuda()` tensor, `torch.cuda.set_device()`, or another library touching CUDA first) by the time `enforce_reproducibility()` runs. Note that this only catches initialization that went through PyTorch's own lazy-init path - if some non-PyTorch library created a raw CUDA context first, `is_initialized()` won't know about it. Therefore, try to run `enforce_reproducibility()` as early as possible in your program.

  NVIDIA's cuBLAS docs sanction exactly two values for deterministic behavior, and both involve a memory-vs-performance tradeoff - a smaller workspace leaves cuBLAS with fewer algorithm choices, which can limit performance, while a larger workspace gives it more room to pick a faster algorithm at the cost of extra GPU memory:
  - `:16:8` - deterministic, minimal extra memory, but the smaller workspace may limit performance
  - `:4096:8` - deterministic, costs ~24MiB of extra GPU memory, but leaves more performance headroom (the more commonly used of the two, and what's used above)

  There's no third, cost-free option - determinism is structurally at odds with letting cuBLAS freely pick whichever algorithm/workspace layout is fastest across concurrent streams, so pinning it down always gives up some of that freedom. `:4096:8` tends to have the smaller performance penalty of the two (more workspace room means fewer algorithms get ruled out), but "smaller" isn't "zero" - the actual hit is workload- and `matmul`-shape-dependent, and can range from negligible to significant (some reports show throughput cut roughly in half on specific workloads).

  The general format is `:[SIZE]:[COUNT]`, and you'll see it chained into multiple `SIZE:COUNT` pairs (e.g. the default, unset behavior is equivalent to `CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8`). That chained form comes from cuBLAS's own internal workspace-pool config, where each pair originally described a distinct bucket size/count so the pool could efficiently serve both large and small workspace requests. PyTorch, however, just sums every `SIZE * COUNT` pair it finds into a single total byte count and hands that one fixed-size buffer to cuBLAS - so `:4096:2:16:8` simply means `2*4096 + 8*16` KiB in total, and chaining more pairs only changes *how many KiB you add up*, not the determinism behavior. In practice you don't need to chain anything yourself: pick one of NVIDIA's two blessed single-pair values above, or set `:0:0` to force cuBLAS to not use any workspace at all (not recommended - this is the most restrictive and slowest option).

- `torch.use_deterministic_algorithms(True)` makes cuDNN convolutions deterministic and it additionally covers hundreds of other ops throughout PyTorch (`index_add_`, `scatter_`, interpolation, etc.). Instead of silently doing its best, it raises a `RuntimeError` the moment it hits an op with no deterministic implementation, so nondeterminism can't sneak in unnoticed.

- `torch.backends.cudnn.benchmark = False` is a separate, complementary knob that stops cuDNN from timing several convolution algorithms on the first call with each new input shape and caching whichever was fastest. That benchmarking step is itself a source of run-to-run variation - which algorithm "wins" the timing race can depend on incidental system noise - independently of whether the algorithm it settles on is deterministic.

> **PyTorch version note**: `torch.use_deterministic_algorithms()` requires **PyTorch >= 1.8**. On PyTorch < 1.8, fall back to the narrower `torch.backends.cudnn.deterministic = True`, which only makes cuDNN convolutions deterministic and leaves everything else (`matmul`s, `index_add_`, `scatter_`, etc.) untouched - `CUBLAS_WORKSPACE_CONFIG` is independent of the PyTorch version and still applies:
> ```python
> if use_seed:
>     os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
>     torch.backends.cudnn.benchmark     = False
>     torch.backends.cudnn.deterministic = True # PyTorch < 1.8 fallback - cuDNN convolutions only
> ```

Resources:
- [NVIDIA cuBLAS docs - Reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility) - the canonical explanation of cuBLAS non-determinism and the accepted `CUBLAS_WORKSPACE_CONFIG` values
- [`torch.use_deterministic_algorithms`](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html) - the full list of affected ops and how `warn_only` works
- [PyTorch CUDA environment variables reference](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html) - the exact `:[SIZE]:[COUNT]` format spec for `CUBLAS_WORKSPACE_CONFIG`
- [PyTorch Reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html) - ties this together with seeding, `cudnn.deterministic` and `cudnn.benchmark` for the full picture

If your code also touches other accelerator backends or frameworks besides CPU/CUDA - e.g. NPU, XPU, or TensorFlow - seed their RNGs too:
```python
    torch.npu.manual_seed_all(seed)
    torch.xpu.manual_seed_all(seed)
    tf.random.set_seed(seed)
```

When you rerun the same code again and again to solve some problem, set a specific seed at the beginning of your code with:
```
enforce_reproducibility(42)
```

As mentioned above, **this is for debug only**: it activates various `torch` flags that help with determinism, but can slow things down, so you don't want this in production.

However, you can call this instead to use in production:
```
enforce_reproducibility()
```
i.e. w/o the explicit seed. And then it'll pick a random seed and log it! So if something happens in production, you can now reproduce the same RNG state the issue occurred under. And no performance penalty this time, as the determinism settings (`cudnn.benchmark` / `use_deterministic_algorithms` / `CUBLAS_WORKSPACE_CONFIG`) are only set if you provided the seed explicitly. Say it logged:
```
Using seed: 1234
```
you then just need to change the code to:
```
enforce_reproducibility(1234)
```
and you will get the same RNGs setup.

As mentioned earlier, there could be many other RNGs involved in a system - for example, if you want the data to be fed in the same order for a `DataLoader`, you need [to have its seed set as well](https://pytorch.org/docs/stable/notes/randomness.html#dataloader).

See also [Floating point math discrepancies on different devices](../../debug/pytorch.md#floating-point-math-discrepancies-on-different-devices) in the debugging chapter.

## Reproduce the software and system environment

This methodology is useful when discovering some discrepancy in outcomes - quality or a throughput for example.

The idea is to log the key components of the environment used to launch a training (or inference) so that if at a later stage it needs to be reproduced exactly as it was it can be done.

Since there is a huge variety of systems and components being used it's impossible to prescribe a way that will always work. So let's discuss one possible recipe and you can then adapt it to your particular environment.

This is added to your slurm launcher script (or whatever other way you use to launch the training) - this is Bash script:

```bash
SAVE_DIR=/tmp # edit to a real persistent path
export REPRO_DIR=$SAVE_DIR/repro/$SLURM_JOB_ID
mkdir -p $REPRO_DIR
# 1. modules (writes to stderr) (remove if you don't use lmod or similar modules implementation)
module list 2> $REPRO_DIR/modules.txt
# 2. shell env vars
/usr/bin/printenv | sort > $REPRO_DIR/env.txt
# 3. pip (this includes devel installs SHA)
pip freeze > $REPRO_DIR/requirements.txt
# 4. uncommitted diff in git clones installed into conda
perl -nle 'm|"file://(.*?/([^/]+))"| && qx[cd $1; if [ ! -z "\$(git diff)" ]; then git diff > \$REPRO_DIR/$2.diff; fi]' $CONDA_PREFIX/lib/python*/site-packages/*.dist-info/direct_url.json
```

As you can see this recipe is used in a SLURM environment, so every new training will dump the environment specific to the SLURM job. But you can adapt it to any other environment.

Let's expend on each step in the recipe:

1. We save which `modules` were loaded, e.g. in cloud cluster/HPC setups you're like to be loading the CUDA and cuDNN libraries using this.

   If you don't use `modules` then remove that entry

2. We dump the shell environment variables. This can be crucial since a single env var like `LD_PRELOAD` or `LD_LIBRARY_PATH` could make a huge impact on performance in some environments

3. We then dump the python environment packages and their versions - this should work with any virtual python environment like `conda`, `venv` or even if you don't use a virtual environment.

   If you use `uv` instead of `pip`, switch to `uv pip freeze` as it'd be much faster.

4. If you use a devel install with `pip install -e .` it doesn't know anything about the git clone repository it was installed from other than its git SHA. But the issue is that it's likely that you have modified the files locally and now `pip freeze` will miss those changes. So this part will go through all packages that are not installed into the conda environment (we find them by looking inside `site-packages/*.dist-info/direct_url.json`) and, for each one that has uncommitted changes, save its `git diff` to `$REPRO_DIR/<package>.diff`.

To save the `apt` packages add:
```bash
apt list --installed > $REPRO_DIR/apt-packages.txt
```

If using `conda` an additionally useful tool is [conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md) which helps you to find out the exact differences 2 conda environments have.

Anecdotally, me and my colleague were getting very different training TFLOPS on a cloud cluster running the exact same code - literally launching the same slurm script from the same shared directory. We first compared our conda environments using [conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md) and found some differences - I installed the exact packages she had to match her environment and it was still showing a huge performance difference. We then compared the output of `printenv` and discovered that I had `LD_PRELOAD` set up whereas she didn't - and that made a huge difference since this particular cloud provider required multiple env vars to be set to custom paths to get the most of their hardware.
