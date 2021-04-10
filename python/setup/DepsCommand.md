# Python setup.py extensions / goodies

This is part of the `setup.py` files

## Report selective package dependency groups

[DepsCommand](DepsCommand.py) - this one implements - a custom distutils command to print selective dependency groups. Other than via `pip` it's very difficult to get the requirements out by running `setup.py`, so this `distutils` command extends `setup.py` to support

```
    # show available dependency groups:
    python setup.py -q deps

    # print dependency list for specified groups
    python setup.py -q deps --dep-groups=core,vision

```

This is useful if you want to feed these outputs directly to `pip`:
```
pip install $(python setup.py -q deps --dep-groups=core,vision)
```
While we can do the same with `pip install package[core,vision]` it forces the installation of the main package. This allows you install just the dependencies.

or if you just want to know which `extras` are available, e.g.:
```
$ python setup.py -q deps
Available dependency groups: core, text, vision
```
The command was originally developed for [fastai v1](https://github.com/fastai/fastai1).

if you want to use it just rename it `setup.py` and adapt to your project, or copy-n-paste the parts that you need.

Here is the full original doc:

### Custom dependencies

If for any reason you don't want to install all of `fastai`'s dependencies, since, perhaps, you have limited disk space on your remote instance, here is how you can install only the dependencies that you need.

1. First, install `fastai` without its dependencies using either `pip` or `conda`:

   ```
   # pip
   pip install --no-deps fastai==1.0.61
   # conda
   conda install --no-deps -c fastai fastai=1.0.61
   ```

2. The rest of this section assumes you're inside the `fastai` git repo, since that's where `setup.py` resides. If you don't have the repository checked out, do:

   ```
   git clone https://github.com/fastai/fastai1
   cd fastai
   tools/run-after-git-clone
   ```

3. Next, find out which groups of dependencies you want:

   ```
   python setup.py -q deps
   ```
   You should get something like:
   ```
   Available dependency groups: core, text, vision
   ```

   You need to use at least the `core` group.

   Do note that the `deps` command is a custom `distutils` extension, i.e. it only works in the `fastai` setup.

4. Finally, install the custom dependencies for the desired groups.

   For the sake of this demonstration, let's say you want to get the core dependencies (`core`), plus dependencies specific to computer vision (`vision`). The following command will give you the up-to-date dependencies for these two groups:

   ```
   python setup.py -q deps --dep-groups=core,vision
   ```
   It will return something like:
   ```
   Pillow beautifulsoup4 bottleneck dataclasses;python_version<'3.7' fastprogress>=0.1.18 matplotlib numexpr numpy>=1.12 nvidia-ml-py3 packaging pandas pyyaml requests scipy torch>=1.0.0 torchvision typing
   ```
   which can be fed directly to `pip install`:

   ```
   pip install $(python setup.py -q deps --dep-groups=core,vision)
   ```

   Since conda uses a slightly different syntax/package names, to get the same output suitable for conda, add `--dep-conda`:

   ```
   python setup.py -q deps --dep-groups=core,vision --dep-conda
   ```

   If your shell doesn't support `$()` syntax, it most likely will support backticks, which are deprecated in modern `bash`. (The two are equivalent, but `$()` has a superior flexibility). If that's your situation, use the following syntax instead:

   ```
   pip install `python setup.py -q deps --dep-groups=core,vision`
   ```

* Manual copy-n-paste case:

   If, instead of feeding the output directly to `pip` or `conda`, you want to do it manually via copy-n-paste, you need to quote the arguments, in which case add the `--dep-quote` option, which will do it for you:

   ```
   # pip:
   python setup.py -q deps --dep-groups=core,vision --dep-quote
   # conda:
   python setup.py -q deps --dep-groups=core,vision --dep-quote --dep-conda
   ```

   So the output for pip will look like:
   ```
   "Pillow" "beautifulsoup4" "bottleneck" "dataclasses;python_version<'3.7'" "fastprogress>=0.1.18" "matplotlib" "numexpr" "numpy>=1.12" "nvidia-ml-py3" "packaging" "pandas" "pyyaml" "requests" "scipy" "torch>=1.0.0" "torchvision" "typing"
   ```

* Summary:

   pip selective dependency installation:
   ```
   pip install --no-deps fastai==1.0.61
   pip install $(python setup.py -q deps --dep-groups=core,vision)
   ```

   same for conda:
   ```
   conda install --no-deps -c fastai fastai=1.0.61
   conda install -c pytorch -c fastai $(python setup.py -q deps --dep-conda --dep-groups=core,vision)
   ```

   adjust the `--dep-groups` argument to match your needs.


* Full usage:

   ```
   # show available dependency groups:
   python setup.py -q deps

   # print dependency list for specified groups
   python setup.py -q deps --dep-groups=core,vision

   # see all options:
   python setup.py -q deps --help
   ```
