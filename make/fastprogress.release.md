# Build and Release Instructions



## One Time Preparation

You can skip this step if you have done it once already on the system you're making the release from.

1. You need to register (free) with:

    - [PyPI](https://pypi.org/account/register/)
    - [TestPyPI](https://test.pypi.org/account/register/)
    - [anaconda.org](https://anaconda.org/)

    After registration, to upload to fastai project, you will need to ask Jeremy to add your username to PyPI and anaconda.

2. Create file `~/.pypirc` with the following content:

    ```
    [distutils]
    index-servers =
      pypi
      testpypi

    [testpypi]
    repository: https://test.pypi.org/legacy/
    username: your testpypi username
    password: your testpypi password

    [pypi]
    username: your pypi username
    password: your pypi password
    ```

3. You can also setup your client to have transparent access to anaconda tools, see https://anaconda.org/YOURUSERNAME/settings/access (adjust the url to insert your username).

    You don't really need it, as the anaconda client cashes your credentials so you need to login only infrequently.





## Quick Release Process

No matter which release process you follow, always remember to start with:

```
git pull
```

This does it all:
```
make release
```

If this was a bug fix, remember to update `fastai` dependency files: `conda/meta.yaml` and `setup.py` and also `imports/core.py` with this release's `fastprogress` version number.

## TL;DR

Here is a step-by-step release process. You can follow these steps from the beginning, or if something goes wrong during the auto-pilot simply continue with where it left off.

1. Install/update build tools

   ```
   make tools-update
   ```

2. Test code:
   ```
   make git-pull
   make test
   make git-not-dirty || echo "Commit changes before proceeding"
   ```

   The next stage requires a clean tree to start with, so commit any uncommitted code. If you `git stash` make sure to rerun `make test`.

3. Bump and Tag and Commit:

   ```
   make git-not-dirty && make bump && make commit-tag
   ```

   This will do patch-level bump, for major/minor bump targets see below.

4. Release:

   ```
   make dist
   make upload
   ```

5. Test uploads by installing them:

   ```
   make test-install
   ```

6. Update fastai repo

   If this was a bug fix or a change of API, update the following four `fastai` dependency files with this release's `fastprogress` version number:
   1. `imports/core.py`
   2. `conda/meta.yaml`
   3. `setup.py`
   4. `docs/install.md` (#custom-dependencies)



## Detailed information

The following is needed if the combined release instructions were failing. So that each step can be done separately.


### Bump the version

You can either edit `fastprogress/version.py` and change the version number by hand.

Or run one of these `make` targets:

   Target            | Function
   ------------------| --------------------------------------------
   bump-major        | bump major level; remove .devX if any
   bump-minor        | bump minor level; remove .devX if any
   bump-patch        | bump patch level unless has .devX, then don't bump, but remove .devX
   bump              | alias to bump-patch (as it's used often)
   bump-post-release | add .post1 or bump post-release level .post2, .post3, ...
   bump-major-dev    | bump major level and add .dev0
   bump-minor-dev    | bump minor level and add .dev0
   bump-patch-dev    | bump patch level and add .dev0
   bump-dev          | alias to bump-patch-dev (as it's used often)

e.g.:

```
make bump
```

We use the semver version convention w/ python adjustment to `.devX`, instead of `-devX`:

* release: `major.minor.patch`, 0.1.10
* dev or rc: `major.minor.patch.devX`, 0.1.10.dev0

For fastprogress, due to its simplicity and usage, there is probably no need for intermediary `.devX` stage. So just normal `bump` will do when a new version is released.



### PyPI details

To build a PyPI package and release it on [pypi.org/](https://pypi.org/project/fastprogress/):

1. Build the pip packages (source and wheel)

   ```
   make dist-pypi
   ```

2. Publish:

   ```
   make upload-pypi
   ```

   Note: PyPI won't allow re-uploading the same package filename, even if it's a minor fix. If you delete the file from pypi or test.pypi it still won't let you do it. So either a patch-level version needs to be bumped (A.B.C++) or some [post release string added](https://www.python.org/dev/peps/pep-0440/#post-releases) in `version.py`.

3. Test that the uploaded package is found and gets installed:

   Test the webpage so that the description looks correct: [https://pypi.org/project/fastprogress/](https://pypi.org/project/fastprogress/)

   Test installation:

   ```
   pip install fastprogress
   ```



### Conda details

To build a Conda package and release it on [anaconda.org](https://anaconda.org/fastai/fastprogress):

1. Build the fastprogress conda package:

   ```
   make dist-conda

   ```

2. Upload

   ```
   make upload-conda

   ```

3. Test that the uploaded package is found and gets installed:

   Test the webpage so that the description looks correct: [https://pypi.org/project/fastprogress/](https://pypi.org/project/fastprogress/)

   Test installation:

   ```
   conda install -c fastai fastprogress
   ```

### Others

`make clean` removes any intermediary build artifacts.

`make` will show all possible targets with a short description of what they do.
