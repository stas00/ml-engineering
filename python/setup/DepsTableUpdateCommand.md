# Python setup.py extensions / goodies

This is part of the `setup.py` files

## Group dependencies and Maintain dependencies table and Runtime dependency checks

[DepsTableUpdateCommand](DepsTableUpdateCommand.py) - this one implements - a custom distutils command to have a single source of all dependents package versions.

- to create/update the dependency table
```
python setup.py deps_table_update
```

If you need to quickly access the data from this table in a shell, you can do so easily with:
```
python -c 'import sys; from transformers.dependency_versions_table import deps; \
print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets
```

Just pass the desired package names to that script as it's shown with 2 packages above.

You can then feed this for example to `pip`:
```
 pip install -U $(python -c 'import sys; from transformers.dependency_versions_table import deps; \
print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets)
```

This setup also shows another way to maintain a complex set of dependencies.

## check dependencies at run time

To check at run time:
```

import sys

from .dependency_versions_table import deps
from .utils.versions import require_version, require_version_core


# define which module versions we always want to check at run time
# (usually the ones defined in `install_requires` in setup.py)
#
# order specific notes:
# - tqdm must be checked before tokenizers

pkgs_to_check_at_runtime = "python tqdm regex sacremoses requests packaging filelock numpy tokenizers".split()
if sys.version_info < (3, 7):
    pkgs_to_check_at_runtime.append("dataclasses")
if sys.version_info < (3, 8):
    pkgs_to_check_at_runtime.append("importlib_metadata")

for pkg in pkgs_to_check_at_runtime:
    if pkg in deps:
        if pkg == "tokenizers":
            # must be loaded here, or else tqdm check may fail
            from .file_utils import is_tokenizers_available

            if not is_tokenizers_available():
                continue  # not required, check version only if installed

        require_version_core(deps[pkg])
    else:
        raise ValueError(f"can't find {pkg} in {deps.keys()}, check dependency_versions_table.py")
```


where `utils/versions.py` is:

```

import operator
import re
import sys
from typing import Optional

from packaging import version


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


ops = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


def _compare_versions(op, got_ver, want_ver, requirement, pkg, hint):
    if got_ver is None:
        raise ValueError("got_ver is None")
    if want_ver is None:
        raise ValueError("want_ver is None")
    if not ops[op](version.parse(got_ver), version.parse(want_ver)):
        raise ImportError(
            f"{requirement} is required for a normal functioning of this module, but found {pkg}=={got_ver}.{hint}"
        )


def require_version(requirement: str, hint: Optional[str] = None) -> None:
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.
    The installed module version comes from the `site-packages` dir via `importlib_metadata`.
    Args:
        requirement (:obj:`str`): pip style definition, e.g.,  "tokenizers==0.9.4", "tqdm>=4.27", "numpy"
        hint (:obj:`str`, `optional`): what suggestion to print in case of requirements not being met
    Example::
       require_version("pandas>1.1.2")
       require_version("numpy>1.18.5", "this is important to have for whatever reason")
    """

    hint = f"\n{hint}" if hint is not None else ""

    # non-versioned check
    if re.match(r"^[\w_\-\d]+$", requirement):
        pkg, op, want_ver = requirement, None, None
    else:
        match = re.findall(r"^([^!=<>\s]+)([\s!=<>]{1,2}.+)", requirement)
        if not match:
            raise ValueError(
                f"requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23, but got {requirement}"
            )
        pkg, want_full = match[0]
        want_range = want_full.split(",")  # there could be multiple requirements
        wanted = {}
        for w in want_range:
            match = re.findall(r"^([\s!=<>]{1,2})(.+)", w)
            if not match:
                raise ValueError(
                    f"requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23, but got {requirement}"
                )
            op, want_ver = match[0]
            wanted[op] = want_ver
            if op not in ops:
                raise ValueError(f"{requirement}: need one of {list(ops.keys())}, but got {op}")

    # special case
    if pkg == "python":
        got_ver = ".".join([str(x) for x in sys.version_info[:3]])
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
        return

    # check if any version is installed
    try:
        got_ver = importlib_metadata.version(pkg)
    except importlib_metadata.PackageNotFoundError:
        raise importlib_metadata.PackageNotFoundError(
            f"The '{requirement}' distribution was not found and is required by this application. {hint}"
        )

    # check that the right version is installed if version number or a range was provided
    if want_ver is not None:
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
```
