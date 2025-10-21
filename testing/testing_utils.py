# I developed the bulk of this library while I worked at HF

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio  # noqa
import contextlib
import importlib.util
import inspect
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import unittest
from distutils.util import strtobool
from io import StringIO
from pathlib import Path
from typing import Iterator, Union
from unittest import mock
from unittest.case import SkipTest

import numpy as np
from packaging import version
from parameterized import parameterized


try:
    import torch

    _torch_available = True
except Exception:
    _torch_available = False


def is_torch_available():
    return _torch_available


def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        # KEY isn't set, default to `default`.
        _value = default
    else:
        # KEY is set, convert it to True or False.
        try:
            _value = strtobool(value)
        except ValueError:
            # More values are supported, but let's keep the message simple.
            raise ValueError(f"If set, {key} must be yes or no.")
    return _value


def parse_int_from_env(key, default=None):
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = int(value)
        except ValueError:
            raise ValueError(f"If set, {key} must be a int.")
    return _value


def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)
    else:
        return test_case


def require_torch_no_gpus(test_case):
    """
    Decorator marking a test that requires a setup without GPUs (in PyTorch). These tests are skipped on a machine with GPUs.

    To run *only* the no gpu tests, assuming all test names contain no_gpu: $ pytest -sv ./tests -k "no_gpu"
    """
    import torch

    if is_torch_available() and torch.cuda.device_count() > 0:
        return unittest.skip("test requires an environment w/o GPUs")(test_case)
    else:
        return test_case


def require_torch_multi_gpu(test_case):
    """
    Decorator marking a test that requires a multi-GPU setup (in PyTorch). These tests are skipped on a machine without
    multiple GPUs.

    To run *only* the multi_gpu tests, assuming all test names contain multi_gpu: $ pytest -sv ./tests -k "multi_gpu"
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.cuda.device_count() < 2:
        return unittest.skip("test requires multiple GPUs")(test_case)
    else:
        return test_case


def require_torch_non_multi_gpu(test_case):
    """
    Decorator marking a test that requires 0 or 1 GPU setup (in PyTorch).
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.cuda.device_count() > 1:
        return unittest.skip("test requires 0 or 1 GPU")(test_case)
    else:
        return test_case


def require_torch_up_to_2_gpus(test_case):
    """
    Decorator marking a test that requires 0 or 1 or 2 GPU setup (in PyTorch).
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.cuda.device_count() > 2:
        return unittest.skip("test requires 0 or 1 or 2 GPUs")(test_case)
    else:
        return test_case


if is_torch_available():
    # Set env var CUDA_VISIBLE_DEVICES="" to force cpu-mode
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    torch_device = None


def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch."""
    if torch_device != "cuda":
        return unittest.skip("test requires CUDA")(test_case)
    else:
        return test_case


def is_deepspeed_available():
    return importlib.util.find_spec("deepspeed") is not None


def require_deepspeed(test_case):
    """
    Decorator marking a test that requires deepspeed
    """
    if not is_deepspeed_available():
        return unittest.skip("test requires deepspeed")(test_case)
    else:
        return test_case


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


def require_bnb(test_case):
    """
    Decorator marking a test that requires bitsandbytes
    """
    if not is_bnb_available():
        return unittest.skip("test requires bitsandbytes from https://github.com/facebookresearch/bitsandbytes")(
            test_case
        )
    else:
        return test_case


def require_bnb_non_decorator():
    """
    Non-Decorator function that would skip a test if bitsandbytes is missing
    """
    if not is_bnb_available():
        raise SkipTest("Test requires bitsandbytes from https://github.com/facebookresearch/bitsandbytes")


def set_seed(seed: int = 42):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available


def get_gpu_count():
    """
    Return the number of available gpus (regardless of whether torch or tf is used)
    """
    if is_torch_available():
        import torch

        return torch.cuda.device_count()
    else:
        return 0


def torch_assert_equal(actual, expected, **kwargs):
    """
    Compare two tensors or non-tensor numbers for their equality.
    
    Add msg=blah to add an additional comment to when assert fails.
    """
    # assert_close was added around pt-1.9, it does better checks - e.g. will check that dimensions dtype, device and layout match
    return torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, **kwargs)


def torch_assert_close(actual, expected, **kwargs):
    """
    Compare two tensors or non-tensor numbers for their closeness.

    Add msg=blah to add an additional comment to when assert fails.

    For default values of `rtol` and `atol` which are dtype dependent, see the table at https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close
    For example for bf16 it is `rtol=1.6e-2` and `atol=1e-5`.
    
    The check doesn't assert when `|a - b| <= (atol + rtol * |b|)`
    """
    # assert_close was added around pt-1.9, it does better checks - e.g. will check that dimensions dtype, device and layout match
    return torch.testing.assert_close(actual, expected, **kwargs)


def is_torch_bf16_available():
    # from https://github.com/huggingface/transformers/blob/26eb566e43148c80d0ea098c76c3d128c0281c16/src/transformers/file_utils.py#L301
    if is_torch_available():
        import torch

        if not torch.cuda.is_available() or torch.version.cuda is None:
            return False
        if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
            return False
        if int(torch.version.cuda.split(".")[0]) < 11:
            return False
        if not version.parse(torch.__version__) >= version.parse("1.09"):
            return False
        return True
    else:
        return False


def require_torch_bf16(test_case):
    """Decorator marking a test that requires CUDA hardware supporting bf16 and PyTorch >= 1.9."""
    if not is_torch_bf16_available():
        return unittest.skip("test requires CUDA hardware supporting bf16 and PyTorch >= 1.9")(test_case)
    else:
        return test_case


def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path

    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.

    """
    # this function caller's __file__
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))
    if append_path:
        return os.path.join(tests_dir, append_path)
    else:
        return tests_dir


def parameterized_custom_name_func_join_params(func, param_num, param):
    """
    customize the test name generator function as we want all params to appear in the sub-test
    name, as by default it shows only the first param or for multiple params it just uses a unique sequence of ids and no params at all.

    Usage:

    @parameterized.expand(
        [
            (0, True),
            (0, False),
            (1, True),
        ],
        name_func=parameterized_custom_name_func_join_params,
    )
    def test_determinism_wrt_rank(self, num_workers, pad_dataset):

    which gives:

    test_determinism_wrt_rank_0_true
    test_determinism_wrt_rank_0_false
    test_determinism_wrt_rank_1_true

    """
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"

# When any function contains print() calls that get overwritten, like progress bars,
# a special care needs to be applied, since under pytest -s captured output (capsys
# or contextlib.redirect_stdout) contains any temporary printed strings, followed by
# \r's. This helper function ensures that the buffer will contain the same output
# with and without -s in pytest, by turning:
# foo bar\r tar mar\r final message
# into:
# final message
# it can handle a single string or a multiline buffer
def apply_print_resets(buf):
    return re.sub(r"^.*\r", "", buf, 0, re.M)

class CaptureStd:
    """
    Context manager to capture:

    - stdout: replay it, clean it up and make it available via ``obj.out``
    - stderr: replay it and make it available via ``obj.err``
    - combined: combined the chosen streams and make it available via ``obj.combined``

    init arguments:

    - out - capture stdout:`` True``/``False``, default ``True``
    - err - capture stdout: ``True``/``False``, default ``True``
    - replay - whether to replay or not: ``True``/``False``, default ``True``. By default each
    captured stream gets replayed back on context's exit, so that one can see what the test was
    doing. If this is a not wanted behavior and the captured data shouldn't be replayed, pass
    ``replay=False`` to disable this feature.

    Examples::

        # to capture stdout only with auto-replay
        with CaptureStdout() as cs:
            print("Secret message")
        assert "message" in cs.out

        # to capture stderr only with auto-replay
        import sys
        with CaptureStderr() as cs:
            print("Warning: ", file=sys.stderr)
        assert "Warning" in cs.err

        # to capture both streams with auto-replay
        with CaptureStd() as cs:
            print("Secret message")
            print("Warning: ", file=sys.stderr)
        assert "message" in cs.out
        assert "Warning" in cs.err

        # to capture just one of the streams, and not the other, with auto-replay
        with CaptureStd(err=False) as cs:
            print("Secret message")
        assert "message" in cs.out
        # but best use the stream-specific subclasses

        # to capture without auto-replay
        with CaptureStd(replay=False) as cs:
            print("Secret message")
        assert "message" in cs.out

        # sometimes it's easier to not try to figure out if it's stdout or stderr, and yet at
        # other times the software may send the same output to stderr or stdout depending on
        # environment, so to make the test robust a combined entry of both streams is available

    """

    def __init__(self, out=True, err=True, replay=True):
        self.replay = replay

        if out:
            self.out_buf = StringIO()
            self.out = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.out_buf = None
            self.out = "not capturing stdout"

        if err:
            self.err_buf = StringIO()
            self.err = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.err_buf = None
            self.err = "not capturing stderr"

            self.combined = "error: CaptureStd context is unfinished yet, called too early"

    def __enter__(self):
        if self.out_buf is not None:
            self.out_old = sys.stdout
            sys.stdout = self.out_buf

        if self.err_buf is not None:
            self.err_old = sys.stderr
            sys.stderr = self.err_buf

        self.combined = ""

        return self

    def __exit__(self, *exc):
        if self.out_buf is not None:
            sys.stdout = self.out_old
            captured = self.out_buf.getvalue()
            if self.replay:
                sys.stdout.write(captured)
            self.out = apply_print_resets(captured)
            self.combined += self.out

        if self.err_buf is not None:
            sys.stderr = self.err_old
            captured = self.err_buf.getvalue()
            if self.replay:
                sys.stderr.write(captured)
            self.err = captured
            self.combined += self.err

    def __repr__(self):
        msg = ""
        if self.out_buf:
            msg += f"stdout: {self.out}\n"
        if self.err_buf:
            msg += f"stderr: {self.err}\n"
        return msg


# in tests it's the best to capture only the stream that's wanted, otherwise
# it's easy to miss things, so unless you need to capture both streams, use the
# subclasses below (less typing). Or alternatively, configure `CaptureStd` to
# disable the stream you don't need to test.


class CaptureStdout(CaptureStd):
    """Same as CaptureStd but captures only stdout"""

    def __init__(self, replay=True):
        super().__init__(err=False, replay=replay)


class CaptureStderr(CaptureStd):
    """Same as CaptureStd but captures only stderr"""

    def __init__(self, replay=True):
        super().__init__(out=False, replay=replay)


class CaptureLogger:
    """
    Context manager to capture `logging` streams

    Args:

    - logger: 'logging` logger object

    Results:
        The captured output is available via `self.out`

    Example::

        >>> from transformers import logging
        >>> from transformers.testing_utils import CaptureLogger

        >>> msg = "Testing 1, 2, 3"
        >>> logging.set_verbosity_info()
        >>> logger = logging.get_logger("transformers.models.bart.tokenization_bart")
        >>> with CaptureLogger(logger) as cl:
        ...     logger.info(msg)
        >>> assert cl.out, msg+"\n"
    """

    def __init__(self, logger):
        self.logger = logger
        self.io = StringIO()
        self.sh = logging.StreamHandler(self.io)
        self.out = ""

    def __enter__(self):
        self.logger.addHandler(self.sh)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.sh)
        self.out = self.io.getvalue()

    def __repr__(self):
        return f"captured: {self.out}\n"


@contextlib.contextmanager
# adapted from https://stackoverflow.com/a/64789046/9201239
def ExtendSysPath(path: Union[str, os.PathLike]) -> Iterator[None]:
    """
    Temporary add given path to `sys.path`.

    Usage ::

       with ExtendSysPath('/path/to/dir'):
           mymodule = importlib.import_module('mymodule')

    """

    path = os.fspath(path)
    try:
        sys.path.insert(0, path)
        yield
    finally:
        sys.path.remove(path)


class TestCasePlus(unittest.TestCase):
    """This class extends `unittest.TestCase` with additional features.

    Feature 1: A set of fully resolved important file and dir path accessors.

    In tests often we need to know where things are relative to the current test file, and it's not trivial since the
    test could be invoked from more than one directory or could reside in sub-directories with different depths. This
    class solves this problem by sorting out all the basic paths and provides easy accessors to them:

    * ``pathlib`` objects (all fully resolved):

       - ``test_file_path`` - the current test file path (=``__file__``)
       - ``test_file_dir`` - the directory containing the current test file
       - ``tests_dir`` - the directory of the ``tests`` test suite
       - ``data_dir`` - the directory of the ``tests/data`` test suite
       - ``repo_root_dir`` - the directory of the repository
       - ``src_dir`` - the directory where the ``m4`` sub-dir resides (same as repo_root_dir in this case)

    * stringified paths---same as above but these return paths as strings, rather than ``pathlib`` objects:

       - ``test_file_path_str``
       - ``test_file_dir_str``
       - ``tests_dir_str``
       - ``data_dir_str``
       - ``repo_root_dir_str``
       - ``src_dir_str``

    Feature 2: Flexible auto-removable temporary dirs which are guaranteed to get removed at the end of test.

    1. Create a unique temporary dir:

    ::

        def test_whatever(self):
            tmp_dir = self.get_auto_remove_tmp_dir()

    ``tmp_dir`` will contain the pathlib path to the created temporary dir. It will be automatically
    removed at the end of the test.


    2. Create a temporary dir of my choice, ensure it's empty before the test starts and don't
    empty it after the test.

    ::

        def test_whatever(self):
            tmp_dir = self.get_auto_remove_tmp_dir("./xxx")

    This is useful for debug when you want to monitor a specific directory and want to make sure the previous tests
    didn't leave any data in there.

    3. You can override the first two options by directly overriding the ``before`` and ``after`` args, leading to the
       following behavior:

    ``before=True``: the temporary dir will always be cleared at the beginning of the test.

    ``before=False``: if the temporary dir already existed, any existing files will remain there.

    ``after=True``: the temporary dir will always be deleted at the end of the test.

    ``after=False``: the temporary dir will always be left intact at the end of the test.

    Use `self.get_auto_remove_tmp_dir_str()` instead if you want the returned value to be a non-pathlib version.

    Note 1: In order to run the equivalent of ``rm -r`` safely, only subdirs of the project repository checkout are
    allowed if an explicit ``tmp_dir`` is used, so that by mistake no ``/tmp`` or similar important part of the
    filesystem will get nuked. i.e. please always pass paths that start with ``./``

    Note 2: Each test can register multiple temporary dirs and they all will get auto-removed, unless requested
    otherwise.

    Feature 3: Get a copy of the ``os.environ`` object that sets up ``PYTHONPATH`` specific to the current test suite.
    This is useful for invoking external programs from the test suite - e.g. distributed training.


    ::
        def test_whatever(self):
            env = self.get_env()

    """

    def setUp(self):
        # get_auto_remove_tmp_dir feature:
        self.teardown_tmp_dirs = []

        # figure out the resolved paths for repo_root, tests,  etc.
        self._test_file_path = inspect.getfile(self.__class__)
        path = Path(self._test_file_path).resolve()
        self._test_file_dir = path.parents[0]
        for up in [1, 2, 3]:
            tmp_dir = path.parents[up]
            if (tmp_dir / "m4").is_dir() and (tmp_dir / "tests").is_dir():
                break
        if tmp_dir:
            self._repo_root_dir = tmp_dir
        else:
            raise ValueError(f"can't figure out the root of the repo from {self._test_file_path}")
        self._tests_dir = self._repo_root_dir / "tests"
        self._data_dir = self._repo_root_dir / "tests" / "test_data"
        self._src_dir = self._repo_root_dir  # m4 doesn't use "src/" prefix in the repo

    @property
    def test_file_path(self):
        return self._test_file_path

    @property
    def test_file_path_str(self):
        return str(self._test_file_path)

    @property
    def test_file_dir(self):
        return self._test_file_dir

    @property
    def test_file_dir_str(self):
        return str(self._test_file_dir)

    @property
    def tests_dir(self):
        return self._tests_dir

    @property
    def tests_dir_str(self):
        return str(self._tests_dir)

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def data_dir_str(self):
        return str(self._data_dir)

    @property
    def repo_root_dir(self):
        return self._repo_root_dir

    @property
    def repo_root_dir_str(self):
        return str(self._repo_root_dir)

    @property
    def src_dir(self):
        return self._src_dir

    @property
    def src_dir_str(self):
        return str(self._src_dir)

    def get_env(self):
        """
        Return a copy of the ``os.environ`` object that sets up ``PYTHONPATH`` correctly. This is useful
        for invoking external programs from the test suite - e.g. distributed training.

        It always inserts ``.`` first, then ``./tests`` depending on the test suite type and
        finally the preset ``PYTHONPATH`` if any (all full resolved paths).

        """
        env = os.environ.copy()
        paths = [self.src_dir_str]
        paths.append(self.tests_dir_str)
        paths.append(env.get("PYTHONPATH", ""))

        env["PYTHONPATH"] = ":".join(paths)
        return env

    def get_auto_remove_tmp_dir(self, tmp_dir=None, before=None, after=None):
        """
        Args:
            tmp_dir (:obj:`string`, `optional`):
                if :obj:`None`:

                   - a unique temporary path will be created
                   - sets ``before=True`` if ``before`` is :obj:`None`
                   - sets ``after=True`` if ``after`` is :obj:`None`
                else:

                   - :obj:`tmp_dir` will be created
                   - sets ``before=True`` if ``before`` is :obj:`None`
                   - sets ``after=False`` if ``after`` is :obj:`None`
            before (:obj:`bool`, `optional`):
                If :obj:`True` and the :obj:`tmp_dir` already exists, make sure to empty it right away if :obj:`False`
                and the :obj:`tmp_dir` already exists, any existing files will remain there.
            after (:obj:`bool`, `optional`):
                If :obj:`True`, delete the :obj:`tmp_dir` at the end of the test if :obj:`False`, leave the
                :obj:`tmp_dir` and its contents intact at the end of the test.

        Returns:
            tmp_dir(:obj:`string`): either the same value as passed via `tmp_dir` or the path to the auto-selected tmp
            dir
        """
        if tmp_dir is not None:
            # defining the most likely desired behavior for when a custom path is provided.
            # this most likely indicates the debug mode where we want an easily locatable dir that:
            # 1. gets cleared out before the test (if it already exists)
            # 2. is left intact after the test
            if before is None:
                before = True
            if after is None:
                after = False

            # to avoid nuking parts of the filesystem, only relative paths are allowed
            if not tmp_dir.startswith("./"):
                raise ValueError(
                    f"`tmp_dir` can only be a relative path, i.e. `./some/path`, but received `{tmp_dir}`"
                )

            # using provided path
            tmp_dir = Path(tmp_dir).resolve()

            # ensure the dir is empty to start with
            if before is True and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

            tmp_dir.mkdir(parents=True, exist_ok=True)

        else:
            # defining the most likely desired behavior for when a unique tmp path is auto generated
            # (not a debug mode), here we require a unique tmp dir that:
            # 1. is empty before the test (it will be empty in this situation anyway)
            # 2. gets fully removed after the test
            if before is None:
                before = True
            if after is None:
                after = True

            # using unique tmp dir (always empty, regardless of `before`)
            tmp_dir = Path(tempfile.mkdtemp())

        if after is True:
            # register for deletion
            self.teardown_tmp_dirs.append(tmp_dir)

        return tmp_dir

    def get_auto_remove_tmp_dir_str(self, *args, **kwargs):
        return str(self.get_auto_remove_tmp_dir(*args, **kwargs))

    def tearDown(self):
        # get_auto_remove_tmp_dir feature: remove registered temp dirs
        for path in self.teardown_tmp_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self.teardown_tmp_dirs = []


def mockenv(**kwargs):
    """
    this is a convenience wrapper, that allows this:

    @mockenv(RUN_SLOW=True, USE_TF=False)
    def test_something():
        run_slow = os.getenv("RUN_SLOW", False)
        use_tf = os.getenv("USE_TF", False)

    Additionally see `mockenv_context` to use a context manager

    """
    return mock.patch.dict(os.environ, kwargs)


# from https://stackoverflow.com/a/34333710/9201239
@contextlib.contextmanager
def mockenv_context(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place. Similar to mockenv

    The ``os.environ`` dictionary is updated in-place so that the modification is sure to work in all situations.

    Args:
      remove: Environment variables to remove.
      update: Dictionary of environment variables and values to add/update.

    Example:

    with mockenv_context(FOO="1"):
        execute_subprocess_async(cmd, env=self.get_env())
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


# --- test network helper functions --- #


def get_xdist_worker_id():
    """
    when run under pytest-xdist returns the worker id (int), otherwise returns 0
    """
    worker_id_string = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    return int(worker_id_string[2:])  # strip "gw"


DEFAULT_MASTER_PORT = 10999


def get_unique_port_number():
    """
    When the test suite runs under pytest-xdist we need to make sure that concurrent tests won't use
    the same port number. We can accomplish that by using the same base and always adding the xdist
    worker id to it, or 0 if not running under pytest-xdist
    """
    return DEFAULT_MASTER_PORT + get_xdist_worker_id()


# --- test IO helper functions --- #


def write_file(file, content):
    with open(file, "w") as f:
        f.write(content)


def read_json_file(file):
    with open(file, "r") as fh:
        return json.load(fh)


def replace_str_in_file(file, text_to_search, replacement_text):
    file = Path(file)
    text = file.read_text()
    text = text.replace(text_to_search, replacement_text)
    file.write_text(text)


# --- pytest conf functions --- #

"""
This is a hack to get `pytest` to write out individual reports

To activate add to `tests/conftest.py`:

```
import pytest

def pytest_addoption(parser):
    from testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)
```

and then run:

```
pytest --make-reports=mytests tests
```

then check the individual reports under `reports/mytests/`

```
$ ls -1 reports/mytests/
durations.txt
errors.txt
failures_line.txt
failures_long.txt
failures_short.txt
stats.txt
summary_short.txt
warnings.txt
```

so now instead of having only a single output from `pytest` with everything together, you can now have each type of report saved into each own file.

"""


# to avoid multiple invocation from tests/conftest.py and examples/conftest.py - make sure it's called only once
pytest_opt_registered = {}


def pytest_addoption_shared(parser):
    """
    This function is to be called from `conftest.py` via `pytest_addoption` wrapper that has to be defined there.

    It allows loading both `conftest.py` files at once without causing a failure due to adding the same `pytest`
    option.

    """
    option = "--make-reports"
    if option not in pytest_opt_registered:
        parser.addoption(
            option,
            action="store",
            default=False,
            help="generate report files. The value of this option is used as a prefix to report names",
        )
        pytest_opt_registered[option] = 1


def pytest_terminal_summary_main(tr, id):
    """
    Generate multiple reports at the end of test suite run - each report goes into a dedicated file in the current
    directory. The report files are prefixed with the test suite name.

    This function emulates --duration and -rA pytest arguments.

    This function is to be called from `conftest.py` via `pytest_terminal_summary` wrapper that has to be defined
    there.

    Args:
    - tr: `terminalreporter` passed from `conftest.py`
    - id: unique id like `tests` or `examples` that will be incorporated into the final reports filenames - this is
      needed as some jobs have multiple runs of pytest, so we can't have them overwrite each other.

    NB: this functions taps into a private _pytest API and while unlikely, it could break should pytest do internal
    changes - also it calls default internal methods of terminalreporter which can be hijacked by various `pytest-`
    plugins and interfere.

    """
    from _pytest.config import create_terminal_writer

    if not len(id):
        id = "tests"

    config = tr.config
    orig_writer = config.get_terminal_writer()
    orig_tbstyle = config.option.tbstyle
    orig_reportchars = tr.reportchars

    dir = f"reports/{id}"
    Path(dir).mkdir(parents=True, exist_ok=True)
    report_files = {
        k: f"{dir}/{k}.txt"
        for k in [
            "durations",
            "errors",
            "failures_long",
            "failures_short",
            "failures_line",
            "passes",
            "stats",
            "summary_short",
            "warnings",
        ]
    }

    # custom durations report
    # note: there is no need to call pytest --durations=XX to get this separate report
    # adapted from https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/runner.py#L66
    dlist = []
    for replist in tr.stats.values():
        for rep in replist:
            if hasattr(rep, "duration"):
                dlist.append(rep)
    if dlist:
        dlist.sort(key=lambda x: x.duration, reverse=True)
        with open(report_files["durations"], "w") as f:
            durations_min = 0.05  # sec
            f.write("slowest durations\n")
            for i, rep in enumerate(dlist):
                if rep.duration < durations_min:
                    f.write(f"{len(dlist)-i} durations < {durations_min} secs were omitted")
                    break
                f.write(f"{rep.duration:02.2f}s {rep.when:<8} {rep.nodeid}\n")

    def summary_failures_short(tr):
        # expecting that the reports were --tb=long (default) so we chop them off here to the last frame
        reports = tr.getreports("failed")
        if not reports:
            return
        tr.write_sep("=", "FAILURES SHORT STACK")
        for rep in reports:
            msg = tr._getfailureheadline(rep)
            tr.write_sep("_", msg, red=True, bold=True)
            # chop off the optional leading extra frames, leaving only the last one
            longrepr = re.sub(r".*_ _ _ (_ ){10,}_ _ ", "", rep.longreprtext, 0, re.M | re.S)
            tr._tw.line(longrepr)
            # note: not printing out any rep.sections to keep the report short

    # use ready-made report funcs, we are just hijacking the filehandle to log to a dedicated file each
    # adapted from https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/terminal.py#L814
    # note: some pytest plugins may interfere by hijacking the default `terminalreporter` (e.g.
    # pytest-instafail does that)

    # report failures with line/short/long styles
    config.option.tbstyle = "auto"  # full tb
    with open(report_files["failures_long"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    # config.option.tbstyle = "short" # short tb
    with open(report_files["failures_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        summary_failures_short(tr)

    config.option.tbstyle = "line"  # one line per error
    with open(report_files["failures_line"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    with open(report_files["errors"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_errors()

    with open(report_files["warnings"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_warnings()  # normal warnings
        tr.summary_warnings()  # final warnings

    tr.reportchars = "wPpsxXEf"  # emulate -rA (used in summary_passes() and short_test_summary())

    # Skip the `passes` report, as it starts to take more than 5 minutes, and sometimes it timeouts on CircleCI if it
    # takes > 10 minutes (as this part doesn't generate any output on the terminal).
    # (also, it seems there is no useful information in this report, and we rarely need to read it)
    # with open(report_files["passes"], "w") as f:
    #     tr._tw = create_terminal_writer(config, f)
    #     tr.summary_passes()

    with open(report_files["summary_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.short_test_summary()

    with open(report_files["stats"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_stats()

    # restore:
    tr._tw = orig_writer
    tr.reportchars = orig_reportchars
    config.option.tbstyle = orig_tbstyle


# --- distributed testing functions --- #


class _RunOutput:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


async def _read_stream(stream, callback):
    while True:
        line = await stream.readline()
        if line:
            callback(line)
        else:
            break


async def _stream_subprocess(cmd, env=None, stdin=None, timeout=None, quiet=False, echo=False) -> _RunOutput:
    if echo:
        print("\nRunning: ", " ".join(cmd))

    p = await asyncio.create_subprocess_exec(
        cmd[0],
        *cmd[1:],
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # note: there is a warning for a possible deadlock when using `wait` with huge amounts of data in the pipe
    # https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.asyncio.subprocess.Process.wait
    #
    # If it starts hanging, will need to switch to the following code. The problem is that no data
    # will be seen until it's done and if it hangs for example there will be no debug info.
    # out, err = await p.communicate()
    # return _RunOutput(p.returncode, out, err)

    out = []
    err = []

    def tee(line, sink, pipe, label=""):
        line = line.decode("utf-8").rstrip()
        sink.append(line)
        if not quiet:
            print(label, line, file=pipe)

    # XXX: the timeout doesn't seem to make any difference here
    await asyncio.wait(
        [
            _read_stream(p.stdout, lambda line: tee(line, out, sys.stdout, label="stdout:")),
            _read_stream(p.stderr, lambda line: tee(line, err, sys.stderr, label="stderr:")),
        ],
        timeout=timeout,
    )
    return _RunOutput(await p.wait(), out, err)


def execute_subprocess_async(cmd, env=None, stdin=None, timeout=180, quiet=False, echo=True) -> _RunOutput:
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        _stream_subprocess(cmd, env=env, stdin=stdin, timeout=timeout, quiet=quiet, echo=echo)
    )

    cmd_str = " ".join(cmd)
    if result.returncode > 0:
        stderr = "\n".join(result.stderr)
        raise RuntimeError(
            f"'{cmd_str}' failed with returncode {result.returncode}\n\n"
            f"The combined stderr from workers follows:\n{stderr}"
        )

    # check that the subprocess actually did run and produced some output, should the test rely on
    # the remote side to do the testing
    if not result.stdout and not result.stderr:
        raise RuntimeError(f"'{cmd_str}' produced no output.")

    return result
