""" NicerTrace - an improved Trace package """

"""
To try it in action and to get a sense of how it can help you just run:
python trace/NicerTrace.py
"""


import datetime
import os
import socket
import sys
import sysconfig
import time
import trace


class NicerTrace(trace.Trace):
    # as the 2 paths overlap the longer with site-packages needs to be first
    py_dirs = [sysconfig.get_paths().get(k) for k in ["purelib", "stdlib"]]
    site_packages_dir = sysconfig.get_paths()["purelib"]
    stdlib_dir = sysconfig.get_paths()["stdlib"]

    def __init__(self, *args, packages_to_include=None, log_pids=False, **kwargs):
        """normal init plus added package/dir exclusion overrides:

        While preserving the original behavior a new optional arg is added `packages_to_include`
        with the following behavior:

        1. if ignoredirs is a list the original trace behavior is used - only those dirs and subdirs will be excluded
        2. if ignoredirs is None and packages_to_include is None - everything is included
        3. if packages_to_include="uninstalled" all packages found under  /.../site-packages will be excluded. I couldn't find a way to exclude core python packages under /.../lib/python3.8 since it'd then exclude site-packages as well
        3. if packages_to_include=["PIL", "numpy", "pytorch"] all packages found under  /.../site-packages, and /.../lib/python3.8 will be excluded except the packages that were listed to be included - use top-level package name here
        4. if packages_to_include=None, everything under /.../site-packages, and /.../lib/python3.8 will be excluded and any packages that are installed via `pip install -e .` will be included

        """
        ignoredirs = kwargs.get("ignoredirs", None)

        if ignoredirs is not None and len(ignoredirs) > 1:
            if packages_to_include is not None:
                raise ValueError("can't have both ignoredirs and packages_to_include not None")
            kwargs["ignoredirs"] = ignoredirs
        elif packages_to_include is None:
            kwargs["ignoredirs"] = None
        elif packages_to_include == "uninstalled":
            kwargs["ignoredirs"] = self.stdlib_dir  # everything including python core packages
        else:
            # exclude all of /.../lib/python3.8 and sub-paths from /.../site-packages, and
            packages = os.listdir(self.site_packages_dir)
            packages_to_exclude = set(packages) - set(packages_to_include)
            dirs_to_exclude = [
                f"{self.site_packages_dir}/{dir}" for dir in sorted(packages_to_exclude) if not dir.endswith("-info")
            ]
            # note, no way to exclude python core packages in this situation because
            # sysconfig.get_paths()'s' purelib is a subset of stdlib :(, so excluding only site-packages
            kwargs["ignoredirs"] = dirs_to_exclude

        # not packages, but final module names like Image from Image.py
        # mods_to_exclude = []

        # print("\n".join(kwargs["ignoredirs"]))

        super().__init__(*args, **kwargs)
        self.log_pids = log_pids

    def strip_py_dirs(self, path):
        """strips python path prefix like /.../site-packages, and /.../lib/python3.8 if any matches"""
        for prefix in self.py_dirs:
            if path.startswith(prefix):
                return path.replace(prefix + "/", "")
        return path

    def globaltrace_lt(self, frame, why, arg):
        """Handler for call events.
        If the code block being entered is to be ignored, returns `None',
        else returns self.localtrace.

        This is an override to properly show full package names:
        1. if it's under site-packages or core python dir - convert to package name
        2. otherwise show full path to the python file - usually uninstalled packages

        Additionally enter frames now include the line number since some packages have multiple
        methods that have the same name and there is no telling which one of them was called.

        It was written against https://github.com/python/cpython/blob/3.8/Lib/trace.py. If you're
        using a different python version you may have to adapt it should the core implementation
        change (but it's unlikely)

        """
        if why == "call":
            code = frame.f_code
            # print(f"\n\n{frame.f_code=}")
            # print(dir(code))

            filename = frame.f_globals.get("__file__", None)
            if filename:
                lineno = code.co_firstlineno
                # python's trace fails to get the full package name - let's fix it
                # strip the common path of python library
                modulename = self.strip_py_dirs(filename)
                if filename != modulename:
                    # the package was installed under /.../site-packages, /.../lib/python3.8
                    modulename, ext = os.path.splitext(modulename)
                    modulename = modulename.replace("/", ".")
                else:
                    # still full path, because the package is not installed
                    modulename = filename

                if modulename is not None:
                    # XXX: ignoremods may not work now as before
                    ignore_it = self.ignore.names(filename, modulename)
                    if not ignore_it:
                        if self.trace:
                            if self.log_pids:
                                print(os.getpid(), end=" ")

                            print(f"        {modulename}:{lineno} {code.co_name}")
                        return self.localtrace
            else:
                return None

    def localtrace_trace_and_count(self, frame, why, arg):
        """
        Overriding the default method.

        Using hh:mm:ss format for timestamps (instead of secs) as it's more readable when the trace is run for hours

        XXX: ideally it would be nice not to repeat the same module name on every line, but when I tried
        that I discovered that globaltrace_lt doesn't necessarily frame all the local calls, since
        localtrace_trace_and_count may continue printing local calls from an earlier frame w/o
        notifying that the context has changed. So we are forced to reprint the module name on each
        line to keep at least the incomplete context.

        Ideally there should an indication of a frame change before all the local prints

        Read the disclaimer in globaltrace_lt that this was tested with py-3.8

        """
        if why == "line":
            # record the file name and line number of every trace
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            key = filename, lineno
            self.counts[key] = self.counts.get(key, 0) + 1
            basename = os.path.basename(filename)
            if self.log_pids:
                print(os.getpid(), end=" ")
            if self.start_time:
                delta_time = trace._time() - self.start_time
                delta_time = str(datetime.timedelta(seconds=delta_time)).split(".")[0]
                print(delta_time, end=" ")
            print(f"{basename}:{lineno:>6}: {trace.linecache.getline(filename, lineno)}", end="")
        return self.localtrace

# -------------------------------- #


class Tee:
    """
    A helper class to tee print's output into a file.
    Usage:
    sys.stdout = Tee(filename)
    """

    def __init__(self, filename):
        self.stdout = sys.stdout
        self.file = open(filename, "a")

    def __getattr__(self, attr):
        return getattr(self.stdout, attr)

    def write(self, msg):
        # comment out the next line if you don't want to write to stdout
        self.stdout.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        # comment out the next line if you don't want to write to stdout
        self.stdout.flush()
        self.file.flush()


# -------------------------------- #

import time

from PIL import Image

def main():
    img = Image.new("RGB", (4, 4))
    time.sleep(1)
    img1 = img.convert("RGB")

# or if you want to try another version of main:

# from transformers import AutoConfig
# def main():
#     c = AutoConfig.from_pretrained("t5-small")

if __name__ == "__main__":
    # enable the trace
    if 1:
        cwd = os.path.realpath(".")
        pid = os.getpid()
        hostname = socket.gethostname()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        trace_output_file = f"{cwd}/trace-{hostname}-{local_rank}-{pid}.txt"

        # run the new command using the given tracer
        sys.stdout = Tee(trace_output_file)

        # create a Trace object, telling it what to ignore, and whether to
        # do tracing or line-counting or both.
        # tracer = trace.Trace(
        tracer = NicerTrace(
            # ignoredirs=dirs_to_exclude, # don't set this one if you use packages_to_include
            # ignoremods=mods_to_exclude,
            trace=1,
            count=1,
            timing=True,
            # log_pids=True, useful if you fork workers and want to tell which process the trace belongs to
            packages_to_include=["PIL"],
        )

        # string with commands to run - passed to exec()
        tracer.run("main()")
        # or to use the function interface to call main with args, kwargs
        # tracer.runfunc(main, *args, **kwds))
    else:
        main()
