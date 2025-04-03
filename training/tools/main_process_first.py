"""
Tooling for dealing with efficient dataset loading in a multi-process, potentially multi-node environment with shared and local filesystems.

For notes please see https://github.com/stas00/ml-engineering/blob/master/training/datasets.md#preprocessing-and-caching-datasets-on-the-main-process

"""


import os
from contextlib import contextmanager
from pathlib import Path
import torch.distributed as dist

def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", 0))

def get_global_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

# delay the local filesystems lookup until it's needed
node_fs_types = None

local_node_fs_types = ["ext", "ext2", "ext3", "ext4", "reiserfs", "jfs", "xfs", "zfs", "xfs", "btrfs", "ntfs", "overlay"]
def is_local_fs(path):
    """ returns True if the `path` resides on the local fs or False otherwise """
    global node_fs_types
    if node_fs_types is None:
        from psutil import disk_partitions
        node_fs_types = {Path(r.mountpoint):r.fstype for r in disk_partitions(all=True)}

    return True if path_to_fs_type(path) in local_node_fs_types else False

def path_to_fs_type(path):
    """
    Given a fs `path` returns the fs type (ext, ext2, etc.) it resides on.
    Note that in this implementation non-existing paths will return the fs type of `/` (which often will be mapped to "overlay")
    This is useful since as long as partitions are mounted already you can detect the type of the fs ven before the sub-dirs were created
    """
    path = Path(path).resolve()
    if path.is_symlink():
        path = path.readlink() # py3.9+

    # assuming at the end we percolate to `/` which is always there so the exit condition is assured
    if path in node_fs_types:
        return node_fs_types[path]

    return path_to_fs_type(path.parent)

def is_main_process_by_path(path):
    if is_local_fs(path):
        return is_local_main_process()
    else:
        return is_global_main_process()

def is_local_main_process():
    return get_local_rank() == 0

def is_global_main_process():
    return dist.get_rank() == 0

@contextmanager
def _goes_first(is_main: bool):
    if not is_main:
        dist.barrier()

    yield

    if is_main:
        dist.barrier()


@contextmanager
def main_process_by_path_first(path):
    """
    Lets the global or the local main process go first inside a with block. The decision which to use is based on the `path`. If the `path` is on a local non-shared fs, we use the local main process. If the path is on the shared fs then it's a global main process.

    The other processes will enter the with block after the defined above main process exits.

    Important: since this context manager uses a barrier it can't be used around code that requires all ranks to work in sync - e.g. gather, barrier, etc. - it'd lead to a deadlock

    Example:

        import time
        with main_process_by_path_first("/shared_fs/cache"):
            # This will be printed first by global process 0 then in a seemingly
            # random order by the other processes.
            # we presume in this example the path is on a shared fs
            global_rank = torch.distributed.get_rank()
            print(f"This will be printed by process {global_rank}")
            time.sleep(5) # emulate actual work
    """
    if is_local_fs(path):
        with _goes_first(is_local_main_process()):
            yield
    else:
        with _goes_first(is_global_main_process()):
            yield

@contextmanager
def global_main_process_first():
    """
    Lets the global main process go first inside a with block.

    The other processes will enter the with block after the global main process exits.

    Important: since this context manager uses a barrier it can't be used around code that requires all ranks to work in sync - e.g. gather, barrier, etc. - it'd lead to a deadlock

    Example:

        import time
        global_rank = torch.distributed.get_rank()
        with global_main_process_first():
            # This will be printed first by global process 0 then in a seemingly
            # random order by the other processes.
            print(f"This will be printed by process {global_rank}")
            time.sleep(5) # emulate actual work
    """
    with _goes_first(is_global_main_process()):
        yield

@contextmanager
def local_main_process_first():
    """
    Lets the local main process go inside a with block.

    The other processes will enter the with block after the local main process exits.

    Important: since this context manager uses a barrier it can't be used around code that requires all ranks to work in sync - e.g. gather, barrier, etc. - it'd lead to a deadlock

    Example:

        import time
        local_rank = get_local_rank()
        with local_main_process_first():
            # This will be printed first by local process 0 then in a seemingly
            # random order by the other processes.
            print(f"This will be printed by process {local_rank}")
            time.sleep(5) # emulate actual work
    """
    with _goes_first(is_local_main_process()):
        yield

if __name__ == "__main__":
    # to test run:
    #
    # python -u -m torch.distributed.run --nproc_per_node=2 --rdzv_endpoint localhost:6000  --rdzv_backend c10d main_process_first.py

    dist.init_process_group()

    import time
    global_rank = get_global_rank()
    local_rank  = get_local_rank()

    def ds_load_emulate():
        print("Loading dataset")
        time.sleep(2)

    if global_rank == 0:
        print("\n\n*** Demo global_main_process_first")
    with global_main_process_first():
        print(f"Running on global rank {global_rank}")
        ds_load_emulate()
    dist.barrier()

    if global_rank == 0:
        print("\n\n*** Demo local_main_process_first")
    with local_main_process_first():
        print(f"Running on local rank {local_rank}")
        ds_load_emulate()
    dist.barrier()

    if global_rank == 0:
        print("\n\n*** Demo is_main_process_by_path")
    path = "./"
    rank_type = "main" if is_main_process_by_path(path) else "non-main"
    with main_process_by_path_first(path):
        print(f"Running on {rank_type} rank local={local_rank} global={global_rank}")
        ds_load_emulate()
    dist.barrier()

    dist.destroy_process_group()
