# If you have ever done multi-gpu work and tried to `print` for debugging you quickly discovered
# that some messages get interleaved and are impossible to make sense of. Especially so if you're
# using `print` to debug values.
#
# This simple solution that uses the good old `flock` solves the interleaving problem. To use this
# version of print you can either do:
#
# from printflock import printflock
# import torch.distributed as dist
# printflock(f"{dist.get_rank()}: my long debug message")
#
# or you can override `print` with a better one:
#
# from printflock import printflock as print
# import torch.distributed as dist
# print(f"{dist.get_rank()}: my long debug message")
#

import builtins
import fcntl

def printflock(*args, **kwargs):
    """
    This is a wrapper around the built-in Python `print` which calls `flock` before calling
    `print` and unlocks it immediately after. This wrapper is useful for when each rank needs to
    print a message without getting it interleaved with prints from other ranks.
    The lock file is the file this wrapper is defined in.
    The output order will be random per rank.

    Example:
        >>> # assuming 4 GPUs
        >>> world_size = dist.get_world_size()
        >>> rank = dist.get_rank()
        >>> printflock(f"This is a very long message from rank {rank}/{world_size}")
       This is a very long message from rank 0/4
       This is a very long message from rank 2/4
       This is a very long message from rank 3/4
       This is a very long message from rank 1/4

    It can also be used to override normal `print`:

    from printflock import printflock as print

    and then you don't need to change anything in your code.
    """

    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
