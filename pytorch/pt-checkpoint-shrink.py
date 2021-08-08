#!/usr/bin/env python

# this script fixes checkpoints which for some reason stored tensors with storage larger than their view at the moment of saving.

import argparse
import torch
import glob
import os

debug = 0

# load to cpu
device = torch.device('cpu')

def get_pt_files(checkpoint_dir):

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Directory '{checkpoint_dir}' doesn't exist")

    pt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))

    if len(pt_files) == 0:
        raise FileNotFoundError(
            f"can't find '*.pt' files in directory '{checkpoint_dir}'")

    return pt_files

import collections.abc
def shrink_dict_values(d, prefix=""):
    for k, v in d.items():
        k_full = f"{prefix}.{k}" if len(prefix) else k
        if isinstance(v, collections.abc.Mapping):
            shrink_dict_values(v, k_full)
        else:
            if debug:
                print(f"{k_full}")
            if v is not None and torch.is_tensor(v):
                d[k] = v.clone() # drop any unused storage

def shrink_pt_file(f):
    size_before = os.path.getsize(f)
    sd = torch.load(f, map_location=device)
    shrink_dict_values(sd)
    torch.save(sd, f)
    size_after = os.path.getsize(f)
    size_delta = size_before - size_after
    if debug:
        print(f"before {size_before / 2**20:.2f}MB, after {size_after / 2**20:.2f}MB, saved {size_delta / 2**20:.2f}MB")
    return size_before, size_after, size_delta

def checkpoint_shrink(checkpoint_dir):
    """
    Args:
        - ``ds_checkpoint_dir``: path to the deepspeed checkpoint folder (where the optimizer files are)
    """
    print(f"Processing zero checkpoint '{checkpoint_dir}'")
    pt_files = get_pt_files(checkpoint_dir)
    before, after, delta = 0, 0, 0
    for f in pt_files:
        size_before, size_after, size_delta = shrink_pt_file(f)
        before += size_before
        after  += size_after
        delta  += size_delta
    print(f"before {before / 2**20:.2f}MB, after {after / 2**20:.2f}MB, saved {delta / 2**20:.2f}MB")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="path to the desired checkpoint folder, e.g., path/checkpoints/global_step10")
    parser.add_argument("-d", "--debug", action='store_true', help="enable debug")
    args = parser.parse_args()

    debug = args.debug

    checkpoint_shrink(args.checkpoint_dir)
