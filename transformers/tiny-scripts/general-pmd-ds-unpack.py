# unpack the desired datasets records into a filesystem-based subdir structure which can then be
# used to create a synthetic dataset of desired records. Each record can now be easily modified on
# the filesystem before being packed back into a dataset
#
# each records is a subdir
# each part of the record is:
# image.jpg
# text.txt
# meta.txt
# source.txt
#
# .null extension is when the slot is empty
#
# Example:
# python general-pmd-ds-unpack.py \
# --dataset_name_or_path /hf/m4-master/data/general_pmd/image/localized_narratives__ADE20k/train/00000-00002 \
# --ids 1,4-10 --target_path data


from argparse import ArgumentParser
from collections import defaultdict
from datasets import load_from_disk, Dataset
from pathlib import Path
from pprint import pprint
import gc
import numpy as np
import os
import psutil
import sys
import torchvision.transforms as transforms
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# LOCALIZE ME!
DATA_ROOT = "/hf/m4-master/data/cm4"

DATASET_PATH = f"{DATA_ROOT}/cm4-10000-v0.1"

parser = ArgumentParser()

parser.add_argument("--dataset_name_or_path", required=True, type=str, help="source dataset_name_or_path")
parser.add_argument("--target_path", required=False, default="output", type=str, help="path to where to unpack")
parser.add_argument("--ids", required=False, default="0", type=str, help="which ids to extract. example: 1,2,5-7,10")
args = parser.parse_args()

def list2range(s):
    """
    list2range('1,2,5-7,10')
    [1, 2, 5, 6, 7, 10]
    # from https://stackoverflow.com/a/6405711/9201239
    """
    return sum(((list(range(*[int(j) + k for k,j in enumerate(i.split('-'))]))
         if '-' in i else [int(i)]) for i in s.split(',')), [])

def unpack(args, idx, row):
    #pprint(row)

    path = f"{args.target_path}/{idx}"
    Path(path).mkdir(parents=True, exist_ok=True)

    # all items are text, except 'image'

    img = row["image"]
    basename = f"{path}/image"
    ext = "null" if img is None else "jpg"
    file = f"{basename}.{ext}"
    with open(file, "w") as fh:
        if img is not None:
            img.save(fh, 'jpeg')

    for col in ['meta', 'source', 'text']:
        item = row[col]
        basename = f"{path}/{col}"
        ext = "null" if item is None else "txt"
        file = f"{basename}.{ext}"
        with open(file, "w") as fh:
            if item is not None:
                fh.write(item)

def dump_example_shapes(idx, row):
    """ dump the row stats """
    shapes = {}

    img = row["image"]
    shapes["image"] = 0 if img is None else "x".join(map(str, img.size))

    for col in ['meta', 'source', 'text']:
        item = row[col]
        shapes[col] = 0 if item is None else len(item)

    summary = ", ".join([f"{k}: {v:>9}" for k,v in shapes.items()])
    print(f"rec{idx:>6}: {summary}")


ids_range = list2range(args.ids)

ds = load_from_disk(args.dataset_name_or_path)
#rows = ds[ids_range]

#pprint(rows[1])

for idx, id in enumerate(ids_range):
    unpack(args, id, ds[id])
    dump_example_shapes(id, ds[id])
    #sys.exit()

ds.info.write_to_directory(args.target_path)
