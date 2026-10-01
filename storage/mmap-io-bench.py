#!/usr/bin/env python3

#
# usage:
#
# pip install datasets numpy
# python mmap-io-bench.py
#
# Builds a 1GiB Hugging Face datasets Arrow shard set under each path below, then prints the markdown table from the mmap vs sequential dataset
# reads section. Edit the paths to match your local NVMe and network file system mounts.
#
# This script will only work on Linux because of posix_fadvise(DONTNEED)

import os
import time
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

# IMPORTANT: edit the paths to the target folders to reflect the actual NVME, Network fs (e.g. Lustre), and you can add more key/value entries to this dict
bench_roots = {
    "local NVMe": Path("./mmap-io-bench-local"),
    "Network FS": Path("./mmap-io-bench-network"),
}

# Optionally you can edit these as well
nrows = 32768          # 32Ki
row_bytes = 32 * 1024  # 1GiB payload
read_chunk = 2**20     # 1MiB

method_labels = [
    "sequential `read()`, whole file",
    "mmap, rows in order",
    "mmap, rows shuffled",
]

times = {label: {} for label in method_labels}

# Drop cached pages so each timed run reads from the file system, not RAM.
def drop_page_cache(paths):
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


# Run all three read patterns on each file system root.
for fs_label, root in bench_roots.items():
    root.mkdir(parents=True, exist_ok=True)
    arrow_dir = root / "arrow"
    # One-time setup: 1GiB of text split across a few Arrow shards, same layout HF datasets uses on disk.
    if not (arrow_dir / "dataset_info.json").exists():
        row = "a" * row_bytes
        Dataset.from_dict({"text": [row] * nrows}).save_to_disk(str(arrow_dir))

    arrow_files = sorted(arrow_dir.rglob("*.arrow"))

    # Table row 1: read every shard start to finish with ordinary read() calls.
    drop_page_cache(arrow_files)

    t0 = time.perf_counter()
    for path in arrow_files:
        with open(path, "rb") as f:
            while f.read(read_chunk):
                pass
    times[method_labels[0]][fs_label] = time.perf_counter() - t0

    # Table row 2: default HF datasets path - memory-mapped Arrow, rows visited in order.
    ds = load_from_disk(str(arrow_dir))
    drop_page_cache(arrow_files)

    t0 = time.perf_counter()
    for row in ds:
        len(row["text"])
    times[method_labels[1]][fs_label] = time.perf_counter() - t0

    # Table row 3: shuffled training - random row index into the mapped dataset.
    drop_page_cache(arrow_files)

    idx = np.random.default_rng(0).permutation(len(ds))
    t0 = time.perf_counter()
    for i in idx:
        len(ds[int(i)]["text"])
    times[method_labels[2]][fs_label] = time.perf_counter() - t0
    del ds

# Print the chapter table: wall times per root, slowdown vs sequential read() on the same root.
fs_labels = list(bench_roots)
header = "| How the 1GiB was read | " + " | ".join(fs_labels) + " | " + " | ".join(["Slowdown"] * len(fs_labels)) + " |"
rule_parts = [":------------------------------"] + ["---------:"] * len(fs_labels) + ["-------:"] * len(fs_labels)
rule = "| " + " | ".join(rule_parts) + " |"

print(header)
print(rule)

for label in method_labels:
    cells = [label]
    seq = times[method_labels[0]]
    for fs_label in fs_labels:
        cells.append(f"{times[label][fs_label]:7.2f}s")
    for fs_label in fs_labels:
        t = times[label][fs_label]
        base = seq[fs_label]
        slowdown = 0 if label == method_labels[0] else (t / base - 1) * 100
        cells.append(f"{round(slowdown):d}%")
    print("| " + " | ".join(cells) + " |")
