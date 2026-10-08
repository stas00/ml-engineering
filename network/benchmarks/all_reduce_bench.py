#!/usr/bin/env python

"""
all_reduce_bench.py has been renamed to torch-dist-bench.py, which still benchmarks all_reduce by default.

Its new location is: https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/torch-dist-bench.py

This stub only stays so that the old links and launch commands tell you where the script went.
"""

import os
import sys
from pathlib import Path

new_url = "https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/torch-dist-bench.py"
new_path = Path(__file__).resolve().parent / "torch-dist-bench.py"

# under a launcher every rank runs this, so only one prints
if int(os.environ.get("RANK", "0")) == 0:
    print(f"all_reduce_bench.py has been renamed to torch-dist-bench.py, run it instead:\n"
          f"- {new_path}{'' if new_path.exists() else ' (download it first)'}\n"
          f"- {new_url}", file=sys.stderr)
sys.exit(1)
