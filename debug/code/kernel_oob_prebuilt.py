#!/usr/bin/env python
import torch
from torch.utils.cpp_extension import load

# built the way a PyPI wheel ships: release, NO -lineinfo / -G
ext = load(
    name="kernel_oob_prebuilt",
    sources=["kernel_oob.cu"],
    verbose=False,
)
ext.run()
torch.cuda.synchronize()
