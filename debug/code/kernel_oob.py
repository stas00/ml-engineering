#!/usr/bin/env python
import torch
from torch.utils.cpp_extension import load

ext = load(
    name="kernel_oob",
    sources=["kernel_oob.cu"],
    extra_cuda_cflags=["-lineinfo"],
    verbose=True,
)
ext.run()
torch.cuda.synchronize()
