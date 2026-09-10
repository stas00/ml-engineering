#!/usr/bin/env python
import torch
from torch.utils.cpp_extension import load

# full device debug info so cuda-gdb can break inside the kernel
ext = load(
    name="kernel_oob_gdb",
    sources=["kernel_oob.cu"],
    extra_cuda_cflags=["-G"],
    verbose=False,
)
ext.run()
torch.cuda.synchronize()
