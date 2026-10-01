#!/usr/bin/env python3
import time
import torch

BATCH = 1
SEQLENS = [1024, 2048, 4096, 8192, 16384, 32768]
NHEADS = 32
NHEADS_KV = 8
HEAD_DIM = 128
CAUSAL = True
DTYPE = torch.bfloat16
WARMUP = 5
ITERS = 20

major, _ = torch.cuda.get_device_capability()
if major == 9:
    from flash_attn_interface import flash_attn_func
    BACKEND = "fa3"
elif major == 10:
    from flash_attn.cute import flash_attn_func
    BACKEND = "fa4"
else:
    raise RuntimeError(f"unsupported compute capability {major}.x")

torch.manual_seed(0)
dev = "cuda"
print(f"# gpu={torch.cuda.get_device_name()} backend={BACKEND} causal={CAUSAL} "
      f"nheads={NHEADS} nheads_kv={NHEADS_KV} head_dim={HEAD_DIM}")


def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS


for seqlen in SEQLENS:
    q = torch.randn(BATCH, seqlen, NHEADS, HEAD_DIM, device=dev, dtype=DTYPE, requires_grad=True)
    k = torch.randn(BATCH, seqlen, NHEADS_KV, HEAD_DIM, device=dev, dtype=DTYPE, requires_grad=True)
    v = torch.randn(BATCH, seqlen, NHEADS_KV, HEAD_DIM, device=dev, dtype=DTYPE, requires_grad=True)
    dout = torch.randn(BATCH, seqlen, NHEADS, HEAD_DIM, device=dev, dtype=DTYPE)

    def fwd():
        o = flash_attn_func(q, k, v, causal=CAUSAL)
        return o[0] if isinstance(o, tuple) else o

    def fwd_bwd():
        fwd().backward(dout)

    f = 4 * BATCH * seqlen**2 * NHEADS * HEAD_DIM
    if CAUSAL:
        f //= 2

    t_fwd = bench(fwd)
    t_fb = bench(fwd_bwd)
    print(f'{{"seqlen": {seqlen}, '
          f'"fwd_ms": {t_fwd*1000:.2f}, "fwd_tflops": {f/t_fwd/1e12:.2f}, '
          f'"fwd_bwd_ms": {t_fb*1000:.2f}, "fwd_bwd_tflops": {3.5*f/t_fb/1e12:.2f}}}')
