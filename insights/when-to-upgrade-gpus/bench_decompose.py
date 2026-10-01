#!/usr/bin/env python3
"""Single-clock step decomposition: attention (FlashAttention kernels) vs everything else.

Splits the real DeepSpeed training step onto ONE clock -- measured wall-clock step time --
so buckets sum to the total by construction (unlike measuring attention in isolation and the
step end-to-end: different clocks that don't reconcile, which is the bug this replaces).

Method:
  1. Measure wall-clock step time T (fwd+bwd+optimizer), profiler off.
  2. torch.profiler one window; classify CUDA kernels by name:
       attention = name contains "flash" (FA3 on Hopper / FA4 on Blackwell)
       comm      = name contains "nccl" (ZeRO grad/param comms)
       compute   = everything else on the compute stream (dense GEMMs, norm/rope/swiglu, CE, optimizer)
  3. NCCL comm overlaps backward compute (overlap_comm=True) -> it's hidden behind the wall clock,
     so we split T by each bucket's share of *compute-stream* busy time (comm excluded):
       t_att  = T * att_time / (att_time + compute_time)
       t_rest = T - t_att            # dense compute + any non-overlapped comm tail
  4. bucket TFLOPS = bucket_FLOPs / bucket_time ; total = total_FLOPs / T (consistent blend).

Deep-dive config (right-sized 8k): ZeRO-2, no activation checkpointing, no offload, no Liger.
Run on both GPUs with the SAME command; each prints its own row:

    deepspeed --num_gpus=8 bench_decompose.py
"""

import importlib.metadata as md
import os
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import torch
import deepspeed
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity

# ---- config (deep-dive: right-sized 8k) -------------------------------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B"
BATCH_SIZE = 1
SEQ_LEN = int(os.environ.get("SEQ_LEN", 8192))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 3))
TIME_STEPS = int(os.environ.get("TIME_STEPS", 10))
PROFILE_STEPS = int(os.environ.get("PROFILE_STEPS", 5))
ZERO_STAGE = int(os.environ.get("ZERO_STAGE", 2))
OFFLOAD_OPTIMIZER = int(os.environ.get("OFFLOAD_OPTIMIZER", 0))
GRAD_CHECKPOINT = int(os.environ.get("GRAD_CHECKPOINT", 0))
USE_LIGER = int(os.environ.get("USE_LIGER", 0))
LR = 1e-4
SEED = 42
# bf16 dense-matmul peak TFLOPS (for MFU), per GPU family
PEAK_TFLOPS = {9: 989.0, 10: 2250.0}  # H200 (sm90) / B200 (sm100)
# -----------------------------------------------------------------------------------------


def resolve_model(model_id: str) -> str:
    local = os.path.join(os.environ.get("DATA_FAST", os.path.expanduser("~/base-models")), model_id)
    return local if os.path.isfile(os.path.join(local, "config.json")) else model_id


def flops_split(config):
    """(attn_flops, dense_flops) per step in units of 1e12, causal (coeff 6) MFU convention."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    q_size = config.num_attention_heads * head_dim
    k_size = v_size = config.num_key_value_heads * head_dim
    mlp_N = config.hidden_size * config.intermediate_size * 3
    attn_linear_N = config.hidden_size * (q_size + k_size + v_size + q_size)
    dense_N = (mlp_N + attn_linear_N) * config.num_hidden_layers + config.vocab_size * config.hidden_size
    tokens = BATCH_SIZE * SEQ_LEN
    dense_flops = 6 * dense_N * tokens
    attn_flops = 6 * BATCH_SIZE * SEQ_LEN**2 * head_dim * config.num_attention_heads * config.num_hidden_layers
    return attn_flops / 1e12, dense_flops / 1e12


def detect_backend(device=0) -> str:
    major, _ = torch.cuda.get_device_capability(device)
    if major == 9:
        return "flash_attention_3"
    if major == 10:
        return "flash_attention_4"
    raise RuntimeError(f"unsupported compute capability {major}.x")


def build_model(backend: str):
    path = resolve_model(MODEL_ID)
    if USE_LIGER:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model = AutoLigerKernelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=backend)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=backend)
    if GRAD_CHECKPOINT:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


def ds_config():
    zero = {"stage": ZERO_STAGE, "overlap_comm": True, "contiguous_gradients": True}
    if OFFLOAD_OPTIMIZER:
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    return {
        "train_micro_batch_size_per_gpu": BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "optimizer": {"type": "AdamW", "params": {"lr": LR, "betas": [0.9, 0.999], "eps": 1e-8}},
        "zero_optimization": zero,
    }


def main():
    deepspeed.init_distributed()
    rank = dist.get_rank()
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)

    backend = detect_backend()
    model = build_model(backend)
    engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config())

    attn_tflop, dense_tflop = flops_split(model.config)
    total_tflop = attn_tflop + dense_tflop
    ids = torch.randint(0, model.config.vocab_size - 256, (BATCH_SIZE, SEQ_LEN), device=device)
    batch = {"input_ids": ids, "labels": ids}

    def one_step():
        loss = engine(**batch).loss
        engine.backward(loss)
        engine.step()

    for _ in range(WARMUP_STEPS):
        one_step()
    torch.cuda.synchronize()

    # (1) wall-clock step time, profiler OFF
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(TIME_STEPS):
        one_step()
    torch.cuda.synchronize()
    T_step = (time.perf_counter() - t0) / TIME_STEPS
    peak_mem = torch.tensor(torch.cuda.max_memory_reserved(device) / 2**30, device=device)
    dist.all_reduce(peak_mem, op=dist.ReduceOp.MAX)
    peak_mem = peak_mem.item()

    # (2) kernel breakdown, profiler ON
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(PROFILE_STEPS):
            one_step()
        torch.cuda.synchronize()

    if rank != 0:
        return

    ka = prof.key_averages()
    att_us = sum(e.self_device_time_total for e in ka if "flash" in e.key.lower())
    comm_us = sum(e.self_device_time_total for e in ka if "nccl" in e.key.lower())
    all_us = sum(e.self_device_time_total for e in ka)
    compute_us = all_us - comm_us  # compute-stream busy time (comm overlaps, excluded)

    # (3) split measured wall time by compute-stream share
    f_att = att_us / compute_us
    t_att = f_att * T_step
    t_rest = T_step - t_att

    major = torch.cuda.get_device_capability(device)[0]
    peak = PEAK_TFLOPS[major]

    def row(name, tflop, t):
        tflops = tflop / t
        return f"{name:16s} {tflop:7.1f} TFLOP  {t*1e3:8.2f} ms  {tflops:8.1f} TFLOPS  MFU {tflops/peak*100:5.1f}%"

    def ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "n/a"

    print("=" * 100)
    print(f"# gpu={torch.cuda.get_device_name()} backend={backend} "
          f"zero={ZERO_STAGE} gc={GRAD_CHECKPOINT} offload={OFFLOAD_OPTIMIZER} liger={USE_LIGER} "
          f"seq={SEQ_LEN} time_steps={TIME_STEPS} profile_steps={PROFILE_STEPS}")
    fa = "flash-attn-3" if backend == "flash_attention_3" else "flash-attn-4"
    print(f"# python={platform.python_version()} torch={torch.__version__} cuda={torch.version.cuda} "
          f"transformers={ver('transformers')} deepspeed={ver('deepspeed')} "
          f"liger-kernel={ver('liger-kernel')} {fa}={ver(fa)}")
    print(f"# wall step={T_step*1e3:.2f} ms   peak_mem={peak_mem:.1f} GiB   "
          f"attention FLOP share={attn_tflop/total_tflop*100:.1f}%   "
          f"attention wall share={f_att*100:.1f}%   "
          f"comm={comm_us/all_us*100:.1f}% of kernel time (overlapped, hidden in wall)")
    print("-" * 100)
    print(row("attention(FA)", attn_tflop, t_att))
    print(row("everything else", dense_tflop, t_rest))
    print(row("TOTAL", total_tflop, T_step))
    print("-" * 100)
    print("# top CUDA kernels by self time:")
    rows = sorted(ka, key=lambda e: -e.self_device_time_total)[:12]
    for e in rows:
        tag = "ATT" if "flash" in e.key.lower() else ("NCL" if "nccl" in e.key.lower() else "   ")
        print(f"#  {tag} {e.self_device_time_total/all_us*100:5.1f}%  {e.key[:80]}")
    print("=" * 100)


if __name__ == "__main__":
    main()
