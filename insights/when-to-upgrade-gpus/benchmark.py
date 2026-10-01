#!/usr/bin/env python3
"""FA3 vs FA4 DeepSpeed fwd/bwd/step benchmark: real Llama-3 weights, fake data.
Attention backend auto-detected from GPU: Hopper -> FA3, Blackwell -> FA4.
Edit the config block below, or override via env vars (SEQ_LEN, STEPS, ...), then:

    deepspeed --num_gpus=8 benchmark.py

Set MEM_DEBUG=1 for staged memory probes + a torch memory snapshot on rank 0.
"""

import importlib.metadata as md
import json
import os
import platform
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic cuBLAS
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import torch
import deepspeed
import torch.distributed as dist

# ---- config -----------------------------------------------------------------------------
#MODEL_SIZE = "tiny"
MODEL_SIZE = "8b"

BATCH_SIZE = 1
SEQ_LEN = int(os.environ.get("SEQ_LEN", 8192))
STEPS = int(os.environ.get("STEPS", 10))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 2))
ZERO_STAGE = int(os.environ.get("ZERO_STAGE", 3))
OFFLOAD_OPTIMIZER = int(os.environ.get("OFFLOAD_OPTIMIZER", 1))  # ZeRO offload optimizer states to CPU
GRAD_CHECKPOINT = int(os.environ.get("GRAD_CHECKPOINT", 1))  # recompute activations in bwd instead of storing them
LR = 1e-4
SEED = 42
USE_LIGER = int(os.environ.get("USE_LIGER", 1))  # fuse RMSNorm/RoPE/SwiGLU/CE via liger-kernel
MEM_DEBUG = int(os.environ.get("MEM_DEBUG", 0))  # staged mem probes + snapshot on rank 0
OUTPUT = None  # path to append per-step JSON records, or None
# -----------------------------------------------------------------------------------------

MODEL_IDS = {
    "tiny": "tiny-random/llama-3",
    "8b": "meta-llama/Llama-3.1-8B",
}


def resolve_model(model_id: str) -> str:
    """Prefer a local staged copy (see install.sh) if present; else the hub id (HF cache)."""
    local = os.path.join(os.environ.get("DATA_FAST", os.path.expanduser("~/base-models")), model_id)
    return local if os.path.isfile(os.path.join(local, "config.json")) else model_id


def estimate_tflos(config) -> float:
    """Model fwd+bwd FLOPs (1e12) for dense Llama/GQA, ported from arctic_training's
    FlopsCounter._estimate_qwen2_flops (adapted from verl; same dense GQA shape as Llama).
    Attention coeff is 6 (causal), not 12: FlashAttention computes only the lower triangle,
    so full-s^2 counting would ~double-count attention and inflate MFU at long context.
    Model-FLOPs (MFU) count -- gradient-checkpoint recompute is excluded (that would be HFU)."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    q_size = config.num_attention_heads * head_dim
    k_size = v_size = config.num_key_value_heads * head_dim

    mlp_N = config.hidden_size * config.intermediate_size * 3
    attn_linear_N = config.hidden_size * (q_size + k_size + v_size + q_size)
    emd_and_lm_head_N = config.vocab_size * config.hidden_size
    dense_N = (mlp_N + attn_linear_N) * config.num_hidden_layers + emd_and_lm_head_N

    tokens = BATCH_SIZE * SEQ_LEN
    dense_flops = 6 * dense_N * tokens
    attn_flops = 6 * BATCH_SIZE * SEQ_LEN**2 * head_dim * config.num_attention_heads * config.num_hidden_layers
    return (dense_flops + attn_flops) / 1e12


def env_line(backend: str) -> str:
    def ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "n/a"
    fa = "flash-attn-3" if backend == "flash_attention_3" else "flash-attn-4"
    return (f"# env: python={platform.python_version()} torch={torch.__version__} "
            f"cuda={torch.version.cuda} transformers={ver('transformers')} "
            f"deepspeed={ver('deepspeed')} liger-kernel={ver('liger-kernel')} "
            f"{fa}={ver(fa)}")


def detect_backend(device=0) -> str:
    """HF-native attn_implementation per GPU: Hopper -> FA3, Blackwell -> FA4."""
    major, _ = torch.cuda.get_device_capability(device)
    if major == 9:
        return "flash_attention_3"
    if major == 10:
        return "flash_attention_4"
    raise RuntimeError(f"unsupported compute capability {major}.x; need Hopper (9) or Blackwell (10)")


def build_model(backend: str) -> torch.nn.Module:
    model_path = resolve_model(MODEL_IDS[MODEL_SIZE])
    if USE_LIGER:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model = AutoLigerKernelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation=backend)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation=backend)
    if GRAD_CHECKPOINT:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # from_pretrained returns eval mode; train mode gates gc + liger fused-CE
    return model


def set_seed(seed: int) -> None:
    """Reproducible init + data. warn_only so FA's non-deterministic bwd warns instead
    of raising; bitwise determinism isn't guaranteed with flash-attn, but runs are stable."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def fake_batch(config, device, step: int) -> dict:
    """Random content tokens, fixed dense seq (no padding -> every position does real work).
    Llama-3 keeps bos/eos/pad + reserved specials in the top 256 ids; sample below them so
    no pad/special tokens leak in."""
    g = torch.Generator().manual_seed(SEED + step)
    n_special = 256 if config.vocab_size >= 128256 else 0
    ids = torch.randint(0, config.vocab_size - n_special, (BATCH_SIZE, SEQ_LEN),
                        generator=g).to(device)
    return {"input_ids": ids, "labels": ids}


def ds_config() -> dict:
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


def gib(x):
    return x / 2**30


def mem(tag, device, rank):
    if MEM_DEBUG and rank == 0:
        print(f"# mem[{tag:22s}] alloc={gib(torch.cuda.memory_allocated(device)):6.2f} "
              f"reserved={gib(torch.cuda.memory_reserved(device)):6.2f} "
              f"peak_alloc={gib(torch.cuda.max_memory_allocated(device)):6.2f} GiB")


def main():
    deepspeed.init_distributed()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(device)

    backend = detect_backend()

    set_seed(SEED)  # reproducible init + data; same weights on FA3 vs FA4 -> losses comparable
    model = build_model(backend)
    mem("after_build", device, rank)
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(),
                                            config=ds_config())
    mem("after_ds_init", device, rank)

    if rank == 0:
        print(env_line(backend))
        print(f"# gpu={torch.cuda.get_device_name()} attn_backend={backend} liger={USE_LIGER} "
              f"gc={GRAD_CHECKPOINT} offload_optim={OFFLOAD_OPTIMIZER} zero={ZERO_STAGE} "
              f"model_size={MODEL_SIZE} world_size={world_size} "
              f"batch_size={BATCH_SIZE} seq_len={SEQ_LEN}")

    tflos_per_gpu = estimate_tflos(model.config)
    output_file = open(OUTPUT, "a") if (OUTPUT and rank == 0) else None
    for step in range(WARMUP_STEPS + STEPS):
        batch = fake_batch(model.config, device, step)
        if MEM_DEBUG and step == WARMUP_STEPS:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.memory._record_memory_history(max_entries=200000)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = engine(**batch)
        loss = out.loss
        mem("after_fwd", device, rank)
        if MEM_DEBUG and step == WARMUP_STEPS and rank == 0:
            snap = f"mem_snapshot_seq{SEQ_LEN}_gc{GRAD_CHECKPOINT}_lg{USE_LIGER}.pickle"
            torch.cuda.memory._dump_snapshot(snap)
            print(f"# wrote {snap}")
        engine.backward(loss)
        mem("after_bwd", device, rank)
        engine.step()
        mem("after_step", device, rank)
        torch.cuda.synchronize()
        exec_time = time.perf_counter() - t0
        if MEM_DEBUG and step == WARMUP_STEPS:
            torch.cuda.memory._record_memory_history(enabled=None)

        peak = torch.tensor(torch.cuda.max_memory_reserved(device) / 2**30, device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if rank == 0:
            record = {"step": step, "warmup": step < WARMUP_STEPS,
                      "exec_time_ms": f"{exec_time * 1000:.2f}",
                      "tflops": f"{tflos_per_gpu / exec_time:.2f}",
                      "peak_mem_gib": f"{peak.item():.1f}"}
            print(json.dumps(record))
            if output_file:
                output_file.write(json.dumps(record) + "\n")
    if output_file:
        output_file.close()


if __name__ == "__main__":
    main()
