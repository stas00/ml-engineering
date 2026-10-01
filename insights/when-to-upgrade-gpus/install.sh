#!/usr/bin/env bash
# Unified installer for the FA3/FA4 benchmark stack, for BOTH GPU types:
#   Hopper    (H100/H200)  -> FlashAttention-3
#   Blackwell (B200/B300)  -> FlashAttention-4
# Pins the identical torch/CUDA/deps on every box so the FA3-vs-FA4 comparison is
# apples-to-apples; the attention backend is the only intended difference (auto-detected).
#
# Usage: ./install.sh [--force fa3|fa4]
set -euo pipefail

TORCH=2.13.0
TRANSFORMERS=5.14.1
DEEPSPEED=0.19.2
LIGER=0.8.0
CU130=https://download.pytorch.org/whl/cu130

FORCE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# DeepSpeed JIT-builds cpu_adam with the system nvcc, which must match torch's CUDA major
# (13). Point the `cuda` alternative at 13.0. Non-fatal: boxes already on 13 (or without
# this alternative) just skip it.
sudo update-alternatives --set cuda /usr/local/cuda-13.0 2>/dev/null || true

echo "== installing torch==$TORCH (cu130) =="
uv pip install --upgrade "torch==$TORCH" --index-url "$CU130"

# A stale torchvision/torchaudio built against an older torch triggers a circular-import
# crash ("torchvision has no attribute 'extension'") the moment transformers imports
# PreTrainedModel. The benchmark is text-only, so remove them.
echo "== removing stale torchvision/torchaudio (not needed) =="
uv pip uninstall torchvision torchaudio 2>/dev/null || true

echo "== installing matched deps =="
uv pip install --upgrade "transformers==$TRANSFORMERS" "deepspeed==$DEEPSPEED" "liger-kernel==$LIGER"

# DeepSpeed JIT-compiles ops (e.g. cpu_adam, used by CPU optimizer offload) and caches them
# under ~/.cache/torch_extensions. After a torch upgrade the stale op fails to load ->
# "'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'". Clear so it rebuilds against
# torch $TORCH. (If the rebuild fails, nvcc/CUDA-dev is missing: prebuild instead with
# `DS_BUILD_CPU_ADAM=1 uv pip install --force-reinstall --no-deps --no-build-isolation
# deepspeed==$DEEPSPEED`.)
echo "== clearing stale JIT-compiled deepspeed ops =="
rm -rf ~/.cache/torch_extensions/* 2>/dev/null || true

detect_backend() {
  python3 -c "
import torch
major, _ = torch.cuda.get_device_capability()
print('fa3' if major == 9 else 'fa4' if major == 10 else 'unsupported')
"
}
BACKEND="${FORCE:-$(detect_backend)}"

if [[ "$BACKEND" == "fa3" ]]; then
  echo "== installing FlashAttention-3 (Hopper, cu130) =="
  # --no-deps so the FA3 wheel can't drag torch off the pin above
  uv pip install --no-deps flash-attn-3 --index-url "$CU130"
elif [[ "$BACKEND" == "fa4" ]]; then
  echo "== installing FlashAttention-4 (Blackwell, pure-python CuTe DSL) =="
  uv pip install --pre "flash-attn-4[cu13]"
else
  echo "unsupported GPU compute capability; need Hopper (9.x) or Blackwell (10.x)" >&2
  exit 1
fi

# Stage the model onto a fast local disk ($DATA_FAST) so training loads from local disk rather
# than a shared/network cache -> faster load. Re-copy when missing. cp -rL dereferences HF blob
# symlinks so the copy is self-contained. The benchmarks auto-prefer $DATA_FAST/<id> when present
# (else fall back to the HF cache). Override HF_CACHE / DATA_FAST for your environment.
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
DATA_FAST="${DATA_FAST:-$HOME/base-models}"
stage_model() {
  local id=$1 hub dest snap
  hub="$HF_CACHE/hub/models--${id//\//--}"
  dest="$DATA_FAST/$id"
  if [[ -f "$dest/config.json" ]]; then echo "   already staged: $dest"; return; fi
  if [[ ! -d "$hub" ]]; then echo "   WARNING: $id not cached in $HF_CACHE (run: HF_HOME=$HF_CACHE hf download $id)"; return; fi
  snap="$hub/snapshots/$(cat "$hub/refs/main")"
  mkdir -p "$dest" && cp -rL "$snap"/. "$dest"/
  echo "   staged $id -> $dest"
}
echo "== staging model to $DATA_FAST (local disk) =="
stage_model "meta-llama/Llama-3.1-8B"

echo "== verify =="
python3 - <<'PY'
import torch, importlib.metadata as md
print("torch     ", torch.__version__, "| cuda", torch.version.cuda)
print("gpu       ", torch.cuda.get_device_name())
for p in ("transformers", "deepspeed", "liger-kernel", "flash-attn-3", "flash-attn-4"):
    try: print(f"{p:13s}", md.version(p))
    except md.PackageNotFoundError: pass
# import + tiny call catches ABI mismatches immediately
major, _ = torch.cuda.get_device_capability()
if major == 9:
    from flash_attn_interface import flash_attn_func
else:
    from flash_attn.cute import flash_attn_func
q = torch.randn(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16)
o = flash_attn_func(q, q, q, causal=True)
o = o[0] if isinstance(o, tuple) else o
print("FA OK     ", tuple(o.shape))
PY
echo "== done =="
