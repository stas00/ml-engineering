#!/usr/bin/env bash
# Sweep the sequence lengths from README.md and log per-step JSON to a file.
# Auto-detects GPU/backend inside benchmark.py; run the same script on H200 and
# B200 (after ./install.sh on each) to get an apples-to-apples comparison.
#
# Usage: ./run_sweep.sh                     # default sweep, 8 GPUs
#        SEQLENS="8192 262144" ./run_sweep.sh
#        GPUS=4 STEPS=5 ./run_sweep.sh
set -euo pipefail

SEQLENS="${SEQLENS:-196608}"
#SEQLENS="${SEQLENS:-8192 32768 65536 131072 262144}"
GPUS="${GPUS:-8}"
STEPS="${STEPS:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
PORT="${PORT:-29540}"

gpu=$(python3 -c "import torch; print(torch.cuda.get_device_name().replace(' ','_'))" 2>/dev/null || echo gpu)
OUT="${OUT:-results_${gpu}_$(date +%Y%m%d_%H%M%S).log}"
echo "logging to $OUT"

for S in $SEQLENS; do
  echo "===== seq_len=$S =====" | tee -a "$OUT"
  SEQ_LEN=$S STEPS=$STEPS WARMUP_STEPS=$WARMUP_STEPS HF_HUB_OFFLINE=1 \
    deepspeed --master_port "$((PORT++))" --num_gpus="$GPUS" benchmark.py 2>&1 \
    | grep -E '# env|# gpu|"step"|out of memory|Error' | tee -a "$OUT" || true
done

echo "done -> $OUT"
