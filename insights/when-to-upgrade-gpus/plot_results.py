#!/usr/bin/env python3
"""Plot FA3(H200) vs FA4(B200) steady-state results vs sequence length.

Data: Llama-3.1-8B, 8 GPU, ZeRO-3 + activation checkpointing + CPU-offload + Liger-Kernel,
mbs=1 (all rows share this one config, so every speedup ratio is apples-to-apples)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

seqlen = [8192, 32768, 65536, 131072, 196608, 262144]
h200 = [141, 324, 391, 420, 422, 419]          # TFLOPS/GPU (causal MFU)
b200 = [167, 524, 719, 836, 852, 852]           # TFLOPS/GPU (causal MFU)
speedup = [1.19, 1.62, 1.84, 1.99, 2.02, 2.03]
attn_share = [13, 36, 53, 70, 77, 82]           # % of model FLOPs (causal)
h200_mem = [16.3, 25.8, 37.5, 66.5, 95.7, 124.5]  # peak GiB (max reserved, worst rank)
b200_mem = [16.3, 25.9, 37.6, 66.7, 95.7, 124.6]
xlab = ["8k", "32k", "64k", "128k", "192k", "256k"]
x = range(len(seqlen))

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

ax[0, 0].plot(x, b200, "o-", color="#76b900", lw=2, label="B200 (FA4)")
ax[0, 0].plot(x, h200, "s-", color="#1f77b4", lw=2, label="H200 (FA3)")
ax[0, 0].set_title("Throughput per GPU")
ax[0, 0].set_ylabel("TFLOPS / GPU")
ax[0, 0].set_ylim(0, 1000)
ax[0, 0].legend()

ax[0, 1].plot(x, speedup, "o-", color="#d62728", lw=2)
ax[0, 1].axhline(2.28, ls="--", color="gray", lw=1)
ax[0, 1].text(0.05, 2.30, "2.28x hardware peak ratio", color="gray", fontsize=9)
ax[0, 1].axhline(2.0, ls=":", color="black", lw=1)
for xi, s in zip(x, speedup):
    ax[0, 1].annotate(f"{s:.2f}x", (xi, s), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax[0, 1].set_title("B200 / H200 speedup")
ax[0, 1].set_ylabel("speedup (x)")
ax[0, 1].set_ylim(1.0, 2.4)

ax[1, 0].plot(x, attn_share, "o-", color="#9467bd", lw=2)
for xi, a in zip(x, attn_share):
    ax[1, 0].annotate(f"{a}%", (xi, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax[1, 0].set_title("Attention share of FLOPs (O(s\u00b2))")
ax[1, 0].set_ylabel("attention share (%)")
ax[1, 0].set_ylim(0, 100)

ax[1, 1].plot(x, b200_mem, "o-", color="#76b900", lw=2, label="B200")
ax[1, 1].plot(x, h200_mem, "s--", color="#1f77b4", lw=2, label="H200")
ax[1, 1].axhline(141, ls="--", color="#1f77b4", lw=1, alpha=0.6)
ax[1, 1].text(0.05, 143, "H200 141 GiB HBM", color="#1f77b4", fontsize=9)
ax[1, 1].axhline(180, ls="--", color="#76b900", lw=1, alpha=0.6)
ax[1, 1].text(0.05, 182, "B200 180 GiB HBM", color="#76b900", fontsize=9)
ax[1, 1].set_title("Peak memory / GPU (same on both)")
ax[1, 1].set_ylabel("peak reserved (GiB)")
ax[1, 1].set_ylim(0, 195)
ax[1, 1].legend(loc="center right")

for a in ax.flat:
    a.set_xticks(list(x))
    a.set_xticklabels(xlab)
    a.set_xlabel("sequence length")
    a.grid(True, alpha=0.3)

fig.suptitle("Llama-3.1-8B, 8xGPU, ZeRO-3 + activation checkpointing + CPU-offload + Liger-Kernel, mbs=1 (unpacked)", fontsize=12)
fig.tight_layout()
fig.savefig("results_plot.png", dpi=130, bbox_inches="tight")
print("wrote results_plot.png")
