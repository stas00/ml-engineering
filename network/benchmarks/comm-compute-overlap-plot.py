#!/usr/bin/env python

"""
Draws the figures of comm-compute-overlap-bench.py's results in network/exposed-communication.md: each run's overlapped
step as a horizontal bar of its forward and backward compute alone, with the communication exposed in each phase and
after the optimizer step stacked after the phase it belongs to.

Each figure only puts together runs of the same model with the same tokens and sequence length, so that every bar
of a figure does the same work. Where the compute alone is the same in every row, each bar is labeled with its
compute slowdown, and where it differs, as between two schemes, with its overlapped step, which is what decides
between them.

The numbers are the medians the benchmark printed, in ms: the compute alone and the exposed communication of
forward and backward, the exposed post-step all_gather of ZeRO-1 and ZeRO-2, and the step's compute alone and
overlapped time, which the benchmark measures separately from the phases, so the segments of a bar can add up to
slightly more or less than its label.

Usage:

    python comm-compute-overlap-plot.py [--out-dir images]

Requires matplotlib.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SEGMENTS = (
    ("Forward compute", "#1f5fa8"),
    ("Forward exposed comm", "#e8590c"),
    ("Backward compute", "#74a9e0"),
    ("Backward exposed comm", "#f5a26b"),
    ("Post-step all_gather", "#a33a00"),
)

# row: label, forward compute, forward exposed, backward compute, backward exposed, post-step exposed,
#      step compute alone, step overlapped[, the compute slowdown the benchmark printed, where the ms above round to
#      a different one]
FIGURES = {
    "dp-seqlen8192": dict(
        title="Data-parallel patterns, 8192 tokens per GPU",
        subtitle="8x B200, 32 layers of Llama-3.1-8B, 8192 tokens per GPU in sequences of 8192",
        label="slowdown",
        rows=[
            ("fsdp reshard_after_forward=0", 116.61, 11.44, 273.57, 14.99, 0, 389.92, 414.93),
            ("ddp", 116.49, 0, 273.30, 30.87, 0, 389.31, 416.73),
            ("fsdp", 116.51, 11.80, 274.24, 23.19, 0, 390.15, 424.31),
            ("fsdp separate_rs_group=1", 116.55, 11.61, 273.95, 26.02, 0, 389.65, 426.84),
            ("ds-zero2 overlap_comm=1", 116.45, 0, 273.45, 21.42, 22.02, 389.30, 429.91),
            ("ds-zero3", 116.62, 12.00, 273.41, 32.57, 0, 389.82, 431.00),
            ("ds-zero2", 116.56, 0, 273.40, 44.83, 21.18, 389.13, 451.66),
        ]),
    "dp-seqlen16384": dict(
        title="Data-parallel patterns, 16384 tokens per GPU",
        subtitle="8x B200, 32 layers of Llama-3.1-8B, 16384 tokens per GPU in sequences of 16384",
        label="slowdown",
        rows=[
            ("fsdp reshard_after_forward=0", 262.11, 11.43, 637.95, 12.04, 0, 898.90, 921.14),
            ("fsdp", 262.55, 10.63, 637.76, 17.79, 0, 898.86, 926.74),
            ("ddp", 261.46, 0, 636.81, 36.81, 0, 897.91, 927.75),
            ("ds-zero2 overlap_comm=1", 262.48, 0, 636.56, 16.26, 21.65, 896.53, 928.50),
            ("fsdp separate_rs_group=1", 262.08, 10.92, 636.33, 23.22, 0, 897.62, 930.11),
            ("ds-zero3", 260.08, 16.00, 635.53, 32.23, 0, 894.26, 937.18),
            ("ds-zero2", 261.25, 0, 636.74, 45.75, 21.53, 895.94, 957.74),
        ]),
    "dp-1-vs-4-nodes": dict(
        title="Data-parallel patterns over 1 and 4 nodes",
        subtitle="8x B200 nodes, 32 layers of Llama-3.1-8B, 8192 tokens per GPU in sequences of 8192, from separate "
                 "jobs",
        label="slowdown", pairs=True,
        rows=[
            ("fsdp reshard_after_forward=0, 1 node", 116.61, 11.44, 273.57, 14.99, 0, 389.92, 414.93),
            ("fsdp reshard_after_forward=0, 4 nodes", 117.14, 13.94, 274.73, 19.55, 0, 390.66, 423.41),
            ("ddp, 1 node", 116.49, 0, 273.30, 30.87, 0, 389.31, 416.73),
            ("ddp, 4 nodes", 117.28, 0, 274.47, 67.00, 0, 391.21, 454.01),
            ("fsdp, 1 node", 116.51, 11.80, 274.24, 23.19, 0, 390.15, 424.31),
            ("fsdp, 4 nodes", 117.36, 14.42, 275.09, 35.22, 0, 391.66, 440.02),
            ("fsdp separate_rs_group=1, 1 node", 116.55, 11.61, 273.95, 26.02, 0, 389.65, 426.84),
            ("fsdp separate_rs_group=1, 4 nodes", 117.33, 14.17, 274.47, 32.23, 0, 391.02, 435.92),
            ("ds-zero2 overlap_comm=1, 1 node", 116.45, 0, 273.45, 21.42, 22.02, 389.30, 429.91),
            ("ds-zero2 overlap_comm=1, 4 nodes", 117.19, 0, 274.23, 28.46, 48.13, 390.55, 463.59),
            ("ds-zero3, 1 node", 116.62, 12.00, 273.41, 32.57, 0, 389.82, 431.00),
            ("ds-zero3, 4 nodes", 117.61, 16.65, 274.50, 51.12, 0, 390.66, 457.55),
            ("ds-zero2, 1 node", 116.56, 0, 273.40, 44.83, 21.18, 389.13, 451.66),
            ("ds-zero2, 4 nodes", 117.19, 0, 274.30, 81.87, 41.22, 390.51, 509.19),
        ]),
    "dp-1-vs-4-nodes-seqlen16384": dict(
        title="Data-parallel patterns over 1 and 4 nodes, 16384 tokens per GPU",
        subtitle="8x B200 nodes, 32 layers of Llama-3.1-8B, 16384 tokens per GPU in sequences of 16384, from "
                 "separate jobs",
        label="slowdown", pairs=True,
        rows=[
            ("fsdp reshard_after_forward=0, 1 node", 262.11, 11.43, 637.95, 12.04, 0, 898.90, 921.14),
            ("fsdp reshard_after_forward=0, 4 nodes", 264.25, 12.57, 639.36, 17.21, 0, 901.34, 929.07),
            ("fsdp, 1 node", 262.55, 10.63, 637.76, 17.79, 0, 898.86, 926.74),
            ("fsdp, 4 nodes", 264.04, 13.48, 640.55, 26.62, 0, 903.36, 940.60),
            ("ddp, 1 node", 261.46, 0, 636.81, 36.81, 0, 897.91, 927.75),
            ("ddp, 4 nodes", 264.49, 0, 639.74, 89.19, 0, 901.22, 981.92),
            ("ds-zero2 overlap_comm=1, 1 node", 262.48, 0, 636.56, 16.26, 21.65, 896.53, 928.50),
            ("ds-zero2 overlap_comm=1, 4 nodes", 264.00, 0, 639.31, 22.12, 46.02, 901.35, 959.64),
            ("fsdp separate_rs_group=1, 1 node", 262.08, 10.92, 636.33, 23.22, 0, 897.62, 930.11),
            ("fsdp separate_rs_group=1, 4 nodes", 264.54, 12.24, 638.52, 30.89, 0, 900.35, 942.04),
            ("ds-zero3, 1 node", 260.08, 16.00, 635.53, 32.23, 0, 894.26, 937.18),
            ("ds-zero3, 4 nodes", 262.93, 18.09, 638.22, 54.79, 0, 899.93, 964.54),
            ("ds-zero2, 1 node", 261.25, 0, 636.74, 45.75, 21.53, 895.94, 957.74),
            ("ds-zero2, 4 nodes", 264.69, 0, 639.33, 91.03, 44.88, 901.36, 1026.20),
        ]),
    "tp-sp-megatron": dict(
        title="Tensor parallelism, with and without Megatron-LM's sequence parallelism",
        subtitle="8x B200, 32 layers of Llama-3.1-8B, tp=8, the tokens per group in one sequence",
        label="step", pairs=True,
        rows=[
            ("tp, 8192 tokens", 18.92, 17.73, 38.92, 11.55, 0, 57.73, 87.14),
            ("sp-megatron, 8192 tokens", 16.14, 17.98, 35.02, 17.96, 0, 51.12, 87.10),
            ("tp, 16384 tokens", 37.52, 24.74, 90.51, 12.71, 0, 127.77, 165.50),
            ("sp-megatron, 16384 tokens", 32.16, 26.61, 82.04, 25.75, 0, 113.76, 166.55),
            ("tp, 32768 tokens", 88.51, 42.98, 215.54, 25.23, 0, 303.58, 372.27),
            ("sp-megatron, 32768 tokens", 77.06, 47.46, 202.62, 45.69, 0, 278.79, 372.67),
        ]),
    "sp-ulysses-cp-rsa": dict(
        title="One sequence split over 8 GPUs: DeepSpeed Ulysses and ring attention",
        subtitle="8x B200, 32 layers of Llama-3.1-8B, sp=8 or cp=8, one sequence per group",
        label="step", pairs=True,
        rows=[
            ("sp-ulysses, sequence of 16384", 33.00, 4.82, 80.54, 5.72, 0, 113.33, 124.04),
            ("cp-rsa, sequence of 16384", 46.03, 9.60, 98.32, 20.56, 0, 143.54, 174.44),
            ("sp-ulysses, sequence of 32768", 80.08, 8.29, 197.65, 4.49, 0, 276.71, 290.73),
            ("cp-rsa, sequence of 32768", 101.41, 9.18, 230.73, 21.10, 0, 330.20, 362.26),
            ("sp-ulysses, sequence of 65536", 214.41, 11.65, 555.95, 12.06, 0, 769.76, 794.09),
            ("cp-rsa, sequence of 65536", 261.47, 2.69, 613.03, 36.93, 0, 871.12, 912.78),
        ]),
    "ep-qwen3-30b-a3b": dict(
        title="Expert parallelism, Qwen3-30B-A3B, with 1 and 2 micro-batches",
        subtitle="8x B200, 32 layers, ep=8, the tokens per GPU in sequences of half the tokens",
        label="step", pairs=True,
        rows=[
            ("4096 tokens, 1 micro-batch", 33.07, 15.25, 54.87, 14.19, 0, 87.91, 117.36),
            ("4096 tokens, 2 micro-batches", 36.31, 4.92, 58.19, 7.98, 0, 94.47, 107.38),
            ("8192 tokens, 1 micro-batch", 65.52, 27.91, 118.21, 24.46, 0, 183.46, 236.10),
            ("8192 tokens, 2 micro-batches", 68.14, 8.48, 121.82, 13.07, 0, 189.90, 211.75),
            ("16384 tokens, 1 micro-batch", 143.82, 49.83, 274.07, 50.55, 0, 417.47, 518.12),
            ("16384 tokens, 2 micro-batches", 144.82, 18.59, 276.73, 22.94, 0, 421.25, 462.87),
        ]),
    "ep-mixtral-8x7b": dict(
        title="Expert parallelism, Mixtral-8x7B, with 1 and 2 micro-batches",
        subtitle="8x B200, 32 layers, ep=8, the tokens per GPU in sequences of half the tokens",
        label="step", pairs=True,
        rows=[
            ("4096 tokens, 1 micro-batch", 100.84, 9.20, 222.62, 6.67, 0, 323.08, 339.12),
            ("4096 tokens, 2 micro-batches", 103.25, 3.35, 225.48, 4.19, 0, 328.76, 336.15),
            ("8192 tokens, 1 micro-batch", 204.98, 15.77, 457.01, 16.54, 0, 660.27, 694.29),
            ("8192 tokens, 2 micro-batches", 209.11, 6.64, 464.73, 11.08, 0, 673.45, 691.69),
        ]),
    "ep-1-vs-4-nodes": dict(
        title="Expert parallelism, Qwen3-30B-A3B, over 1 and 4 nodes",
        subtitle="ep=8 over 1 8x B200 node and ep=32 over 4, from separate jobs, 32 layers, the tokens per GPU in "
                 "sequences of half the tokens",
        label="step", pairs=True,
        rows=[
            ("4096 tokens, 1 micro-batch, 1 node", 33.07, 15.25, 54.87, 14.19, 0, 87.91, 117.36),
            ("4096 tokens, 1 micro-batch, 4 nodes", 33.05, 152.14, 55.07, 151.22, 0, 88.06, 391.41),
            ("4096 tokens, 2 micro-batches, 1 node", 36.31, 4.92, 58.19, 7.98, 0, 94.47, 107.38),
            ("4096 tokens, 2 micro-batches, 4 nodes", 36.20, 136.36, 57.70, 117.84, 0, 93.83, 348.03),
            ("8192 tokens, 1 micro-batch, 1 node", 65.52, 27.91, 118.21, 24.46, 0, 183.46, 236.10),
            ("8192 tokens, 1 micro-batch, 4 nodes", 65.65, 291.45, 118.40, 284.93, 0, 183.73, 760.32),
            ("8192 tokens, 2 micro-batches, 1 node", 68.14, 8.48, 121.82, 13.07, 0, 189.90, 211.75),
            ("8192 tokens, 2 micro-batches, 4 nodes", 68.05, 264.12, 121.66, 219.02, 0, 189.60, 672.77),
        ]),
}

def draw(name, fig_def, out_dir):
    rows, pairs = fig_def["rows"], fig_def.get("pairs", False)
    # a gap between pairs, so that each pair reads as one comparison
    ys, y = [], 0.0
    for i in range(len(rows)):
        if pairs and i and i % 2 == 0:
            y += 0.5
        ys.append(y)
        y += 1
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.42 * y))
    widest = 0
    for (label, *v), y in zip(rows, ys):
        segments, step, overlapped = v[:5], v[5], v[6]
        slowdown = v[7] if len(v) > 7 else 100 * (overlapped - step) / step
        left = 0
        for value, (_, color) in zip(segments, SEGMENTS):
            if value:
                ax.barh(y, value, left=left, color=color, height=0.75, edgecolor="white", linewidth=0.5)
                left += value
        widest = max(widest, left)
        text = f"+{slowdown:.1f}%" if fig_def["label"] == "slowdown" else f"{overlapped:.2f}ms"
        ax.text(left, y, f" {text}", va="center", fontsize=8)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, widest * 1.12)
    ax.set_xlabel("ms", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", linewidth=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    what = ("compute slowdown" if fig_def["label"] == "slowdown" else "overlapped step")
    ax.set_title(f"{fig_def['title']}\n{fig_def['subtitle']} - each bar labeled with its {what}",
                 fontsize=9, loc="left")
    used = {i for r in rows for i, value in enumerate(r[1:6]) if value}
    fig.tight_layout()
    fig.legend(handles=[Patch(color=color, label=label) for i, (label, color) in enumerate(SEGMENTS) if i in used],
               fontsize=7, ncol=len(used), loc="upper center", bbox_to_anchor=(0.5, 0), frameon=False)
    path = out_dir / f"comm-compute-overlap-{name}-8xb200.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out-dir", default=Path(__file__).parent / "images", type=Path, help="where to save the PNGs")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig_def in FIGURES.items():
        draw(name, fig_def, args.out_dir)

if __name__ == "__main__":
    main()
