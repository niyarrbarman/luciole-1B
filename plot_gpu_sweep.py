#!/usr/bin/env python3
"""Combine the SSA-backward sweep TensorBoard logs into one figure + summary.

Usage:
    .venv/bin/python plot_gpu_sweep.py [tb_logs_dir] [out.png]
"""
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB_DIR = sys.argv[1] if len(sys.argv) > 1 else "tb_logs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "gpu_sweep.png"

# Measured steady-state s/step (from checkpoint-save deltas, warmup dropped)
SPS = {"w4s2": 29.2, "w4s3": 28.4, "w4s4": 27.4, "w8s2": 109.6, "w8s3": 47.4, "w8s4": 47.4}
H200 = 14.4  # reference s/step

# Fixed color per config: w4* cool (good family), w8* warm (bad family)
COLORS = {"w4s2": "#1f77b4", "w4s3": "#17becf", "w4s4": "#2ca02c",
          "w8s2": "#d62728", "w8s3": "#ff7f0e", "w8s4": "#9467bd"}
ORDER = ["w4s2", "w4s3", "w4s4", "w8s2", "w8s3", "w8s4"]


def cfg_of(path):
    m = re.search(r"(w\d+s\d+)", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def load(path):
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    for t in tags:
        s = ea.Scalars(t)
        out[t] = ([e.step for e in s], [e.value for e in s])
    return out


runs = {}
for d in sorted(glob.glob(os.path.join(TB_DIR, "*"))):
    if os.path.isdir(d):
        runs[cfg_of(d)] = load(d)
order = [c for c in ORDER if c in runs] + [c for c in runs if c not in ORDER]

PANELS = [
    ("gpu/util_pct", "SM utilization (%)", "gpu/util_pct"),
    ("gpu/sm_clock_mhz", "SM clock (MHz)", "gpu/sm_clock_mhz"),
    ("gpu/power_w", "Power draw (W)", "gpu/power_w"),
    ("gpu/temp_c", "Temp (°C)", "gpu/temp_c"),
    ("gpu/mem_reserved_gb", "Mem reserved (GB) — dashed=peak", "gpu/mem_reserved_gb"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.ravel()

for ax, (tag, title, _) in zip(axes, PANELS):
    for c in order:
        if tag in runs[c]:
            x, y = runs[c][tag]
            ax.plot(x, y, label=c, color=COLORS.get(c), lw=1.8, marker="o", ms=3)
    if tag == "gpu/mem_reserved_gb":
        for c in order:
            if "gpu/mem_max_reserved_gb" in runs[c]:
                x, y = runs[c]["gpu/mem_max_reserved_gb"]
                ax.plot(x, y, color=COLORS.get(c), lw=1.0, ls="--", alpha=0.6)
        ax.axhline(184, color="k", ls=":", lw=1, alpha=0.5)
        ax.text(0, 185, "184 GB cap", fontsize=8, alpha=0.7)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    if tag == "gpu/util_pct":
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, ncol=2, title="warps/stages")

# Final panel: s/step bar (the outcome)
ax = axes[5]
cfgs = [c for c in order if c in SPS]
vals = [SPS[c] for c in cfgs]
bars = ax.bar(cfgs, vals, color=[COLORS.get(c) for c in cfgs])
ax.axhline(H200, color="k", ls="--", lw=1.2)
ax.text(len(cfgs) - 0.5, H200 + 2, f"H200 = {H200}", fontsize=9, ha="right")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
ax.set_title("steady-state s/step (lower=better)", fontsize=11, weight="bold")
ax.set_ylabel("s/step")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("SSA backward sweep on B200 (GB200) — 1B model, mbs 8, 16 GPUs",
             fontsize=14, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"\nsaved -> {os.path.abspath(OUT)}\n")

# Numeric summary (steady state = drop first 3 samples)
print(f"{'config':7s} {'util%':>7s} {'clk MHz':>8s} {'power W':>8s} {'temp C':>7s} "
      f"{'memGB':>7s} {'s/step':>7s}")
for c in order:
    r = runs[c]

    def mean(tag, skip=3):
        if tag not in r:
            return float("nan")
        v = r[tag][1][skip:] or r[tag][1]
        return sum(v) / len(v)

    print(f"{c:7s} {mean('gpu/util_pct'):7.1f} {mean('gpu/sm_clock_mhz'):8.0f} "
          f"{mean('gpu/power_w'):8.0f} {mean('gpu/temp_c'):7.0f} "
          f"{mean('gpu/mem_reserved_gb'):7.1f} {SPS.get(c, float('nan')):7.1f}")
