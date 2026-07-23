#!/usr/bin/env python3
"""Publication-style benchmark comparison plots: SSA vs Softmax, matching
the styling of loss_plot_formal.py (serif font, thin lines, clean spines)."""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

HERE = pathlib.Path(__file__).parent
OUT = HERE
OUT.mkdir(exist_ok=True)

DATA = json.load(open(HERE / "benchmark_data.json"))
TM = DATA["task_metrics"]
STEPS = [100, 200, 300, 400, 500, 715]

# color families -------------------------------------------------------
SSA_BLUE = "#0072B2"        # primary SSA color (matches loss plot)
SSA_BLUE_DARK = "#004C73"
SOFTMAX_RED = "#C0392B"     # primary softmax color
SOFTMAX_RED_DARK = "#7B241C"
INK = "#1f2937"
GRID_GRAY = "#9aa0a6"


def run_name(prefix: str, step: int) -> str:
    if prefix == "luciole":
        merged = 3 if step in (500, 715) else 2
        return f"luciole_{step}k (softmax) [merged:{merged}]"
    merged = 3 if step == 715 else 2
    return f"ssa_luciole_{step}k (ssa_triton) [merged:{merged}]"


SOFTMAX_RUNS = [run_name("luciole", s) for s in STEPS]
SSA_RUNS = [run_name("ssa_luciole", s) for s in STEPS]

PERPLEXITY_TASKS = {"wikitext", "french_bench_opus_perplexity"}
EXCLUDE_FROM_COMPOSITE = PERPLEXITY_TASKS | {"gsm8k"}

CORE_TASKS = [
    "arc_challenge", "arc_easy", "boolq", "hellaswag",
    "lambada_openai", "openbookqa", "record", "winogrande",
]

TASK_LABELS = {
    "arc_challenge": "ARC-Challenge",
    "arc_easy": "ARC-Easy",
    "boolq": "BoolQ",
    "hellaswag": "HellaSwag",
    "openbookqa": "OpenBookQA",
    "lambada_openai": "LAMBADA",
    "winogrande": "WinoGrande",
    "record": "ReCoRD (F1)",
}

# All 28 tasks used for the composite average, excluding gsm8k (near-zero,
# noise-floor at this scale) and the two perplexity-scored tasks (different
# unit than accuracy, can't share an axis with the rest).
ALL_TASKS = [
    "arc_challenge", "arc_easy", "boolq", "cb", "copa", "hellaswag",
    "lambada_openai", "multirc", "openbookqa", "record", "truthfulqa_mc2",
    "wic", "winogrande", "wsc",
    "french_bench_arc_challenge", "french_bench_boolqa", "french_bench_fquadv2",
    "french_bench_fquadv2_bool", "french_bench_fquadv2_genq",
    "french_bench_fquadv2_hasAns", "french_bench_grammar",
    "french_bench_hellaswag", "french_bench_multifquad",
    "french_bench_reading_comp", "french_bench_topic_based_nli",
    "french_bench_trivia", "french_bench_vocab", "french_bench_xnli",
]

TASK_LABELS_ALL = {
    "arc_challenge": "ARC-Challenge",
    "arc_easy": "ARC-Easy",
    "boolq": "BoolQ",
    "cb": "CB",
    "copa": "COPA",
    "hellaswag": "HellaSwag",
    "lambada_openai": "LAMBADA",
    "multirc": "MultiRC",
    "openbookqa": "OpenBookQA",
    "record": "ReCoRD (F1)",
    "truthfulqa_mc2": "TruthfulQA (MC2)",
    "wic": "WiC",
    "winogrande": "WinoGrande",
    "wsc": "WSC",
    "french_bench_arc_challenge": "FR ARC-Challenge",
    "french_bench_boolqa": "FR BoolQA",
    "french_bench_fquadv2": "FR FQuAD v2 (F1)",
    "french_bench_fquadv2_bool": "FR FQuAD v2 Bool",
    "french_bench_fquadv2_genq": "FR FQuAD v2 GenQ (F1)",
    "french_bench_fquadv2_hasAns": "FR FQuAD v2 HasAns (F1)",
    "french_bench_grammar": "FR Grammar",
    "french_bench_hellaswag": "FR HellaSwag",
    "french_bench_multifquad": "FR MultiFQuAD (F1)",
    "french_bench_reading_comp": "FR Reading Comp",
    "french_bench_topic_based_nli": "FR Topic NLI",
    "french_bench_trivia": "FR Trivia (F1)",
    "french_bench_vocab": "FR Vocab",
    "french_bench_xnli": "FR XNLI",
}


def pick_metric(metrics: dict) -> str:
    for pref in ("acc_norm,none", "acc,none", "f1,none", "exact_match,flexible-extract", "rouge1,none"):
        if pref in metrics:
            return pref
    raise KeyError("no usable metric")


def task_series(task: str, runs: list[str]) -> np.ndarray:
    metrics = TM[task]
    key = pick_metric(metrics)
    return np.array([metrics[key][r] for r in runs], dtype=float)


def composite_series(runs: list[str], tasks: list[str]) -> np.ndarray:
    vals = np.array([task_series(t, runs) for t in tasks])
    return vals.mean(axis=0)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "grid.color": GRID_GRAY,
            "lines.linewidth": 1.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": True,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
        }
    )


def find_crossover(ssa: np.ndarray, sm: np.ndarray):
    """First step where ssa >= sm; returns interpolated (x, y) and index."""
    for i in range(1, len(STEPS)):
        if ssa[i] >= sm[i]:
            x0, x1 = STEPS[i - 1], STEPS[i]
            y0s, y1s = sm[i - 1], sm[i]
            y0a, y1a = ssa[i - 1], ssa[i]
            denom = (y1a - y0a) - (y1s - y0s)
            t = (y0s - y0a) / denom if denom != 0 else 0.5
            t = min(max(t, 0.0), 1.0)
            cx = x0 + t * (x1 - x0)
            cy = y0a + t * (y1a - y0a)
            return cx, cy, i
    return None


# ---------------------------------------------------------------------
# Figure 1: composite benchmark average vs training steps (headline)
# ---------------------------------------------------------------------
def fig_composite_progression():
    all_tasks = [t for t in TM if t not in EXCLUDE_FROM_COMPOSITE]
    ssa_comp = composite_series(SSA_RUNS, all_tasks) * 100
    sm_comp = composite_series(SOFTMAX_RUNS, all_tasks) * 100

    fig, ax = plt.subplots(figsize=(6.75, 4.15))

    ax.plot(STEPS, sm_comp, color=SOFTMAX_RED, linestyle="--", marker="o",
            markersize=4.5, markerfacecolor="white", markeredgewidth=1.3,
            markeredgecolor=SOFTMAX_RED, label="Softmax", solid_capstyle="round")
    ax.plot(STEPS, ssa_comp, color=SSA_BLUE, linestyle="-", marker="o",
            markersize=4.5, markerfacecolor=SSA_BLUE, markeredgewidth=0,
            label="SSA", solid_capstyle="round")

    all_vals = np.concatenate([ssa_comp, sm_comp])
    y_min, y_max = float(all_vals.min()), float(all_vals.max())
    pad_lo = 0.10 * (y_max - y_min)
    pad_hi = 0.18 * (y_max - y_min)
    ax.set_xlim(70, 745)
    ax.set_ylim(y_min - pad_lo, y_max + pad_hi)

    cross = find_crossover(ssa_comp, sm_comp)
    if cross is not None:
        cx, cy, _ = cross
        ax.axvline(cx, color=GRID_GRAY, linewidth=0.9, linestyle=":", zorder=0)
        ax.annotate(
            "SSA overtakes\nSoftmax",
            xy=(cx, cy), xycoords="data",
            xytext=(0.30, 0.26), textcoords="axes fraction",
            fontsize=8.7, color=INK, ha="left",
            arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=0.8,
                             shrinkA=2, shrinkB=4, connectionstyle="arc3,rad=0.28"),
        )

    gap = ssa_comp[-1] - sm_comp[-1]
    ax.annotate(
        f"+{gap:.1f} pts at 715k steps",
        xy=(715, ssa_comp[-1]), xycoords="data",
        xytext=(0.98, 0.90), textcoords="axes fraction",
        fontsize=8.7, color=SSA_BLUE_DARK, ha="right", fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=SSA_BLUE_DARK, linewidth=0.8,
                         shrinkA=2, shrinkB=4, connectionstyle="arc3,rad=-0.15"),
    )

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Benchmark average accuracy (%)")
    ax.set_title(
        "SSA vs. Softmax — aggregate benchmark accuracy across training\n"
        f"({len(all_tasks)}-task average, English + French benchmarks)",
        pad=10, fontsize=10,
    )
    ax.set_xticks(STEPS)
    ax.set_xticklabels([f"{s}k" for s in STEPS])
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8)
    ax.legend(loc="lower right", frameon=False, handlelength=2.4)
    fig.savefig(OUT / "fig1_composite_progression.pdf")
    fig.savefig(OUT / "fig1_composite_progression.png")
    plt.close(fig)

    print("Fig 1 — composite average (%)")
    for s, a, b in zip(STEPS, ssa_comp, sm_comp):
        print(f"  {s:>4}k  SSA={a:6.2f}  Softmax={b:6.2f}  diff={a - b:+.2f}")


# ---------------------------------------------------------------------
# Figure 2: small multiples of individual benchmark tasks
# ---------------------------------------------------------------------
def fig_task_small_multiples():
    tasks = CORE_TASKS
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.2), sharex=True)
    axes = axes.flatten()

    for ax, task in zip(axes, tasks):
        ssa = task_series(task, SSA_RUNS) * 100
        sm = task_series(task, SOFTMAX_RUNS) * 100
        ax.plot(STEPS, sm, color=SOFTMAX_RED, linestyle="--", marker="o",
                markersize=3.2, markerfacecolor="white", markeredgewidth=1.0,
                markeredgecolor=SOFTMAX_RED, linewidth=1.4)
        ax.plot(STEPS, ssa, color=SSA_BLUE, linestyle="-", marker="o",
                markersize=3.2, markerfacecolor=SSA_BLUE, markeredgewidth=0,
                linewidth=1.4)
        ax.set_title(TASK_LABELS[task], fontsize=9.5, pad=4)
        ax.set_xticks(STEPS)
        ax.set_xticklabels([f"{s}k" for s in STEPS], fontsize=7, rotation=0)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="both", which="major", length=3, width=0.7)
        ax.grid(True, alpha=0.15)

    fig.supxlabel("Training steps", fontsize=10, y=0.02)
    fig.supylabel("Accuracy (%)", fontsize=10, x=0.005)

    handles = [
        plt.Line2D([0], [0], color=SOFTMAX_RED, linestyle="--", marker="o",
                    markersize=4.5, markerfacecolor="white", markeredgewidth=1.2,
                    markeredgecolor=SOFTMAX_RED, label="Softmax"),
        plt.Line2D([0], [0], color=SSA_BLUE, linestyle="-", marker="o",
                    markersize=4.5, markerfacecolor=SSA_BLUE, markeredgewidth=0,
                    label="SSA"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=10, handlelength=2.4)
    fig.suptitle("")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.94))
    fig.savefig(OUT / "fig2_task_small_multiples.pdf")
    fig.savefig(OUT / "fig2_task_small_multiples.png")
    plt.close(fig)
    print("Fig 2 saved (8-task small multiples)")


# ---------------------------------------------------------------------
# Figure 3: final head-to-head comparison at 715k steps
# ---------------------------------------------------------------------
def fig_final_comparison():
    tasks = CORE_TASKS
    ssa_final = np.array([task_series(t, [SSA_RUNS[-1]])[0] for t in tasks]) * 100
    sm_final = np.array([task_series(t, [SOFTMAX_RUNS[-1]])[0] for t in tasks]) * 100

    order = np.argsort(ssa_final - sm_final)[::-1]
    tasks_sorted = [tasks[i] for i in order]
    ssa_sorted = ssa_final[order]
    sm_sorted = sm_final[order]
    labels = [TASK_LABELS[t] for t in tasks_sorted]

    all_tasks = [t for t in TM if t not in EXCLUDE_FROM_COMPOSITE]
    ssa_comp = composite_series([SSA_RUNS[-1]], all_tasks)[0] * 100
    sm_comp = composite_series([SOFTMAX_RUNS[-1]], all_tasks)[0] * 100

    labels = labels + ["", f"Overall avg.\n({len(all_tasks)} tasks)"]
    ssa_sorted = np.append(ssa_sorted, [np.nan, ssa_comp])
    sm_sorted = np.append(sm_sorted, [np.nan, sm_comp])

    y = np.arange(len(labels))
    height = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.barh(y + height / 2, sm_sorted, height=height, color=SOFTMAX_RED,
            label="Softmax", zorder=3)
    ax.barh(y - height / 2, ssa_sorted, height=height, color=SSA_BLUE,
            label="SSA", zorder=3)

    for yi, (a, b) in enumerate(zip(ssa_sorted, sm_sorted)):
        if np.isnan(a):
            continue
        ax.text(a + 0.6, yi - height / 2, f"{a:.1f}", va="center", ha="left",
                fontsize=7.3, color=SSA_BLUE_DARK)
        ax.text(b + 0.6, yi + height / 2, f"{b:.1f}", va="center", ha="left",
                fontsize=7.3, color=SOFTMAX_RED_DARK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Final comparison at 715k training steps", pad=10)
    ax.set_xlim(0, max(np.nanmax(ssa_sorted), np.nanmax(sm_sorted)) * 1.15)
    ax.grid(True, axis="x", alpha=0.18)
    ax.grid(False, axis="y")
    ax.axhline(len(tasks) + 0.5, color=GRID_GRAY, linewidth=0.8, linestyle="-")
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, fontsize=9.5)
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_final_comparison_715k.pdf")
    fig.savefig(OUT / "fig3_final_comparison_715k.png")
    plt.close(fig)

    print("Fig 3 — final @715k (%)")
    for t, a, b in zip(tasks_sorted, ssa_final[order], sm_final[order]):
        print(f"  {TASK_LABELS[t]:<14} SSA={a:6.2f}  Softmax={b:6.2f}  diff={a - b:+.2f}")
    print(f"  {'Overall avg':<14} SSA={ssa_comp:6.2f}  Softmax={sm_comp:6.2f}  diff={ssa_comp - sm_comp:+.2f}")


# ---------------------------------------------------------------------
# Figure 4: small multiples across ALL 28 tasks (English + French)
# ---------------------------------------------------------------------
def fig_task_all_grid():
    tasks = ALL_TASKS
    ncols, nrows = 7, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.5, 8.6), sharex=True)
    axes = axes.flatten()

    for ax, task in zip(axes, tasks):
        ssa = task_series(task, SSA_RUNS) * 100
        sm = task_series(task, SOFTMAX_RUNS) * 100
        ax.plot(STEPS, sm, color=SOFTMAX_RED, linestyle="--", marker="o",
                markersize=2.8, markerfacecolor="white", markeredgewidth=0.9,
                markeredgecolor=SOFTMAX_RED, linewidth=1.2)
        ax.plot(STEPS, ssa, color=SSA_BLUE, linestyle="-", marker="o",
                markersize=2.8, markerfacecolor=SSA_BLUE, markeredgewidth=0,
                linewidth=1.2)
        ax.set_title(TASK_LABELS_ALL[task], fontsize=8.3, pad=3)
        ax.set_xticks(STEPS)
        ax.set_xticklabels([f"{s}k" for s in STEPS], fontsize=6, rotation=45)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="both", which="major", length=2.5, width=0.6)
        ax.grid(True, alpha=0.15)

    for ax in axes[len(tasks):]:
        ax.axis("off")

    fig.supxlabel("Training steps", fontsize=10.5, y=0.01)
    fig.supylabel("Accuracy (%)", fontsize=10.5, x=0.005)

    handles = [
        plt.Line2D([0], [0], color=SOFTMAX_RED, linestyle="--", marker="o",
                    markersize=4.5, markerfacecolor="white", markeredgewidth=1.2,
                    markeredgecolor=SOFTMAX_RED, label="Softmax"),
        plt.Line2D([0], [0], color=SSA_BLUE, linestyle="-", marker="o",
                    markersize=4.5, markerfacecolor=SSA_BLUE, markeredgewidth=0,
                    label="SSA"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.03), fontsize=11, handlelength=2.4)
    fig.suptitle(
        "SSA vs. Softmax at each training checkpoint",
        fontsize=10.5, y=1.065,
    )
    fig.tight_layout(rect=(0.015, 0.025, 1, 0.955))
    fig.savefig(OUT / "fig4_all28_tasks_progression.pdf")
    fig.savefig(OUT / "fig4_all28_tasks_progression.png")
    plt.close(fig)
    print(f"Fig 4 saved ({len(tasks)}-task grid)")


# ---------------------------------------------------------------------
# Figure 5: final head-to-head comparison at 715k, all 28 tasks, no average
# ---------------------------------------------------------------------
def fig_final_comparison_all():
    tasks = ALL_TASKS
    ssa_final = np.array([task_series(t, [SSA_RUNS[-1]])[0] for t in tasks]) * 100
    sm_final = np.array([task_series(t, [SOFTMAX_RUNS[-1]])[0] for t in tasks]) * 100

    order = np.argsort(ssa_final - sm_final)[::-1]
    tasks_sorted = [tasks[i] for i in order]
    ssa_sorted = ssa_final[order]
    sm_sorted = sm_final[order]
    labels = [TASK_LABELS_ALL[t] for t in tasks_sorted]

    y = np.arange(len(labels))
    height = 0.36

    fig, ax = plt.subplots(figsize=(7.6, 10.2))
    ax.barh(y + height / 2, sm_sorted, height=height, color=SOFTMAX_RED,
            label="Softmax", zorder=3)
    ax.barh(y - height / 2, ssa_sorted, height=height, color=SSA_BLUE,
            label="SSA", zorder=3)

    for yi, (a, b) in enumerate(zip(ssa_sorted, sm_sorted)):
        ax.text(a + 0.7, yi - height / 2, f"{a:.1f}", va="center", ha="left",
                fontsize=6.8, color=SSA_BLUE_DARK)
        ax.text(b + 0.7, yi + height / 2, f"{b:.1f}", va="center", ha="left",
                fontsize=6.8, color=SOFTMAX_RED_DARK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.3)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("comparison at 715k steps", pad=10)
    ax.set_xlim(0, max(ssa_sorted.max(), sm_sorted.max()) * 1.15)
    ax.grid(True, axis="x", alpha=0.18)
    ax.grid(False, axis="y")
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, fontsize=9.5)
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_all28_tasks_final_comparison_715k.pdf")
    fig.savefig(OUT / "fig5_all28_tasks_final_comparison_715k.png")
    plt.close(fig)

    print(f"Fig 5 — final @715k, all {len(tasks)} tasks (%)")
    for t, a, b in zip(tasks_sorted, ssa_sorted, sm_sorted):
        print(f"  {TASK_LABELS_ALL[t]:<24} SSA={a:6.2f}  Softmax={b:6.2f}  diff={a - b:+.2f}")
    wins = int((ssa_sorted > sm_sorted).sum())
    print(f"  SSA ahead on {wins}/{len(tasks)} tasks at 715k steps")


if __name__ == "__main__":
    setup_style()
    fig_composite_progression()
    fig_task_small_multiples()
    fig_final_comparison()
    fig_task_all_grid()
    fig_final_comparison_all()
    print(f"\nSaved outputs to {OUT}")
