#!/usr/bin/env python3
"""Generate a publication-style PDF comparing Softmax and SSA training loss."""

import argparse
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plot_loss import LOG_DIR, parse_logs

OUTPUT_PDF = pathlib.Path(__file__).parent / "loss_plot_formal.pdf"
SOFTMAX_CSV = pathlib.Path(__file__).parent / "softmax_convergence_phase1.csv"
SSA_EMA_ALPHA = 0.003
SSA_MEDIAN_WINDOW = 2001
SOFTMAX_EMA_ALPHA = 0.25
SOFTMAX_MEDIAN_WINDOW = 7


def _ema(values, alpha: float) -> np.ndarray:
    out = np.empty(len(values), dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _downsample_indices(n_rows: int, max_points: int) -> np.ndarray:
    if n_rows <= max_points:
        return np.arange(n_rows)
    stride = max(1, math.ceil(n_rows / max_points))
    idx = np.arange(0, n_rows, stride)
    if idx[-1] != n_rows - 1:
        idx = np.append(idx, n_rows - 1)
    return idx


def _smooth_loss(values, *, median_window: int, ema_alpha: float) -> np.ndarray:
    window = max(1, int(median_window))
    if window % 2 == 0:
        window += 1
    filtered = (
        pd.Series(values, dtype="float64")
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    return _ema(filtered, ema_alpha)


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": True,
            "path.simplify_threshold": 0.5,
        }
    )


def _load_ssa_loss(ema_alpha: float, median_window: int) -> pd.DataFrame:
    df = parse_logs(LOG_DIR)
    return pd.DataFrame(
        {
            "tokens_B": df["tokens_B"].to_numpy(dtype=float),
            "loss": _smooth_loss(
                df["reduced_train_loss"].to_numpy(dtype=float),
                median_window=median_window,
                ema_alpha=ema_alpha,
            ),
        }
    )


def _load_softmax_loss(
    csv_path: pathlib.Path, *, ema_alpha: float, median_window: int
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing softmax CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"training_tokens", "training_loss"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    df = df.sort_values("training_tokens").reset_index(drop=True)
    return pd.DataFrame(
        {
            "tokens_B": df["training_tokens"].to_numpy(dtype=float) / 1e9,
            "loss": _smooth_loss(
                df["training_loss"].to_numpy(dtype=float),
                median_window=median_window,
                ema_alpha=ema_alpha,
            ),
        }
    )


def make_formal_plot(
    *,
    output: pathlib.Path,
    softmax_csv: pathlib.Path,
    start_tokens_b: float,
    ssa_ema_alpha: float,
    ssa_median_window: int,
    softmax_ema_alpha: float,
    softmax_median_window: int,
    max_points: int,
    include_inset: bool,
) -> tuple[float, float, float]:
    _setup_style()

    ssa = _load_ssa_loss(ssa_ema_alpha, ssa_median_window)
    softmax = _load_softmax_loss(
        softmax_csv,
        ema_alpha=softmax_ema_alpha,
        median_window=softmax_median_window,
    )
    ssa_main = ssa[ssa["tokens_B"] >= start_tokens_b].reset_index(drop=True)
    softmax_main = softmax[softmax["tokens_B"] >= start_tokens_b].reset_index(drop=True)
    if ssa_main.empty or softmax_main.empty:
        raise RuntimeError(f"No rows remain after --start-tokens-b {start_tokens_b}")

    fig, ax = plt.subplots(figsize=(6.75, 3.05))

    ssa_idx = _downsample_indices(len(ssa_main), max_points)
    ax.plot(
        ssa_main["tokens_B"].to_numpy()[ssa_idx],
        ssa_main["loss"].to_numpy()[ssa_idx],
        color="#0072B2",
        label="SSA",
        solid_capstyle="round",
    )
    ax.plot(
        softmax_main["tokens_B"],
        softmax_main["loss"],
        color="#D55E00",
        linestyle="--",
        linewidth=1.65,
        label="Softmax",
        solid_capstyle="round",
    )

    ax.set_xlabel("Training tokens (B)")
    ax.set_ylabel("Training loss")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2.0, width=0.6)
    ax.legend(loc="upper left", frameon=False, handlelength=2.4)

    main_loss = np.concatenate(
        [
            ssa_main["loss"].to_numpy(dtype=float),
            softmax_main["loss"].to_numpy(dtype=float),
        ]
    )
    y_min = float(np.nanmin(main_loss))
    y_max = float(np.nanmax(main_loss))
    pad = 0.04 * (y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(
        start_tokens_b,
        float(max(ssa["tokens_B"].max(), softmax["tokens_B"].max())),
    )

    if include_inset and start_tokens_b > 0:
        axins = inset_axes(ax, width="36%", height="46%", loc="upper right", borderpad=0.8)
        full_idx = _downsample_indices(len(ssa), min(max_points, 4000))
        axins.plot(
            ssa["tokens_B"].to_numpy()[full_idx],
            ssa["loss"].to_numpy()[full_idx],
            color="#0072B2",
            linewidth=0.95,
        )
        axins.plot(
            softmax["tokens_B"],
            softmax["loss"],
            color="#D55E00",
            linestyle="--",
            linewidth=0.9,
        )
        axins.set_title("Full range", fontsize=7, pad=2)
        axins.xaxis.set_major_locator(MaxNLocator(3))
        axins.yaxis.set_major_locator(MaxNLocator(3))
        axins.tick_params(axis="both", labelsize=6.5, length=2.5, width=0.6, pad=1)
        axins.grid(True, alpha=0.12)
        axins.set_xlim(
            0.0,
            float(max(ssa["tokens_B"].max(), softmax["tokens_B"].max())),
        )
        for spine in axins.spines.values():
            spine.set_linewidth(0.6)

    fig.savefig(output)
    plt.close(fig)

    return float(ssa["tokens_B"].max()), float(softmax["tokens_B"].max()), float(softmax["loss"].iloc[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a publication-style PDF comparing Softmax and SSA loss"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=OUTPUT_PDF,
        help=f"Output PDF path (default: {OUTPUT_PDF})",
    )
    parser.add_argument(
        "--softmax-csv",
        type=pathlib.Path,
        default=SOFTMAX_CSV,
        help=f"Softmax convergence CSV path (default: {SOFTMAX_CSV})",
    )
    parser.add_argument(
        "--start-tokens-b",
        type=float,
        default=20.0,
        help="Main-panel first token count in billions. The inset shows the full range.",
    )
    parser.add_argument(
        "--ssa-ema-alpha",
        type=float,
        default=SSA_EMA_ALPHA,
        help="EMA smoothing factor for the SSA loss curve",
    )
    parser.add_argument(
        "--ssa-median-window",
        type=int,
        default=SSA_MEDIAN_WINDOW,
        help="Centered rolling-median window, in SSA training steps",
    )
    parser.add_argument(
        "--softmax-ema-alpha",
        type=float,
        default=SOFTMAX_EMA_ALPHA,
        help="EMA smoothing factor for the Softmax loss curve",
    )
    parser.add_argument(
        "--softmax-median-window",
        type=int,
        default=SOFTMAX_MEDIAN_WINDOW,
        help="Centered rolling-median window, in Softmax CSV rows",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=12000,
        help="Maximum points in the vector path after uniform downsampling",
    )
    parser.add_argument(
        "--no-inset",
        action="store_true",
        help="Do not include the full-range inset",
    )
    args = parser.parse_args()

    ssa_tokens_b, softmax_tokens_b, softmax_final_loss = make_formal_plot(
        output=args.output,
        softmax_csv=args.softmax_csv,
        start_tokens_b=args.start_tokens_b,
        ssa_ema_alpha=args.ssa_ema_alpha,
        ssa_median_window=args.ssa_median_window,
        softmax_ema_alpha=args.softmax_ema_alpha,
        softmax_median_window=args.softmax_median_window,
        max_points=args.max_points,
        include_inset=not args.no_inset,
    )
    print(f"Saved {args.output}")
    print(f"SSA tokens: {ssa_tokens_b:.2f}B")
    print(f"Softmax tokens: {softmax_tokens_b:.2f}B (final loss {softmax_final_loss:.4f})")


if __name__ == "__main__":
    main()
