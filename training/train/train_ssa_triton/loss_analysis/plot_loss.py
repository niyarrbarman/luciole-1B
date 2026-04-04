#!/usr/bin/env python3
"""Parse SLURM logs from nemotron-1B SSA Triton training and plot loss curves.

Produces an interactive Plotly HTML file with:
  - Train loss vs global step (primary x-axis)
  - Train loss vs tokens consumed (secondary x-axis)
  - Learning rate overlay
  - Smoothed (EMA) loss curve
  - SLURM job boundaries as vertical markers
"""

import argparse
import re
import glob
import pathlib

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOG_DIR = pathlib.Path(__file__).parent / "slurm_logs"
SEQ_LENGTH = 4096  # tokens per sample
OUTPUT_HTML = pathlib.Path(__file__).parent / "loss_plot.html"
EMA_ALPHA = 0.01  # smoothing factor (lower = smoother)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
LINE_RE = re.compile(
    r"Training epoch \d+, iteration (\d+)/\d+ \|"
    r" lr: ([\d.e+-]+) \|"
    r" consumed_samples: (\d+) \|"
    r" global_batch_size: (\d+) \|"
    r" global_step: (\d+) \|"
    r" reduced_train_loss: ([\d.e+-]+)"
)


def parse_logs(log_dir: pathlib.Path) -> pd.DataFrame:
    rows = []
    for fpath in sorted(log_dir.glob("tr_nemo1b_ssa_triton_*.out")):
        job_id = fpath.stem.rsplit("_", 1)[-1]
        for line in fpath.open():
            m = LINE_RE.search(line)
            if m:
                rows.append(
                    {
                        "iteration": int(m.group(1)),
                        "lr": float(m.group(2)),
                        "consumed_samples": int(m.group(3)),
                        "global_batch_size": int(m.group(4)),
                        "global_step": int(m.group(5)),
                        "reduced_train_loss": float(m.group(6)),
                        "job_id": job_id,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No training lines found in {log_dir}")

    # Deduplicate: when jobs overlap on restart, keep the *last* logged value
    # for each global_step (from the job that ran further).
    df = df.sort_values(["global_step", "job_id"]).drop_duplicates(
        subset="global_step", keep="last"
    )
    df = df.sort_values("global_step").reset_index(drop=True)

    # Derived columns
    df["tokens"] = df["consumed_samples"] * SEQ_LENGTH
    df["tokens_B"] = df["tokens"] / 1e9  # billions

    # EMA smoothed loss
    ema = []
    s = df["reduced_train_loss"].iloc[0]
    for val in df["reduced_train_loss"]:
        s = EMA_ALPHA * val + (1 - EMA_ALPHA) * s
        ema.append(s)
    df["loss_ema"] = ema

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plot(df: pd.DataFrame, *, smooth: bool = True) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- Per-step loss ---
    raw_opacity = 0.3 if smooth else 0.6
    fig.add_trace(
        go.Scattergl(
            x=df["global_step"],
            y=df["reduced_train_loss"],
            mode="lines",
            line=dict(width=0.5, color=f"rgba(99,110,250,{raw_opacity})"),
            name="Per-step loss",
            customdata=df[["tokens_B", "lr"]].values,
            hovertemplate=(
                "Step: %{x:,}<br>"
                "Loss: %{y:.4f}<br>"
                "Tokens: %{customdata[0]:.2f}B<br>"
                "LR: %{customdata[1]:.2e}<br>"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # --- Smoothed loss (EMA) ---
    if smooth:
        fig.add_trace(
            go.Scattergl(
                x=df["global_step"],
                y=df["loss_ema"],
                mode="lines",
                line=dict(color="rgb(99,110,250)", width=2),
                name="Smoothed loss",
                customdata=df[["tokens_B", "lr"]].values,
                hovertemplate=(
                    "Step: %{x:,}<br>"
                    "Loss (EMA): %{y:.4f}<br>"
                    "Tokens: %{customdata[0]:.2f}B<br>"
                    "LR: %{customdata[1]:.2e}<br>"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    # --- Learning rate on secondary y-axis ---
    fig.add_trace(
        go.Scattergl(
            x=df["global_step"],
            y=df["lr"],
            mode="lines",
            line=dict(color="rgba(239,85,59,0.6)", width=1.5, dash="dot"),
            name="Learning rate",
            hovertemplate="Step: %{x:,}<br>LR: %{y:.2e}<extra></extra>",
        ),
        secondary_y=True,
    )

    # --- SLURM job boundaries ---
    job_starts = df.groupby("job_id")["global_step"].min().sort_values()
    # Only mark boundaries for jobs that contributed >100 steps (skip tiny restarts)
    job_counts = df["job_id"].value_counts()
    sig_jobs = job_counts[job_counts > 100].index
    for job_id, step in job_starts.items():
        if job_id in sig_jobs:
            fig.add_vline(
                x=step,
                line=dict(color="rgba(150,150,150,0.3)", width=1, dash="dash"),
                annotation=dict(
                    text=f"Job {job_id}",
                    font=dict(size=8, color="gray"),
                    textangle=-90,
                    yanchor="top",
                ),
            )

    # --- Layout ---
    last_step = df["global_step"].iloc[-1]
    last_tokens_B = df["tokens_B"].iloc[-1]
    fig.update_layout(
        title=dict(
            text=(
                f"Nemotron-1B SSA Triton — Training Loss<br>"
                f"<sup>{last_step:,} steps · {last_tokens_B:.1f}B tokens</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(
            title="Global Step",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis=dict(
            title="Train Loss",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis2=dict(
            title="Learning Rate",
            showgrid=False,
        ),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
        autosize=True,
        margin=dict(l=60, r=60, t=80, b=50),
    )

    # --- Secondary x-axis for tokens (top) ---
    # We build a mapping step→tokens_B for tick labels
    step_min, step_max = df["global_step"].min(), df["global_step"].max()
    tok_min, tok_max = df["tokens_B"].min(), df["tokens_B"].max()

    # Add a transparent trace on xaxis2 for the tokens axis
    fig.add_trace(
        go.Scattergl(
            x=df["tokens_B"],
            y=df["loss_ema"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            xaxis="x2",
        ),
        secondary_y=False,
    )
    fig.update_layout(
        xaxis2=dict(
            title="Tokens (B)",
            overlaying="x",
            side="top",
            showgrid=False,
            range=[tok_min, tok_max],
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot training loss from SLURM logs")
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Show only per-step loss, skip the EMA smoothed curve",
    )
    args = parser.parse_args()

    print(f"Parsing logs from {LOG_DIR} ...")
    df = parse_logs(LOG_DIR)
    print(
        f"  {len(df):,} unique steps  "
        f"(step {df['global_step'].min()} → {df['global_step'].max()})  "
        f"from {df['job_id'].nunique()} SLURM jobs"
    )
    print(
        f"  Tokens: {df['tokens_B'].iloc[0]:.2f}B → {df['tokens_B'].iloc[-1]:.2f}B"
    )

    fig = make_plot(df, smooth=not args.raw_only)
    fig.write_html(
        str(OUTPUT_HTML),
        auto_open=False,
        full_html=True,
        include_plotlyjs=True,
        config={"responsive": True},
        default_width="100vw",
        default_height="100vh",
    )
    print(f"\nPlot saved to {OUTPUT_HTML}")
    print("Open in a browser to interact with the chart.")


if __name__ == "__main__":
    main()
