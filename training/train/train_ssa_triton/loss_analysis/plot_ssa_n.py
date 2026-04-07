#!/usr/bin/env python3
"""Parse SLURM logs and plot SSA n values per layer over training.

Produces an interactive Plotly HTML with one line per layer.
By default excludes layer 23 (divergent scale). Use --all-layers to show it
in a separate subplot.
"""

import argparse
import re
import pathlib

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOG_DIR = pathlib.Path(__file__).parent / "slurm_logs"
SEQ_LENGTH = 4096
OUTPUT_HTML = pathlib.Path(__file__).parent / "ssa_n_plot.html"
OUTLIER_LAYER = 23

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
STEP_RE = re.compile(r"Step (\d+) - SSA n values:")
LAYER_RE = re.compile(r"Layer (\d+): n = ([\d.e+-]+)")

# Also parse training lines to build step → tokens mapping
TRAIN_RE = re.compile(
    r"consumed_samples: (\d+) \|.*global_step: (\d+)"
)


def parse_logs(log_dir: pathlib.Path):
    # Parse SSA n values
    n_rows = []
    for fpath in sorted(log_dir.glob("tr_nemo1b_ssa_triton_*.out")):
        job_id = fpath.stem.rsplit("_", 1)[-1]
        current_step = None
        for line in fpath.open():
            m = STEP_RE.search(line)
            if m:
                current_step = int(m.group(1))
                continue
            if current_step is not None:
                m = LAYER_RE.search(line)
                if m:
                    n_rows.append({
                        "global_step": current_step,
                        "layer": int(m.group(1)),
                        "n_value": float(m.group(2)),
                        "job_id": job_id,
                    })
                    if int(m.group(1)) == 23:  # last layer
                        current_step = None
                else:
                    current_step = None

    # Parse step → consumed_samples for tokens axis
    tok_rows = []
    for fpath in sorted(log_dir.glob("tr_nemo1b_ssa_triton_*.out")):
        for line in fpath.open():
            m = TRAIN_RE.search(line)
            if m:
                tok_rows.append({
                    "global_step": int(m.group(2)),
                    "consumed_samples": int(m.group(1)),
                })

    df_n = pd.DataFrame(n_rows)
    if df_n.empty:
        raise RuntimeError(f"No SSA n values found in {log_dir}")

    # Deduplicate
    df_n = df_n.sort_values(["global_step", "layer", "job_id"]).drop_duplicates(
        subset=["global_step", "layer"], keep="last"
    ).sort_values(["global_step", "layer"]).reset_index(drop=True)

    # Build step→tokens_B mapping
    df_tok = pd.DataFrame(tok_rows).drop_duplicates(subset="global_step", keep="last")
    df_tok["tokens_B"] = df_tok["consumed_samples"] * SEQ_LENGTH / 1e9
    step_to_tokens = df_tok.set_index("global_step")["tokens_B"].to_dict()
    df_n["tokens_B"] = df_n["global_step"].map(step_to_tokens)

    return df_n


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
# Qualitative palette for 24 layers
COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3",
    "#FF6692", "#B6E880", "#FF97FF", "#FECB52", "#1F77B4", "#FF7F0E",
    "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
    "#BCBD22", "#17BECF", "#AEC7E8", "#FFBB78", "#98DF8A", "#C5B0D5",
]


def _add_layer_traces(fig, df_n, layers, *, row=None, col=None):
    """Add one Scattergl trace per layer."""
    kwargs = {}
    if row is not None:
        kwargs["row"] = row
        kwargs["col"] = col
    for layer in layers:
        dl = df_n[df_n["layer"] == layer].sort_values("global_step")
        fig.add_trace(
            go.Scattergl(
                x=dl["global_step"],
                y=dl["n_value"],
                mode="lines+markers",
                line=dict(color=COLORS[layer % len(COLORS)], width=1.5),
                marker=dict(size=3),
                name=f"Layer {layer}",
                customdata=dl[["tokens_B"]].values,
                hovertemplate=(
                    f"Layer {layer}<br>"
                    "Step: %{x:,}<br>"
                    "n = %{y:.4f}<br>"
                    "Tokens: %{customdata[0]:.2f}B<br>"
                    "<extra></extra>"
                ),
            ),
            **kwargs,
        )


def make_plot(df_n: pd.DataFrame, *, all_layers: bool = False) -> go.Figure:
    last_step = df_n["global_step"].max()
    tok_min = df_n["tokens_B"].min()
    tok_max = df_n["tokens_B"].max()

    normal_layers = sorted(l for l in df_n["layer"].unique() if l != OUTLIER_LAYER)

    if all_layers:
        # Two subplots: top = layers 0-22, bottom = layer 23
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.75, 0.25],
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=["Layers 0–22", f"Layer {OUTLIER_LAYER} (outlier)"],
        )
        _add_layer_traces(fig, df_n, normal_layers, row=1, col=1)
        _add_layer_traces(fig, df_n, [OUTLIER_LAYER], row=2, col=1)

        fig.update_yaxes(title_text="SSA n value", row=1, col=1,
                         showgrid=True, gridcolor="rgba(200,200,200,0.3)")
        fig.update_yaxes(title_text="SSA n value", row=2, col=1,
                         showgrid=True, gridcolor="rgba(200,200,200,0.3)")
        fig.update_xaxes(title_text="Global Step", row=2, col=1,
                         showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    else:
        fig = go.Figure()
        _add_layer_traces(fig, df_n, normal_layers)
        fig.update_layout(
            xaxis=dict(
                title="Global Step",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.3)",
            ),
            yaxis=dict(
                title="SSA n value",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.3)",
            ),
        )

    fig.update_layout(
        title=dict(
            text=(
                f"Nemotron-1B SSA Triton — SSA n Values per Layer<br>"
                f"<sup>{last_step:,} steps · {tok_max:.1f}B tokens"
                f"{'' if all_layers else f' (layer {OUTLIER_LAYER} excluded)'}</sup>"
            ),
            x=0.5,
        ),
        template="plotly_white",
        legend=dict(
            title="Layer",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        hovermode="closest",
        autosize=True,
        margin=dict(l=60, r=120, t=80, b=50),
    )

    # Tokens axis on top
    dl0 = df_n[df_n["layer"] == 0].sort_values("global_step")
    if all_layers:
        fig.add_trace(
            go.Scattergl(
                x=dl0["tokens_B"], y=dl0["n_value"],
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip", xaxis="x3",
            ),
            row=1, col=1,
        )
        fig.update_layout(
            xaxis3=dict(
                title="Tokens (B)", overlaying="x", side="top",
                showgrid=False, range=[tok_min, tok_max],
            ),
        )
    else:
        fig.add_trace(
            go.Scattergl(
                x=dl0["tokens_B"], y=dl0["n_value"],
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip", xaxis="x2",
            )
        )
        fig.update_layout(
            xaxis2=dict(
                title="Tokens (B)", overlaying="x", side="top",
                showgrid=False, range=[tok_min, tok_max],
            ),
        )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot SSA n values from SLURM logs")
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help=f"Include layer {OUTLIER_LAYER} in a separate subplot (excluded by default)",
    )
    args = parser.parse_args()

    print(f"Parsing logs from {LOG_DIR} ...")
    df_n = parse_logs(LOG_DIR)

    steps = sorted(df_n["global_step"].unique())
    layers = sorted(df_n["layer"].unique())
    print(
        f"  {len(steps)} logged steps  "
        f"(step {steps[0]} → {steps[-1]})  "
        f"{len(layers)} layers"
    )

    fig = make_plot(df_n, all_layers=args.all_layers)
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
