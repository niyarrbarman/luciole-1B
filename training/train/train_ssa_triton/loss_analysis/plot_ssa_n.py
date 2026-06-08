#!/usr/bin/env python3
"""Parse SLURM logs and plot SSA n values per layer over training.

Produces an interactive Plotly HTML with one line per layer.
By default excludes layer 23 (divergent scale). Use --all-layers to show it
in a separate subplot.
"""

import argparse
import json
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

# Also parse training lines for tokens / LR / job ownership
TRAIN_RE = re.compile(
    r"lr: ([\d.e+-]+) \| consumed_samples: (\d+) \|.*?global_step: (\d+)"
)


def parse_logs(log_dir: pathlib.Path):
    n_rows = []
    train_rows = []
    for fpath in sorted(log_dir.glob("tr_nemo1b_ssa_triton_*.out")):
        job_id = fpath.stem.rsplit("_", 1)[-1]
        current_step = None
        for line in fpath.open():
            m = STEP_RE.search(line)
            if m:
                current_step = int(m.group(1))
                continue
            if current_step is not None:
                ml = LAYER_RE.search(line)
                if ml:
                    n_rows.append({
                        "global_step": current_step,
                        "layer": int(ml.group(1)),
                        "n_value": float(ml.group(2)),
                        "job_id": job_id,
                    })
                    if int(ml.group(1)) == 23:
                        current_step = None
                else:
                    current_step = None
            mt = TRAIN_RE.search(line)
            if mt:
                train_rows.append({
                    "lr": float(mt.group(1)),
                    "consumed_samples": int(mt.group(2)),
                    "global_step": int(mt.group(3)),
                    "job_id": job_id,
                })

    df_n = pd.DataFrame(n_rows)
    if df_n.empty:
        raise RuntimeError(f"No SSA n values found in {log_dir}")
    df_n = df_n.sort_values(["global_step", "layer", "job_id"]).drop_duplicates(
        subset=["global_step", "layer"], keep="last"
    ).sort_values(["global_step", "layer"]).reset_index(drop=True)

    df_t = pd.DataFrame(train_rows)
    df_t = (
        df_t.sort_values(["global_step", "job_id"])
        .drop_duplicates(subset="global_step", keep="last")
        .sort_values("global_step")
        .reset_index(drop=True)
    )
    df_t["tokens_B"] = df_t["consumed_samples"] * SEQ_LENGTH / 1e9

    step_to_tokens = df_t.set_index("global_step")["tokens_B"].to_dict()
    df_n["tokens_B"] = df_n["global_step"].map(step_to_tokens)

    return df_n, df_t


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


def _build_grid_specs(df_n: pd.DataFrame):
    """Build per-layer Plotly specs (small for grid, big for fullscreen)."""
    layers = [int(l) for l in sorted(df_n["layer"].unique())]
    small, big = {}, {}
    for layer in layers:
        dl = df_n[df_n["layer"] == layer].sort_values("global_step")
        color = COLORS[layer % len(COLORS)]
        x = dl["global_step"].tolist()
        y = dl["n_value"].tolist()
        tokens = dl["tokens_B"].tolist()

        small[layer] = {
            "data": [{
                "type": "scatter",
                "x": x, "y": y,
                "mode": "lines",
                "line": {"color": color, "width": 1.2},
                "hovertemplate": f"L{layer} step %{{x:,}}<br>n=%{{y:.4f}}<extra></extra>",
            }],
            "layout": {
                "title": {"text": f"Layer {layer}", "font": {"size": 12}, "x": 0.5, "y": 0.95},
                "margin": {"l": 38, "r": 8, "t": 26, "b": 26},
                "template": "plotly_white",
                "showlegend": False,
            },
        }
        big[layer] = {
            "data": [{
                "type": "scattergl",
                "x": x, "y": y,
                "customdata": [[t] for t in tokens],
                "mode": "lines+markers",
                "line": {"color": color, "width": 1.8},
                "marker": {"size": 3},
                "hovertemplate": (
                    f"Layer {layer}<br>"
                    "Step: %{x:,}<br>n = %{y:.4f}<br>"
                    "Tokens: %{customdata[0]:.2f}B<extra></extra>"
                ),
            }],
            "layout": {
                "title": {"text": f"Layer {layer} — SSA n", "x": 0.5},
                "xaxis": {"title": "Global Step", "showgrid": True, "gridcolor": "rgba(200,200,200,0.3)"},
                "yaxis": {"title": "SSA n value", "showgrid": True, "gridcolor": "rgba(200,200,200,0.3)"},
                "template": "plotly_white",
                "margin": {"l": 70, "r": 30, "t": 70, "b": 60},
            },
        }
    return layers, small, big


def write_grid_html(
    df_n: pd.DataFrame,
    df_t: pd.DataFrame,
    last_step: int,
    tok_max: float,
    output_path: pathlib.Path,
):
    """Write a self-contained HTML with clickable 6x4 grid + fullscreen + LR/job toggles."""
    layers, small_specs, big_specs = _build_grid_specs(df_n)

    # Downsampled LR (small for grid, more detail for fullscreen)
    lr_small_step = max(1, len(df_t) // 500)
    lr_small_x = df_t["global_step"].iloc[::lr_small_step].astype(int).tolist()
    lr_small_y = df_t["lr"].iloc[::lr_small_step].astype(float).tolist()
    lr_big_step = max(1, len(df_t) // 4000)
    lr_big_x = df_t["global_step"].iloc[::lr_big_step].astype(int).tolist()
    lr_big_y = df_t["lr"].iloc[::lr_big_step].astype(float).tolist()

    # Significant job boundaries (>100 steps each)
    job_starts = df_t.groupby("job_id")["global_step"].min().sort_values()
    job_counts = df_t["job_id"].value_counts()
    sig_jobs = set(job_counts[job_counts > 100].index)
    job_bounds = [int(s) for j, s in job_starts.items() if j in sig_jobs]

    template = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SSA n per Layer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font-family: system-ui, -apple-system, sans-serif; }
  #header { padding: 8px 16px; border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 18px; height: 56px; flex-wrap: wrap; }
  #title { font-size: 15px; font-weight: 600; }
  #subtitle { font-size: 11px; color: #666; }
  #back-btn { padding: 6px 14px; border: 1px solid #ccc; background: #f7f7f7; border-radius: 4px; cursor: pointer; font-size: 13px; }
  #back-btn:hover { background: #ececec; }
  .toggle { font-size: 13px; user-select: none; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }
  .toggle input { cursor: pointer; }
  #grid { display: grid; grid-template-columns: repeat(4, 1fr); grid-template-rows: repeat(6, 1fr); gap: 6px; padding: 6px; height: calc(100vh - 56px); }
  .cell { border: 1px solid #eee; border-radius: 4px; cursor: pointer; transition: border-color 0.15s, box-shadow 0.15s; min-width: 0; min-height: 0; overflow: hidden; }
  .cell:hover { border-color: #636efa; box-shadow: 0 2px 6px rgba(99,110,250,0.15); }
  #fullscreen { display: none; height: calc(100vh - 56px); padding: 6px; }
  #full-plot { width: 100%; height: 100%; }
</style>
</head>
<body>
<div id="header">
  <div>
    <div id="title">Nemotron-1B SSA Triton — SSA n per Layer</div>
    <div id="subtitle">__SUBTITLE__</div>
  </div>
  <label class="toggle"><input type="checkbox" id="toggle-lr" checked> Learning rate</label>
  <label class="toggle"><input type="checkbox" id="toggle-jobs" checked> Job boundaries</label>
  <button id="back-btn" style="display:none;" onclick="showGrid()">&larr; Back to grid</button>
</div>
<div id="grid"></div>
<div id="fullscreen"><div id="full-plot"></div></div>

<script>
const SMALL = __SMALL__;
const BIG = __BIG__;
const LAYERS = __LAYERS__;
const LR_SMALL = {x: __LR_SMALL_X__, y: __LR_SMALL_Y__};
const LR_BIG = {x: __LR_BIG_X__, y: __LR_BIG_Y__};
const JOB_BOUNDS = __JOB_BOUNDS__;

let showLR = true;
let showJobs = true;

const grid = document.getElementById('grid');
const fullscreen = document.getElementById('fullscreen');
const fullPlot = document.getElementById('full-plot');
const backBtn = document.getElementById('back-btn');
let currentBigLayer = null;

function makeShapes() {
  if (!showJobs) return [];
  return JOB_BOUNDS.map(s => ({
    type: 'line', xref: 'x', x0: s, x1: s, yref: 'paper', y0: 0, y1: 1,
    line: {color: 'rgba(150,150,150,0.4)', width: 1, dash: 'dash'},
  }));
}

function lrTraceSmall() {
  return {
    type: 'scatter', x: LR_SMALL.x, y: LR_SMALL.y, yaxis: 'y2', mode: 'lines',
    line: {color: 'rgba(239,85,59,0.6)', width: 1, dash: 'dot'},
    hoverinfo: 'skip', showlegend: false, name: 'LR',
    visible: showLR ? true : false,
  };
}
function lrTraceBig() {
  return {
    type: 'scattergl', x: LR_BIG.x, y: LR_BIG.y, yaxis: 'y2', mode: 'lines',
    line: {color: 'rgba(239,85,59,0.6)', width: 1.5, dash: 'dot'},
    hovertemplate: 'Step: %{x:,}<br>LR: %{y:.2e}<extra></extra>',
    name: 'Learning rate',
    visible: showLR ? true : false,
  };
}

LAYERS.forEach(layer => {
  const cell = document.createElement('div');
  cell.className = 'cell';
  cell.id = 'cell-' + layer;
  cell.onclick = () => showFull(layer);
  grid.appendChild(cell);
  const data = [SMALL[layer].data[0], lrTraceSmall()];
  const layout = Object.assign({}, SMALL[layer].layout, {
    shapes: makeShapes(),
    yaxis2: {overlaying: 'y', side: 'right', showgrid: false, visible: false},
  });
  Plotly.newPlot(cell, data, layout, {responsive: true, displayModeBar: false});
});

function showFull(layer) {
  grid.style.display = 'none';
  fullscreen.style.display = 'block';
  backBtn.style.display = 'inline-block';
  currentBigLayer = layer;
  const data = [BIG[layer].data[0], lrTraceBig()];
  const layout = Object.assign({}, BIG[layer].layout, {
    shapes: makeShapes(),
    yaxis2: {title: 'Learning rate', overlaying: 'y', side: 'right', showgrid: false},
  });
  Plotly.newPlot(fullPlot, data, layout, {responsive: true});
}

function showGrid() {
  fullscreen.style.display = 'none';
  grid.style.display = 'grid';
  backBtn.style.display = 'none';
  Plotly.purge(fullPlot);
  currentBigLayer = null;
  LAYERS.forEach(l => Plotly.Plots.resize(document.getElementById('cell-' + l)));
}

function applyToggles() {
  const shapes = makeShapes();
  LAYERS.forEach(l => {
    const cell = document.getElementById('cell-' + l);
    Plotly.restyle(cell, {visible: showLR}, [1]);
    Plotly.relayout(cell, {shapes: shapes});
  });
  if (currentBigLayer !== null) {
    Plotly.restyle(fullPlot, {visible: showLR}, [1]);
    Plotly.relayout(fullPlot, {shapes: shapes});
  }
}

document.getElementById('toggle-lr').addEventListener('change', e => {
  showLR = e.target.checked; applyToggles();
});
document.getElementById('toggle-jobs').addEventListener('change', e => {
  showJobs = e.target.checked; applyToggles();
});

window.addEventListener('resize', () => {
  if (grid.style.display !== 'none') {
    LAYERS.forEach(l => Plotly.Plots.resize(document.getElementById('cell-' + l)));
  } else {
    Plotly.Plots.resize(fullPlot);
  }
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && fullscreen.style.display === 'block') showGrid();
});
</script>
</body>
</html>"""

    html = (
        template
        .replace("__SUBTITLE__", f"{last_step:,} steps · {tok_max:.1f}B tokens · click a layer to expand · Esc/Back to return")
        .replace("__SMALL__", json.dumps(small_specs))
        .replace("__BIG__", json.dumps(big_specs))
        .replace("__LAYERS__", json.dumps(layers))
        .replace("__LR_SMALL_X__", json.dumps(lr_small_x))
        .replace("__LR_SMALL_Y__", json.dumps(lr_small_y))
        .replace("__LR_BIG_X__", json.dumps(lr_big_x))
        .replace("__LR_BIG_Y__", json.dumps(lr_big_y))
        .replace("__JOB_BOUNDS__", json.dumps(job_bounds))
    )
    output_path.write_text(html)


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
        "--combined",
        action="store_true",
        help="Single overlay plot with all layers (default is clickable 6x4 grid)",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help=f"For --combined mode: include layer {OUTLIER_LAYER} in a separate subplot",
    )
    args = parser.parse_args()

    print(f"Parsing logs from {LOG_DIR} ...")
    df_n, df_t = parse_logs(LOG_DIR)

    steps = sorted(df_n["global_step"].unique())
    layers = sorted(df_n["layer"].unique())
    print(
        f"  {len(steps)} logged SSA steps  "
        f"(step {steps[0]} → {steps[-1]})  "
        f"{len(layers)} layers · {len(df_t):,} training-step rows"
    )

    if args.combined:
        fig = make_plot(df_n, all_layers=args.all_layers)
        fig.write_html(
            str(OUTPUT_HTML),
            auto_open=False,
            full_html=True,
            include_plotlyjs="cdn",
            config={"responsive": True},
            default_width="100vw",
            default_height="100vh",
        )
    else:
        last_step = df_n["global_step"].max()
        tok_max = df_n["tokens_B"].max()
        write_grid_html(df_n, df_t, last_step, tok_max, OUTPUT_HTML)

    print(f"\nPlot saved to {OUTPUT_HTML}")
    print("Open in a browser to interact with the chart.")


if __name__ == "__main__":
    main()
