#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required to plot benchmark metrics. "
        "Install it with 'pip install matplotlib'."
    ) from exc


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _expand_inputs(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        raw_path = Path(raw)
        if any(ch in raw for ch in "*?[]"):
            matches = sorted(Path().glob(raw))
            paths.extend([m for m in matches if m.is_file() and m.suffix == ".json"])
            continue

        if raw_path.is_dir():
            paths.extend(sorted(raw_path.glob("*.json")))
        elif raw_path.is_file():
            paths.append(raw_path)

    # Preserve order while removing duplicates.
    deduped: List[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            deduped.append(rp)
            seen.add(rp)
    return deduped


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_model_runs(payload: dict, source_path: Path) -> List[dict]:
    if isinstance(payload.get("models"), list):
        runs = []
        for idx, model_payload in enumerate(payload["models"], start=1):
            if isinstance(model_payload, dict):
                run = dict(model_payload)
                run["_source_file"] = str(source_path)
                run["_source_stem"] = source_path.stem
                run["_source_index"] = idx
                runs.append(run)
        return runs

    run = dict(payload)
    run["_source_file"] = str(source_path)
    run["_source_stem"] = source_path.stem
    run["_source_index"] = 1
    return [run]


def _run_label(run: dict) -> str:
    model_name = run.get("model_name") or "unknown_model"
    model_type = run.get("model_type") or "unknown_type"
    stem = run.get("_source_stem") or "results"
    src_idx = run.get("_source_index", 1)
    return f"{model_name} ({model_type}) [{stem}#{src_idx}]"


def _collect_task_metrics(run: dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    results = run.get("results", {})
    task_results = results.get("results", {}) if isinstance(results, dict) else {}

    if not isinstance(task_results, dict):
        return out

    for task_name, metrics in task_results.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            if metric_name == "alias":
                continue
            if _is_number(value):
                out[f"task::{task_name}::{metric_name}"] = float(value)
    return out


def _collect_run_metrics(run: dict) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # Useful run-level settings and timings.
    top_level_numeric = [
        "batch_size",
        "max_length",
        "num_fewshot",
        "limit",
        "gsm8k_limit",
        "gsm8k_random_seed",
    ]
    for key in top_level_numeric:
        value = run.get(key)
        if _is_number(value):
            out[f"run::{key}"] = float(value)

    timing = run.get("timing", {})
    if isinstance(timing, dict):
        for key, value in timing.items():
            if key in {"groups", "tasks", "elapsed_human", "estimated_full_dataset_human"}:
                continue
            if _is_number(value):
                out[f"timing::{key}"] = float(value)

    forward_perf = run.get("forward_performance", {})
    if isinstance(forward_perf, dict):
        for key, value in forward_perf.items():
            if _is_number(value):
                out[f"forward::{key}"] = float(value)

    return out


def _metric_display_name(metric_key: str) -> str:
    parts = metric_key.split("::")
    if len(parts) == 3 and parts[0] == "task":
        _, task, metric = parts
        return f"{task} | {metric}"
    if len(parts) == 2:
        scope, name = parts
        return f"{scope} | {name}"
    return metric_key


def _plot_metric_chunk(
    metric_keys: List[str],
    metric_values: Dict[str, Dict[str, float]],
    run_labels: List[str],
    output_file: Path,
    title_prefix: str,
    dpi: int,
) -> None:
    n = len(metric_keys)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 3.8 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, metric_key in enumerate(metric_keys):
        ax = axes_flat[idx]
        display_name = _metric_display_name(metric_key)
        per_run_values = metric_values[metric_key]

        y_values = [per_run_values.get(label, float("nan")) for label in run_labels]
        x_values = list(range(len(run_labels)))

        ax.bar(x_values, y_values)
        ax.set_title(display_name, fontsize=10)
        ax.set_xticks(x_values)
        ax.set_xticklabels(run_labels, rotation=60, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, axis="y", alpha=0.25)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"{title_prefix} ({n} metrics)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)


def build_plots(
    input_paths: List[Path],
    output_dir: Path,
    max_plots_per_figure: int,
    dpi: int,
    title_prefix: str,
    include_run_metrics: bool,
) -> Tuple[int, int, List[Path]]:
    all_runs: List[dict] = []
    for path in input_paths:
        payload = _load_json(path)
        all_runs.extend(_extract_model_runs(payload, path))

    if not all_runs:
        raise ValueError("No runs found in the provided input files.")

    run_labels = [_run_label(run) for run in all_runs]
    metric_values: Dict[str, Dict[str, float]] = defaultdict(dict)

    for run, label in zip(all_runs, run_labels):
        for metric_key, value in _collect_task_metrics(run).items():
            metric_values[metric_key][label] = value

        if include_run_metrics:
            for metric_key, value in _collect_run_metrics(run).items():
                metric_values[metric_key][label] = value

    metric_keys = sorted(metric_values.keys())
    if not metric_keys:
        raise ValueError("No numeric metrics found in the provided run files.")

    output_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: List[Path] = []
    num_pages = math.ceil(len(metric_keys) / max_plots_per_figure)
    for page_idx in range(num_pages):
        start = page_idx * max_plots_per_figure
        end = min((page_idx + 1) * max_plots_per_figure, len(metric_keys))
        chunk = metric_keys[start:end]
        page_path = output_dir / f"benchmark_metrics_page_{page_idx + 1:02d}.png"
        _plot_metric_chunk(
            metric_keys=chunk,
            metric_values=metric_values,
            run_labels=run_labels,
            output_file=page_path,
            title_prefix=title_prefix,
            dpi=dpi,
        )
        figure_paths.append(page_path)

    # Save a CSV-like manifest for easier post-processing.
    manifest_path = output_dir / "benchmark_metrics_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("metric\t" + "\t".join(run_labels) + "\n")
        for metric_key in metric_keys:
            values = [str(metric_values[metric_key].get(label, "")) for label in run_labels]
            f.write(metric_key + "\t" + "\t".join(values) + "\n")

    return len(all_runs), len(metric_keys), figure_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all numeric benchmark metrics for one or more run JSON files. "
            "Accepts single-run payloads and combined multi-model payloads."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Input JSON files, directories, and/or glob patterns. "
            "Examples: results/*.json benchmark_results_99ksteps_ssa_fixed"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_plots"),
        help="Directory where generated plot pages and manifest are written.",
    )
    parser.add_argument(
        "--max-plots-per-figure",
        type=int,
        default=12,
        help="Maximum number of metric subplots per output image.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="Benchmark Metrics Comparison",
        help="Prefix used in generated figure titles.",
    )
    parser.add_argument(
        "--no-run-metrics",
        action="store_true",
        help="Only plot task metrics (skip timing and forward-performance metrics).",
    )

    args = parser.parse_args()

    if args.max_plots_per_figure <= 0:
        raise ValueError("--max-plots-per-figure must be > 0")

    input_paths = _expand_inputs(args.inputs)
    if not input_paths:
        raise ValueError("No JSON input files were found from the provided inputs.")

    run_count, metric_count, output_files = build_plots(
        input_paths=input_paths,
        output_dir=args.output_dir,
        max_plots_per_figure=args.max_plots_per_figure,
        dpi=args.dpi,
        title_prefix=args.title_prefix,
        include_run_metrics=not args.no_run_metrics,
    )

    print(f"Loaded {run_count} run(s) from {len(input_paths)} JSON file(s)")
    print(f"Discovered {metric_count} numeric metric(s)")
    print("Generated plot pages:")
    for path in output_files:
        print(f"  - {path}")
    print(f"Manifest: {args.output_dir / 'benchmark_metrics_manifest.tsv'}")


if __name__ == "__main__":
    main()
