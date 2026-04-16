#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def expand_inputs(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        candidate = Path(raw)

        if any(ch in raw for ch in "*?[]"):
            matches = sorted(Path().glob(raw))
            paths.extend([m for m in matches if m.is_file() and m.suffix == ".json"])
            continue

        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.json")))
        elif candidate.is_file() and candidate.suffix == ".json":
            paths.append(candidate)

    deduped: List[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            deduped.append(rp)
            seen.add(rp)
    return deduped


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def unpack_runs(payload: dict, source_path: Path) -> List[dict]:
    if isinstance(payload.get("models"), list):
        out = []
        for idx, model_payload in enumerate(payload["models"], start=1):
            if isinstance(model_payload, dict):
                item = dict(model_payload)
                item["_source_path"] = str(source_path)
                item["_source_stem"] = source_path.stem
                item["_source_index"] = idx
                out.append(item)
        return out

    item = dict(payload)
    item["_source_path"] = str(source_path)
    item["_source_stem"] = source_path.stem
    item["_source_index"] = 1
    return [item]


def run_label(run: dict) -> str:
    model_name = run.get("model_name") or "unknown_model"
    model_type = run.get("model_type") or "unknown_type"
    merged_count = len(run.get("_merged_sources", []))
    if merged_count > 1:
        return f"{model_name} ({model_type}) [merged:{merged_count}]"

    source_stem = run.get("_source_stem") or "results"
    source_index = run.get("_source_index", 1)
    return f"{model_name} ({model_type}) [{source_stem}#{source_index}]"


def merge_mapping_dicts(base: dict, incoming: dict, *, prefer: str = "last") -> dict:
    out = dict(base)
    for key, value in incoming.items():
        if key not in out:
            out[key] = value
            continue

        # Values are equal; nothing to do.
        if out[key] == value:
            continue

        # For nested dicts, merge recursively.
        if isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = merge_mapping_dicts(out[key], value, prefer=prefer)
            continue

        if prefer == "last":
            out[key] = value

    return out


def merge_run_payloads(base: dict, incoming: dict) -> dict:
    merged = dict(base)

    merged_sources = list(merged.get("_merged_sources", []))
    if not merged_sources and merged.get("_source_path"):
        merged_sources = [merged["_source_path"]]
    incoming_source = incoming.get("_source_path")
    if incoming_source and incoming_source not in merged_sources:
        merged_sources.append(incoming_source)
    merged["_merged_sources"] = merged_sources

    # Keep first non-empty scalar by default unless explicitly recomputed below.
    for key, value in incoming.items():
        if key.startswith("_"):
            continue
        if key not in merged or merged[key] is None:
            merged[key] = value

    # Merge task list (preserve insertion order).
    base_tasks = merged.get("tasks") if isinstance(merged.get("tasks"), list) else []
    incoming_tasks = incoming.get("tasks") if isinstance(incoming.get("tasks"), list) else []
    merged["tasks"] = list(dict.fromkeys([*base_tasks, *incoming_tasks]))

    # Merge timing fields by summation where relevant.
    if isinstance(merged.get("timing"), dict) or isinstance(incoming.get("timing"), dict):
        timing_base = merged.get("timing") if isinstance(merged.get("timing"), dict) else {}
        timing_in = incoming.get("timing") if isinstance(incoming.get("timing"), dict) else {}
        timing_out = dict(timing_base)

        for key in ["elapsed_seconds", "estimated_full_dataset_seconds"]:
            b = timing_base.get(key)
            i = timing_in.get(key)
            if is_number(b) or is_number(i):
                timing_out[key] = float(b or 0.0) + float(i or 0.0)

        # Preserve per-task and per-group details from both runs.
        timing_out["groups"] = [
            *(timing_base.get("groups") if isinstance(timing_base.get("groups"), list) else []),
            *(timing_in.get("groups") if isinstance(timing_in.get("groups"), list) else []),
        ]
        timing_out["tasks"] = [
            *(timing_base.get("tasks") if isinstance(timing_base.get("tasks"), list) else []),
            *(timing_in.get("tasks") if isinstance(timing_in.get("tasks"), list) else []),
        ]

        if is_number(timing_out.get("elapsed_seconds")):
            total_elapsed = timing_out["elapsed_seconds"]
            timing_out["elapsed_human"] = f"{int(total_elapsed // 3600)}h {int((total_elapsed % 3600) // 60)}m {int(total_elapsed % 60)}s"
        if is_number(timing_out.get("estimated_full_dataset_seconds")):
            total_estimated = timing_out["estimated_full_dataset_seconds"]
            timing_out["estimated_full_dataset_human"] = f"{int(total_estimated // 3600)}h {int((total_estimated % 3600) // 60)}m {int(total_estimated % 60)}s"

        merged["timing"] = timing_out

    # Merge forward performance; recompute rate metrics after summation.
    if isinstance(merged.get("forward_performance"), dict) or isinstance(incoming.get("forward_performance"), dict):
        forward_base = merged.get("forward_performance") if isinstance(merged.get("forward_performance"), dict) else {}
        forward_in = incoming.get("forward_performance") if isinstance(incoming.get("forward_performance"), dict) else {}
        forward_out = dict(forward_base)

        additive_fields = ["calls", "input_tokens", "prep_seconds", "forward_seconds", "output_seconds", "total_seconds"]
        for key in additive_fields:
            b = forward_base.get(key)
            i = forward_in.get(key)
            if is_number(b) or is_number(i):
                forward_out[key] = float(b or 0.0) + float(i or 0.0)

        calls = forward_out.get("calls")
        total_seconds = forward_out.get("total_seconds")
        forward_seconds = forward_out.get("forward_seconds")
        input_tokens = forward_out.get("input_tokens")
        if is_number(calls) and calls > 0 and is_number(total_seconds):
            forward_out["avg_call_seconds"] = float(total_seconds) / float(calls)
        if is_number(forward_seconds) and forward_seconds > 0 and is_number(input_tokens):
            forward_out["forward_tokens_per_second"] = float(input_tokens) / float(forward_seconds)

        merged["forward_performance"] = forward_out

    # Merge detailed results structure.
    if isinstance(merged.get("results"), dict) or isinstance(incoming.get("results"), dict):
        res_base = merged.get("results") if isinstance(merged.get("results"), dict) else {}
        res_in = incoming.get("results") if isinstance(incoming.get("results"), dict) else {}
        res_out = dict(res_base)

        for key in [
            "results",
            "group_subtasks",
            "configs",
            "versions",
            "n-shot",
            "higher_is_better",
            "n-samples",
        ]:
            base_map = res_base.get(key) if isinstance(res_base.get(key), dict) else {}
            in_map = res_in.get(key) if isinstance(res_in.get(key), dict) else {}
            res_out[key] = merge_mapping_dicts(base_map, in_map, prefer="last")

        # Keep most recent date if present.
        if is_number(res_base.get("date")) and is_number(res_in.get("date")):
            res_out["date"] = max(float(res_base["date"]), float(res_in["date"]))
        elif "date" in res_in and "date" not in res_out:
            res_out["date"] = res_in["date"]

        # Keep config from first run unless absent.
        if not isinstance(res_out.get("config"), dict) and isinstance(res_in.get("config"), dict):
            res_out["config"] = res_in["config"]

        merged["results"] = res_out

    return merged


def merge_runs_by_identity(runs: List[dict], merge_key: str) -> List[dict]:
    if merge_key == "none":
        return runs

    grouped: Dict[str, dict] = {}
    output_order: List[str] = []

    for run in runs:
        model_name = str(run.get("model_name") or "unknown_model")
        model_type = str(run.get("model_type") or "unknown_type")
        checkpoint = str(run.get("checkpoint") or "")

        if merge_key == "model":
            key = f"{model_name}::{model_type}"
        else:
            key = f"{model_name}::{model_type}::{checkpoint}"

        if key not in grouped:
            grouped[key] = dict(run)
            source_path = run.get("_source_path")
            grouped[key]["_merged_sources"] = [source_path] if source_path else []
            output_order.append(key)
        else:
            grouped[key] = merge_run_payloads(grouped[key], run)

    return [grouped[key] for key in output_order]


def collect_report_data(runs: List[dict]) -> dict:
    run_labels = [run_label(r) for r in runs]
    run_model_names = {
        label: (run.get("model_name") or "unknown_model")
        for run, label in zip(runs, run_labels)
    }

    task_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    task_stderr: Dict[str, Dict[str, Dict[str, float]]] = {}
    scoped_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        "run": {},
        "timing": {},
        "forward": {},
    }

    def put(scope_dict: Dict[str, Dict[str, float]], metric_name: str, label: str, value: float):
        scope_dict.setdefault(metric_name, {})[label] = value

    for run, label in zip(runs, run_labels):
        # Task metrics
        results = run.get("results", {})
        task_results = results.get("results", {}) if isinstance(results, dict) else {}
        if isinstance(task_results, dict):
            for task_name, metrics in task_results.items():
                if not isinstance(metrics, dict):
                    continue
                for metric_name, value in metrics.items():
                    if metric_name == "alias":
                        continue
                    if is_number(value):
                        if "_stderr" in metric_name:
                            base_metric_name = metric_name.replace("_stderr", "", 1)
                            task_stderr.setdefault(task_name, {}).setdefault(base_metric_name, {})[label] = float(value)
                        else:
                            task_metrics.setdefault(task_name, {}).setdefault(metric_name, {})[label] = float(value)

        # Run-level numeric fields
        for key in ["batch_size", "max_length", "num_fewshot", "limit", "gsm8k_limit", "gsm8k_random_seed"]:
            value = run.get(key)
            if is_number(value):
                put(scoped_metrics["run"], key, label, float(value))

        # Timing numeric fields
        timing = run.get("timing", {})
        if isinstance(timing, dict):
            for key, value in timing.items():
                if key in {"groups", "tasks", "elapsed_human", "estimated_full_dataset_human"}:
                    continue
                if is_number(value):
                    put(scoped_metrics["timing"], key, label, float(value))

        # Forward perf fields
        forward = run.get("forward_performance", {})
        if isinstance(forward, dict):
            for key, value in forward.items():
                if is_number(value):
                    put(scoped_metrics["forward"], key, label, float(value))

    return {
        "runs": run_labels,
      "run_model_names": run_model_names,
        "task_metrics": task_metrics,
        "task_stderr": task_stderr,
        "scoped_metrics": scoped_metrics,
    }


def build_html(report_data: dict, title: str) -> str:
    payload = json.dumps(report_data)
    title_escaped = title.replace("<", "&lt;").replace(">", "&gt;")
    template = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>__TITLE__</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    :root {{
      --bg: #f7f5ef;
      --panel: #fffdf7;
      --ink: #1f2937;
      --accent: #0f766e;
      --muted: #6b7280;
      --border: #d1d5db;
    }}
    body {{
      margin: 0;
      font-family: \"IBM Plex Sans\", \"Segoe UI\", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 15% 10%, #efe9dc 0%, transparent 40%),
        radial-gradient(circle at 80% 90%, #e6f3ee 0%, transparent 30%),
        var(--bg);
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 20px;
    }}
    .title {{
      margin: 0 0 10px;
      font-size: 32px;
      letter-spacing: 0.2px;
    }}
    .subtitle {{
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: 360px 1fr;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }}
    .controls {{
      padding: 14px;
      position: sticky;
      top: 10px;
      height: fit-content;
    }}
    .control {{
      margin-bottom: 12px;
    }}
    label {{
      display: block;
      font-size: 12px;
      margin-bottom: 6px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    select, input[type=\"text\"] {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--border);
      background: white;
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 14px;
    }}
    .run-list {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: white;
      max-height: 250px;
      overflow: auto;
      padding: 6px;
    }}
    .run-item {{
      display: flex;
      gap: 6px;
      align-items: center;
      padding: 4px 2px;
      font-size: 12px;
    }}
    .btn-row {{
      display: flex;
      gap: 8px;
      margin-bottom: 8px;
    }}
    button {{
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
    }}
    .content {{
      padding: 12px;
    }}
    #chart {{
      width: 100%;
      height: 620px;
    }}
    .table-wrap {{
      margin-top: 10px;
      overflow: auto;
      max-height: 260px;
      border-top: 1px solid var(--border);
      padding-top: 8px;
    }}
    .summary-wrap {{
      margin-top: 8px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fff;
      padding: 10px;
    }}
    .summary-title {{
      margin: 0 0 6px;
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      background: white;
      border: 1px solid var(--border);
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      text-align: left;
      padding: 6px 8px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f9fafb;
      z-index: 1;
    }}
    .muted {{
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 1100px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .controls {{
        position: static;
      }}
      #chart {{
        height: 500px;
      }}
    }}
  </style>
</head>
<body>
<div class=\"wrap\">
  <h1 class=\"title\">__TITLE__</h1>
  <p class=\"subtitle\">Interactive benchmark explorer: choose scope, task, metric, and visible runs.</p>

  <div class=\"grid\">
    <aside class=\"panel controls\">
      <div class=\"control\">
        <label for=\"scopeSelect\">Metric Scope</label>
        <select id=\"scopeSelect\">
          <option value=\"task\">Task metrics</option>
          <option value=\"timing\">Timing metrics</option>
          <option value=\"forward\">Forward metrics</option>
          <option value=\"run\">Run settings</option>
        </select>
      </div>

      <div class=\"control\" id=\"taskControl\">
        <label for=\"taskSelect\">Task</label>
        <select id=\"taskSelect\"></select>
      </div>

      <div class=\"control\">
        <label for=\"metricSelect\">Metric</label>
        <select id=\"metricSelect\"></select>
      </div>

      <div class=\"control\">
        <label for=\"runFilter\">Filter Runs</label>
        <input id=\"runFilter\" type=\"text\" placeholder=\"Type to filter runs...\" />
      </div>

      <div class=\"control\">
        <div class=\"btn-row\">
          <button id=\"selectAllBtn\" type=\"button\">Select all</button>
          <button id=\"selectNoneBtn\" type=\"button\">Select none</button>
        </div>
        <div class=\"run-list\" id=\"runList\"></div>
      </div>

      <p class=\"muted\" id=\"selectionSummary\"></p>
    </aside>

    <section class=\"panel content\">
      <div id=\"chart\"></div>

      <div class=\"table-wrap\">
        <table id=\"valueTable\">
          <thead>
            <tr><th>Run</th><th>Value</th></tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
      
      <div class="summary-wrap" id="summaryWrap">
        <h3 class="summary-title">Significant Wins Summary (95% CI)</h3>
        <p class="muted" id="summaryHint"></p>
        <div class="table-wrap" style="max-height: 300px; border-top: 0; padding-top: 0; margin-top: 6px;">
          <table id="summaryTable">
            <thead>
              <tr>
                <th>Task</th>
                <th>Metric</th>
                <th>Winner</th>
                <th>Winner value</th>
                <th>Best challenger</th>
                <th>Challenger value</th>
                <th>CI separation</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>

      <div class="summary-wrap" id="trendWrap">
        <h3 class="summary-title">Task Order Trend</h3>
        <p class="muted" id="trendHint"></p>
        <div class="table-wrap" style="max-height: 300px; border-top: 0; padding-top: 0; margin-top: 6px;">
          <table id="trendTable">
            <thead>
              <tr>
                <th>Task</th>
                <th>Metric used</th>
                <th>Order</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </section>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;

const scopeSelect = document.getElementById('scopeSelect');
const taskControl = document.getElementById('taskControl');
const taskSelect = document.getElementById('taskSelect');
const metricSelect = document.getElementById('metricSelect');
const runFilter = document.getElementById('runFilter');
const runList = document.getElementById('runList');
const selectAllBtn = document.getElementById('selectAllBtn');
const selectNoneBtn = document.getElementById('selectNoneBtn');
const selectionSummary = document.getElementById('selectionSummary');
const valueTableBody = document.querySelector('#valueTable tbody');
const summaryHint = document.getElementById('summaryHint');
const summaryTableBody = document.querySelector('#summaryTable tbody');
const trendHint = document.getElementById('trendHint');
const trendTableBody = document.querySelector('#trendTable tbody');
const CI_MULTIPLIER = 1.96;

function uniqueSorted(values) {{
  return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
}}

function isPercentLikeMetric(metricName) {{
  return /(acc|accuracy|exact_match|f1|em,|precision|recall)/i.test(metricName) && !/(perplex|loss|ppl)/i.test(metricName);
}}

function isLowerBetterMetric(metricName) {{
  return /(perplex|ppl|loss|error|wer|bpb|bits_per_byte)/i.test(metricName);
}}

function shouldScaleTaskMetricToPercent(metricName, values) {{
  if (isPercentLikeMetric(metricName)) return true;
  const finiteValues = values.filter(v => Number.isFinite(v));
  return finiteValues.length > 0 && finiteValues.every(v => v >= 0 && v <= 1);
}}

function maybeScale(value, scaleToPercent) {{
  if (!Number.isFinite(value)) return null;
  return scaleToPercent ? value * 100 : value;
}}

function formatMetricValue(value, scaleToPercent) {{
  if (!Number.isFinite(value)) return 'N/A';
  const scaled = maybeScale(value, scaleToPercent);
  const suffix = scaleToPercent ? '%' : '';
  if (Math.abs(scaled) >= 1000 || (Math.abs(scaled) > 0 && Math.abs(scaled) < 0.001)) return `${scaled.toExponential(3)}${suffix}`;
  return `${scaled.toFixed(6).replace(/0+$/, '').replace(/\.$/, '')}${suffix}`;
}}

function clearElement(el) {{
  while (el.firstChild) el.removeChild(el.firstChild);
}}

function option(parent, value, text) {{
  const el = document.createElement('option');
  el.value = value;
  el.textContent = text;
  parent.appendChild(el);
}}

function buildRunChecklist() {{
  clearElement(runList);
  for (const label of DATA.runs) {{
    const row = document.createElement('label');
    row.className = 'run-item';
    row.dataset.label = label.toLowerCase();

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = label;
    cb.checked = true;
    cb.addEventListener('change', updateView);

    const span = document.createElement('span');
    span.textContent = label;

    row.appendChild(cb);
    row.appendChild(span);
    runList.appendChild(row);
  }}
}}

function visibleRunCheckboxes() {{
  return Array.from(runList.querySelectorAll('label.run-item')).filter(r => r.style.display !== 'none');
}}

function selectedRuns() {{
  return Array.from(runList.querySelectorAll('input[type=checkbox]'))
    .filter(cb => cb.checked)
    .map(cb => cb.value);
}}

function metricNamesForTask(taskName) {{
  const taskData = DATA.task_metrics[taskName] || {{}};
  return Object.keys(taskData).sort((a, b) => a.localeCompare(b));
}}

function metricNamesForTaskAll() {{
  const out = [];
  for (const taskName of Object.keys(DATA.task_metrics)) {{
    out.push(...Object.keys(DATA.task_metrics[taskName] || {{}}));
  }}
  return uniqueSorted(out);
}}

function metricNamesForScope(scope) {{
  return Object.keys(DATA.scoped_metrics[scope] || {{}}).sort((a, b) => a.localeCompare(b));
}}

function setOptionsPreserve(selectEl, values) {{
  const prev = selectEl.value;
  clearElement(selectEl);
  for (const v of values) option(selectEl, v, v);
  if (values.includes(prev)) selectEl.value = prev;
}}

function refreshTaskAndMetricSelectors() {{
  const scope = scopeSelect.value;

  if (scope === 'task') {{
    taskControl.style.display = '';
    const tasks = Object.keys(DATA.task_metrics).sort((a, b) => a.localeCompare(b));
    const taskOptions = ['__ALL_TASKS__', ...tasks];
    setOptionsPreserve(taskSelect, taskOptions);

    const selectedTask = taskSelect.value || '__ALL_TASKS__';
    const metrics = selectedTask === '__ALL_TASKS__' ? metricNamesForTaskAll() : metricNamesForTask(selectedTask);
    setOptionsPreserve(metricSelect, metrics);
  }} else {{
    taskControl.style.display = 'none';
    const metrics = metricNamesForScope(scope);
    setOptionsPreserve(metricSelect, metrics);
  }}
}}

function formatNumber(value) {{
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) return value.toExponential(3);
  return value.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
}}

function getTaskMetricValues(taskName, metricName) {{
  if (taskName === '__ALL_TASKS__') return null;
  return (DATA.task_metrics[taskName] || {{}})[metricName] || {{}};
}}

function getTaskMetricStderr(taskName, metricName) {{
  return (DATA.task_stderr?.[taskName] || {{}})[metricName] || {{}};
}}

function isSignificantlyBetter(bestVal, bestSe, otherVal, otherSe, lowerIsBetter) {{
  if (!Number.isFinite(bestVal) || !Number.isFinite(otherVal)) return false;
  if (!Number.isFinite(bestSe) || !Number.isFinite(otherSe)) return false;
  const bestCi = CI_MULTIPLIER * bestSe;
  const otherCi = CI_MULTIPLIER * otherSe;

  if (lowerIsBetter) {{
    return (bestVal + bestCi) < (otherVal - otherCi);
  }}
  return (bestVal - bestCi) > (otherVal + otherCi);
}}

function clearSummaryTable() {{
  while (summaryTableBody.firstChild) summaryTableBody.removeChild(summaryTableBody.firstChild);
}}

function displayRunName(runLabel) {{
  return (DATA.run_model_names && DATA.run_model_names[runLabel]) || runLabel;
}}

function clearTrendTable() {{
  while (trendTableBody.firstChild) trendTableBody.removeChild(trendTableBody.firstChild);
}}

function chooseTrendMetric(taskMetrics) {{
  const preferred = [
    'acc,none',
    'acc_norm,none',
    'exact_match,strict-match',
    'exact_match,flexible-extract',
    'f1,none',
    'em,none',
    'perplexity,none',
    'loss,none',
  ];
  for (const metric of preferred) {{
    if (Object.prototype.hasOwnProperty.call(taskMetrics, metric)) return metric;
  }}
  const allMetrics = Object.keys(taskMetrics).sort((a, b) => a.localeCompare(b));
  return allMetrics.length > 0 ? allMetrics[0] : null;
}}

function buildTrendTable(rows) {{
  clearTrendTable();
  for (const row of rows) {{
    const tr = document.createElement('tr');
    const cols = [row.task, row.metric, row.order];
    for (const col of cols) {{
      const td = document.createElement('td');
      td.textContent = col;
      tr.appendChild(td);
    }}
    trendTableBody.appendChild(tr);
  }}
}}

function buildSummaryTable(rows) {{
  clearSummaryTable();
  for (const row of rows) {{
    const tr = document.createElement('tr');
    const cols = [
      row.task,
      row.metric,
      displayRunName(row.winner),
      row.winnerValue,
      displayRunName(row.challenger),
      row.challengerValue,
      row.separation,
    ];
    for (const col of cols) {{
      const td = document.createElement('td');
      td.textContent = col;
      tr.appendChild(td);
    }}
    summaryTableBody.appendChild(tr);
  }}
}}

function updateSignificanceSummary(runs) {{
  const summaryRows = [];
  const tasks = Object.keys(DATA.task_metrics).sort((a, b) => a.localeCompare(b));

  for (const task of tasks) {{
    const metrics = Object.keys(DATA.task_metrics[task] || {{}}).sort((a, b) => a.localeCompare(b));

    for (const metric of metrics) {{
      const values = (DATA.task_metrics[task] || {{}})[metric] || {{}};
      const stderr = getTaskMetricStderr(task, metric);

      const candidates = runs
        .map(run => ({{ run, value: values[run], stderr: stderr[run] }}))
        .filter(item => Number.isFinite(item.value) && Number.isFinite(item.stderr));

      if (candidates.length < 2) continue;

      const lowerIsBetter = isLowerBetterMetric(metric);
      candidates.sort((a, b) => lowerIsBetter ? a.value - b.value : b.value - a.value);

      const winner = candidates[0];
      const challenger = candidates[1];
      const winnerBeatsAll = candidates.slice(1).every(other =>
        isSignificantlyBetter(winner.value, winner.stderr, other.value, other.stderr, lowerIsBetter)
      );

      if (!winnerBeatsAll) continue;

      const winnerCi = CI_MULTIPLIER * winner.stderr;
      const challengerCi = CI_MULTIPLIER * challenger.stderr;
      const scaleToPercent = shouldScaleTaskMetricToPercent(metric, candidates.map(c => c.value));

      const separationRaw = lowerIsBetter
        ? (challenger.value - challengerCi) - (winner.value + winnerCi)
        : (winner.value - winnerCi) - (challenger.value + challengerCi);

      summaryRows.push({
        task,
        metric,
        winner: winner.run,
        winnerValue: `${formatMetricValue(winner.value, scaleToPercent)} ± ${formatMetricValue(winnerCi, scaleToPercent)}`,
        challenger: challenger.run,
        challengerValue: `${formatMetricValue(challenger.value, scaleToPercent)} ± ${formatMetricValue(challengerCi, scaleToPercent)}`,
        separation: formatMetricValue(separationRaw, scaleToPercent),
      });
    }}
  }}

  summaryRows.sort((a, b) => (a.task + a.metric).localeCompare(b.task + b.metric));
  buildSummaryTable(summaryRows);

  if (summaryRows.length === 0) {{
    summaryHint.textContent = 'No statistically significant winner found (among selected runs with available stderr).';
  }} else {{
    summaryHint.textContent = `${summaryRows.length} significant winner case(s) found among selected runs.`;
  }}
}}

function updateTrendSummary(runs) {{
  const trendRows = [];
  const tasks = Object.keys(DATA.task_metrics).sort((a, b) => a.localeCompare(b));

  for (const task of tasks) {{
    const taskMetrics = DATA.task_metrics[task] || {{}};
    const metric = chooseTrendMetric(taskMetrics);
    if (!metric) continue;

    const values = taskMetrics[metric] || {{}};
    const candidates = runs
      .map(run => ({{ run, value: values[run] }}))
      .filter(item => Number.isFinite(item.value));

    if (candidates.length === 0) continue;

    const lowerIsBetter = isLowerBetterMetric(metric);
    candidates.sort((a, b) => lowerIsBetter ? a.value - b.value : b.value - a.value);

    trendRows.push({
      task,
      metric,
      order: candidates.map(item => displayRunName(item.run)).join(' > '),
    });
  }}

  buildTrendTable(trendRows);
  if (trendRows.length === 0) {{
    trendHint.textContent = 'No task ordering available for the currently selected runs.';
  }} else {{
    trendHint.textContent = `${trendRows.length} task trend row(s) generated from selected runs.`;
  }}
}}

function buildTable(rows) {{
  clearElement(valueTableBody);
  for (const row of rows) {{
    const tr = document.createElement('tr');
    const tdRun = document.createElement('td');
    const tdVal = document.createElement('td');
    tdRun.textContent = row.run;
    tdVal.textContent = formatMetricValue(row.value, row.scaleToPercent);
    tr.appendChild(tdRun);
    tr.appendChild(tdVal);
    valueTableBody.appendChild(tr);
  }}
}}

function updateView() {{
  refreshTaskAndMetricSelectors();

  const scope = scopeSelect.value;
  const metric = metricSelect.value;
  const runs = selectedRuns();
  selectionSummary.textContent = `${runs.length} selected run(s)`;
  updateSignificanceSummary(runs);
  updateTrendSummary(runs);

  if (!metric) {{
    Plotly.react('chart', [], {{title: 'No metric available'}});
    buildTable([]);
    return;
  }}

  if (scope === 'task') {{
    const taskName = taskSelect.value;

    if (taskName === '__ALL_TASKS__') {{
      const allTasks = Object.keys(DATA.task_metrics).sort((a, b) => a.localeCompare(b));
      const z = allTasks.map(task => runs.map(run => {{
        const values = (DATA.task_metrics[task] || {{}})[metric] || {{}};
        const v = values[run];
        const scaleToPercent = shouldScaleTaskMetricToPercent(metric, Object.values(values));
        return Number.isFinite(v) ? maybeScale(v, scaleToPercent) : null;
      }}));
      const scaleToPercent = shouldScaleTaskMetricToPercent(metric, allTasks.flatMap(task => Object.values((DATA.task_metrics[task] || {{}})[metric] || {{}})));

      Plotly.react('chart', [{
        type: 'heatmap',
        x: runs,
        y: allTasks,
        z,
        colorscale: 'Viridis',
        colorbar: {{title: scaleToPercent ? `${metric} (%)` : metric}},
        hovertemplate: 'Task=%{{y}}<br>Run=%{{x}}<br>Value=%{{z}}<extra></extra>'
      }}], {{
        title: `Task metric heatmap: ${metric}`,
        margin: {{l: 150, r: 20, t: 50, b: 150}},
        xaxis: {{tickangle: 45}},
      }}, {{responsive: true}});

      buildTable([]);
      return;
    }}

    const values = getTaskMetricValues(taskName, metric);
    const stderrValues = getTaskMetricStderr(taskName, metric);
    const rawValues = runs.map(run => values[run]);
    const scaleToPercent = shouldScaleTaskMetricToPercent(metric, rawValues.concat(Object.values(stderrValues)));
    const rows = runs.map(run => ({{ run, value: values[run], scaleToPercent }}));
    const errorValues = runs.map(run => {
      const stderr = stderrValues[run];
      return Number.isFinite(stderr) ? maybeScale(stderr * CI_MULTIPLIER, scaleToPercent) : null;
    });

    Plotly.react('chart', [{
      type: 'bar',
      x: runs,
      y: rows.map(r => maybeScale(r.value, scaleToPercent)),
      marker: {{color: '#0f766e'}},
      error_y: {{
        type: 'data',
        array: errorValues,
        visible: true,
        thickness: 1.2,
        color: '#334155',
      }},
      hovertemplate: scaleToPercent ? 'Run=%{{x}}<br>Value=%{{y:.2f}}%<extra></extra>' : 'Run=%{{x}}<br>Value=%{{y}}<extra></extra>'
    }}], {{
      title: `${taskName} | ${metric}`,
      margin: {{l: 60, r: 20, t: 50, b: 170}},
      xaxis: {{tickangle: 45}},
      yaxis: scaleToPercent ? {{automargin: true, range: [0, 100], ticksuffix: '%'}} : {{automargin: true}},
    }}, {{responsive: true}});

    buildTable(rows);
    return;
  }}

  const values = (DATA.scoped_metrics[scope] || {{}})[metric] || {{}};
  const rows = runs.map(run => ({{ run, value: values[run], scaleToPercent: false }}));

  Plotly.react('chart', [{
    type: 'bar',
    x: runs,
    y: rows.map(r => Number.isFinite(r.value) ? r.value : null),
    marker: {{color: '#1d4ed8'}},
    hovertemplate: 'Run=%{{x}}<br>Value=%{{y}}<extra></extra>'
  }}], {{
    title: `${scope} | ${metric}`,
    margin: {{l: 60, r: 20, t: 50, b: 170}},
    xaxis: {{tickangle: 45}},
    yaxis: {{automargin: true}},
  }}, {{responsive: true}});

  buildTable(rows);
}}

scopeSelect.addEventListener('change', updateView);
taskSelect.addEventListener('change', updateView);
metricSelect.addEventListener('change', updateView);

runFilter.addEventListener('input', () => {{
  const needle = runFilter.value.toLowerCase().trim();
  const rows = Array.from(runList.querySelectorAll('label.run-item'));
  for (const row of rows) {{
    const show = !needle || row.dataset.label.includes(needle);
    row.style.display = show ? '' : 'none';
  }}
}});

selectAllBtn.addEventListener('click', () => {{
  for (const row of visibleRunCheckboxes()) row.querySelector('input').checked = true;
  updateView();
}});

selectNoneBtn.addEventListener('click', () => {{
  for (const row of visibleRunCheckboxes()) row.querySelector('input').checked = false;
  updateView();
}});

buildRunChecklist();
refreshTaskAndMetricSelectors();
updateView();
</script>
</body>
</html>
"""
    # Normalize escaped braces in template first; do not mutate JSON payload.
    normalized_template = template.replace("{{", "{").replace("}}", "}")
    return normalized_template.replace("__TITLE__", title_escaped).replace("__PAYLOAD__", payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an interactive HTML benchmark report from one or multiple result JSON files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input JSON files, directories and/or glob patterns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_report.html"),
        help="Output HTML report path.",
    )
    parser.add_argument(
        "--title",
        default="Luciole Benchmark Interactive Report",
        help="Title displayed in the report.",
    )
    parser.add_argument(
        "--merge-runs-by",
        choices=["none", "model", "model+checkpoint"],
        default="model+checkpoint",
        help="Merge input runs that represent the same model identity before generating the report.",
    )

    args = parser.parse_args()

    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("No JSON files found from the provided inputs.")

    runs: List[dict] = []
    for path in input_paths:
        payload = load_json(path)
        runs.extend(unpack_runs(payload, path))

    if not runs:
        raise SystemExit("No benchmark run payloads were found in the provided files.")

    merge_key = args.merge_runs_by
    if merge_key == "model+checkpoint":
        merge_key = "model_checkpoint"
    runs = merge_runs_by_identity(runs, merge_key)

    report_data = collect_report_data(runs)
    html = build_html(report_data, args.title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")

    print(f"Loaded {len(runs)} run(s) from {len(input_paths)} input JSON file(s) after merge mode '{args.merge_runs_by}'")
    print(f"Wrote interactive report: {args.output}")


if __name__ == "__main__":
    main()
