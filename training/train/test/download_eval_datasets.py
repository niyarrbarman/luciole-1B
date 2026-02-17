#!/usr/bin/env python3
"""
Download Hugging Face datasets needed by run_benchmarks_triton_v4.py.

Run this inside your apptainer shell once, then benchmark with
HF_DATASETS_OFFLINE=1.
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


SUPERGLUE_CORE_TASKS = [
    "boolq",
    "cb",
    "copa",
    "multirc",
    "record",
    "rte",
    "wic",
    "wsc",
]
SUPERGLUE_DIAGNOSTIC_TASKS = ["axb", "axg"]
SUPERGLUE_ALL_TASKS = SUPERGLUE_CORE_TASKS + SUPERGLUE_DIAGNOSTIC_TASKS


BENCHMARK_GROUPS = {
    "quick": ["arc_easy"],
    "standard": ["arc_easy", "arc_challenge", "hellaswag", "winogrande"],
    "leaderboard": [
        "arc_challenge",
        "hellaswag",
        "truthfulqa",
        "winogrande",
        "gsm8k",
    ],
    "superglue_core": SUPERGLUE_CORE_TASKS,
    "superglue": SUPERGLUE_ALL_TASKS,
    "all": [
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "truthfulqa",
        "gsm8k",
        "openbookqa",
        "lambada",
    ]
    + SUPERGLUE_ALL_TASKS,
}

# Multiple candidates are used where dataset IDs differ across lm-eval/datasets versions.
# The *first* candidate that succeeds is kept.  Order matters: put the name that
# the container's lm-eval YAML actually uses first so the cache directory matches.
TASK_DATASET_CANDIDATES = {
    "arc_easy": [("allenai/ai2_arc", "ARC-Easy"), ("ai2_arc", "ARC-Easy")],
    "arc_challenge": [
        ("allenai/ai2_arc", "ARC-Challenge"),
        ("ai2_arc", "ARC-Challenge"),
    ],
    "hellaswag": [("Rowan/hellaswag", None), ("hellaswag", None)],
    "piqa": [("piqa", None)],
    "winogrande": [("winogrande", "winogrande_xl")],
    "mmlu": [("cais/mmlu", "all"), ("hendrycks_test", None)],
    "truthfulqa": [
        ("truthfulqa/truthful_qa", "multiple_choice"),
        ("truthful_qa", "multiple_choice"),
    ],
    "gsm8k": [("gsm8k", "main")],
    "boolq": [("super_glue", "boolq"), ("google/boolq", None)],
    "cb": [("super_glue", "cb")],
    "copa": [("super_glue", "copa")],
    "multirc": [("super_glue", "multirc")],
    "record": [("super_glue", "record")],
    # In lm-eval 0.4.10 on this container, task `rte` resolves to GLUE RTE.
    # Cache GLUE first, then SuperGLUE as fallback for compatibility.
    "rte": [("glue", "rte"), ("super_glue", "rte")],
    "wic": [("super_glue", "wic")],
    "wsc": [("super_glue", "wsc")],
    "axb": [("super_glue", "axb")],
    "axg": [("super_glue", "axg")],
    "openbookqa": [("allenai/openbookqa", "main"), ("openbookqa", "main")],
    "lambada": [("EleutherAI/lambada_openai", "default"), ("lambada", None)],
}


def resolve_tasks(task_arg: str) -> list[str]:
    normalized = task_arg.strip().lower()
    if normalized in BENCHMARK_GROUPS:
        return BENCHMARK_GROUPS[normalized]

    tasks = [t.strip().lower() for t in normalized.split(",") if t.strip()]
    invalid = [t for t in tasks if t not in TASK_DATASET_CANDIDATES]
    if invalid:
        raise ValueError(f"Unknown tasks: {invalid}")
    return tasks


def _cache_dir() -> str:
    """Return the HF datasets cache directory."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(hf_home, "datasets")


def _is_cached(dataset_path: str, dataset_name: str | None) -> bool:
    """Check whether a dataset/config pair already has arrow files in the cache."""
    cache = _cache_dir()
    # HF datasets stores "owner/repo" as "owner___repo" on disk
    dir_name = dataset_path.replace("/", "___")
    config = dataset_name or "default"
    candidate = os.path.join(cache, dir_name, config)
    if os.path.isdir(candidate):
        # Look for at least one .arrow file to confirm it's a real cache entry
        for entry in os.listdir(candidate):
            subdir = os.path.join(candidate, entry)
            if os.path.isdir(subdir):
                if any(f.endswith(".arrow") for f in os.listdir(subdir)):
                    return True
    return False


def download_task_dataset(task: str) -> tuple[bool, str]:
    from datasets import load_dataset

    last_error = ""
    for dataset_path, dataset_name in TASK_DATASET_CANDIDATES[task]:
        # Skip download if already cached
        if _is_cached(dataset_path, dataset_name):
            label = f"{dataset_path}/{dataset_name}"
            logger.info("Task %s already cached (%s), skipping download", task, label)
            return True, f"{label} (cached)"

        try:
            logger.info(
                "Downloading task=%s with dataset=%s name=%s",
                task,
                dataset_path,
                dataset_name,
            )
            dataset_dict = load_dataset(dataset_path, dataset_name)
            split_names = (
                list(dataset_dict.keys()) if hasattr(dataset_dict, "keys") else []
            )
            logger.info("Cached task=%s splits=%s", task, split_names)
            return True, f"{dataset_path}/{dataset_name}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Failed candidate for task=%s dataset=%s name=%s -> %s",
                task,
                dataset_path,
                dataset_name,
                last_error,
            )

    return False, last_error


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets required for lm-eval benchmarks (offline-ready cache)."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="standard",
        help=(
            "Task list or group: quick, standard, leaderboard, superglue_core, "
            "superglue, all, or comma-separated task names"
        ),
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default=None,
        help="Optional HF_HOME path (if set, script exports it before downloading)",
    )
    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
        logger.info("Using HF_HOME=%s", args.hf_home)

    tasks = resolve_tasks(args.tasks)
    logger.info("Preparing datasets for tasks: %s", tasks)

    failures = []
    for task in tasks:
        ok, info = download_task_dataset(task)
        if ok:
            logger.info("Task %s cached using %s", task, info)
        else:
            failures.append((task, info))

    print("\n" + "=" * 70)
    if failures:
        print("Dataset download completed with failures:")
        for task, err in failures:
            print(f"  - {task}: {err}")
        print("=" * 70)
        sys.exit(1)

    print("All requested datasets cached successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
