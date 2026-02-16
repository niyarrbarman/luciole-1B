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


BENCHMARK_GROUPS = {
    "quick": ["arc_easy"],
    "standard": ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"],
    "leaderboard": [
        "arc_challenge",
        "hellaswag",
        "mmlu",
        "truthfulqa",
        "winogrande",
        "gsm8k",
    ],
    "all": [
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "piqa",
        "winogrande",
        "mmlu",
        "truthfulqa",
        "gsm8k",
        "boolq",
        "openbookqa",
    ],
}

# Multiple candidates are used where dataset IDs differ across lm-eval/datasets versions.
TASK_DATASET_CANDIDATES = {
    "arc_easy": [("ai2_arc", "ARC-Easy")],
    "arc_challenge": [("ai2_arc", "ARC-Challenge")],
    "hellaswag": [("Rowan/hellaswag", None), ("hellaswag", None)],
    "piqa": [("piqa", None)],
    "winogrande": [("winogrande", "winogrande_xl")],
    "mmlu": [("cais/mmlu", "all"), ("hendrycks_test", None)],
    "truthfulqa": [
        ("truthfulqa/truthful_qa", "multiple_choice"),
        ("truthful_qa", "multiple_choice"),
    ],
    "gsm8k": [("gsm8k", "main")],
    "boolq": [("google/boolq", None), ("boolq", None)],
    "openbookqa": [("allenai/openbookqa", "main"), ("openbookqa", "main")],
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


def download_task_dataset(task: str) -> tuple[bool, str]:
    from datasets import load_dataset

    last_error = ""
    for dataset_path, dataset_name in TASK_DATASET_CANDIDATES[task]:
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
        help="Task list or group: quick, standard, leaderboard, all, or comma-separated task names",
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
