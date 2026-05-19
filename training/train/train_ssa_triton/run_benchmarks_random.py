from functools import cached_property
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "/work/m24047/m24047brmn/tokenizers/luciole_50k"

# Mapping from the short/old dataset name that lm-eval task YAMLs reference
# to the fully-qualified HF Hub name under which the data was actually cached.
# The HF datasets library stores "owner/repo" as "owner___repo" on disk; when
# running offline it only resolves cached entries whose directory name matches.
# If a YAML says ``dataset_path: hellaswag`` but the cache dir is
# ``Rowan___hellaswag``, the lookup fails in offline mode.  We fix this by
# creating a symlink ``<cache>/hellaswag -> <cache>/Rowan___hellaswag``.
_DATASET_CACHE_ALIASES = {
    # short (YAML) name  →  qualified (cached) name
    "hellaswag": "Rowan___hellaswag",
    "truthful_qa": "truthfulqa___truthful_qa",
    "openbookqa": "allenai___openbookqa",
    "lambada_openai": "EleutherAI___lambada_openai",
    "orange_sum": "EdinburghNLP___orange_sum",
    "xnli": "facebook___xnli",
}


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

FRENCH_BENCH_BENCHMARKS = {
    "french_bench_arc_challenge": {
        "task_name": "french_bench_arc_challenge",
        "description": "FrenchBench ARC Challenge",
        "num_fewshot": 3,
    },
    "french_bench_boolqa": {
        "task_name": "french_bench_boolqa",
        "description": "FrenchBench BoolQA",
        "num_fewshot": 3,
    },
    "french_bench_fquadv2": {
        "task_name": "french_bench_fquadv2",
        "description": "FrenchBench FQuADv2 extractive QA",
        "num_fewshot": 3,
    },
    "french_bench_fquadv2_bool": {
        "task_name": "french_bench_fquadv2_bool",
        "description": "FrenchBench FQuADv2 yes/no",
        "num_fewshot": 3,
    },
    "french_bench_fquadv2_genq": {
        "task_name": "french_bench_fquadv2_genq",
        "description": "FrenchBench FQuADv2 question generation",
        "num_fewshot": 3,
    },
    "french_bench_fquadv2_hasAns": {
        "task_name": "french_bench_fquadv2_hasAns",
        "description": "FrenchBench FQuADv2 answerability",
        "num_fewshot": 3,
    },
    "french_bench_grammar": {
        "task_name": "french_bench_grammar",
        "description": "FrenchBench grammar",
        "num_fewshot": 3,
    },
    "french_bench_hellaswag": {
        "task_name": "french_bench_hellaswag",
        "description": "FrenchBench HellaSwag",
        "num_fewshot": 3,
    },
    "french_bench_multifquad": {
        "task_name": "french_bench_multifquad",
        "description": "FrenchBench MultiFQuAD",
        "num_fewshot": 3,
    },
    "french_bench_opus_perplexity": {
        "task_name": "french_bench_opus_perplexity",
        "description": "FrenchBench OPUS perplexity",
        "num_fewshot": 0,
    },
    "french_bench_orangesum_abstract": {
        "task_name": "french_bench_orangesum_abstract",
        "description": "FrenchBench OrangeSum abstract generation",
        "num_fewshot": 3,
    },
    "french_bench_orangesum_title": {
        "task_name": "french_bench_orangesum_title",
        "description": "FrenchBench OrangeSum title generation",
        "num_fewshot": 3,
    },
    "french_bench_reading_comp": {
        "task_name": "french_bench_reading_comp",
        "description": "FrenchBench reading comprehension",
        "num_fewshot": 3,
    },
    "french_bench_topic_based_nli": {
        "task_name": "french_bench_topic_based_nli",
        "description": "FrenchBench topic-based NLI",
        "num_fewshot": 3,
    },
    "french_bench_trivia": {
        "task_name": "french_bench_trivia",
        "description": "FrenchBench trivia",
        "num_fewshot": 3,
    },
    "french_bench_vocab": {
        "task_name": "french_bench_vocab",
        "description": "FrenchBench vocabulary",
        "num_fewshot": 3,
    },
    "french_bench_wikitext_fr": {
        "task_name": "french_bench_wikitext_fr",
        "description": "FrenchBench WikiText-FR perplexity",
        "num_fewshot": 0,
    },
    "french_bench_xnli": {
        "task_name": "french_bench_xnli",
        "description": "FrenchBench XNLI (fr)",
        "num_fewshot": 3,
    },
}
FRENCH_BENCH_TASKS = list(FRENCH_BENCH_BENCHMARKS.keys())
FRENCH_BENCH_PERPLEXITY_TASKS = [
    "french_bench_opus_perplexity",
    "french_bench_wikitext_fr",
]


def _ensure_dataset_cache_symlinks() -> None:
    """Create symlinks in the HF datasets cache so that short dataset names
    used by lm-eval task YAMLs resolve to their fully-qualified cached copies.

    This is only relevant when ``HF_DATASETS_OFFLINE=1``.
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_dir = Path(hf_home) / "datasets"
    if not cache_dir.is_dir():
        return

    for short_name, qualified_name in _DATASET_CACHE_ALIASES.items():
        source = cache_dir / qualified_name
        target = cache_dir / short_name
        if source.is_dir() and not target.exists():
            try:
                target.symlink_to(source)
                logger.info("Created dataset cache symlink: %s -> %s", target, source)
            except OSError as exc:
                logger.warning(
                    "Could not create dataset cache symlink %s -> %s: %s",
                    target,
                    source,
                    exc,
                )
        elif target.exists():
            logger.debug("Dataset cache entry already exists: %s", target)


def _patch_datasets_list_feature_type() -> None:
    """Register 'List' as an alias for 'Sequence' in the datasets feature registry.

    Some cached Arrow tables (notably SuperGLUE/ReCoRD) embed ``_type: "List"``
    in their schema metadata.  Recent ``datasets`` releases removed the ``List``
    feature type, causing:
        ``ValueError: Feature type 'List' not found``

    Monkey-patching ``_FEATURE_TYPES`` fixes this at the root — it covers both
    ``dataset_info.json`` and Arrow-schema-embedded metadata.
    """
    try:
        from datasets.features.features import _FEATURE_TYPES

        if "List" not in _FEATURE_TYPES:
            from datasets.features import Sequence

            _FEATURE_TYPES["List"] = Sequence
            logger.info("Enabled datasets compatibility shim: List -> Sequence")
    except (ImportError, AttributeError):
        pass


AVAILABLE_BENCHMARKS = {
    "arc_easy": {
        "task_name": "arc_easy",
        "description": "ARC Easy - Science questions from standardized tests",
        "num_fewshot": 25,
    },
    "arc_challenge": {
        "task_name": "arc_challenge",
        "description": "ARC Challenge - Harder science questions",
        "num_fewshot": 25,
    },
    "hellaswag": {
        "task_name": "hellaswag",
        "description": "HellaSwag - Commonsense reasoning about situations",
        "num_fewshot": 10,
    },
    "piqa": {
        "task_name": "piqa",
        "description": "PIQA - Physical commonsense reasoning",
        "num_fewshot": 5,
    },
    "winogrande": {
        "task_name": "winogrande",
        "description": "WinoGrande - Pronoun resolution benchmark",
        "num_fewshot": 5,
    },
    "truthfulqa": {
        "task_name": "truthfulqa_mc2",
        "description": "TruthfulQA - Measuring model truthfulness",
        "num_fewshot": 0,
    },
    "gsm8k": {
        "task_name": "gsm8k",
        "description": "GSM8K - Grade school math word problems",
        "num_fewshot": 5,
    },
    "boolq": {
        "task_name": "boolq",
        "description": "BoolQ - Boolean question answering",
        "num_fewshot": 0,
    },
    "cb": {
        "task_name": "cb",
        "description": "CB - CommitmentBank natural language inference",
        "num_fewshot": 0,
    },
    "copa": {
        "task_name": "copa",
        "description": "COPA - Causal reasoning with alternatives",
        "num_fewshot": 0,
    },
    "multirc": {
        "task_name": "multirc",
        "description": "MultiRC - Multi-sentence reading comprehension",
        "num_fewshot": 0,
    },
    "record": {
        "task_name": "record",
        "description": "ReCoRD - Reading comprehension with commonsense reasoning",
        "num_fewshot": 0,
    },
    "rte": {
        "task_name": "rte",
        "description": "RTE - Recognizing textual entailment",
        "num_fewshot": 0,
    },
    "wic": {
        "task_name": "wic",
        "description": "WiC - Word-in-context disambiguation",
        "num_fewshot": 0,
    },
    "wsc": {
        "task_name": "wsc",
        "description": "WSC - Winograd Schema Challenge coreference",
        "num_fewshot": 0,
    },
    "axb": {
        "task_name": "axb",
        "description": "AX-b - Broad-coverage diagnostics from SuperGLUE",
        "num_fewshot": 0,
    },
    "axg": {
        "task_name": "axg",
        "description": "AX-g - Winogender diagnostics from SuperGLUE",
        "num_fewshot": 0,
    },
    "openbookqa": {
        "task_name": "openbookqa",
        "description": "OpenBookQA - Elementary science questions",
        "num_fewshot": 0,
    },
    "lambada": {
        "task_name": "lambada_openai",
        "description": "LAMBADA - Word prediction requiring broad context",
        "num_fewshot": 0,
    },
    **FRENCH_BENCH_BENCHMARKS,
}

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
    "french_bench": FRENCH_BENCH_TASKS,
    "french_bench_perplexity": FRENCH_BENCH_PERPLEXITY_TASKS,
    "french_bench_all": FRENCH_BENCH_TASKS,
    "all": [
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "truthfulqa",
        # "gsm8k",
        "openbookqa",
        "lambada",
    ]
    + SUPERGLUE_ALL_TASKS,
}


def get_task_list(task_string: str) -> List[str]:
    if task_string.lower() in BENCHMARK_GROUPS:
        return BENCHMARK_GROUPS[task_string.lower()]

    raw_items = [t.strip().lower() for t in task_string.split(",") if t.strip()]
    tasks = []
    invalid_tasks = []
    
    for item in raw_items:
        if item in BENCHMARK_GROUPS:
            tasks.extend(BENCHMARK_GROUPS[item])
        elif item in AVAILABLE_BENCHMARKS:
            tasks.append(item)
        else:
            invalid_tasks.append(item)
            
    if invalid_tasks:
        logger.warning("Unknown tasks or groups: %s", invalid_tasks)
        logger.info("Available tasks: %s", list(AVAILABLE_BENCHMARKS.keys()))
        logger.info("Available groups: %s", list(BENCHMARK_GROUPS.keys()))
        
    return list(dict.fromkeys(tasks)) # remove duplicates


def _merge_lm_eval_results(results_list: List[dict]) -> dict:
    """Merge multiple lm-eval result payloads into one."""
    if not results_list:
        return {}
    if len(results_list) == 1:
        return results_list[0]

    merged = {}
    for result in results_list:
        for key, value in result.items():
            if isinstance(value, dict):
                merged.setdefault(key, {})
                merged[key].update(value)
            elif isinstance(value, list):
                merged.setdefault(key, [])
                merged[key].extend(value)
            else:
                merged[key] = value
    return merged


def _build_eval_groups(task_names: List[str], default_limit: Optional[int], gsm8k_limit: Optional[int]):
    """
    Split tasks so gsm8k can use its own limit without affecting other tasks.
    """
    gsm8k_task_name = AVAILABLE_BENCHMARKS["gsm8k"]["task_name"]
    has_gsm8k = gsm8k_task_name in task_names

    regular_tasks = [t for t in task_names if t != gsm8k_task_name]
    groups = []
    if regular_tasks:
        groups.append(
            {
                "name": "regular",
                "tasks": regular_tasks,
                "limit": default_limit,
            }
        )

    if has_gsm8k:
        groups.append(
            {
                "name": gsm8k_task_name,
                "tasks": [gsm8k_task_name],
                "limit": gsm8k_limit if gsm8k_limit is not None else default_limit,
            }
        )

    return groups


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"

    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _estimate_task_total_docs(task) -> Optional[int]:
    split_candidates = (
        ("has_test_docs", "test_docs", "test"),
        ("has_validation_docs", "validation_docs", "validation"),
        ("has_training_docs", "training_docs", "train"),
    )

    def _safe_len(value):
        try:
            return len(value)
        except (TypeError, AttributeError):
            return None

    dataset = getattr(task, "dataset", None)

    for has_name, docs_name, split_name in split_candidates:
        has_fn = getattr(task, has_name, None)
        if callable(has_fn):
            try:
                if not has_fn():
                    continue
            except Exception:
                pass

        docs_fn = getattr(task, docs_name, None)
        if callable(docs_fn):
            try:
                docs = docs_fn()
            except Exception:
                continue

            doc_count = _safe_len(docs)
            if doc_count is not None:
                return doc_count

        if dataset is not None:
            try:
                split_docs = dataset[split_name]
            except Exception:
                split_docs = None

            doc_count = _safe_len(split_docs)
            if doc_count is not None:
                return doc_count

    return None


def _estimate_group_runtime(task_objects, task_names: List[str], group_limit: Optional[int]):
    total_docs = 0
    evaluated_docs = 0

    for task_name in task_names:
        task = task_objects.get(task_name)
        if task is None:
            return None, None

        task_total_docs = _estimate_task_total_docs(task)
        if task_total_docs is None:
            return None, None

        total_docs += task_total_docs
        if group_limit is None:
            evaluated_docs += task_total_docs
        else:
            evaluated_docs += min(task_total_docs, group_limit)

    if total_docs <= 0 or evaluated_docs <= 0:
        return None, None

    return total_docs, evaluated_docs


try:
    from lm_eval.api.model import LM as LMBase
except ImportError:
    LMBase = object


try:
    from lm_eval.models.dummy import DummyLM
except ImportError:
    DummyLM = object

class RandomBaselineLM(DummyLM):
    """Random prediction baseline using lm_eval DummyLM."""

    def __init__(self, tokenizer_name: str, max_length: int = 2048):
        super().__init__()
        self._tokenizer_name_custom = tokenizer_name
        self._max_length = max_length
        self._batch_size = 1

    @cached_property
    def tokenizer(self):
        from functools import cached_property
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            self._tokenizer_name_custom, trust_remote_code=True, use_fast=False
        )

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cpu"

    def get_performance_summary(self) -> dict:
        return {
            "calls": 0, "input_tokens": 0, "prep_seconds": 0.0,
            "forward_seconds": 0.0, "output_seconds": 0.0, "total_seconds": 0.0,
            "avg_call_seconds": 0.0, "forward_tokens_per_second": 0.0,
        }

def run_evaluation(
    tasks: List[str],
    tokenizer_name: str = DEFAULT_TOKENIZER,
    batch_size: int = 1,
    max_length: int = 2048,
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    gsm8k_limit: Optional[int] = 100,
    gsm8k_random_seed: int = 42,
    output_path: Optional[str] = None,
):
    try:
        import lm_eval
    except ImportError:
        logger.error(
            "lm-evaluation-harness not installed. Install with: pip install lm-eval"
        )
        sys.exit(1)

    # Ensure offline dataset cache resolves short names used by lm-eval YAMLs
    _ensure_dataset_cache_symlinks()
    _patch_datasets_list_feature_type()

    model = RandomBaselineLM(
        tokenizer_name=tokenizer_name,
        max_length=max_length,
    )
    perf_summary = None

    task_names = []
    for task in tasks:
        if task in AVAILABLE_BENCHMARKS:
            task_names.append(AVAILABLE_BENCHMARKS[task]["task_name"])
            logger.info(
                "Added task: %s -> %s", task, AVAILABLE_BENCHMARKS[task]["task_name"]
            )
            logger.info("  Description: %s", AVAILABLE_BENCHMARKS[task]["description"])
        else:
            logger.warning("Unknown task: %s, skipping", task)

    if not task_names:
        logger.error("No valid tasks specified")
        sys.exit(1)

    # Pre-check: verify each task's dataset is loadable offline.
    # lm-eval loads all tasks eagerly and fails on the first broken one,
    # so we probe each task individually and drop those that can't load.
    verified_tasks = []
    verified_task_objects = {}
    from lm_eval.tasks import get_task_dict

    for t in task_names:
        try:
            task_dict = get_task_dict([t])
            verified_tasks.append(t)
            verified_task_objects[t] = next(iter(task_dict.values()))
        except Exception as exc:
            logger.warning("Skipping task '%s' (dataset unavailable offline): %s", t, exc)

    if not verified_tasks:
        logger.error("No tasks could be loaded (all datasets unavailable offline)")
        sys.exit(1)

    if len(verified_tasks) < len(task_names):
        skipped = [t for t in task_names if t not in verified_tasks]
        logger.info("Skipped %d task(s) due to offline cache issues: %s", len(skipped), skipped)

    eval_groups = _build_eval_groups(
        task_names=verified_tasks,
        default_limit=limit,
        gsm8k_limit=gsm8k_limit,
    )

    if not eval_groups:
        logger.error("No evaluation groups were generated")
        sys.exit(1)

    logger.info("Running evaluation on tasks: %s", verified_tasks)
    evaluation_start = time.perf_counter()
    partial_results = []
    group_timings = []
    estimated_full_dataset_seconds = 0.0
    have_full_dataset_estimate = True
    for group in eval_groups:
        group_name = group["name"]
        group_tasks = group["tasks"]
        group_limit = group["limit"]

        logger.info(
            "Evaluating group '%s' tasks=%s limit=%s",
            group_name,
            group_tasks,
            group_limit,
        )

        group_start = time.perf_counter()

        eval_kwargs = {
            "model": model,
            "tasks": group_tasks,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
            "limit": group_limit,
            "log_samples": False,
        }

        if group_name == AVAILABLE_BENCHMARKS["gsm8k"]["task_name"]:
            # Keep subset reproducible while avoiding always using the same head slice.
            eval_kwargs["random_seed"] = gsm8k_random_seed
            eval_kwargs["numpy_random_seed"] = gsm8k_random_seed + 1
            eval_kwargs["fewshot_random_seed"] = gsm8k_random_seed + 2
            logger.info(
                "gsm8k subset config: limit=%s, random_seed=%s",
                group_limit,
                gsm8k_random_seed,
            )

        try:
            partial = lm_eval.simple_evaluate(**eval_kwargs)
        except Exception as exc:
            logger.error("Evaluation failed for group '%s': %s", group_name, exc)
            raise

        group_elapsed = time.perf_counter() - group_start
        group_total_docs, group_evaluated_docs = _estimate_group_runtime(
            task_objects=verified_task_objects,
            task_names=group_tasks,
            group_limit=group_limit,
        )

        group_estimated_full_seconds = None
        if group_total_docs is not None and group_evaluated_docs is not None:
            if group_evaluated_docs > 0:
                group_estimated_full_seconds = (
                    group_elapsed * group_total_docs / group_evaluated_docs
                )
                estimated_full_dataset_seconds += group_estimated_full_seconds
            else:
                have_full_dataset_estimate = False
        else:
            have_full_dataset_estimate = False

        group_timings.append(
            {
                "name": group_name,
                "tasks": group_tasks,
                "limit": group_limit,
                "elapsed_seconds": group_elapsed,
                "elapsed_human": _format_duration(group_elapsed),
                "evaluated_docs": group_evaluated_docs,
                "total_docs": group_total_docs,
                "estimated_full_dataset_seconds": group_estimated_full_seconds,
                "estimated_full_dataset_human": _format_duration(
                    group_estimated_full_seconds
                ),
            }
        )

        if group_estimated_full_seconds is not None:
            logger.info(
                "Completed group '%s' in %s (evaluated %s docs, full dataset estimate %s)",
                group_name,
                _format_duration(group_elapsed),
                f"{group_evaluated_docs}/{group_total_docs}",
                _format_duration(group_estimated_full_seconds),
            )
        else:
            logger.info(
                "Completed group '%s' in %s (full dataset estimate unavailable)",
                group_name,
                _format_duration(group_elapsed),
            )

        partial_results.append(partial)

    results = _merge_lm_eval_results(partial_results)
    total_elapsed_seconds = time.perf_counter() - evaluation_start
    if not have_full_dataset_estimate:
        estimated_full_dataset_seconds = None

    task_timing_rows = []
    for group in group_timings:
        group_tasks = group["tasks"]
        group_elapsed = group["elapsed_seconds"]
        group_evaluated_docs = group["evaluated_docs"]

        for task_name in group_tasks:
            task = verified_task_objects.get(task_name)
            if task is None:
                continue

            task_total_docs = _estimate_task_total_docs(task)
            if task_total_docs is None:
                continue

            task_limit = group["limit"]
            task_evaluated_docs = task_total_docs if task_limit is None else min(task_total_docs, task_limit)
            if group_evaluated_docs > 0:
                task_elapsed_seconds = group_elapsed * task_evaluated_docs / group_evaluated_docs
            else:
                task_elapsed_seconds = None

            if task_elapsed_seconds is not None and task_evaluated_docs > 0:
                task_estimated_full_seconds = task_elapsed_seconds * task_total_docs / task_evaluated_docs
            else:
                task_estimated_full_seconds = None

            task_timing_rows.append(
                {
                    "task": task_name,
                    "limit": task_limit,
                    "evaluated_docs": task_evaluated_docs,
                    "total_docs": task_total_docs,
                    "elapsed_seconds": task_elapsed_seconds,
                    "elapsed_human": _format_duration(task_elapsed_seconds),
                    "estimated_full_dataset_seconds": task_estimated_full_seconds,
                    "estimated_full_dataset_human": _format_duration(
                        task_estimated_full_seconds
                    ),
                }
            )

    print("\n" + "=" * 70)
    print("RANDOM BASELINE EVALUATION RESULTS")
    print("=" * 70)
    print("-" * 70)
    for task_name, task_results in results.get("results", {}).items():
        print(f"\n{task_name}:")
        print("-" * 50)
        for metric, value in task_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    if task_timing_rows:
        print("\nTASK TIMING")
        print("-" * 70)
        print(f"{'task':<20} {'elapsed':<12} {'full-est':<12} {'docs':<14}")
        print("-" * 70)
        for row in task_timing_rows:
            docs_text = f"{row['evaluated_docs']}/{row['total_docs']}"
            print(
                f"{row['task']:<20} {row['elapsed_human']:<12} {row['estimated_full_dataset_human']:<12} {docs_text:<14}"
            )
    print("\n" + "=" * 70)

    logger.info(
        "Finished random baseline evaluation in %s",
        _format_duration(total_elapsed_seconds),
    )
    perf_summary = model.get_performance_summary()
    logger.info(
        "Forward perf summary: calls=%s input_tokens=%s prep=%.3fs forward=%.3fs output=%.3fs total=%.3fs avg_call=%s forward_toks_per_s=%s",
        perf_summary["calls"],
        perf_summary["input_tokens"],
        perf_summary["prep_seconds"],
        perf_summary["forward_seconds"],
        perf_summary["output_seconds"],
        perf_summary["total_seconds"],
        f"{perf_summary['avg_call_seconds']:.4f}s" if perf_summary["avg_call_seconds"] is not None else "n/a",
        f"{perf_summary['forward_tokens_per_second']:.1f}" if perf_summary["forward_tokens_per_second"] is not None else "n/a",
    )
    if estimated_full_dataset_seconds is not None:
        logger.info(
            "Estimated full-dataset runtime: %s",
            _format_duration(estimated_full_dataset_seconds),
        )
    else:
        logger.info(
            "Estimated full-dataset runtime unavailable",
        )

    payload = {
        "evaluation_type": "random_baseline",
        "model_name": "random_baseline",
        "model_type": "baseline",
        "checkpoint": "n/a",
        "tasks": verified_tasks,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "max_length": max_length,
        "limit": limit,
        "gsm8k_limit": gsm8k_limit,
        "gsm8k_random_seed": gsm8k_random_seed,
        "timing": {
            "elapsed_seconds": total_elapsed_seconds,
            "elapsed_human": _format_duration(total_elapsed_seconds),
            "estimated_full_dataset_seconds": estimated_full_dataset_seconds,
            "estimated_full_dataset_human": _format_duration(
                estimated_full_dataset_seconds
            ),
            "groups": group_timings,
            "tasks": task_timing_rows,
        },
        "forward_performance": perf_summary,
        "results": results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        logger.info("Results saved to: %s", output_path)

    return payload


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run random baseline evaluation on LM Evaluation Harness benchmarks."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_easy",
        help=(
            "Comma-separated list of tasks or group name "
            "(quick, standard, leaderboard, superglue_core, superglue, "
            "french_bench, french_bench_perplexity, french_bench_all, all)"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (overrides task defaults)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit examples per task"
    )
    parser.add_argument(
        "--gsm8k-limit",
        type=int,
        default=100,
        help=(
            "When gsm8k is selected, evaluate only this many examples. "
            "Set to 0 or a negative number to disable gsm8k-specific override."
        ),
    )
    parser.add_argument(
        "--gsm8k-random-seed",
        type=int,
        default=42,
        help="Seed used for gsm8k subset evaluation randomness.",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save results JSON"
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    tasks = get_task_list(args.tasks)
    if not tasks:
        logger.error("No valid tasks specified")
        parser.print_help()
        sys.exit(1)

    logger.info("Tasks to evaluate: %s", tasks)
    effective_gsm8k_limit = args.gsm8k_limit if args.gsm8k_limit and args.gsm8k_limit > 0 else None
    logger.info(
        "GSM8K subset config: limit=%s (seed=%s)",
        effective_gsm8k_limit,
        args.gsm8k_random_seed,
    )

    result = run_evaluation(
        tasks=tasks,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        gsm8k_limit=effective_gsm8k_limit,
        gsm8k_random_seed=args.gsm8k_random_seed,
        output_path=args.output,
    )

    print("\n" + "=" * 70)
    if args.output:
        print(f"Results saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
