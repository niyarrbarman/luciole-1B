import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from eval_perplexity import (
    cleanup_parallel_state as cleanup_default_parallel_state,
    load_model as load_baby_luciole_model,
)
from eval_perplexity_triton import (
    cleanup_parallel_state as cleanup_triton_parallel_state,
    load_model as load_triton_model,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "/work/m24047/m24047brmn/tokenizers/luciole_50k"
MODEL_TYPE_SSA_TRITON_V4 = "ssa_triton"
MODEL_TYPE_SSA = "ssa"
MODEL_TYPE_SOFTMAX = "softmax"
MODEL_TYPES = (MODEL_TYPE_SSA_TRITON_V4, MODEL_TYPE_SSA, MODEL_TYPE_SOFTMAX)
GSM8K_BENCHMARK_KEY = "gsm8k"

DEFAULT_SSA_TRITON_V4_CHECKPOINT = (
    "/tmpdir/m24047brmn/nemo_1b/output/baby_luciole-ssa-triton/checkpoints/"
    "baby_luciole-ssa-triton-step=0023999"
)
DEFAULT_SOFTMAX_CHECKPOINT = (
    "/tmpdir/m24047brmn/nemo_1b/output/baby_luciole-softmax-test/checkpoints/"
    "baby_luciole-softmax-test-step=0020998-last"
)


@dataclass
class ModelSpec:
    name: str
    model_type: str
    checkpoint_path: str

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


def _parse_model_spec(value: str) -> ModelSpec:
    """
    Parse model spec in format:
        name|model_type|/path/to/checkpoint
    """
    parts = [part.strip() for part in value.split("|", 2)]
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"Invalid --model value '{value}'. Expected: name|model_type|checkpoint_path"
        )

    name, model_type, checkpoint_path = parts
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Invalid model_type '{model_type}' in --model '{value}'. "
            f"Valid types: {MODEL_TYPES}"
        )

    return ModelSpec(name=name, model_type=model_type, checkpoint_path=checkpoint_path)


def _sanitize_model_name(name: str) -> str:
    safe_chars = []
    for char in name:
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("._")
    return sanitized or "model"


def _build_model_specs(args) -> List[ModelSpec]:
    if args.model:
        return [_parse_model_spec(value) for value in args.model]

    if args.checkpoint:
        model_name = args.model_name or Path(args.checkpoint).name
        return [
            ModelSpec(
                name=model_name,
                model_type=args.model_type,
                checkpoint_path=args.checkpoint,
            )
        ]

    logger.info(
        "No --model/--checkpoint provided; using default SSA and softmax checkpoints"
    )
    return [
        ModelSpec(
            name="baby_luciole_ssa_triton",
            model_type=MODEL_TYPE_SSA_TRITON_V4,
            checkpoint_path=DEFAULT_SSA_TRITON_V4_CHECKPOINT,
        ),
        ModelSpec(
            name="baby_luciole_softmax",
            model_type=MODEL_TYPE_SOFTMAX,
            checkpoint_path=DEFAULT_SOFTMAX_CHECKPOINT,
        ),
    ]


def _model_output_path(base_output_path: Optional[str], model_name: str, multi_model: bool) -> Optional[str]:
    if not base_output_path:
        return None

    if not multi_model:
        return base_output_path

    output = Path(base_output_path)
    suffix = output.suffix or ".json"
    filename = f"{output.stem}_{_sanitize_model_name(model_name)}{suffix}"
    return str(output.with_name(filename))


def _cleanup_parallel_state():
    for cleanup_fn in [cleanup_triton_parallel_state, cleanup_default_parallel_state]:
        try:
            cleanup_fn()
        except Exception as exc:
            logger.debug("Parallel state cleanup via %s failed: %s", cleanup_fn.__name__, exc)


def get_task_list(task_string: str) -> List[str]:
    if task_string.lower() in BENCHMARK_GROUPS:
        return BENCHMARK_GROUPS[task_string.lower()]

    tasks = [t.strip().lower() for t in task_string.split(",") if t.strip()]
    invalid_tasks = [t for t in tasks if t not in AVAILABLE_BENCHMARKS]
    if invalid_tasks:
        logger.warning("Unknown tasks: %s", invalid_tasks)
        logger.info("Available tasks: %s", list(AVAILABLE_BENCHMARKS.keys()))
        logger.info("Available groups: %s", list(BENCHMARK_GROUPS.keys()))
        tasks = [t for t in tasks if t in AVAILABLE_BENCHMARKS]
    return tasks


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
    gsm8k_task_name = AVAILABLE_BENCHMARKS[GSM8K_BENCHMARK_KEY]["task_name"]
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


class BabyLucioleLM(LMBase):
    """lm-eval wrapper for Baby Luciole checkpoints (SSA Triton / SSA / softmax)."""

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = MODEL_TYPE_SSA_TRITON_V4,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        device: str = "cuda",
        batch_size: int = 1,
        max_length: int = 2048,
        compiled_bda: bool = False,
        force_contiguous_qkv: bool = True,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length

        logger.info(
            "Initializing BabyLucioleLM wrapper (model_type=%s, checkpoint=%s)...",
            model_type,
            checkpoint_path,
        )

        if model_type == MODEL_TYPE_SSA_TRITON_V4:
            self.model, self.tokenizer = load_triton_model(
                checkpoint_path=checkpoint_path,
                tokenizer_name=tokenizer_name,
                device=device,
                compiled_bda=compiled_bda,
                force_contiguous_qkv=force_contiguous_qkv,
            )
        elif model_type in (MODEL_TYPE_SSA, MODEL_TYPE_SOFTMAX):
            if compiled_bda or not force_contiguous_qkv:
                logger.warning(
                    "compiled_bda/force_contiguous_qkv are only used for %s; ignoring for %s",
                    MODEL_TYPE_SSA_TRITON_V4,
                    model_type,
                )

            self.model, self.tokenizer = load_baby_luciole_model(
                checkpoint_path=checkpoint_path,
                tokenizer_name=tokenizer_name,
                device=device,
                use_ssa=(model_type == MODEL_TYPE_SSA),
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self._setup_tokenizer()
        logger.info("BabyLucioleLM wrapper initialized successfully")

    def _setup_tokenizer(self):
        if hasattr(self.tokenizer, "vocab_size"):
            self.vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, "vocab"):
            self.vocab_size = len(self.tokenizer.vocab)
        else:
            self.vocab_size = 50000

        if hasattr(self.tokenizer, "eos_id"):
            self._eot_token_id = self.tokenizer.eos_id
        elif hasattr(self.tokenizer, "eos_token_id"):
            self._eot_token_id = self.tokenizer.eos_token_id
        else:
            self._eot_token_id = 2

        if hasattr(self.tokenizer, "bos_id"):
            self.prefix_token_id = self.tokenizer.bos_id
        elif hasattr(self.tokenizer, "bos_token_id"):
            self.prefix_token_id = self.tokenizer.bos_token_id
        else:
            self.prefix_token_id = 1

        logger.info(
            "Tokenizer setup: vocab_size=%s, eot_token_id=%s, prefix_token_id=%s",
            self.vocab_size,
            self._eot_token_id,
            self.prefix_token_id,
        )

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_length(self):
        return self._max_length

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    @property
    def eot_token_id(self):
        return self._eot_token_id

    @property
    def max_gen_toks(self):
        return 256

    def tok_encode(
        self,
        string: str,
        left_truncate_len: int = None,
        add_special_tokens: bool = None,
    ) -> List[int]:
        if hasattr(self.tokenizer, "text_to_ids"):
            tokens = self.tokenizer.text_to_ids(string)
        else:
            tokens = self.tokenizer.encode(string)

        if left_truncate_len is not None and len(tokens) > left_truncate_len:
            tokens = tokens[-left_truncate_len:]

        return tokens

    def tok_decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        if hasattr(self.tokenizer, "ids_to_text"):
            return self.tokenizer.ids_to_text(tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
            attention_mask = torch.ones_like(input_ids, device=self.device)

            if hasattr(self.model, "module") and self.model.module is not None:
                outputs = self.model.module(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )

            if hasattr(outputs, "logits"):
                return outputs.logits
            if isinstance(outputs, torch.Tensor):
                return outputs
            return outputs[0]

    def loglikelihood(self, requests) -> List[tuple]:
        results = []
        for request in requests:
            if hasattr(request, "args"):
                context, continuation = request.args
            else:
                context, continuation = request

            context_ids = self.tok_encode(context)
            continuation_ids = self.tok_encode(continuation)

            full_ids = context_ids + continuation_ids
            if len(full_ids) > self.max_length:
                full_ids = full_ids[-self.max_length :]
                context_ids = full_ids[: -len(continuation_ids)]

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
            logits = self._model_call(input_ids)
            log_probs = torch.log_softmax(logits, dim=-1)

            continuation_start = len(context_ids)
            continuation_logprobs = []
            greedy_tokens = []

            for i, token_id in enumerate(continuation_ids):
                pos = continuation_start + i - 1
                if 0 <= pos < log_probs.shape[1]:
                    continuation_logprobs.append(log_probs[0, pos, token_id].item())
                    greedy_tokens.append(log_probs[0, pos].argmax().item() == token_id)

            total_logprob = sum(continuation_logprobs)
            is_greedy = all(greedy_tokens) if greedy_tokens else False
            results.append((total_logprob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests) -> List[tuple]:
        results = []
        for request in requests:
            if hasattr(request, "args"):
                text = request.args[0]
            else:
                text = request

            tokens = self.tok_encode(text)
            if len(tokens) > self.max_length:
                tokens = tokens[-self.max_length :]

            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            logits = self._model_call(input_ids)
            log_probs = torch.log_softmax(logits, dim=-1)

            total_logprob = 0.0
            for i in range(1, len(tokens)):
                total_logprob += log_probs[0, i - 1, tokens[i]].item()
            results.append((total_logprob,))
        return results

    def generate_until(self, requests) -> List[str]:
        results = []
        for request in requests:
            if hasattr(request, "args"):
                context = request.args[0]
                gen_kwargs = request.args[1] if len(request.args) > 1 else {}
            else:
                context, gen_kwargs = request

            until = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", 128)
            temperature = gen_kwargs.get("temperature", 0.0)

            input_ids = self.tok_encode(context)
            if len(input_ids) > self.max_length - max_gen_toks:
                input_ids = input_ids[-(self.max_length - max_gen_toks) :]

            input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            generated_ids = input_ids.clone()

            for _ in range(max_gen_toks):
                logits = self._model_call(generated_ids)
                next_token_logits = logits[:, -1, :]

                if temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if next_token.item() == self.eot_token_id:
                    break

                current_text = self.tok_decode(
                    generated_ids[0, input_ids.shape[1] :].tolist()
                )
                if any(stop_str in current_text for stop_str in until):
                    break

            generated_text = self.tok_decode(
                generated_ids[0, input_ids.shape[1] :].tolist()
            )
            for stop_str in until:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0]
            results.append(generated_text)

        return results


def run_evaluation(
    model_spec: ModelSpec,
    tasks: List[str],
    tokenizer_name: str = DEFAULT_TOKENIZER,
    device: str = "cuda",
    batch_size: int = 1,
    max_length: int = 2048,
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    gsm8k_limit: Optional[int] = 100,
    gsm8k_random_seed: int = 42,
    output_path: Optional[str] = None,
    compiled_bda: bool = False,
    force_contiguous_qkv: bool = True,
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

    model = BabyLucioleLM(
        checkpoint_path=model_spec.checkpoint_path,
        model_type=model_spec.model_type,
        tokenizer_name=tokenizer_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        compiled_bda=compiled_bda,
        force_contiguous_qkv=force_contiguous_qkv,
    )

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

        if group_name == AVAILABLE_BENCHMARKS[GSM8K_BENCHMARK_KEY]["task_name"]:
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
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Model name:   {model_spec.name}")
    print(f"Model type:   {model_spec.model_type}")
    print(f"Checkpoint:   {model_spec.checkpoint_path}")
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
        "Finished evaluation for model '%s' in %s",
        model_spec.name,
        _format_duration(total_elapsed_seconds),
    )
    if estimated_full_dataset_seconds is not None:
        logger.info(
            "Estimated full-dataset runtime for model '%s': %s",
            model_spec.name,
            _format_duration(estimated_full_dataset_seconds),
        )
    else:
        logger.info(
            "Estimated full-dataset runtime for model '%s' unavailable",
            model_spec.name,
        )

    payload = {
        "model_name": model_spec.name,
        "model_type": model_spec.model_type,
        "checkpoint": model_spec.checkpoint_path,
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
        description=(
            "Run LM Evaluation Harness benchmarks on Baby Luciole models "
            "(SSA Triton / SSA / softmax)."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model spec in format 'name|model_type|checkpoint_path'. "
            "Repeat --model to evaluate multiple models in one run."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to NeMo checkpoint directory for single-model mode. "
            "Ignored when --model is provided."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=MODEL_TYPE_SSA_TRITON_V4,
        choices=MODEL_TYPES,
        help=(
            "Single-model mode type used with --checkpoint "
            f"(choices: {MODEL_TYPES})."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional display name for single-model mode (--checkpoint).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_easy",
        help=(
            "Comma-separated list of tasks or group name "
            "(quick, standard, leaderboard, superglue_core, superglue, all)"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to load model on"
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
    parser.add_argument(
        "--compiled-bda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable compiled bias-dropout-add path used in SSA Triton layer spec",
    )
    parser.add_argument(
        "--force-contiguous-qkv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Materialize contiguous Q/K/V tensors before Triton attention kernel",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("CUDA requested but not available. Please run on a GPU node.")
        sys.exit(1)

    tasks = get_task_list(args.tasks)
    if not tasks:
        logger.error("No valid tasks specified")
        parser.print_help()
        sys.exit(1)

    try:
        model_specs = _build_model_specs(args)
    except ValueError as exc:
        logger.error("%s", exc)
        parser.print_help()
        sys.exit(1)

    logger.info("Tasks to evaluate: %s", tasks)
    effective_gsm8k_limit = args.gsm8k_limit if args.gsm8k_limit and args.gsm8k_limit > 0 else None
    logger.info(
        "gsm8k subset override: limit=%s (seed=%s)",
        effective_gsm8k_limit,
        args.gsm8k_random_seed,
    )
    logger.info(
        "Models to evaluate: %s",
        [
            {
                "name": spec.name,
                "type": spec.model_type,
                "checkpoint": spec.checkpoint_path,
            }
            for spec in model_specs
        ],
    )

    all_results = []
    multi_model = len(model_specs) > 1

    try:
        for spec in model_specs:
            logger.info(
                "Starting evaluation for model '%s' (%s)",
                spec.name,
                spec.model_type,
            )

            per_model_output = _model_output_path(
                base_output_path=args.output,
                model_name=spec.name,
                multi_model=multi_model,
            )

            model_result = run_evaluation(
                model_spec=spec,
                tasks=tasks,
                tokenizer_name=args.tokenizer,
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                num_fewshot=args.num_fewshot,
                limit=args.limit,
                gsm8k_limit=effective_gsm8k_limit,
                gsm8k_random_seed=args.gsm8k_random_seed,
                output_path=per_model_output,
                compiled_bda=args.compiled_bda,
                force_contiguous_qkv=args.force_contiguous_qkv,
            )
            all_results.append(model_result)

            # Reset Megatron parallel/distributed state before loading next model.
            _cleanup_parallel_state()

    finally:
        _cleanup_parallel_state()

    if multi_model:
        print("\n" + "=" * 70)
        print("MULTI-MODEL BENCHMARK SUMMARY")
        print("=" * 70)
        for model_result in all_results:
            task_count = len(model_result.get("results", {}).get("results", {}))
            print(
                f"- {model_result['model_name']} ({model_result['model_type']}): "
                f"{task_count} evaluated task(s)"
            )
        print("=" * 70)

        if args.output:
            combined_payload = {
                "tasks_requested": tasks,
                "num_models": len(all_results),
                "models": all_results,
            }
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(combined_payload, handle, indent=2, default=str)
            logger.info("Combined results saved to: %s", args.output)


if __name__ == "__main__":
    main()
