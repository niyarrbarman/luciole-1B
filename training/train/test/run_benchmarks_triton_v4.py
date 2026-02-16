import argparse
import json
import logging
import os
import sys
from typing import List, Optional

import torch
from eval_perplexity_triton_v4 import cleanup_parallel_state, load_model

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


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
    "mmlu": {
        "task_name": "mmlu",
        "description": "MMLU - 57 subjects covering STEM, humanities, social sciences",
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
    "openbookqa": {
        "task_name": "openbookqa",
        "description": "OpenBookQA - Elementary science questions",
        "num_fewshot": 0,
    },
}

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
    "all": list(AVAILABLE_BENCHMARKS.keys()),
}


def get_task_list(task_string: str) -> List[str]:
    if task_string.lower() in BENCHMARK_GROUPS:
        return BENCHMARK_GROUPS[task_string.lower()]

    tasks = [t.strip().lower() for t in task_string.split(",")]
    invalid_tasks = [t for t in tasks if t not in AVAILABLE_BENCHMARKS]
    if invalid_tasks:
        logger.warning("Unknown tasks: %s", invalid_tasks)
        logger.info("Available tasks: %s", list(AVAILABLE_BENCHMARKS.keys()))
        logger.info("Available groups: %s", list(BENCHMARK_GROUPS.keys()))
        tasks = [t for t in tasks if t in AVAILABLE_BENCHMARKS]
    return tasks


try:
    from lm_eval.api.model import LM as LMBase
except ImportError:
    LMBase = object


class BabyLucioleSSATritonV4LM(LMBase):
    """lm-eval wrapper around Baby Luciole SSA Triton-v4 checkpoint."""

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_name: str = "/work/m24047/m24047brmn/tokenizers/luciole_50k",
        device: str = "cuda",
        batch_size: int = 1,
        max_length: int = 2048,
        compiled_bda: bool = False,
        force_contiguous_qkv: bool = True,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length

        logger.info("Initializing BabyLucioleSSATritonV4LM wrapper...")
        self.model, self.tokenizer = load_model(
            checkpoint_path=checkpoint_path,
            tokenizer_name=tokenizer_name,
            device=device,
            compiled_bda=compiled_bda,
            force_contiguous_qkv=force_contiguous_qkv,
        )
        self._setup_tokenizer()
        logger.info("BabyLucioleSSATritonV4LM wrapper initialized successfully")

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
    checkpoint_path: str,
    tasks: List[str],
    tokenizer_name: str = "/work/m24047/m24047brmn/tokenizers/luciole_50k",
    device: str = "cuda",
    batch_size: int = 1,
    max_length: int = 2048,
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
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

    model = BabyLucioleSSATritonV4LM(
        checkpoint_path=checkpoint_path,
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

    logger.info("Running evaluation on tasks: %s", task_names)
    try:
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=task_names,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
            log_samples=False,
        )
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        raise

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    for task_name, task_results in results.get("results", {}).items():
        print(f"\n{task_name}:")
        print("-" * 50)
        for metric, value in task_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    print("\n" + "=" * 70)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, default=str)
        logger.info("Results saved to: %s", output_path)

    return results


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run LM Evaluation Harness benchmarks on Baby Luciole SSA Triton-v4 model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to NeMo checkpoint directory",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_easy",
        help="Comma-separated list of tasks or a preset group name",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="/work/m24047/m24047brmn/tokenizers/luciole_50k",
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
        "--output", type=str, default=None, help="Path to save results JSON"
    )
    parser.add_argument(
        "--compiled-bda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable compiled bias-dropout-add path used in SSA Triton-v4 layer spec",
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

    logger.info("Tasks to evaluate: %s", tasks)
    try:
        run_evaluation(
            checkpoint_path=args.checkpoint,
            tasks=tasks,
            tokenizer_name=args.tokenizer,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            output_path=args.output,
            compiled_bda=args.compiled_bda,
            force_contiguous_qkv=args.force_contiguous_qkv,
        )
    finally:
        cleanup_parallel_state()


if __name__ == "__main__":
    main()
