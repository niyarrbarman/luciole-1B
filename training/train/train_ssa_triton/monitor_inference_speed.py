import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monitor_inference_speed")

MODEL_TYPE_SSA_TRITON = "ssa_triton"
MODEL_TYPE_SSA = "ssa"
MODEL_TYPE_SOFTMAX = "softmax"
MODEL_TYPES = (MODEL_TYPE_SSA_TRITON, MODEL_TYPE_SSA, MODEL_TYPE_SOFTMAX)
_CAUSAL_MASK_CACHE: Dict[Tuple[str, int], torch.Tensor] = {}


def _parse_int_list(value: str) -> List[int]:
    values = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _get_model_parameter_dtype(model: Any) -> str:
    module = model.module if hasattr(model, "module") and model.module is not None else model
    try:
        first_param = next(module.parameters())
    except StopIteration:
        return "unknown"
    return str(first_param.dtype)


def _count_parameters(model: Any) -> int:
    module = model.module if hasattr(model, "module") and model.module is not None else model
    return sum(p.numel() for p in module.parameters())


def _resolve_vocab_size(tokenizer: Any) -> int:
    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size is not None:
        return int(tokenizer.vocab_size)
    if hasattr(tokenizer, "vocab"):
        return len(tokenizer.vocab)
    return 50000


def _get_causal_mask(device: torch.device, seq_len: int) -> torch.Tensor:
    key = (str(device), seq_len)
    cached = _CAUSAL_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    # True means masked for TE/Nemo causal attention.
    mask = torch.triu(
        torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device),
        diagonal=1,
    )
    _CAUSAL_MASK_CACHE[key] = mask
    return mask


def _safe_forward(
    module: Any,
    model_type: str,
    input_ids: torch.Tensor,
    attention_mask_4d: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    position_ids: torch.Tensor,
):
    """Try forward signatures/mask formats for model compatibility."""

    if model_type in (MODEL_TYPE_SSA_TRITON, MODEL_TYPE_SSA):
        attempts = [
            # NeMo/TE-style, positional
            (input_ids, position_ids, attention_mask_4d),
            # NeMo/TE-style, keyword
            {"input_ids": input_ids, "position_ids": position_ids, "attention_mask": attention_mask_4d},
            # Triton wrapper style
            {"input_ids": input_ids, "attention_mask": attention_mask_2d, "position_ids": position_ids},
            {"input_ids": input_ids, "position_ids": position_ids},
            {"input_ids": input_ids, "attention_mask": attention_mask_2d},
            {"input_ids": input_ids},
        ]
    else:
        # HuggingFace CausalLM path: keyword calls only to avoid positional mismatch.
        attempts = [
            {"input_ids": input_ids, "attention_mask": attention_mask_2d},
            {"input_ids": input_ids, "attention_mask": attention_mask_2d, "position_ids": position_ids},
            {"input_ids": input_ids, "position_ids": position_ids},
            {"input_ids": input_ids},
        ]

    last_exc = None
    for call_args in attempts:
        try:
            if isinstance(call_args, tuple):
                return module(*call_args)
            return module(**call_args)
        except TypeError as exc:
            last_exc = exc
            continue
        except RuntimeError as exc:
            # Retry with another signature for known mask-shape incompatibilities.
            err = str(exc).lower()
            if "size of tensor a" in err and "must match the size of tensor b" in err:
                last_exc = exc
                last_exc = exc
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("All forward attempts failed without an explicit exception")


def _forward_pass(model: Any, input_ids: torch.Tensor, model_type: str) -> torch.Tensor:
    module = model.module if hasattr(model, "module") and model.module is not None else model
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
    attention_mask_2d = torch.ones_like(input_ids, device=input_ids.device)
    attention_mask_4d = _get_causal_mask(input_ids.device, seq_len)

    with torch.no_grad():
        outputs = _safe_forward(
            module,
            model_type,
            input_ids,
            attention_mask_4d,
            attention_mask_2d,
            position_ids,
        )

    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, torch.Tensor):
        return outputs
    return outputs[0]


def _is_oom_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _benchmark(
    model: Any,
    model_type: str,
    device: str,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    warmup_iters: int,
    measure_iters: int,
) -> Dict[str, Any]:
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    for _ in range(warmup_iters):
        _forward_pass(model, input_ids, model_type)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    timings = []
    for _ in range(measure_iters):
        start = time.perf_counter()
        _forward_pass(model, input_ids, model_type)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    avg_latency = sum(timings) / len(timings)
    tokens_per_step = batch_size * seq_len
    tokens_per_second = tokens_per_step / avg_latency if avg_latency > 0 else 0.0

    peak_gb = None
    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_gb = peak_bytes / (1024**3)

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "avg_latency_s": avg_latency,
        "tokens_per_step": tokens_per_step,
        "tokens_per_second": tokens_per_second,
        "peak_memory_gb": peak_gb,
    }


def _try_batch(
    model: Any,
    model_type: str,
    device: str,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    warmup_iters: int,
    measure_iters: int,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    try:
        result = _benchmark(
            model=model,
            model_type=model_type,
            device=device,
            vocab_size=vocab_size,
            batch_size=batch_size,
            seq_len=seq_len,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
        )
        return True, result, None
    except RuntimeError as exc:
        if _is_oom_error(exc):
            _cleanup_cuda()
            return False, None, "oom"
        _cleanup_cuda()
        return False, None, f"runtime_error: {exc.__class__.__name__}"


def _find_max_safe_batch(
    model: Any,
    model_type: str,
    device: str,
    vocab_size: int,
    seq_len: int,
    start_batch: int,
    max_batch_cap: int,
    warmup_iters: int,
    measure_iters: int,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    low = 0
    low_result = None
    high = None

    batch = max(1, start_batch)
    while batch <= max_batch_cap:
        ok, result, err = _try_batch(
            model=model,
            model_type=model_type,
            device=device,
            vocab_size=vocab_size,
            batch_size=batch,
            seq_len=seq_len,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
        )
        if ok:
            low = batch
            low_result = result
            batch *= 2
            continue

        if err == "oom":
            high = batch
            break

    if high is None:
        return low, low_result

    left = low + 1
    right = high - 1
    while left <= right:
        mid = (left + right) // 2
        ok, result, err = _try_batch(
            model=model,
            model_type=model_type,
            device=device,
            vocab_size=vocab_size,
            batch_size=mid,
            seq_len=seq_len,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
        )
        if ok:
            low = mid
            low_result = result
            left = mid + 1
        elif err == "oom":
            right = mid - 1
        else:
            break

    return low, low_result


def _load_model(args):
    if args.model_type == MODEL_TYPE_SSA_TRITON:
        from eval_perplexity_triton import cleanup_parallel_state, load_model

        model, tokenizer = load_model(
            checkpoint_path=args.checkpoint,
            tokenizer_name=args.tokenizer,
            device=args.device,
            compiled_bda=args.compiled_bda,
            force_contiguous_qkv=args.force_contiguous_qkv,
        )
        cleanup_fn = cleanup_parallel_state
    else:
        from eval_perplexity import cleanup_parallel_state, load_model

        model, tokenizer = load_model(
            checkpoint_path=args.checkpoint,
            tokenizer_name=args.tokenizer,
            device=args.device,
            use_ssa=(args.model_type == MODEL_TYPE_SSA),
        )
        cleanup_fn = cleanup_parallel_state

    return model, tokenizer, cleanup_fn


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple forward-pass inference speed monitor for Baby Luciole models"
    )
    parser.add_argument("--model-type", type=str, choices=MODEL_TYPES, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--seq-lens", type=str, default="2048")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)

    parser.add_argument("--auto-find-max-batch", action="store_true")
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--max-batch-cap", type=int, default=1024)

    parser.add_argument("--compiled-bda", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-contiguous-qkv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-bf16", action="store_true")

    parser.add_argument("--output", type=str, default=None)
    return parser


def main() -> None:
    args = get_parser().parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")

    torch.set_float32_matmul_precision("high")

    seq_lens = _parse_int_list(args.seq_lens)
    batch_sizes = _parse_int_list(args.batch_sizes)

    logger.info("Loading model type=%s checkpoint=%s", args.model_type, args.checkpoint)
    model, tokenizer, cleanup_fn = _load_model(args)

    try:
        if args.force_bf16:
            model = model.to(dtype=torch.bfloat16)
            logger.info("Forced model weights to bfloat16")

        model_dtype = _get_model_parameter_dtype(model)
        num_params = _count_parameters(model)
        vocab_size = _resolve_vocab_size(tokenizer)

        logger.info("Model parameter dtype: %s", model_dtype)
        logger.info("Model parameters: %d", num_params)
        logger.info("Tokenizer vocab size: %d", vocab_size)

        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_idx)
            total_mem_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
            logger.info("CUDA device: %s (%.1f GB)", device_name, total_mem_gb)

        results = {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "device": args.device,
            "model_dtype": model_dtype,
            "num_parameters": num_params,
            "vocab_size": vocab_size,
            "seq_lens": seq_lens,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
            "auto_find_max_batch": args.auto_find_max_batch,
            "measurements": [],
            "max_safe_batch": {},
        }

        for seq_len in seq_lens:
            logger.info("----- seq_len=%d -----", seq_len)

            if args.auto_find_max_batch:
                max_batch, max_batch_result = _find_max_safe_batch(
                    model=model,
                    model_type=args.model_type,
                    device=args.device,
                    vocab_size=vocab_size,
                    seq_len=seq_len,
                    start_batch=args.start_batch,
                    max_batch_cap=args.max_batch_cap,
                    warmup_iters=max(1, args.warmup_iters // 2),
                    measure_iters=max(2, args.measure_iters // 4),
                )
                results["max_safe_batch"][str(seq_len)] = max_batch
                logger.info("Max safe batch at seq_len=%d: %d", seq_len, max_batch)
                if max_batch_result is not None:
                    max_batch_result["note"] = "measurement_at_max_safe_batch_during_search"
                    results["measurements"].append(max_batch_result)

            for batch_size in batch_sizes:
                ok, result, err = _try_batch(
                    model=model,
                    model_type=args.model_type,
                    device=args.device,
                    vocab_size=vocab_size,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    warmup_iters=args.warmup_iters,
                    measure_iters=args.measure_iters,
                )
                if not ok:
                    logger.warning(
                        "batch=%d seq_len=%d -> %s",
                        batch_size,
                        seq_len,
                        err,
                    )
                    results["measurements"].append(
                        {
                            "batch_size": batch_size,
                            "seq_len": seq_len,
                            "error": err,
                        }
                    )
                    continue

                logger.info(
                    "batch=%d seq_len=%d avg=%.4fs toks/s=%.1f peak_mem_gb=%s",
                    batch_size,
                    seq_len,
                    result["avg_latency_s"],
                    result["tokens_per_second"],
                    f"{result['peak_memory_gb']:.2f}" if result["peak_memory_gb"] is not None else "n/a",
                )
                results["measurements"].append(result)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)
            logger.info("Saved speed report to %s", output_path)

        print("\n=== MAX SAFE BATCH SUMMARY ===")
        if results["max_safe_batch"]:
            for seq_len, max_batch in results["max_safe_batch"].items():
                print(f"seq_len={seq_len}: max_safe_batch={max_batch}")
        else:
            print("auto-find-max-batch disabled")

    finally:
        try:
            cleanup_fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cleanup failed: %s", exc)


if __name__ == "__main__":
    main()
