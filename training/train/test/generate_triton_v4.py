"""
Generate text from Baby Luciole SSA Triton-v4 checkpoints.

This script loads a Baby Luciole model configured with the Triton-v4 SSA
attention layer spec used by `train_ssa_triton.py`, then runs generation for
multiple prompts (default: 5 prompts -> 5 generations).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "/work/m24047/m24047brmn/tokenizers/luciole_50k"
DEFAULT_CHECKPOINT = (
    "/tmpdir/m24047brmn/nemo_1b/output/baby_luciole-ssa-triton-v4/"
    "checkpoints/baby_luciole-ssa-triton-v4-step=0019999-last"
)
DEFAULT_PROMPTS = [
    "Write a concise explanation of why sunrise colors change through the morning.",
    "Give three practical tips to reduce GPU memory usage during training.",
    "Continue this short story in a suspenseful tone: The elevator stopped between floors.",
    "Explain the difference between precision, recall, and F1 with a simple example.",
    "Draft a short email asking to reschedule a meeting to next Tuesday afternoon.",
]


def init_single_gpu_parallel_state(seed: int = 1234, device: str = "cuda"):
    """Initialize Megatron parallel state for single GPU inference."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    backend = "nccl" if device.startswith("cuda") and torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend=backend, world_size=1, rank=0)

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
        )

    if backend == "nccl":
        model_parallel_cuda_manual_seed(seed)
    else:
        torch.manual_seed(seed)

    logger.info("Initialized parallel state with backend %s", backend)


def cleanup_parallel_state():
    """Clean up Megatron parallel state and destroy process group."""
    import torch.distributed as dist
    from megatron.core import parallel_state

    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        logger.info("Destroyed Megatron model parallel state")

    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Destroyed process group")


def get_baby_luciole_config():
    """Return Baby Luciole architecture config."""
    from nemo.collections.llm.gpt.model.nemotron import Nemotron3Config4B

    config = Nemotron3Config4B()
    config.num_layers = 12
    config.num_attention_heads = 24
    config.num_query_groups = 8
    config.hidden_size = 768
    config.ffn_hidden_size = 3072
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.share_embeddings_and_output_weights = True
    config.vocab_size = 50256
    return config


def _resolve_strict_handling():
    strict_value = None
    for module_path in [
        "megatron.core.dist_checkpointing.validation",
        "megatron.core.dist_checkpointing.mapping",
    ]:
        try:
            import importlib

            mod = importlib.import_module(module_path)
            StrictHandling = getattr(mod, "StrictHandling")
            strict_value = StrictHandling.LOG_UNEXPECTED
            break
        except (ImportError, AttributeError):
            continue

    if strict_value is None:
        strict_value = "log_unexpected"

    return strict_value


def _get_target_module(model):
    if hasattr(model, "module") and model.module is not None:
        return model.module
    return model


def _resolve_checkpoint_dir(checkpoint_path: str) -> Path:
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise ValueError(
            f"Expected checkpoint directory (typically ending with .ckpt), got {checkpoint_dir}"
        )

    weights_dir = checkpoint_dir / "weights"
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights not found at {weights_dir}")

    return checkpoint_dir


def load_model(
    checkpoint_path: str,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    device: str = "cuda",
    seed: int = 1234,
    compiled_bda: bool = False,
    force_contiguous_qkv: bool = True,
):
    """Load Baby Luciole Triton-v4 model from NeMo distributed checkpoint."""
    from nemo.collections.llm.gpt.model.nemotron import NemotronModel
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
    from SSA.ssa_triton_v4_layer_specs import (
        get_ssa_triton_v4_gpt_layer_spec as get_ssa_triton_gpt_layer_spec,
    )

    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_path)

    tokenizer_path = checkpoint_dir / "context" / "tokenizer_name.txt"
    if tokenizer_path.exists():
        tokenizer_name = tokenizer_path.read_text(encoding="utf-8").strip()
        logger.info("Loading tokenizer from checkpoint: %s", tokenizer_name)
    else:
        logger.info("No tokenizer in checkpoint, using: %s", tokenizer_name)

    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, use_fast=True)

    config = get_baby_luciole_config()
    config.transformer_layer_spec = get_ssa_triton_gpt_layer_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=False,
        ssa_n=1.5,
        ssa_b=0.8,
        learnable_ssa=True,
        learnable_b=False,
        use_compiled_bda=compiled_bda,
        force_contiguous_qkv=force_contiguous_qkv,
    )
    config.masked_softmax_fusion = False
    logger.info(
        "Triton-v4 SSA config enabled (compiled_bda=%s, force_contiguous_qkv=%s)",
        compiled_bda,
        force_contiguous_qkv,
    )

    init_single_gpu_parallel_state(seed=seed, device=device)

    logger.info("Creating NemotronModel...")
    model = NemotronModel(config=config, tokenizer=tokenizer)

    if hasattr(model, "configure_model"):
        logger.info("Configuring model...")
        model.configure_model()

    logger.info("Loading model weights from %s...", checkpoint_dir)
    from megatron.core.dist_checkpointing import load

    target_module = _get_target_module(model)

    if hasattr(target_module, "sharded_state_dict"):
        sharded_state_dict = target_module.sharded_state_dict()
    else:
        sharded_state_dict = target_module.state_dict()

    try:
        from megatron.core.dist_checkpointing.utils import add_prefix_for_sharding

        add_prefix_for_sharding(sharded_state_dict, "module.")
        logger.info("Added 'module.' prefix to sharded state dict")
    except ImportError:
        sharded_state_dict = {f"module.{k}": v for k, v in sharded_state_dict.items()}

    loaded_state = load(
        sharded_state_dict=sharded_state_dict,
        checkpoint_dir=str(checkpoint_dir / "weights"),
        strict=_resolve_strict_handling(),
    )

    loaded_state_stripped = {}
    for key, value in loaded_state.items():
        if key.startswith("module."):
            loaded_state_stripped[key[7:]] = value
        else:
            loaded_state_stripped[key] = value

    target_module.load_state_dict(loaded_state_stripped, strict=False)
    logger.info("Model weights loaded successfully")

    model = model.to(device)
    model.eval()

    logger.info("Model ready for generation")
    return model, tokenizer


def _build_causal_mask(seq_len: int, device: str):
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    ).unsqueeze(0).unsqueeze(0)


def _decode_tokens(tokenizer, token_ids: list[int]) -> str:
    if hasattr(tokenizer, "ids_to_text"):
        return tokenizer.ids_to_text(token_ids)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _sample_next_token(
    next_token_logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
):
    if greedy:
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    logits = next_token_logits
    if temperature > 0:
        logits = logits / temperature

    if top_k > 0:
        top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        indices_to_remove = logits < top_k_values[..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str,
    greedy: bool,
    compiled_bda: bool,
):
    target_module = _get_target_module(model)

    if hasattr(tokenizer, "text_to_ids"):
        prompt_token_ids = tokenizer.text_to_ids(prompt)
        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    else:
        tokenized = tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)

    generated_ids = input_ids.clone()
    prompt_len = int(input_ids.shape[1])

    eos_id = getattr(tokenizer, "eos_id", None)
    if eos_id is None and hasattr(tokenizer, "eos_token_id"):
        eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        seq_len = int(generated_ids.shape[1])
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(1, -1)
        )
        attention_mask = _build_causal_mask(seq_len=seq_len, device=device)

        if compiled_bda and hasattr(torch, "compiler") and hasattr(
            torch.compiler, "cudagraph_mark_step_begin"
        ):
            torch.compiler.cudagraph_mark_step_begin()

        outputs = target_module(
            input_ids=generated_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        if hasattr(outputs, "logits"):
            next_token_logits = outputs.logits[:, -1, :].clone()
        elif isinstance(outputs, torch.Tensor):
            next_token_logits = outputs[:, -1, :].clone()
        else:
            next_token_logits = outputs[0][:, -1, :].clone()

        next_token = _sample_next_token(
            next_token_logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if eos_id is not None and int(next_token.item()) == int(eos_id):
            break

    all_token_ids = generated_ids[0].tolist()
    completion_ids = all_token_ids[prompt_len:]

    return {
        "prompt": prompt,
        "completion": _decode_tokens(tokenizer, completion_ids),
        "full_text": _decode_tokens(tokenizer, all_token_ids),
        "prompt_tokens": prompt_len,
        "generated_tokens": len(completion_ids),
    }


def _build_prompts(prompt_args: list[str], prompts_file: str | None, context: str, num_prompts: int):
    prompts = []

    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    prompts.append(line)

    if prompt_args:
        prompts.extend(prompt_args)

    if not prompts:
        prompts = list(DEFAULT_PROMPTS)

    if len(prompts) < num_prompts:
        raise ValueError(
            f"Need at least {num_prompts} prompts, but got {len(prompts)}."
        )

    if len(prompts) > num_prompts:
        logger.warning("Received %d prompts; using first %d.", len(prompts), num_prompts)
        prompts = prompts[:num_prompts]

    context = context.strip()
    if not context:
        return prompts

    return [f"{context}\n{p}" for p in prompts]


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate text from Baby Luciole SSA Triton-v4 checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to NeMo checkpoint directory (*.ckpt directory containing weights/)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt text. Repeat this flag to provide multiple prompts.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional text file with one prompt per line.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Optional shared initial context prepended to every prompt.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts/generations to run.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (ignored when --greedy is set).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run generation on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed.",
    )
    parser.add_argument(
        "--compiled-bda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable torch.compile'd BDA path in Triton-v4 layer spec.",
    )
    parser.add_argument(
        "--force-contiguous-qkv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Materialize Q/K/V as contiguous tensors before Triton attention call.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for generations.",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("CUDA requested but not available. Run on a GPU node.")
        sys.exit(1)

    if args.num_prompts <= 0:
        logger.error("--num-prompts must be > 0")
        sys.exit(1)

    prompts = _build_prompts(
        prompt_args=args.prompt or [],
        prompts_file=args.prompts_file,
        context=args.context,
        num_prompts=args.num_prompts,
    )

    temperature = 0.0 if args.greedy else args.temperature
    model = None
    tokenizer = None
    results = []

    try:
        model, tokenizer = load_model(
            checkpoint_path=args.checkpoint,
            tokenizer_name=args.tokenizer,
            device=args.device,
            seed=args.seed,
            compiled_bda=args.compiled_bda,
            force_contiguous_qkv=args.force_contiguous_qkv,
        )

        for idx, prompt in enumerate(prompts, start=1):
            logger.info("Generating for prompt %d/%d", idx, len(prompts))
            result = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
                greedy=args.greedy,
                compiled_bda=args.compiled_bda,
            )
            result["index"] = idx
            results.append(result)

        print("\n" + "=" * 80)
        print("BABY LUCIOLE SSA TRITON-V4 GENERATION RESULTS")
        print("=" * 80)
        for item in results:
            print(f"\n[Prompt {item['index']}]")
            print("-" * 80)
            print(item["prompt"])
            print(f"\n[Generation {item['index']}]")
            print("-" * 80)
            print(item["completion"])
        print("\n" + "=" * 80)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "checkpoint": args.checkpoint,
                "tokenizer": args.tokenizer,
                "num_prompts": args.num_prompts,
                "max_new_tokens": args.max_new_tokens,
                "temperature": temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "greedy": args.greedy,
                "compiled_bda": args.compiled_bda,
                "force_contiguous_qkv": args.force_contiguous_qkv,
                "results": results,
            }
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            logger.info("Saved generations JSON to %s", output_path)

    finally:
        try:
            cleanup_parallel_state()
        except Exception as exc:
            logger.warning("Parallel state cleanup failed: %s", exc)


if __name__ == "__main__":
    main()
