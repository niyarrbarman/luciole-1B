"""
Evaluate perplexity of Baby Luciole SSA Triton checkpoints on FineWeb and Wiki.

This script loads a Baby Luciole model configured with the Triton SSA
attention layer spec used by `train_ssa_triton.py`, then evaluates perplexity
on FineWeb and Wikipedia datasets sequentially.
"""

import argparse
import json
import logging
import math
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
DEFAULT_FW_DATA_PATH = "/tmpdir/m24047brmn/nemo_1b/data_fwe_50k/fineweb_edu_text_document"
DEFAULT_WIKI_DATA_PATH = "/tmpdir/m24047brmn/nemo_1b/data_wiki/wikipedia_en_text_document"


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
    compiled_bda: bool = False,
    force_contiguous_qkv: bool = True,
):
    """Load Baby Luciole Triton model from NeMo distributed checkpoint."""
    from nemo.collections.llm.gpt.model.nemotron import NemotronModel
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
    from SSA.ssa_triton_layer_specs import (
        get_ssa_triton_gpt_layer_spec as get_ssa_triton_gpt_layer_spec,
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
        "Triton SSA config enabled (compiled_bda=%s, force_contiguous_qkv=%s)",
        compiled_bda,
        force_contiguous_qkv,
    )

    init_single_gpu_parallel_state(device=device)

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

    logger.info("Model ready for evaluation")
    return model, tokenizer


def _parse_blend_paths(flattened_data_paths: list[str]) -> list[tuple[float, str]]:
    if len(flattened_data_paths) % 2 != 0:
        raise ValueError(
            "Expected flattened data paths as [weight1, path1, weight2, path2, ...], "
            f"got {flattened_data_paths}"
        )

    pairs = []
    for idx in range(0, len(flattened_data_paths), 2):
        weight = float(flattened_data_paths[idx])
        path = flattened_data_paths[idx + 1]
        pairs.append((weight, path))

    return pairs


def resolve_data_path(dataset_name: str, data_path: str | None, datamix_path: str | None) -> str:
    """Resolve dataset prefix from explicit path or datamix file."""
    if data_path:
        if datamix_path:
            logger.warning(
                "%s: both data path and datamix were provided; using explicit data path %s",
                dataset_name,
                data_path,
            )
        return data_path

    if not datamix_path:
        raise ValueError(
            f"{dataset_name}: provide either --{dataset_name}-data-path or --{dataset_name}-datamix"
        )

    from utils import get_data_paths, read_datamix_file

    loaded_data = read_datamix_file(datamix_path)
    data_paths = get_data_paths(loaded_data)

    if isinstance(data_paths, dict):
        train_paths = data_paths.get("train", [])
    else:
        train_paths = data_paths

    pairs = _parse_blend_paths(train_paths)
    if not pairs:
        raise ValueError(f"{dataset_name}: datamix has no train datasets: {datamix_path}")

    if len(pairs) > 1:
        logger.warning(
            "%s datamix has %d train datasets; selecting highest weight entry",
            dataset_name,
            len(pairs),
        )

    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
    selected_path = pairs[0][1]
    logger.info("%s resolved from datamix: %s", dataset_name, selected_path)
    return selected_path


def load_indexed_data(
    data_path: str,
    seq_length: int = 1024,
    num_samples: int = 1000,
    seed: int = 42,
):
    """Load samples from Megatron indexed dataset prefix (.bin/.idx)."""
    import numpy as np

    try:
        from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
    except ImportError:
        from megatron.core.datasets.indexed_dataset import IndexedDataset as MMapIndexedDataset

    logger.info("Loading indexed dataset from %s", data_path)

    bin_path = f"{data_path}.bin"
    idx_path = f"{data_path}.idx"

    if not (os.path.exists(bin_path) and os.path.exists(idx_path)):
        raise FileNotFoundError(f"Data files not found: {bin_path}, {idx_path}")

    dataset = MMapIndexedDataset(data_path)
    total_docs = len(dataset)
    logger.info("Loaded indexed dataset with %d documents", total_docs)

    np.random.seed(seed)
    num_candidates = min(num_samples * 5, total_docs)
    doc_indices = np.random.choice(total_docs, size=num_candidates, replace=False)

    samples = []
    for doc_idx in doc_indices:
        if len(samples) >= num_samples:
            break

        doc_tokens = dataset[doc_idx]
        if isinstance(doc_tokens, torch.Tensor):
            doc_tokens = doc_tokens.cpu().numpy()
        else:
            doc_tokens = np.asarray(doc_tokens)

        if len(doc_tokens) < seq_length:
            continue

        max_start = len(doc_tokens) - seq_length
        start = np.random.randint(0, max_start + 1)
        sample = torch.tensor(doc_tokens[start : start + seq_length], dtype=torch.long)
        samples.append(sample)

    if not samples:
        raise RuntimeError(
            f"Could not collect any samples with seq_length={seq_length} from {data_path}"
        )

    if len(samples) < num_samples:
        logger.warning(
            "Collected %d samples (requested %d) from %s",
            len(samples),
            num_samples,
            data_path,
        )

    logger.info("Loaded %d samples of length %d", len(samples), seq_length)
    return samples


@torch.no_grad()
def compute_perplexity(
    model,
    samples: list,
    batch_size: int = 8,
    device: str = "cuda",
    compiled_bda: bool = False,
):
    """Compute perplexity over provided token samples."""
    model.eval()
    target_module = _get_target_module(model)

    all_losses = []
    total_tokens = 0
    total_loss = 0.0

    num_batches = (len(samples) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(samples))

        batch_samples = samples[start_idx:end_idx]
        batch = torch.stack(batch_samples).to(device)

        seq_len = batch.shape[1]
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(batch.shape[0], -1)
        )

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        # Mitigate torch.compile + CUDAGraph output reuse issues when compiled BDA is enabled.
        if compiled_bda and hasattr(torch, "compiler") and hasattr(
            torch.compiler, "cudagraph_mark_step_begin"
        ):
            torch.compiler.cudagraph_mark_step_begin()

        outputs = target_module(
            input_ids=batch,
            position_ids=position_ids,
            attention_mask=causal_mask,
        )

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs[0]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        sample_losses = loss.mean(dim=1).cpu().tolist()
        all_losses.extend(sample_losses)

        total_loss += loss.sum().item()
        total_tokens += loss.numel()

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            current_ppl = math.exp(total_loss / total_tokens)
            logger.info(
                "Batch %d/%d: running perplexity = %.4f",
                batch_idx + 1,
                num_batches,
                current_ppl,
            )

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_samples": len(samples),
        "per_sample_losses": all_losses,
    }


def evaluate_dataset(
    model,
    dataset_name: str,
    data_path: str,
    seq_length: int,
    num_samples: int,
    batch_size: int,
    device: str,
    seed: int,
    compiled_bda: bool,
):
    """Evaluate one dataset and return result dict."""
    samples = load_indexed_data(
        data_path=data_path,
        seq_length=seq_length,
        num_samples=num_samples,
        seed=seed,
    )

    logger.info("Computing perplexity on %s (%d samples)", dataset_name, len(samples))
    metrics = compute_perplexity(
        model=model,
        samples=samples,
        batch_size=batch_size,
        device=device,
        compiled_bda=compiled_bda,
    )
    metrics["dataset"] = dataset_name
    metrics["data_path"] = data_path
    return metrics


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate Baby Luciole SSA Triton checkpoint on FineWeb and Wiki"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to NeMo checkpoint directory (*.ckpt directory containing weights/)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="Tokenizer name or path",
    )

    parser.add_argument(
        "--fw-data-path",
        type=str,
        default=DEFAULT_FW_DATA_PATH,
        help="FineWeb indexed dataset prefix (without .bin/.idx)",
    )
    parser.add_argument(
        "--fw-datamix",
        type=str,
        default=None,
        help="Optional FineWeb datamix JSON/YAML; used when --fw-data-path is empty",
    )
    parser.add_argument(
        "--wiki-data-path",
        type=str,
        default=DEFAULT_WIKI_DATA_PATH,
        help="Wiki indexed dataset prefix (without .bin/.idx)",
    )
    parser.add_argument(
        "--wiki-datamix",
        type=str,
        default=None,
        help="Optional Wiki datamix JSON/YAML; used when --wiki-data-path is empty",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples per dataset",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=1024,
        help="Sequence length for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (wiki uses seed+1)",
    )
    parser.add_argument(
        "--compiled-bda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable torch.compile'd BDA path in Triton layer spec",
    )
    parser.add_argument(
        "--force-contiguous-qkv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Materialize Q/K/V as contiguous tensors before Triton attention call",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save aggregated results JSON",
    )

    return parser


def _print_dataset_result(checkpoint: str, result: dict):
    print("\n" + "=" * 70)
    print(f"PERPLEXITY RESULTS - {result['dataset'].upper()}")
    print("=" * 70)
    print(f"Checkpoint:  {checkpoint}")
    print(f"Data path:   {result['data_path']}")
    print(f"Num samples: {result['num_samples']}")
    print(f"Total tokens:{result['total_tokens']}")
    print(f"Average loss:{result['avg_loss']:.4f}")
    print(f"Perplexity:  {result['perplexity']:.4f}")
    print("=" * 70)


def _combined_summary(results: list[dict]) -> dict:
    total_tokens = sum(r["total_tokens"] for r in results)
    total_loss = sum(r["avg_loss"] * r["total_tokens"] for r in results)
    avg_loss = total_loss / total_tokens
    return {
        "num_datasets": len(results),
        "total_tokens": total_tokens,
        "avg_loss": avg_loss,
        "perplexity": math.exp(avg_loss),
    }


def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("CUDA requested but not available. Run on a GPU node.")
        sys.exit(1)

    fw_data_path = resolve_data_path("fw", args.fw_data_path, args.fw_datamix)
    wiki_data_path = resolve_data_path("wiki", args.wiki_data_path, args.wiki_datamix)

    results = []
    model = None

    try:
        model, _ = load_model(
            checkpoint_path=args.checkpoint,
            tokenizer_name=args.tokenizer,
            device=args.device,
            compiled_bda=args.compiled_bda,
            force_contiguous_qkv=args.force_contiguous_qkv,
        )

        eval_plan = [
            ("fineweb", fw_data_path, args.seed),
            ("wiki", wiki_data_path, args.seed + 1),
        ]

        for dataset_name, data_path, seed in eval_plan:
            logger.info("Evaluating %s dataset from %s", dataset_name, data_path)
            dataset_result = evaluate_dataset(
                model=model,
                dataset_name=dataset_name,
                data_path=data_path,
                seq_length=args.seq_length,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed,
                compiled_bda=args.compiled_bda,
            )
            results.append(dataset_result)
            _print_dataset_result(args.checkpoint, dataset_result)

        combined = _combined_summary(results)
        print("\n" + "=" * 70)
        print("COMBINED SUMMARY (FW + WIKI)")
        print("=" * 70)
        print(f"Datasets:    {combined['num_datasets']}")
        print(f"Total tokens:{combined['total_tokens']}")
        print(f"Average loss:{combined['avg_loss']:.4f}")
        print(f"Perplexity:  {combined['perplexity']:.4f}")
        print("=" * 70)

        if args.output:
            output_payload = {
                "checkpoint": args.checkpoint,
                "tokenizer": args.tokenizer,
                "seq_length": args.seq_length,
                "batch_size": args.batch_size,
                "num_samples_per_dataset": args.num_samples,
                "compiled_bda": args.compiled_bda,
                "force_contiguous_qkv": args.force_contiguous_qkv,
                "results": results,
                "combined": combined,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, indent=2)
            logger.info("Results saved to %s", args.output)

    finally:
        try:
            cleanup_parallel_state()
        except Exception as exc:
            logger.warning("Parallel state cleanup failed: %s", exc)


if __name__ == "__main__":
    main()
