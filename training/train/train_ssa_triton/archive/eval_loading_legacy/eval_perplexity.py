"""
Evaluate perplexity of Baby Luciole SSA model on FineWeb dataset.

This script loads a Baby Luciole SSA model from a NeMo checkpoint
and calculates perplexity on FineWeb data.

Usage:
    python eval_perplexity.py --checkpoint /path/to/checkpoint
    python eval_perplexity.py --checkpoint /path/to/checkpoint --num_samples 1000
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def init_single_gpu_parallel_state(seed: int = 1234, device: str = "cuda"):
    """Initialize Megatron parallel state for single GPU inference."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    backend = "nccl" if device.startswith("cuda") and torch.cuda.is_available() else "gloo"

    # Initialize process group if not already initialized
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend=backend, world_size=1, rank=0)

    # Initialize Megatron parallel state for single device (no parallelism)
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
        )

    # Initialize RNG tracker when on GPU
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
    """
    Get the Baby Luciole model configuration.
    
    This matches the config from recipes/baby_luciole.py:
    - 12 layers
    - 24 attention heads
    - 8 query groups (GQA)
    - 768 hidden size
    - 3072 FFN hidden size
    """
    from nemo.collections.llm.gpt.model.nemotron import Nemotron3Config4B

    config = Nemotron3Config4B()
    config.num_layers = 12
    config.num_attention_heads = 24
    config.num_query_groups = 8
    config.hidden_size = 768
    config.ffn_hidden_size = 3072
    config.kv_channels = config.hidden_size // config.num_attention_heads  # 768 / 24 = 32
    config.share_embeddings_and_output_weights = True
    config.vocab_size = 50256
    return config


def load_model(
    checkpoint_path: str,
    tokenizer_name: str = "/work/m24047/m24047brmn/tokenizers/luciole_50k",
    device: str = "cuda",
    use_ssa: bool = True,
):
    """
    Load Baby Luciole SSA model from a NeMo checkpoint.

    Args:
        checkpoint_path: Path to the NeMo checkpoint directory
        tokenizer_name: Name/path of the tokenizer to use
        device: Device to load the model on (default: cuda)
        use_ssa: Whether to use SSA attention (default: True)

    Returns:
        model: The loaded model
        tokenizer: The tokenizer
    """
    from nemo.collections.llm.gpt.model.nemotron import NemotronModel
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer

    # Load tokenizer
    if checkpoint_path and os.path.exists(checkpoint_path):
        tokenizer_path = os.path.join(checkpoint_path, "context", "tokenizer_name.txt")
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "r") as f:
                tokenizer_name = f.read().strip()
            logger.info(f"Loading tokenizer from checkpoint: {tokenizer_name}")
        else:
            logger.info(f"No tokenizer in checkpoint, using default: {tokenizer_name}")
    else:
        logger.info(f"Using tokenizer: {tokenizer_name}")

    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, use_fast=True)

    # Get model config
    config = get_baby_luciole_config()
    logger.info(f"Model config: {config.num_layers} layers, hidden_size={config.hidden_size}")

    # Inject SSA layer spec if needed
    if use_ssa:
        from SSA.ssa_layer_specs import get_ssa_gpt_layer_spec
        
        ssa_layer_spec = get_ssa_gpt_layer_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=False,
            ssa_n=1.5,  # Initial value, will be overwritten by checkpoint
            ssa_b=0.8,
        )
        config.transformer_layer_spec = ssa_layer_spec
        config.masked_softmax_fusion = False
        logger.info("SSA: Using SSADotProductAttention")

    # Initialize single-GPU parallel state
    init_single_gpu_parallel_state(device=device)

    logger.info("Creating NemotronModel...")
    model = NemotronModel(config=config, tokenizer=tokenizer)

    # Configure the model
    if hasattr(model, "configure_model"):
        logger.info("Configuring model...")
        model.configure_model()

    # Load weights from checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model weights from {checkpoint_path}...")
        weights_path = os.path.join(checkpoint_path, "weights")
        if os.path.exists(weights_path):
            try:
                from megatron.core.dist_checkpointing import load
                
                # Get sharded state dict
                if hasattr(model, 'module') and model.module is not None:
                    target_module = model.module
                else:
                    target_module = model
                
                if hasattr(target_module, 'sharded_state_dict'):
                    sharded_state_dict = target_module.sharded_state_dict()
                else:
                    sharded_state_dict = target_module.state_dict()
                
                # Add 'module.' prefix for NeMo checkpoint format
                try:
                    from megatron.core.dist_checkpointing.utils import add_prefix_for_sharding
                    add_prefix_for_sharding(sharded_state_dict, 'module.')
                    logger.info("Added 'module.' prefix to sharded state dict")
                except ImportError:
                    sharded_state_dict = {f"module.{k}": v for k, v in sharded_state_dict.items()}
                
                # Try to get StrictHandling
                strict_value = None
                for module_path in [
                    'megatron.core.dist_checkpointing.validation',
                    'megatron.core.dist_checkpointing.mapping',
                ]:
                    try:
                        import importlib
                        mod = importlib.import_module(module_path)
                        StrictHandling = getattr(mod, 'StrictHandling')
                        strict_value = StrictHandling.LOG_UNEXPECTED
                        break
                    except (ImportError, AttributeError):
                        continue
                
                if strict_value is None:
                    strict_value = "log_unexpected"
                
                # Load checkpoint
                loaded_state = load(
                    sharded_state_dict=sharded_state_dict,
                    checkpoint_dir=weights_path,
                    strict=strict_value,
                )
                
                # Strip 'module.' prefix
                loaded_state_stripped = {}
                for k, v in loaded_state.items():
                    if k.startswith("module."):
                        loaded_state_stripped[k[7:]] = v
                    else:
                        loaded_state_stripped[k] = v
                
                target_module.load_state_dict(loaded_state_stripped, strict=False)
                logger.info("Model weights loaded successfully")
                
            except Exception as e:
                logger.error(f"Checkpoint loading failed: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError("Failed to load checkpoint")
        else:
            raise FileNotFoundError(f"Weights not found at {weights_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = model.to(device)
    model.eval()

    logger.info("Model ready for evaluation")
    return model, tokenizer


def load_fineweb_data(
    data_path: str,
    tokenizer,
    seq_length: int = 1024,
    num_samples: int = 1000,
    seed: int = 42,
):
    """
    Load FineWeb data for perplexity evaluation.
    
    Args:
        data_path: Path to the preprocessed FineWeb data (.bin/.idx files prefix)
        tokenizer: Tokenizer for encoding text
        seq_length: Sequence length for each sample
        num_samples: Number of samples to evaluate
        seed: Random seed for reproducibility
    
    Returns:
        List of token tensors, each of shape [seq_length]
    """
    import numpy as np
    
    # Import MMapIndexedDataset with fallback
    try:
        from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
    except ImportError:
        from megatron.core.datasets.indexed_dataset import IndexedDataset as MMapIndexedDataset
    
    logger.info(f"Loading FineWeb data from {data_path}")
    
    # Check if data_path is a prefix for .bin/.idx files
    bin_path = f"{data_path}.bin"
    idx_path = f"{data_path}.idx"
    
    if os.path.exists(bin_path) and os.path.exists(idx_path):
        # Load indexed dataset using MMapIndexedDataset directly
        dataset = MMapIndexedDataset(data_path)
        total_docs = len(dataset)
        logger.info(f"Loaded indexed dataset with {total_docs} documents")
        
        # Sample more documents than needed (5x) since many are shorter than seq_length
        np.random.seed(seed)
        doc_indices = np.random.choice(total_docs, size=min(num_samples * 5, total_docs), replace=False)
        
        samples = []
        for doc_idx in doc_indices:
            if len(samples) >= num_samples:
                break
            
            doc_tokens = dataset[doc_idx]
            if isinstance(doc_tokens, torch.Tensor):
                doc_tokens = doc_tokens.numpy()
            else:
                doc_tokens = np.array(doc_tokens)
            
            # Only use documents with enough tokens
            if len(doc_tokens) >= seq_length:
                # Take a random starting position
                max_start = len(doc_tokens) - seq_length
                start = np.random.randint(0, max_start + 1)
                sample = torch.tensor(doc_tokens[start:start + seq_length], dtype=torch.long)
                samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples of length {seq_length}")
        return samples
    else:
        raise FileNotFoundError(f"Data files not found: {bin_path}, {idx_path}")


def load_fineweb_streaming(
    num_samples: int = 1000,
    seq_length: int = 1024,
    tokenizer=None,
    seed: int = 42,
):
    """
    Load FineWeb data via HuggingFace datasets for perplexity evaluation.
    
    This is an alternative method using streaming from HuggingFace.
    
    Args:
        num_samples: Number of samples to evaluate
        seq_length: Sequence length for each sample
        tokenizer: Tokenizer for encoding text
        seed: Random seed for reproducibility
    
    Returns:
        List of token tensors, each of shape [seq_length]
    """
    from datasets import load_dataset
    
    logger.info("Loading FineWeb-Edu from HuggingFace (streaming)")
    
    # Load a subset of FineWeb-Edu
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    
    samples = []
    dataset = dataset.shuffle(seed=seed)
    
    for item in dataset:
        if len(samples) >= num_samples:
            break
        
        text = item["text"]
        
        # Tokenize
        if hasattr(tokenizer, "text_to_ids"):
            tokens = tokenizer.text_to_ids(text)
        else:
            tokens = tokenizer.encode(text)
        
        # Only use texts with enough tokens
        if len(tokens) >= seq_length:
            sample = torch.tensor(tokens[:seq_length], dtype=torch.long)
            samples.append(sample)
    
    logger.info(f"Loaded {len(samples)} samples of length {seq_length}")
    return samples


@torch.no_grad()
def compute_perplexity(
    model,
    samples: list,
    batch_size: int = 8,
    device: str = "cuda",
):
    """
    Compute perplexity on a list of token samples.
    
    Args:
        model: The loaded model
        samples: List of token tensors, each of shape [seq_length]
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
    
    Returns:
        dict with perplexity, average loss, and per-sample losses
    """
    model.eval()
    
    # Get the actual model module
    if hasattr(model, 'module') and model.module is not None:
        target_module = model.module
    else:
        target_module = model
    
    all_losses = []
    total_tokens = 0
    total_loss = 0.0
    
    num_batches = (len(samples) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(samples))
        
        batch_samples = samples[start_idx:end_idx]
        batch = torch.stack(batch_samples).to(device)  # [batch, seq_length]
        
        seq_len = batch.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch.shape[0], -1)
        
        # Create proper boolean causal mask: True where masked (upper triangular)
        # Shape: [1, 1, seq_len, seq_len] - broadcast over batch and heads
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), 
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        # Forward pass
        outputs = target_module(
            input_ids=batch,
            position_ids=position_ids,
            attention_mask=causal_mask,
        )
        
        # Get logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs[0]
        
        # Compute loss: compare logits[:, :-1] with targets[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        # Cross-entropy loss per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())  # [batch, seq_length-1]
        
        # Per-sample loss (average over tokens)
        sample_losses = loss.mean(dim=1).cpu().tolist()
        all_losses.extend(sample_losses)
        
        # Accumulate for global average
        total_loss += loss.sum().item()
        total_tokens += loss.numel()
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            current_ppl = math.exp(total_loss / total_tokens)
            logger.info(f"Batch {batch_idx + 1}/{num_batches}: running perplexity = {current_ppl:.4f}")
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_samples": len(samples),
        "per_sample_losses": all_losses,
    }


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity of Baby Luciole SSA model on FineWeb"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to NeMo checkpoint directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="/work/m24047/m24047brmn/tokenizers/luciole_50k",
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to preprocessed FineWeb data (prefix for .bin/.idx files). If not provided, will use HuggingFace streaming.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=1024,
        help="Sequence length for evaluation",
    )
    parser.add_argument(
        "--batch_size",
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
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no_ssa",
        action="store_true",
        help="Do not use SSA attention (use standard attention)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    # Check device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("CUDA requested but not available. Run on a GPU node.")
        sys.exit(1)

    # Load model
    model, tokenizer = load_model(
        checkpoint_path=args.checkpoint,
        tokenizer_name=args.tokenizer,
        device=args.device,
        use_ssa=not args.no_ssa,
    )

    # Load data
    if args.data_path:
        samples = load_fineweb_data(
            data_path=args.data_path,
            tokenizer=tokenizer,
            seq_length=args.seq_length,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    else:
        samples = load_fineweb_streaming(
            num_samples=args.num_samples,
            seq_length=args.seq_length,
            tokenizer=tokenizer,
            seed=args.seed,
        )

    # Compute perplexity
    logger.info(f"Computing perplexity on {len(samples)} samples...")
    results = compute_perplexity(
        model=model,
        samples=samples,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("PERPLEXITY EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num samples: {results['num_samples']}")
    print(f"Total tokens: {results['total_tokens']}")
    print(f"Average loss: {results['avg_loss']:.4f}")
    print(f"Perplexity: {results['perplexity']:.4f}")
    print("=" * 60)

    # Save results if output path specified
    if args.output:
        import json
        output_data = {
            "checkpoint": args.checkpoint,
            "perplexity": results["perplexity"],
            "avg_loss": results["avg_loss"],
            "num_samples": results["num_samples"],
            "total_tokens": results["total_tokens"],
            "seq_length": args.seq_length,
            "batch_size": args.batch_size,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    # Clean up
    cleanup_parallel_state()


if __name__ == "__main__":
    main()
