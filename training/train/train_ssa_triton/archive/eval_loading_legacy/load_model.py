"""
Load Nemotron 1B model architecture for inference on single GPU.

This script loads the Nemotron 1B model from NeMo and provides
text generation capabilities.

Usage:
    python load_model.py --prompt "Your prompt here"
    python load_model.py --checkpoint /path/to/checkpoint --prompt "Your prompt here"
"""

import argparse
import logging
import os
import sys

import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def init_single_gpu_parallel_state(seed: int = 1234, device: str = "cuda"):
    """Initialize Megatron parallel state for single GPU or CPU inference."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    backend = "nccl" if device.startswith("cuda") and torch.cuda.is_available() else "gloo"

    # Initialize process group if not already initialized
    if not dist.is_initialized():
        # Use a fake initialization for single rank
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

    # Initialize RNG tracker when on GPU; otherwise just seed CPU for determinism
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


def get_nemotron_1b_config(num_layers: int = 24):
    """
    Get the Nemotron 1B model configuration.
    
    Args:
        num_layers: Number of transformer layers (default: 24 for full model)
    """
    from nemo.collections.llm.gpt.model.nemotron import Nemotron3Config4B

    config = Nemotron3Config4B()
    # Override to match Nemotron 1B architecture (from recipes/nemotron_1b.py)
    config.num_layers = num_layers  # 24 for full model
    config.num_attention_heads = 32
    config.num_query_groups = 8
    config.hidden_size = 2048
    config.ffn_hidden_size = 8192
    # kv_channels is computed automatically as hidden_size // num_attention_heads = 64
    # Setting to None lets Megatron compute it, or set explicitly for older builds
    config.kv_channels = config.hidden_size // config.num_attention_heads  # 64
    config.share_embeddings_and_output_weights = True
    return config


def load_model(
    checkpoint_path: str = None,
    tokenizer_name: str = "/work/m24047/m24047brmn/tokenizers/minitron-4b",
    device: str = "cuda",
    num_layers: int = 24,
):
    """
    Load Nemotron model from a NeMo checkpoint or initialize from scratch.

    Args:
        checkpoint_path: Path to the NeMo checkpoint directory (optional)
        tokenizer_name: Name/path of the tokenizer to use
        device: Device to load the model on (default: cuda)
        num_layers: Number of transformer layers (default: 24 for full model)

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

    # Get model config (matching train_model_1L.py)
    config = get_nemotron_1b_config(num_layers=num_layers)
    logger.info(f"Model config: {num_layers} layers, hidden_size={config.hidden_size}")

    # Create model
    # Initialize single-GPU parallel state (required by Megatron-LM)
    init_single_gpu_parallel_state(device=device)

    logger.info("Creating Nemotron model...")
    model = NemotronModel(config=config, tokenizer=tokenizer)

    # Configure the model to set up self.module (required for NeMo 2.x models)
    # This is normally done by PyTorch Lightning during setup
    if hasattr(model, "configure_model"):
        logger.info("Configuring model (setting up internal module)...")
        model.configure_model()

    # Load weights from checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model weights from {checkpoint_path}...")
        weights_path = os.path.join(checkpoint_path, "weights")
        if os.path.exists(weights_path):
            # Use Megatron's distributed checkpoint loading API
            try:
                from megatron.core.dist_checkpointing import load
                
                # Try to import StrictHandling from different possible locations
                strict_value = None
                for module_path in [
                    'megatron.core.dist_checkpointing.validation',
                    'megatron.core.dist_checkpointing.mapping',
                ]:
                    try:
                        import importlib
                        mod = importlib.import_module(module_path)
                        StrictHandling = getattr(mod, 'StrictHandling')
                        strict_value = StrictHandling.LOG_UNEXPECTED  # Log but don't fail on unexpected keys
                        logger.info(f"Using StrictHandling from {module_path}")
                        break
                    except (ImportError, AttributeError):
                        continue
                
                if strict_value is None:
                    # If StrictHandling not found anywhere, try using string value
                    # which some versions accept
                    strict_value = "log_unexpected"
                    logger.info("StrictHandling enum not found, using string value 'log_unexpected'")
                
                # Get sharded state dict from the model's internal module
                if hasattr(model, 'module') and model.module is not None: 
                    target_module = model.module
                else:
                    target_module = model
                
                # Get sharded state dict for loading
                if hasattr(target_module, 'sharded_state_dict'):
                    sharded_state_dict = target_module.sharded_state_dict()
                else:
                    # Fallback: use regular state dict
                    sharded_state_dict = target_module.state_dict()
                
                # NeMo saves checkpoints with 'module.' prefix on all keys
                # but GPTModel's sharded_state_dict() returns keys without prefix.
                # Use Megatron's utility to properly add prefix to ShardedTensor objects
                try:
                    from megatron.core.dist_checkpointing.utils import add_prefix_for_sharding
                    add_prefix_for_sharding(sharded_state_dict, 'module.')
                    logger.info("Added 'module.' prefix to sharded state dict")
                except ImportError:
                    # Fallback: manual prefix addition (may not work for all ShardedTensor types)
                    logger.warning("add_prefix_for_sharding not available, trying manual prefix")
                    sharded_state_dict = {f"module.{k}": v for k, v in sharded_state_dict.items()}
                
                # Load the distributed checkpoint
                loaded_state = load(
                    sharded_state_dict=sharded_state_dict,
                    checkpoint_dir=weights_path,
                    strict=strict_value,
                )
                
                # Strip 'module.' prefix from loaded state dict keys
                # since the model expects keys without the prefix
                loaded_state_stripped = {}
                for k, v in loaded_state.items():
                    if k.startswith("module."):
                        loaded_state_stripped[k[7:]] = v  # Remove 'module.' prefix
                    else:
                        loaded_state_stripped[k] = v
                
                # Load the state dict into the model
                target_module.load_state_dict(loaded_state_stripped, strict=False)
                logger.info("Model weights loaded via Megatron dist_checkpointing.load")
                
            except Exception as e:
                logger.error(f"Megatron dist_checkpointing load failed: {e}")
                import traceback
                traceback.print_exc()
                logger.warning("Using random initialization - checkpoint loading failed")
        else:
            logger.warning(
                f"Weights not found at {weights_path}, using random initialization"
            )
    else:
        logger.info("No checkpoint provided, model initialized with random weights")

    model = model.to(device)
    model.eval()

    logger.info("Model ready for inference")
    return model, tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda",
):
    """
    Generate text using the model with manual autoregressive loop.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling probability
        device: Device to run generation on

    Returns:
        str: Generated text
    """
    logger.info(f"Generating with prompt: {prompt[:50]}...")

    # Tokenize input
    if hasattr(tokenizer, "text_to_ids"):
        input_ids = tokenizer.text_to_ids(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

    generated_ids = input_ids.clone()

    # Get EOS token ID
    eos_id = getattr(tokenizer, "eos_id", None)
    if eos_id is None and hasattr(tokenizer, "eos_token_id"):
        eos_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        # Create position_ids
        seq_len = generated_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(
            0
        )
        attention_mask = torch.ones_like(generated_ids, device=device)

        # Forward pass - use the internal module directly if available
        # NeMo models wrap the actual transformer in self.module
        if hasattr(model, "module") and model.module is not None:
            outputs = model.module(
                input_ids=generated_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        else:
            outputs = model(
                input_ids=generated_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        # Get logits for the last token
        if hasattr(outputs, "logits"):
            next_token_logits = outputs.logits[:, -1, :].clone()
        elif isinstance(outputs, torch.Tensor):
            next_token_logits = outputs[:, -1, :].clone()
        else:
            # Handle tuple output (logits, hidden_states, ...)
            next_token_logits = outputs[0][:, -1, :].clone()

        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k_values, _ = torch.topk(
                next_token_logits, min(top_k, next_token_logits.size(-1))
            )
            indices_to_remove = next_token_logits < top_k_values[..., -1, None]
            next_token_logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float("-inf")

        # Sample or greedy
        if temperature > 0:
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Append to generated sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Check for EOS
        if eos_id is not None and next_token.item() == eos_id:
            logger.info(f"EOS token reached at step {step + 1}")
            break

    # Decode
    if hasattr(tokenizer, "ids_to_text"):
        generated_text = tokenizer.ids_to_text(generated_ids[0].tolist())
    else:
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text


def get_parser():
    parser = argparse.ArgumentParser(
        description="Load Nemotron 1B model and generate text"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to NeMo checkpoint directory (optional)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="/work/m24047/m24047brmn/tokenizers/minitron-4b",
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to load model on (cuda or cpu)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=24,
        help="Number of transformer layers (24 for full model, 1 for test)",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    # Check device early to avoid opaque Megatron errors
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("CUDA requested but not available. Please run on a GPU node (e.g., sbatch --gres=gpu:1).")
        sys.exit(1)

    if args.device.startswith("cpu"):
        logger.error("CPU execution is not supported for this Nemotron configuration. Use a GPU device instead.")
        sys.exit(1)

    # Load model
    model, tokenizer = load_model(
        checkpoint_path=args.checkpoint,
        tokenizer_name=args.tokenizer,
        device=args.device,
        num_layers=args.num_layers,
    )

    # Generate
    temperature = 0.0 if args.greedy else args.temperature

    output = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=args.device,
    )

    print(f"\n{'=' * 50}")
    print("Prompt:")
    print(f"{'=' * 50}")
    print(args.prompt)
    print(f"\n{'=' * 50}")
    print("Generated text:")
    print(f"{'=' * 50}")
    print(output)
    print(f"{'=' * 50}")

    # Clean up parallel state
    cleanup_parallel_state()


if __name__ == "__main__":
    main()
