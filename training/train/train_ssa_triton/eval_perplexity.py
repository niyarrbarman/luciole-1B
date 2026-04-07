"""
Evaluate perplexity of Baby Luciole models on FineWeb and Wiki datasets.

This script loads a Baby Luciole model from a NeMo checkpoint
and calculates perplexity. Supports both SSA and Softmax attention.
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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


def _load_checkpoint_model_config(checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
    """Best-effort load of NeMo checkpoint architecture from context/model.yaml."""
    model_yaml_path = checkpoint_dir / "context" / "model.yaml"
    if not model_yaml_path.exists():
        return None

    try:
        import yaml

        with model_yaml_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception as exc:
        logger.warning("Could not parse checkpoint model config %s: %s", model_yaml_path, exc)
        return None

    if not isinstance(payload, dict):
        return None

    config = payload.get("config")
    if isinstance(config, dict):
        return config

    return payload


def get_baby_luciole_config(checkpoint_dir: Optional[Path] = None):
    """Return model config, optionally overridden from checkpoint context metadata."""
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

    if checkpoint_dir is not None:
        checkpoint_cfg = _load_checkpoint_model_config(checkpoint_dir)
        if checkpoint_cfg:
            checkpoint_has_explicit_kv = "kv_channels" in checkpoint_cfg and checkpoint_cfg.get("kv_channels") is not None
            int_fields = (
                "num_layers",
                "num_attention_heads",
                "num_query_groups",
                "hidden_size",
                "ffn_hidden_size",
                "kv_channels",
                "vocab_size",
                "seq_length",
            )
            bool_fields = ("share_embeddings_and_output_weights",)

            for field in int_fields:
                value = checkpoint_cfg.get(field)
                if value is not None:
                    try:
                        setattr(config, field, int(value))
                    except (TypeError, ValueError):
                        logger.warning("Ignoring non-integer checkpoint config value for %s=%r", field, value)

            for field in bool_fields:
                value = checkpoint_cfg.get(field)
                if isinstance(value, bool):
                    setattr(config, field, value)

            # If checkpoint does not explicitly define kv_channels, derive it from hidden/head dims.
            if (not checkpoint_has_explicit_kv) or getattr(config, "kv_channels", None) in (None, 0):
                config.kv_channels = config.hidden_size // config.num_attention_heads

            logger.info(
                "Loaded model architecture from checkpoint context: layers=%s hidden=%s heads=%s q_groups=%s ffn=%s kv=%s vocab=%s",
                config.num_layers,
                config.hidden_size,
                config.num_attention_heads,
                config.num_query_groups,
                config.ffn_hidden_size,
                config.kv_channels,
                config.vocab_size,
            )
        else:
            logger.info("No checkpoint model config found; using Baby Luciole defaults")

    return config


def _resolve_checkpoint_dir(checkpoint_path: str) -> Path:
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise ValueError(
            f"Expected checkpoint directory, got {checkpoint_dir}"
        )

    return checkpoint_dir


def _is_hf_checkpoint(checkpoint_dir: Path) -> bool:
    """Detect HuggingFace format checkpoint."""
    config_path = checkpoint_dir / "config.json"
    has_safetensors = (checkpoint_dir / "model.safetensors").exists()
    has_bin = (checkpoint_dir / "pytorch_model.bin").exists()
    return config_path.exists() and (has_safetensors or has_bin)


def _is_nemo_checkpoint(checkpoint_dir: Path) -> bool:
    """Detect NeMo distributed checkpoint format."""
    weights_dir = checkpoint_dir / "weights"
    return weights_dir.exists()

def _get_target_module(model):
    if hasattr(model, "module") and model.module is not None:
        return model.module
    return model


def _load_hf_model(
    checkpoint_dir: Path,
    tokenizer_name: str,
    device: str,
    use_ssa: bool = False,
):
    """Load HuggingFace format checkpoint using transformers library."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading HuggingFace model from %s", checkpoint_dir)
    
    try:
        config = AutoConfig.from_pretrained(str(checkpoint_dir))
        logger.info(
            "Loaded HF config: layers=%s hidden=%s heads=%s vocab=%s",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.vocab_size,
        )
    except Exception as exc:
        logger.error("Failed to load HF config: %s", exc)
        raise

    if use_ssa:
        raise ValueError(
            "HF checkpoint loading is only supported for softmax models in this script. "
            "Use a NeMo checkpoint for SSA mode."
        )

    # lm-eval needs token logits over the full vocabulary. Use a CausalLM head,
    # not a base transformer model that only returns hidden states.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir),
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
    except Exception:
        # Some custom repos require remote code to register model classes.
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir),
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    logger.info("Model loaded successfully from HuggingFace CausalLM format")

    try:
        tokenizer_path = checkpoint_dir / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
            logger.info("Tokenizer loaded from checkpoint")
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info("Tokenizer loaded from %s", tokenizer_name)
    except Exception as exc:
        logger.warning("Could not load tokenizer; attempting fallback: %s", exc)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model.eval()
    logger.info("Model ready for evaluation (HF format)")
    return model, tokenizer

def load_model(
    checkpoint_path: str,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    device: str = "cuda",
    use_ssa: bool = False,
    enforce_bf16: bool = True,
):
    """Load Baby Luciole model from checkpoint (supports NeMo and HuggingFace formats)."""
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_path)

    if _is_hf_checkpoint(checkpoint_dir):
        logger.info("Detected HuggingFace checkpoint format")
        return _load_hf_model(
            checkpoint_dir,
            tokenizer_name,
            device,
            use_ssa=use_ssa,
        )
    elif _is_nemo_checkpoint(checkpoint_dir):
        logger.info("Detected NeMo distributed checkpoint format")
        return _load_nemo_model(
            checkpoint_dir,
            tokenizer_name,
            device,
            enforce_bf16=enforce_bf16,
        )
    else:
        raise ValueError(
            f"Checkpoint at {checkpoint_dir} is neither HuggingFace nor NeMo format. "
            "Expected 'config.json+model.safetensors' or 'weights/ directory'."
        )


def _load_nemo_model(
    checkpoint_dir: Path,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    device: str = "cuda",
    use_ssa: bool = False,
    enforce_bf16: bool = True,
):
    """Load Baby Luciole model from NeMo distributed checkpoint (standard attention)."""
    from nemo.collections.llm.gpt.model.nemotron import NemotronModel
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer


    tokenizer_path = checkpoint_dir / "context" / "tokenizer_name.txt"
    if tokenizer_path.exists():
        tokenizer_name = tokenizer_path.read_text(encoding="utf-8").strip()
        logger.info("Loading tokenizer from checkpoint: %s", tokenizer_name)
    else:
        logger.info("No tokenizer in checkpoint, using: %s", tokenizer_name)

    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, use_fast=True)

    config = get_baby_luciole_config(checkpoint_dir=checkpoint_dir)

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

    strict_value = _resolve_strict_handling()

    loaded_state = load(
        sharded_state_dict=sharded_state_dict,
        checkpoint_dir=str(checkpoint_dir / "weights"),
        strict=strict_value,
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
    if enforce_bf16:
        model = model.to(dtype=torch.bfloat16)
        logger.info("Converted NeMo model to bfloat16 for inference")
    else:
        logger.info("Keeping NeMo model dtype unchanged (enforce_bf16=False)")
    model.eval()

    logger.info("Model ready for evaluation (NeMo format)")
    return model, tokenizer


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
