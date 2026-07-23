import argparse
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import fiddle
import pytorch_lightning as pl
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipes.recipe_utils import get_recipe  # noqa: E402
from utils import (  # noqa: E402
    check_tokenizer,
    get_data_paths,
    process_datamix_file,
    read_datamix_file,
    save_config,
)

# SSA Triton-fused attention — pinned to triton (tutorial-based kernel).
SSA_KERNEL_VERSION = os.environ.get("SSA_KERNEL_VERSION", "triton")
if SSA_KERNEL_VERSION != "triton":
    raise ValueError(
        f"Unsupported SSA_KERNEL_VERSION={SSA_KERNEL_VERSION!r}. "
        "This launcher is pinned to triton."
    )

from SSA.ssa_triton_kernel import warmup_ssa_triton_kernels as warmup_triton_kernels
from SSA.ssa_triton_layer_specs import (
    get_ssa_triton_gpt_layer_spec as get_ssa_triton_gpt_layer_spec,
)


def find_latest_checkpoint_step(checkpoint_dir: str) -> int:
    """Scan {checkpoint_dir}/checkpoints/ for highest step number. Returns 0 if none found."""
    checkpoint_path = Path(checkpoint_dir) / "checkpoints"
    if not checkpoint_path.exists():
        return 0

    max_step = 0
    step_pattern = re.compile(r"step[=_](\d+)")

    for item in checkpoint_path.iterdir():
        match = step_pattern.search(item.name)
        if match:
            step = int(match.group(1))
            max_step = max(max_step, step)

    return max_step


def parse_args():
    parser = argparse.ArgumentParser(description="SSA Triton-fused training harness")
    parser.add_argument(
        "--datamix",
        default="/tmpdir/m24047brmn/nemo_1b/data/nemo1b_mock_datamix.json",
        type=str,
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        type=str,
        help="Tokenizer path (overrides datamix tokenizer)",
    )
    parser.add_argument("--arch", default="nemotron1b", type=str)
    parser.add_argument("--name", default="nemotron1b-ssa-triton-test", type=str)
    parser.add_argument(
        "--mode",
        default="debug",
        choices=["debug", "benchmark", "phase1", "phase2", "annealing"],
        type=str,
    )
    parser.add_argument(
        "--output_dir", default="/tmpdir/m24047brmn/nemo_1b/output", type=str
    )
    parser.add_argument("--batch_size", "--gbs", default=128, type=int)
    parser.add_argument("--micro_batch_size", "--mbs", default=1, type=int)
    parser.add_argument("--seq_length", default=1024, type=int)
    parser.add_argument("--tensor_parallelism", "--tp", default=1, type=int)
    parser.add_argument("--pipeline_parallelism", "--pp", default=1, type=int)
    parser.add_argument("--context_parallelism", "--cp", default=1, type=int)
    parser.add_argument(
        "--max_steps",
        default=5000,
        type=int,
        help="Absolute training horizon (target global step for this run series).",
    )
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--gpus_per_node", default=1, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument(
        "--base_checkpoint",
        default=None,
        type=str,
        help="Base checkpoint for weight init (phase transitions)",
    )
    parser.add_argument("--fp8", action="store_true", default=False)
    parser.add_argument("--performance_mode", action="store_true", default=False)
    parser.add_argument(
        "--duration", default="00:24:00:00", type=str, help="Walltime DD:HH:MM:SS"
    )
    parser.add_argument("--save_every_n_steps", default=500, type=int)
    parser.add_argument(
        "--global_max_steps",
        default=None,
        type=int,
        help="Deprecated in this launcher; --max_steps is the global LR/training horizon.",
    )
    parser.add_argument(
        "--this_run_max_steps",
        default=None,
        type=int,
        help="Optional per-job step budget; stop after this many optimizer steps in this run.",
    )
    parser.add_argument(
        "--log_ssa_every_n_steps",
        default=1000,
        type=int,
        help="Log SSA n values every N steps",
    )
    parser.add_argument(
        "--log_gpu_every_n_steps",
        default=10,
        type=int,
        help="Log GPU memory/utilization stats to the logger (TensorBoard/W&B) every N steps",
    )
    # SSA hyperparameters
    parser.add_argument(
        "--ssa_n", default=1.5, type=float, help="SSA n param initial value"
    )
    parser.add_argument(
        "--ssa_b", default=0.8, type=float, help="SSA b param initial value"
    )
    parser.add_argument(
        "--learnable_b", action="store_true", default=False, help="Make b learnable"
    )
    parser.add_argument(
        "--disable_compiled_bda",
        action="store_true",
        default=False,
        help="Use eager bias-dropout-add path (disable torch.compile'd BDA).",
    )
    parser.add_argument(
        "--force_contiguous_qkv",
        action="store_true",
        default=False,
        help="Materialize Q/K/V as contiguous tensors before Triton attention call.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=500,
        type=int,
        help="LR scheduler warmup steps override.",
    )
    parser.add_argument(
        "--skip_triton_warmup",
        action="store_true",
        default=False,
        help="Skip Triton pre-warm JIT/autotune step.",
    )

    # Weights & Biases logging (rank 0 only)
    default_wandb_mode = os.environ.get("WANDB_MODE")
    if default_wandb_mode is None:
        default_wandb_mode = "offline" if os.environ.get("SLURM_JOB_ID") else "online"
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project", default=os.environ.get("WANDB_PROJECT", None), type=str
    )
    parser.add_argument(
        "--wandb_entity", default=os.environ.get("WANDB_ENTITY", None), type=str
    )
    parser.add_argument(
        "--wandb_group", default=os.environ.get("WANDB_GROUP", None), type=str
    )
    parser.add_argument(
        "--wandb_run_name", default=os.environ.get("WANDB_NAME", None), type=str
    )
    parser.add_argument(
        "--wandb_tags",
        default=os.environ.get("WANDB_TAGS", ""),
        type=str,
        help="Comma-separated W&B tags.",
    )
    parser.add_argument(
        "--wandb_notes", default=os.environ.get("WANDB_NOTES", None), type=str
    )
    parser.add_argument(
        "--wandb_job_type", default=os.environ.get("WANDB_JOB_TYPE", "train"), type=str
    )
    parser.add_argument(
        "--wandb_dir", default=os.environ.get("WANDB_DIR", None), type=str
    )
    parser.add_argument(
        "--wandb_id", default=os.environ.get("WANDB_RUN_ID", None), type=str
    )
    parser.add_argument(
        "--wandb_resume",
        default=os.environ.get("WANDB_RESUME", "allow"),
        type=str,
        help="W&B resume policy (e.g. allow|must|never).",
    )
    parser.add_argument(
        "--wandb_mode",
        default=default_wandb_mode,
        choices=["online", "offline", "disabled"],
        type=str,
        help="W&B mode (online/offline/disabled). Defaults to $WANDB_MODE or offline on Slurm.",
    )
    parser.add_argument(
        "--wandb_log_model",
        action="store_true",
        default=False,
        help="Log checkpoints as W&B artifacts.",
    )
    return parser.parse_args()


def _get_global_rank() -> int:
    for key in (
        "RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "MV2_COMM_WORLD_RANK",
    ):
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def _parse_wandb_tags(raw: str):
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _resolve_wandb_run_id(experiment_dir: str, provided_id: Optional[str]) -> str:
    if provided_id:
        return provided_id
    run_id_file = Path(experiment_dir) / "wandb_run_id.txt"
    try:
        if run_id_file.exists():
            existing = run_id_file.read_text(encoding="utf-8").strip()
            if existing:
                return existing
    except Exception:
        pass
    run_id = uuid.uuid4().hex
    try:
        run_id_file.write_text(run_id, encoding="utf-8")
    except Exception:
        pass
    return run_id


def _maybe_enable_wandb_logger(
    *, recipe, args, run, logger, experiment_dir: str
) -> None:
    if not getattr(args, "wandb", False):
        return
    if getattr(args, "wandb_mode", "online") == "disabled":
        logger.info("W&B logging requested but disabled via --wandb_mode=disabled.")
        return

    global_rank = _get_global_rank()
    if global_rank != 0:
        # Avoid multiple W&B runs in torchrun/DDP. Rank 0 is enough.
        try:
            recipe.trainer.logger = False
        except Exception:
            pass
        return

    try:
        try:
            from lightning.pytorch.loggers import WandbLogger  # type: ignore
        except Exception:
            from pytorch_lightning.loggers import WandbLogger  # type: ignore
    except Exception as e:
        logger.warning(
            "W&B requested but WandbLogger import failed; continuing without W&B. Error: %s",
            e,
        )
        return

    wandb_dir = args.wandb_dir or os.path.join(experiment_dir, "wandb")
    os.makedirs(wandb_dir, exist_ok=True)

    # Prefer env vars for cross-version compatibility with wandb/Lightning.
    os.environ.setdefault("WANDB_DIR", wandb_dir)
    os.environ.setdefault("WANDB_START_METHOD", "thread")
    os.environ["WANDB_MODE"] = args.wandb_mode

    wandb_id = _resolve_wandb_run_id(experiment_dir, args.wandb_id)
    wandb_name = args.wandb_run_name or args.name
    wandb_group = args.wandb_group or args.name
    wandb_tags = _parse_wandb_tags(args.wandb_tags)

    # Build kwargs defensively across Lightning versions.
    import inspect

    sig = inspect.signature(WandbLogger.__init__)
    params = sig.parameters
    allowed = {k for k in params.keys() if k != "self"}
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    base_kwargs = {
        "project": args.wandb_project,
        "name": wandb_name,
        "save_dir": wandb_dir,
        "id": wandb_id,
        "log_model": args.wandb_log_model,
        # Some Lightning versions use `offline` instead of `mode`.
        "offline": args.wandb_mode == "offline",
    }
    wandb_init_kwargs = {
        "entity": args.wandb_entity,
        "group": wandb_group,
        "tags": wandb_tags if wandb_tags else None,
        "notes": args.wandb_notes,
        "job_type": args.wandb_job_type,
        "resume": args.wandb_resume,
        "config": vars(args),
    }

    cfg_kwargs = {}
    for k, v in base_kwargs.items():
        if v is None:
            continue
        if k in allowed or accepts_var_kwargs:
            cfg_kwargs[k] = v

    if accepts_var_kwargs:
        cfg_kwargs.update({k: v for k, v in wandb_init_kwargs.items() if v is not None})
    else:
        cfg_kwargs.update(
            {
                k: v
                for k, v in wandb_init_kwargs.items()
                if v is not None and k in allowed
            }
        )

    recipe.trainer.logger = run.Config(WandbLogger, **cfg_kwargs)
    logger.info(
        "Enabled W&B logging: project=%s name=%s group=%s mode=%s id=%s dir=%s",
        args.wandb_project,
        wandb_name,
        wandb_group,
        args.wandb_mode,
        wandb_id,
        wandb_dir,
    )


def main():
    args = parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Enforce requested SSA setup:
    # - n is learnable and initialized at 1.5
    # - b is fixed at 0.8 (non-learnable)
    if args.learnable_b:
        logger.warning("Ignoring --learnable_b; b is fixed by policy.")
    if args.ssa_n != 1.5 or args.ssa_b != 0.8:
        logger.warning(
            "Overriding SSA init from (n=%s, b=%s) to fixed policy (n=1.5, b=0.8).",
            args.ssa_n,
            args.ssa_b,
        )
    args.learnable_b = False
    args.ssa_n = 1.5
    args.ssa_b = 0.8

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)

    # Build base recipe
    gpus_per_node = args.gpus_per_node
    num_nodes = args.num_nodes
    recipe = get_recipe(
        arch=args.arch,
        recipe_args=dict(
            dir=args.output_dir,
            name=args.name,
            num_nodes=num_nodes,
            num_gpus_per_node=gpus_per_node,
        ),
        performance_mode_if_possible=args.performance_mode,
    )

    # Data
    if args.tokenizer:
        # --tokenizer provided: skip datamix tokenizer lookup (tokenizer_name.txt may not exist)
        loaded_data = read_datamix_file(args.datamix)
        data_paths = get_data_paths(loaded_data)
        total_tokens = loaded_data.get("total_tokens", None)
        tokenizer_name = args.tokenizer
        logger.info("Using CLI tokenizer override: %s", tokenizer_name)
    else:
        tokenizer_name, data_paths, total_tokens = process_datamix_file(args.datamix)
    # check_tokenizer(tokenizer_name, args.base_checkpoint)

    global_batch_size = args.batch_size
    seq_length = args.seq_length
    tokens_per_batch = seq_length * global_batch_size
    logger.info("Global batch size: %s", global_batch_size)
    logger.info("Sequence length: %s", seq_length)
    logger.info("Tokens per batch: %s", tokens_per_batch)
    logger.info("Total tokens in datamix: %s", total_tokens)

    # In this launcher, --max_steps is the global horizon used by trainer and LR scheduler.
    # --global_max_steps is kept only for backward compatibility and is ignored.
    if args.global_max_steps is not None:
        logger.warning(
            "Ignoring deprecated --global_max_steps=%s; using --max_steps=%s as global horizon.",
            args.global_max_steps,
            args.max_steps,
        )
    global_max_steps = args.max_steps
    logger.info("Global training/LR horizon (from --max_steps): %s", global_max_steps)
    if args.this_run_max_steps is not None and args.this_run_max_steps <= 0:
        raise ValueError("--this_run_max_steps must be > 0 when provided.")
    import nemo_run as run  # noqa: E402
    from nemo import lightning as nl  # noqa: E402
    from nemo.collections.llm.gpt.data import PreTrainingDataModule  # noqa: E402
    from nemo.collections.nlp.modules.common.tokenizer_utils import (
        get_tokenizer,  # noqa: E402
    )

    # Patch megatron optimizer to drop unsupported kwargs (version compatibility)
    try:
        import inspect

        import megatron.core.optimizer as mcore_optim

        if not getattr(mcore_optim, "_patched_get_megatron_optimizer", False):
            _orig = mcore_optim.get_megatron_optimizer
            _accepted = set(inspect.signature(_orig).parameters.keys())

            def _patched(*args, **kwargs):
                return _orig(
                    *args, **{k: v for k, v in kwargs.items() if k in _accepted}
                )

            mcore_optim.get_megatron_optimizer = _patched
            mcore_optim._patched_get_megatron_optimizer = True
            logger.info(
                "Patched megatron.core.optimizer (accepted params: %s)", _accepted
            )
    except Exception as e:
        logger.warning("Could not patch megatron optimizer: %s", e)

    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, use_fast=True)

    recipe.data = run.Config(
        PreTrainingDataModule,
        tokenizer=tokenizer,
        num_workers=8,
        pin_memory=True,
        split="1,0,0",
        paths=data_paths,
        global_batch_size=global_batch_size,
        micro_batch_size=args.micro_batch_size,
        seq_length=seq_length,
        seed=args.seed,
        index_mapping_dir=os.path.join(args.output_dir, "index_mapping"),
    )

    recipe.model.tokenizer = tokenizer
    recipe.model.config.seq_length = seq_length

    # ============================================================
    # SSA Triton: Inject fused Triton attention layer spec
    # Uses a single fused kernel for the entire attention computation:
    #   scale -> causal mask -> SSA transform -> online softmax -> V accumulation
    # No O(S^2) memory, no unfused kernel chain.
    # ============================================================
    ssa_layer_spec = get_ssa_triton_gpt_layer_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=False,
        ssa_n=args.ssa_n,
        ssa_b=args.ssa_b,
        learnable_ssa=True,
        learnable_b=False,
        use_compiled_bda=not args.disable_compiled_bda,
        force_contiguous_qkv=args.force_contiguous_qkv,
    )
    recipe.model.config.transformer_layer_spec = ssa_layer_spec
    # Disable fused softmax (we use our own Triton kernel)
    recipe.model.config.masked_softmax_fusion = False
    logger.info(
        "SSA Triton: Using fused Triton SSA FlashAttention %s (n=%.2f, b=%.2f, compiled_bda=%s)",
        SSA_KERNEL_VERSION,
        args.ssa_n,
        args.ssa_b,
        not args.disable_compiled_bda,
    )
    logger.info("SSA Triton: force_contiguous_qkv=%s", args.force_contiguous_qkv)

    # Max-step policy for Triton launcher:
    # - --max_steps is the absolute global horizon for trainer + scheduler.
    # - Optional --this_run_max_steps limits how many optimizer steps this job executes.
    experiment_dir = os.path.join(args.output_dir, args.name)
    detected_step = find_latest_checkpoint_step(experiment_dir)
    effective_max_steps = global_max_steps

    if detected_step > 0:
        logger.info("Detected checkpoint at step %s", detected_step)
        if detected_step >= effective_max_steps:
            logger.warning(
                "Checkpoint step %s is already >= trainer.max_steps %s; "
                "increase --max_steps to continue training.",
                detected_step,
                effective_max_steps,
            )
        else:
            logger.info(
                "Continuing to absolute trainer.max_steps = %s (remaining this run: %s steps)",
                effective_max_steps,
                effective_max_steps - detected_step,
            )
    else:
        logger.info(
            "Training from scratch, trainer.max_steps = %s", effective_max_steps
        )

    # Trainer — detect vocab size from tokenizer
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = 50256
        logger.warning(
            "Could not detect vocab_size from tokenizer, falling back to %s", vocab_size
        )
    else:
        logger.info("Detected vocab_size from tokenizer: %s", vocab_size)
    recipe.model.config.vocab_size = vocab_size
    recipe.trainer.max_steps = effective_max_steps
    recipe.trainer.val_check_interval = effective_max_steps
    recipe.trainer.limit_val_batches = 0.0
    recipe.trainer.log_every_n_steps = 200
    recipe.trainer.devices = gpus_per_node
    recipe.trainer.strategy.tensor_model_parallel_size = args.tensor_parallelism
    recipe.trainer.strategy.pipeline_model_parallel_size = args.pipeline_parallelism
    recipe.trainer.strategy.context_parallel_size = args.context_parallelism
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = None
    recipe.trainer.strategy.sequence_parallel = False
    recipe.trainer.strategy.pipeline_dtype = torch.bfloat16
    if hasattr(recipe.log, "log_interval"):
        recipe.log.log_interval = 200

    # Remove unsupported optimizer kwargs
    optim_cfg = getattr(recipe, "optim", None)
    if (
        optim_cfg
        and hasattr(optim_cfg, "config")
        and hasattr(optim_cfg.config, "no_weight_decay_cond")
    ):
        try:
            delattr(optim_cfg.config, "no_weight_decay_cond")
        except Exception:
            pass

    # Callbacks
    from nemo.lightning.pytorch.callbacks import GarbageCollectionCallback  # noqa: E402

    from callbacks import (  # noqa: E402
        GPUStatsCallback,
        ProgressiveIntervalCheckpoint,
        RNGStateCallback,
        SSALoggingCallback,
        StatelessTimer,
        StopAfterThisRunMaxStepsCallback,
        WandbMetricsCallback,
    )

    trainer_callbacks = [
        run.Config(StatelessTimer, duration=args.duration),
        run.Config(
            GarbageCollectionCallback, gc_interval_train=100, gc_interval_val=100
        ),
        run.Config(SSALoggingCallback, log_every_n_steps=args.log_ssa_every_n_steps),
        run.Config(GPUStatsCallback, log_every_n_steps=args.log_gpu_every_n_steps),
        run.Config(RNGStateCallback),
    ]
    if args.wandb:
        trainer_callbacks.append(run.Config(WandbMetricsCallback, log_every_n_steps=1))
    if args.this_run_max_steps is not None:
        trainer_callbacks.append(
            run.Config(
                StopAfterThisRunMaxStepsCallback,
                this_run_max_steps=args.this_run_max_steps,
            )
        )
        logger.info(
            "Per-job step cap enabled: this_run_max_steps=%s",
            args.this_run_max_steps,
        )
    recipe.trainer.callbacks = trainer_callbacks

    # Checkpoint config
    recipe.log.ckpt = run.Config(
        ProgressiveIntervalCheckpoint,
        filename=args.name + "-{step:07.0f}",
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=args.save_every_n_steps,
        monitor="step",
        mode="max",
        every_n_epochs=None,
        save_optim_on_train_end=True,
        verbose=True,
    )

    # Resume config
    if args.base_checkpoint:
        logger.info(f"Base checkpoint: {args.base_checkpoint}")
        restore_config = nl.RestoreConfig(
            path=args.base_checkpoint, load_optim_state=False
        )
    else:
        restore_config = None

    # LR scheduler horizon is aligned to trainer.max_steps for consistency.
    # NeMo may overwrite scheduler.max_steps from trainer.max_steps internally.
    if hasattr(recipe.optim, "lr_scheduler"):
        recipe.optim.lr_scheduler.max_steps = effective_max_steps
        recipe.optim.lr_scheduler.warmup_steps = args.warmup_steps
        recipe.optim.config.lr = 3e-4
        recipe.optim.lr_scheduler.min_lr = 6.87e-5
        logger.info(
            "LR scheduler max_steps = %s, warmup_steps = %s, min_lr = %s",
            effective_max_steps,
            args.warmup_steps,
            recipe.optim.lr_scheduler.min_lr,
        )

    recipe.resume = run.Config(
        nl.AutoResume,
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_past_end=True,
        restore_config=restore_config,
    )

    _maybe_enable_wandb_logger(
        recipe=recipe,
        args=args,
        run=run,
        logger=logger,
        experiment_dir=experiment_dir,
    )

    # Save config snapshot (rank 0 only to avoid races under torchrun).
    if _get_global_rank() == 0:
        job_id = os.environ.get("SLURM_JOB_ID", "0")
        job_output = os.path.join(args.output_dir, f"job_{job_id}")
        os.makedirs(job_output, exist_ok=True)
        save_config(job_output, args, recipe)

    # ============================================================
    # Pre-warm Triton kernels (compile ahead of step 0)
    # This runs a tiny dummy fwd+bwd to trigger JIT compilation.
    # Compiled kernels are cached in TRITON_CACHE_DIR.
    # ============================================================
    if torch.cuda.is_available() and not args.skip_triton_warmup:
        # In torchrun multi-GPU jobs, ensure each rank warms up its own GPU.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.device_count() > 0:
            torch.cuda.set_device(local_rank % torch.cuda.device_count())

        # Keep model initialization reproducible vs non-warmup runs:
        # warmup does random tensor allocations, so we snapshot/restore RNG states.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all()
        try:
            # Use the model's actual head dimensions for warmup
            hq = recipe.model.config.num_attention_heads
            hkv = recipe.model.config.num_query_groups
            head_dim = recipe.model.config.kv_channels
            if (
                head_dim is None
                and recipe.model.config.hidden_size is not None
                and hq is not None
            ):
                head_dim = recipe.model.config.hidden_size // hq

            if hq is None or hkv is None or head_dim is None:
                logger.warning(
                    "Skipping Triton warmup (missing dims): Hq=%s, Hkv=%s, D=%s",
                    hq,
                    hkv,
                    head_dim,
                )
            else:
                logger.info(
                    "Warming up Triton kernels on cuda:%s (Hq=%s, Hkv=%s, D=%s)...",
                    torch.cuda.current_device(),
                    hq,
                    hkv,
                    head_dim,
                )
                warmup_triton_kernels(
                    B=2,
                    Hq=hq,
                    Hkv=hkv,
                    N=128,
                    D=head_dim,
                    dtype=torch.bfloat16,
                    device=f"cuda:{torch.cuda.current_device()}",
                )
                logger.info("Triton kernel warmup complete.")
        except Exception as e:
            logger.warning("Triton warmup failed (non-fatal): %s", e)
        finally:
            # Restore RNG so weight init and dataloader seeding stay comparable to baseline.
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state_all(cuda_rng_state)
            logger.info("Restored RNG state after Triton warmup.")
    elif args.skip_triton_warmup:
        logger.info("Skipping Triton warmup (--skip_triton_warmup).")

    # Free GPU memory accumulated during setup/warmup before checkpoint restore
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run
    import time as _time

    _t0 = _time.time()
    recipe_obj = fiddle.build(recipe)
    recipe_obj()
    _elapsed = _time.time() - _t0
    logger.info("Finished SSA Triton training run in %.1f seconds.", _elapsed)


if __name__ == "__main__":
    main()
