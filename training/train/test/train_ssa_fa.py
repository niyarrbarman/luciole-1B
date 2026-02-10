import argparse
import logging
import os
import re
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import fiddle

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipes.recipe_utils import get_recipe  # noqa: E402
from utils import (  # noqa: E402
    check_tokenizer,
    process_datamix_file,
    save_config,
)

# SSA FlexAttention module
from SSA.ssa_flex_layer_specs import get_ssa_flex_gpt_layer_spec


def find_latest_checkpoint_step(checkpoint_dir: str) -> int:
    """Scan {checkpoint_dir}/checkpoints/ for highest step number. Returns 0 if none found."""
    checkpoint_path = Path(checkpoint_dir) / "checkpoints"
    if not checkpoint_path.exists():
        return 0
    
    max_step = 0
    step_pattern = re.compile(r'step[=_](\d+)')
    
    for item in checkpoint_path.iterdir():
        match = step_pattern.search(item.name)
        if match:
            step = int(match.group(1))
            max_step = max(max_step, step)
    
    return max_step


def parse_args():
    parser = argparse.ArgumentParser(description="SSA FlexAttention training harness")
    parser.add_argument("--datamix", default="/tmpdir/m24047brmn/nemo_1b/data/nemo1b_mock_datamix.json", type=str)
    parser.add_argument("--arch", default="nemotron1b", type=str)
    parser.add_argument("--name", default="nemotron1b-ssa-flex-test", type=str)
    parser.add_argument("--mode", default="debug", choices=["debug", "benchmark", "phase1", "phase2", "annealing"], type=str)
    parser.add_argument("--output_dir", default="/tmpdir/m24047brmn/nemo_1b/output", type=str)
    parser.add_argument("--batch_size", "--gbs", default=128, type=int)
    parser.add_argument("--micro_batch_size", "--mbs", default=1, type=int)
    parser.add_argument("--seq_length", default=1024, type=int)
    parser.add_argument("--tensor_parallelism", "--tp", default=1, type=int)
    parser.add_argument("--pipeline_parallelism", "--pp", default=1, type=int)
    parser.add_argument("--context_parallelism", "--cp", default=1, type=int)
    parser.add_argument("--max_steps", default=5000, type=int, help="Steps to run THIS job (per-run)")
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--gpus_per_node", default=1, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--base_checkpoint", default=None, type=str, help="Base checkpoint for weight init")
    parser.add_argument("--fp8", action="store_true", default=False)
    parser.add_argument("--performance_mode", action="store_true", default=False)
    parser.add_argument("--duration", default="00:24:00:00", type=str, help="Walltime DD:HH:MM:SS")
    parser.add_argument("--save_every_n_steps", default=500, type=int)
    parser.add_argument("--global_max_steps", default=None, type=int, help="Total training horizon for LR decay")
    # SSA-specific parameters
    parser.add_argument("--ssa_n", default=1.0, type=float, help="SSA n parameter (initial value, n >= 1)")
    parser.add_argument("--ssa_b", default=1.0, type=float, help="SSA b parameter (initial value, b > 0)")
    parser.add_argument("--ssa_learnable", action="store_true", default=True, help="Make SSA params learnable")
    parser.add_argument("--ssa_fixed", action="store_true", default=False, help="Fix SSA params (not learnable)")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)

    # Build base recipe
    gpus_per_node = args.gpus_per_node
    num_nodes = args.num_nodes
    recipe = get_recipe(
        arch=args.arch,
        recipe_args=dict(dir=args.output_dir, name=args.name, num_nodes=num_nodes, num_gpus_per_node=gpus_per_node),
        performance_mode_if_possible=args.performance_mode,
    )

    # Data
    tokenizer_name, data_paths, total_tokens = process_datamix_file(args.datamix)
    check_tokenizer(tokenizer_name, args.base_checkpoint)

    global_batch_size = args.batch_size
    seq_length = args.seq_length
    tokens_per_batch = seq_length * global_batch_size
    logger.info("Global batch size: %s", global_batch_size)
    logger.info("Sequence length: %s", seq_length)
    logger.info("Tokens per batch: %s", tokens_per_batch)
    logger.info("Total tokens in datamix: %s", total_tokens)
    
    if args.global_max_steps is not None:
        global_max_steps = args.global_max_steps
        logger.info("Using provided global_max_steps: %s", global_max_steps)
    else:
        import math
        global_max_steps = math.floor(total_tokens / tokens_per_batch)
        logger.info("Computed global_max_steps from datamix: %s", global_max_steps)

    from nemo import lightning as nl  # noqa: E402
    from nemo.collections.llm.gpt.data import PreTrainingDataModule  # noqa: E402
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer  # noqa: E402
    import nemo_run as run  # noqa: E402

    # Patch megatron optimizer
    try:
        import inspect
        import megatron.core.optimizer as mcore_optim

        if not getattr(mcore_optim, "_patched_get_megatron_optimizer", False):
            _orig = mcore_optim.get_megatron_optimizer
            _accepted = set(inspect.signature(_orig).parameters.keys())

            def _patched(*args, **kwargs):
                return _orig(*args, **{k: v for k, v in kwargs.items() if k in _accepted})

            mcore_optim.get_megatron_optimizer = _patched
            mcore_optim._patched_get_megatron_optimizer = True
            logger.info("Patched megatron.core.optimizer (accepted params: %s)", _accepted)
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
    # SSA FlexAttention: Inject custom attention layer spec
    # This replaces the default attention with SSAFlexAttention
    # which uses PyTorch's flex_attention with SSA score modification
    # ============================================================
    learnable_ssa = args.ssa_learnable and not args.ssa_fixed
    ssa_layer_spec = get_ssa_flex_gpt_layer_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=False,
        ssa_n=args.ssa_n,
        ssa_b=args.ssa_b,
        learnable_ssa=False,
    )
    recipe.model.config.transformer_layer_spec = ssa_layer_spec
    recipe.model.config.masked_softmax_fusion = False
    logger.info("SSA FlexAttention: n=%.2f, b=%.2f, learnable=%s", args.ssa_n, args.ssa_b, learnable_ssa)

    # Per-run max_steps
    experiment_dir = os.path.join(args.output_dir, args.name)
    detected_step = find_latest_checkpoint_step(experiment_dir)
    
    if detected_step > 0:
        effective_max_steps = min(detected_step + args.max_steps, global_max_steps)
        logger.info(f"Detected checkpoint at step {detected_step}")
        logger.info(f"trainer.max_steps = {detected_step} + {args.max_steps} = {effective_max_steps}")
    else:
        effective_max_steps = min(args.max_steps, global_max_steps)
        logger.info(f"Training from scratch, trainer.max_steps = {effective_max_steps}")

    # Trainer
    recipe.trainer.max_steps = effective_max_steps
    recipe.trainer.val_check_interval = effective_max_steps
    recipe.trainer.limit_val_batches = 0.0
    recipe.trainer.log_every_n_steps = 1
    recipe.trainer.devices = gpus_per_node
    recipe.trainer.strategy.tensor_model_parallel_size = args.tensor_parallelism
    recipe.trainer.strategy.pipeline_model_parallel_size = args.pipeline_parallelism
    recipe.trainer.strategy.context_parallel_size = args.context_parallelism
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = None
    recipe.trainer.strategy.sequence_parallel = False
    recipe.trainer.strategy.pipeline_dtype = torch.bfloat16

    # Remove unsupported optimizer kwargs
    optim_cfg = getattr(recipe, "optim", None)
    if optim_cfg and hasattr(optim_cfg, "config") and hasattr(optim_cfg.config, "no_weight_decay_cond"):
        try:
            delattr(optim_cfg.config, "no_weight_decay_cond")
        except Exception:
            pass

    # Callbacks
    from nemo.lightning.pytorch.callbacks import GarbageCollectionCallback  # noqa: E402
    from callbacks import StatelessTimer, ProgressiveIntervalCheckpoint  # noqa: E402

    recipe.trainer.callbacks = [
        run.Config(StatelessTimer, duration=args.duration),
        run.Config(GarbageCollectionCallback, gc_interval_train=100, gc_interval_val=100),
    ]

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
        restore_config = nl.RestoreConfig(path=args.base_checkpoint, load_optim_state=False)
    else:
        restore_config = None

    # LR scheduler
    if hasattr(recipe.optim, 'lr_scheduler'):
        recipe.optim.lr_scheduler.max_steps = global_max_steps
        logger.info(f"LR scheduler max_steps = {global_max_steps}")
    
    recipe.resume = run.Config(
        nl.AutoResume,
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_past_end=True,
        restore_config=restore_config,
    )

    # Save config snapshot
    job_id = os.environ.get("SLURM_JOB_ID", "0")
    job_output = os.path.join(args.output_dir, f"job_{job_id}")
    os.makedirs(job_output, exist_ok=True)
    save_config(job_output, args, recipe)

    # Run
    recipe_obj = fiddle.build(recipe)
    recipe_obj()
    logger.info("Finished SSA FlexAttention training run.")


if __name__ == "__main__":
    main()
