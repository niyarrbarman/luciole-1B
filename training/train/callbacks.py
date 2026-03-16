# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.timer import Timer

from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging
from nemo.utils.get_rank import get_rank
from nemo import lightning as nl
from datetime import timedelta
from nemo.utils.exp_manager import TimingCallback


def checkpoint_along_step_curve(
    global_step, intervals={1: 1, 50_000: 500, 100_000: 5_000}, else_interval=10_000
):
    for s, interval in intervals.items():
        if global_step <= s:
            return global_step % interval == 0
    return global_step % else_interval == 0


class CustomTimingCallback(TimingCallback):
    def __init__(
        self,
        log_tokens_per_sec: bool = False,
        timer_kwargs={},
        max_training_time_per_step=None,
    ):
        super().__init__(log_tokens_per_sec, timer_kwargs)
        self.max_training_time_per_step = max_training_time_per_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.max_training_time_per_step:
            should_stop = (
                self.timer["train_step_timing"] > self.max_training_time_per_step
            )
            if should_stop:
                logging.info(
                    f"Training step is too large: {self.timer['train_step_timing']} > {self.max_training_time_per_step}. Stopping training."
                )
            should_stop = trainer.strategy.broadcast(should_stop)
            trainer.should_stop = trainer.should_stop or should_stop


class StopAtEndOfPhaseCallback(Callback, IOMixin):
    def __init__(
        self,
        end_step: int = None,
    ):
        self.end_step = end_step

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        if self.end_step:
            should_stop = trainer.global_step >= self.end_step
            if should_stop:
                logging.info(
                    f"End of phase is reached at step: {trainer.global_step}. Stopping training."
                )
            should_stop = trainer.strategy.broadcast(should_stop)
            trainer.should_stop = trainer.should_stop or should_stop


class StopAfterThisRunMaxStepsCallback(Callback, IOMixin):
    """
    Stop training after N optimizer steps in the current job, regardless of resume step.

    This keeps a global LR horizon (trainer.max_steps / scheduler.max_steps) while allowing
    short per-job chunks for Slurm time limits.
    """

    def __init__(self, this_run_max_steps: int):
        self.this_run_max_steps = this_run_max_steps
        self._start_global_step = None

    def on_fit_start(self, trainer, pl_module) -> None:
        self._start_global_step = int(trainer.global_step)
        logging.info(
            "StopAfterThisRunMaxStepsCallback armed: start_global_step=%s, this_run_max_steps=%s, stop_at=%s",
            self._start_global_step,
            self.this_run_max_steps,
            self._start_global_step + self.this_run_max_steps,
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        if self.this_run_max_steps is None or self.this_run_max_steps <= 0:
            return
        if self._start_global_step is None:
            self._start_global_step = int(trainer.global_step)

        local_steps = int(trainer.global_step) - int(self._start_global_step)
        should_stop = local_steps >= int(self.this_run_max_steps)
        if should_stop:
            logging.info(
                "Reached this_run_max_steps=%s at global_step=%s (local_steps=%s). Stopping training.",
                self.this_run_max_steps,
                trainer.global_step,
                local_steps,
            )
        should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop


class ProgressiveIntervalCheckpoint(nl.ModelCheckpoint):
    def __init__(
        self,
        monitor: Optional[str] = "val_loss",
        verbose: bool = True,
        save_last: Optional[Union[bool, Literal["link"]]] = True,
        save_top_k: int = 3,
        save_weights_only: bool = False,  # TODO: check support
        mode: str = "min",
        every_n_epochs: int = None,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        # Save after training, not after validation
        save_on_train_epoch_end: Optional[bool] = False,
        save_optim_on_train_end: Optional[bool] = False,
        always_save_context: bool = True,
        save_context_on_train_end: bool = True,
        every_function_train_steps: Optional[Any] = None,
        **kwargs,
    ):
        self.every_function_train_steps = every_function_train_steps
        super().__init__(
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            every_n_epochs=every_n_epochs,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            save_on_train_epoch_end=save_on_train_epoch_end,
            save_optim_on_train_end=save_optim_on_train_end,
            always_save_context=always_save_context,
            save_context_on_train_end=save_context_on_train_end,
            **kwargs,
        )

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps`"""
        if self._should_skip_saving_checkpoint(trainer):
            return
        # skip_batch = self._every_n_train_steps < 1 or (trainer.global_step % self._every_n_train_steps != 0)
        if self.every_function_train_steps is not None:
            skip_batch = not self.every_function_train_steps(
                global_step=trainer.global_step
            )
        else:
            skip_batch = self._every_n_train_steps < 1 or (
                trainer.global_step % self._every_n_train_steps != 0
            )
        skip_batch = False if trainer.max_steps == trainer.global_step else skip_batch
        train_time_interval = self._train_time_interval
        skip_time = True
        now = time.monotonic()
        if train_time_interval:
            prev_time_check = self._last_time_checked
            skip_time = (
                prev_time_check is None
                or (now - prev_time_check) < train_time_interval.total_seconds()
            )
            # in case we have time differences across ranks
            # broadcast the decision on whether to checkpoint from rank 0 to avoid possible hangs
            skip_time = trainer.strategy.broadcast(skip_time)

        if skip_batch and skip_time:
            return
        if not skip_time:
            self._last_time_checked = now

        monitor_candidates = self._monitor_candidates(trainer)
        self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)


class StatelessTimer(Timer):
    """Extension of PTL timers to be per run."""

    # Override PTL Timer's state dict to not store elapsed time information so that we can
    # restore and continue training.
    def state_dict(self):
        """state_dict"""
        return {}

    def load_state_dict(self, state_dict) -> None:
        """load_state_dict"""
        return


def trace_handler(prof, chakra_device_trace_path):
    """
    Handles the export of profiling traces to a specified directory.

    Args:
        prof: The profiler object containing the trace data.
        chakra_device_trace_path: The path where the trace file will be saved.
    """
    rank = get_rank()
    trace_file = chakra_device_trace_path / f"rank-{rank}.json"
    prof.export_chrome_trace(str(trace_file))
    logging.info(f"Kineto trace saved: {trace_file}")


class PytorchProfilerCallback(Callback, IOMixin):
    """
    A PyTorch Lightning callback for profiling with PyTorch's built-in Profiler and ExecutionTraceObserver.

    This callback enables profiling for specific steps during training using PyTorch's profiler.
    It also captures detailed execution traces using `ExecutionTraceObserver`.
    It ensures proper cleanup, preventing memory leaks or duplicate profiling instances.

    Args:
        start_step (int): Global batch step to start profiling.
        end_step (int): Global batch step to end profiling.
        warmup_steps (int): Number of warmup steps before profiling starts.
        active_steps (int): Number of active profiling steps.
        trace_dir (str): Directory where traces will be saved.
        profiler_kwargs (dict, optional): Additional keyword args to pass to torch.profiler.profile
    """

    def __init__(
        self,
        start_step: int,
        end_step: int,
        warmup_steps: int = 0,
        active_steps: int = 1,
        trace_dir: str = None,
        profiler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if trace_dir is None:
            trace_dir = os.path.join(os.getcwd(), "traces")
            os.makedirs(trace_dir, exist_ok=True)

        if not isinstance(start_step, int) or not isinstance(end_step, int):
            raise TypeError(
                f"start_step and end_step must be integers. Got {type(start_step)}, {type(end_step)}"
            )
        if end_step < start_step:
            raise ValueError("end_step must be greater than or equal to start_step.")

        if not os.path.isdir(trace_dir):
            raise ValueError(f"Chakra trace output path ({trace_dir}) does not exist.")

        self.start_step = start_step
        self.end_step = end_step
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps

        self.trace_dir = Path(trace_dir)
        self.chakra_host_trace_path = self.trace_dir / "host"
        self.chakra_device_trace_path = self.trace_dir / "device"

        self.chakra_host_trace_path.mkdir(parents=True, exist_ok=True)
        self.chakra_device_trace_path.mkdir(parents=True, exist_ok=True)

        self.trace_observer = torch.profiler.ExecutionTraceObserver()

        base_kwargs = {
            "activities": [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            "schedule": torch.profiler.schedule(
                wait=0, warmup=self.warmup_steps, active=self.active_steps
            ),
            "on_trace_ready": lambda prof: trace_handler(
                prof, self.chakra_device_trace_path
            ),
            "execution_trace_observer": self.trace_observer,
        }
        if profiler_kwargs is not None:
            if not isinstance(profiler_kwargs, dict):
                raise TypeError(
                    f"profiler_kwargs must be a dict if provided. Got {type(profiler_kwargs)}"
                )
            base_kwargs.update(profiler_kwargs)

        self.profiler = torch.profiler.profile(**base_kwargs)
        self.is_profiling = False

        logging.info(
            "Chakra profiling initialized:\n"
            f" - Start Step: {self.start_step}\n"
            f" - End Step: {self.end_step}\n"
            f" - Warmup Steps: {self.warmup_steps}\n"
            f" - Active Steps: {self.active_steps}\n"
            f" - Trace Directory: {self.trace_dir}\n"
            f" - Extra profiler kwargs: {profiler_kwargs or {}}"
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        """Chakra trace collection starts."""
        if trainer.global_step == self.start_step:
            if self.is_profiling:
                logging.warning(
                    f"Attempted to start Chakra profiler multiple times at step {trainer.global_step}. Skipping."
                )
                return

            logging.info(
                f"====== Start Chakra profiling at global_step {trainer.global_step} ======"
            )

            trace_file = self.chakra_host_trace_path / f"rank-{get_rank()}.json"
            self.trace_observer.register_callback(str(trace_file))

            self.profiler.start()
            self.is_profiling = True

            logging.info("Chakra Profiler Started.\n")

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        """Chakra trace collection ends."""
        if self.is_profiling:
            if trainer.global_step < self.end_step:
                self.profiler.step()
                logging.info(
                    f"Profiler step executed at global_step {trainer.global_step}"
                )
            else:
                logging.info(
                    f"====== End Chakra profiling at global_step {trainer.global_step} ======"
                )
                self._stop_profiler()

    def _stop_profiler(self):
        if self.is_profiling:
            logging.info("Stopping Chakra Profiler...")
            self.profiler.stop()
            self.is_profiling = False

            try:
                logging.info("Unregistering ExecutionTraceObserver...")
                self.trace_observer.unregister_callback()
            except RuntimeError as e:
                logging.warning(f"ExecutionTraceObserver cleanup failed: {e}")


class WandbMetricsCallback(Callback, IOMixin):
    """
    Explicitly push training metrics (loss, lr, etc.) from trainer.callback_metrics
    to the W&B logger every N steps.  NeMo logs these to stdout but they may not
    flow through Lightning's logger interface to W&B.
    """

    def __init__(self, log_every_n_steps: int = 1):
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        global_step = trainer.global_step
        if global_step % self.log_every_n_steps != 0:
            return
        if get_rank() != 0:
            return

        wandb_logger = None
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError:
            from pytorch_lightning.loggers import WandbLogger
        if trainer.logger is not None:
            if isinstance(trainer.logger, WandbLogger):
                wandb_logger = trainer.logger
            elif hasattr(trainer.logger, '_loggers'):
                for lg in trainer.logger._loggers:
                    if isinstance(lg, WandbLogger):
                        wandb_logger = lg
                        break
        if wandb_logger is None:
            return

        metrics = {}
        cb = trainer.callback_metrics
        for key in ("reduced_train_loss", "lr", "global_step", "consumed_samples",
                     "train_loss", "grad_norm"):
            if key in cb:
                val = cb[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                metrics[key] = val

        # Also grab LR from the optimizer directly in case it's missing
        if "lr" not in metrics:
            try:
                opt = trainer.optimizers
                if isinstance(opt, list):
                    opt = opt[0]
                lr = opt.param_groups[0].get("lr")
                if lr is not None:
                    metrics["lr"] = lr
            except Exception:
                pass

        if metrics:
            wandb_logger.log_metrics(metrics, step=global_step)


class SSALoggingCallback(Callback, IOMixin):
    """
    Callback to log SSA n parameter values at specified intervals.

    Args:
        log_every_n_steps: Log SSA parameters every N training steps
    """

    def __init__(self, log_every_n_steps: int = 1000):
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        global_step = trainer.global_step

        # Only log at specified intervals and on rank 0
        if global_step % self.log_every_n_steps != 0:
            return
        if get_rank() != 0:
            return

        # Find SSA parameters in the model
        n_values = []
        try:
            # Navigate to the model's decoder layers
            model = pl_module
            if hasattr(model, 'module'):
                model = model.module
            if hasattr(model, 'model'):
                model = model.model

            # Collect all ssa_n_raw parameters
            for name, param in model.named_parameters():
                if 'ssa_n_raw' in name:
                    p = param.detach()
                    if p.numel() == 1:
                        n_values.append((name, float(p.item())))
                    else:
                        n_values.append((name, float(p.mean().item())))

            if n_values:
                logging.info(f"Step {global_step} - SSA n values:")
                import re
                wandb_metrics = {}
                for name, val in n_values:
                    # Extract layer number: module.decoder.layers.X.self_attention...
                    match = re.search(r'layers\.(\d+)\.', name)
                    layer_num = match.group(1) if match else '?'
                    logging.info(f"  Layer {layer_num}: n = {val:.6f}")
                    wandb_metrics[f"ssa_n/layer_{layer_num}"] = val

                # Log SSA n values to W&B as well
                if wandb_metrics and trainer.logger is not None:
                    try:
                        trainer.logger.log_metrics(wandb_metrics, step=global_step)
                    except Exception:
                        pass
        except Exception as e:
            logging.warning(f"Could not log SSA params: {e}")
