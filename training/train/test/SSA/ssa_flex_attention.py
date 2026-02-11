# Copyright (c) 2024, FlexAttention-based SSA Implementation
# SSA: Softmax-Substituted Attention using PyTorch FlexAttention

import inspect
import math
import warnings
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide

from torch.nn.attention.flex_attention import create_block_mask, flex_attention


class SSAFlexAttention(MegatronModule):
    """SSA attention using PyTorch FlexAttention with trainable score_mod tensors."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        attention_type: str = "self",
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: str = None,
        # SSA-specific parameters
        ssa_n: float = 1.5,
        ssa_b: float = 0.8,
        learnable_ssa: bool = True,
        learnable_b: bool = False,
        # FlexAttention controls
        use_torch_compile: bool = True,
        torch_compile_mode: str = "max-autotune-no-cudagraphs",
        flex_backend: str = "AUTO",
        force_fp32_score_mod: bool = False,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.learnable_ssa = learnable_ssa
        self.learnable_b = learnable_b
        self.force_fp32_score_mod = force_fp32_score_mod

        assert (
            self.config.context_parallel_size == 1
        ), "Context parallelism is only supported by TEDotProductAttention!"
        assert (
            self.config.window_size is None
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        projection_size = self.config.kv_channels * self.config.num_attention_heads
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        self.enable_gqa = self.num_query_groups_per_partition != self.num_attention_heads_per_partition

        if softmax_scale is None:
            base_softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            base_softmax_scale = softmax_scale

        # Match baseline SSA semantics: query-key layer scaling cancels out.
        self.softmax_scale = base_softmax_scale

        if learnable_ssa:
            self.ssa_n_raw = nn.Parameter(torch.tensor(float(ssa_n)))
            if learnable_b:
                self.ssa_b_raw = nn.Parameter(torch.tensor(float(ssa_b)))
            else:
                self.register_buffer("ssa_b", torch.tensor(float(ssa_b)))
        else:
            self.register_buffer("ssa_n", torch.tensor(float(ssa_n)))
            self.register_buffer("ssa_b", torch.tensor(float(ssa_b)))

        self.attention_dropout_rate = (
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )
        self.attention_dropout = nn.Dropout(self.attention_dropout_rate)
        if self.attention_dropout_rate > 0.0:
            warnings.warn(
                "SSAFlexAttention applies dropout on output context, not attention probabilities. "
                "Set attention_dropout=0.0 for strict parity with baseline SSA.",
                stacklevel=2,
            )

        self._cached_block_mask = None
        self._cached_mask_shape = None

        sig = inspect.signature(flex_attention)
        self._supports_enable_gqa = "enable_gqa" in sig.parameters
        self._supports_kernel_options = "kernel_options" in sig.parameters

        backend = (flex_backend or "AUTO").upper()
        if backend not in {"AUTO", "TRITON", "FLASH", "TRITON_DECODE"}:
            raise ValueError(f"Unsupported flex backend '{flex_backend}'.")
        self._flex_kernel_options = {"BACKEND": backend}

        self._score_mod = self._build_score_mod()
        self._use_torch_compile = bool(use_torch_compile and hasattr(torch, "compile"))
        self._torch_compile_mode = torch_compile_mode
        self._compiled_flex_call = None
        self._compile_failed = False
        self._disable_kernel_options_runtime = False

    def get_ssa_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current SSA parameters n and b."""
        if self.learnable_ssa:
            n = self.ssa_n_raw
            b = self.ssa_b_raw if self.learnable_b else self.ssa_b
        else:
            n, b = self.ssa_n, self.ssa_b
        return n, b

    def _build_score_mod(self) -> Callable:
        """
        Return a score_mod that captures trainable SSA tensors directly.

        This keeps gradient flow to n/b, unlike converting them to Python floats.
        """

        def ssa_score_mod(score, batch, head, q_idx, kv_idx):
            n, b = self.get_ssa_params()
            if self.force_fp32_score_mod:
                score_f = score.float()
                n_f = n.float()
                b_f = b.float()
                out_f = n_f * torch.sign(score_f) * torch.log1p(b_f * torch.abs(score_f))
                return out_f.to(score.dtype)

            n_t = n.to(dtype=score.dtype)
            b_t = b.to(dtype=score.dtype)
            return n_t * torch.sign(score) * torch.log1p(b_t * torch.abs(score))

        return ssa_score_mod

    def _get_causal_block_mask(
        self,
        batch_size: int,
        num_heads: int,
        q_len: int,
        kv_len: int,
        device: torch.device,
    ):
        """Create or retrieve cached causal block mask."""
        shape = (batch_size, num_heads, q_len, kv_len, device.type, device.index)
        if self._cached_block_mask is not None and self._cached_mask_shape == shape:
            return self._cached_block_mask

        def causal_mask_fn(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask_fn,
            B=batch_size,
            H=num_heads,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )
        self._cached_block_mask = block_mask
        self._cached_mask_shape = shape
        return block_mask

    def _flex_call_eager(self, query: Tensor, key: Tensor, value: Tensor, block_mask):
        kwargs = {
            "score_mod": self._score_mod,
            "block_mask": block_mask,
            "scale": self.softmax_scale,
        }
        if self._supports_enable_gqa:
            kwargs["enable_gqa"] = self.enable_gqa
        if self._supports_kernel_options and not self._disable_kernel_options_runtime:
            kwargs["kernel_options"] = self._flex_kernel_options
        try:
            return flex_attention(query, key, value, **kwargs)
        except (TypeError, ValueError, RuntimeError) as exc:
            if "kernel_options" in kwargs and (
                "kernel_options" in str(exc) or "BACKEND" in str(exc)
            ):
                self._disable_kernel_options_runtime = True
                kwargs.pop("kernel_options", None)
                return flex_attention(query, key, value, **kwargs)
            raise

    def _get_flex_call(self):
        if self._compiled_flex_call is not None:
            return self._compiled_flex_call
        if not self._use_torch_compile or self._compile_failed:
            self._compiled_flex_call = self._flex_call_eager
            return self._compiled_flex_call

        try:
            self._compiled_flex_call = torch.compile(
                self._flex_call_eager,
                mode=self._torch_compile_mode,
            )
        except Exception:
            self._compile_failed = True
            self._compiled_flex_call = self._flex_call_eager
        return self._compiled_flex_call

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """
        Forward pass for SSA attention using FlexAttention.

        Inputs are Megatron layout [S, B, H, D]. FlexAttention uses [B, H, S, D].
        """
        assert packed_seq_params is None, "Packed sequence is not supported by SSAFlexAttention."
        assert attention_bias is None, "Attention bias is not supported for SSAFlexAttention."

        sq, batch_size, num_heads_q, _ = query.shape
        sk = key.shape[0]

        query = query.permute(1, 2, 0, 3).contiguous()  # [B, Hq, Sq, D]
        key = key.permute(1, 2, 0, 3).contiguous()      # [B, Hkv, Sk, D]
        value = value.permute(1, 2, 0, 3).contiguous()  # [B, Hkv, Sk, D]

        is_causal = self.attn_mask_type == AttnMaskType.causal
        if attn_mask_type is not None:
            is_causal = attn_mask_type == AttnMaskType.causal

        if is_causal:
            block_mask = self._get_causal_block_mask(
                batch_size=batch_size,
                num_heads=num_heads_q,
                q_len=sq,
                kv_len=sk,
                device=query.device,
            )
        else:
            block_mask = None

        if self.enable_gqa and not self._supports_enable_gqa:
            repeat_factor = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        flex_call = self._get_flex_call()
        context = flex_call(query, key, value, block_mask)

        if self.training and self.attention_dropout_rate > 0:
            if not self.config.sequence_parallel:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    context = self.attention_dropout(context)
            else:
                context = self.attention_dropout(context)

        context = context.permute(2, 0, 1, 3).contiguous()  # [Sq, B, Hq, D]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)
        return context
