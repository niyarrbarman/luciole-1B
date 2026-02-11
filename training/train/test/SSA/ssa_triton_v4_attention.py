# Copyright (c) 2025, SSA Triton Attention v4 - Megatron-compatible module
# Uses the v4 kernel (tutorial-based structure with stage-based causal,
# BLK_SLICE_FACTOR diagonal optimization, and autotuned forward).

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide

from SSA.ssa_flash_attention_v4 import ssa_flash_attention_v4


def _megatron_to_triton_view(t: Tensor, batch_size: int, num_heads: int) -> Tensor:
    """Convert Megatron [S, B, H, D] to Triton [B, H, S, D] via permute (zero-copy)."""
    return t.permute(1, 2, 0, 3)


def _triton_to_megatron_view(t: Tensor) -> Tensor:
    """Convert Triton [B, H, S, D] back to Megatron [S, B, H, D] (zero-copy)."""
    return t.permute(2, 0, 1, 3)


class SSATritonV4Attention(MegatronModule):
    """
    SSA attention using v4 Triton kernels (tutorial-based structure).

    Drop-in replacement for SSADotProductAttention/SSATritonAttention.
    Uses the Flash Attention v2 tutorial structure with SSA integrated:
      - Stage-based causal (off-band vs on-band)
      - BLK_SLICE_FACTOR diagonal optimization
      - Autotuned forward kernel
      - Separate dQ and dKV backward kernels
      - Native GQA
      - Learnable n, b with Kahan-compensated gradients

    SSA formula: w(s) = (1 + b|s|)^(n*sign(s)), P = w/sum(w)
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        attention_type: str = "self",
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: str = None,
        # SSA parameters
        ssa_n: float = 1.5,
        ssa_b: float = 0.8,
        learnable_ssa: bool = True,
        learnable_b: bool = False,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.learnable_ssa = learnable_ssa
        self.learnable_b = learnable_b

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

        self.gqa_ratio = self.num_attention_heads_per_partition // self.num_query_groups_per_partition

        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number

        # SSA parameters
        if learnable_ssa:
            self.ssa_n_raw = nn.Parameter(torch.tensor(float(ssa_n)))
            if learnable_b:
                self.ssa_b_raw = nn.Parameter(torch.tensor(float(ssa_b)))
            else:
                self.register_buffer('ssa_b', torch.tensor(float(ssa_b)))
        else:
            self.register_buffer('ssa_n', torch.tensor(float(ssa_n)))
            self.register_buffer('ssa_b', torch.tensor(float(ssa_b)))

        # Dropout (applied after fused kernel on output)
        dropout_rate = self.config.attention_dropout if attention_dropout is None else attention_dropout
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.dropout_p = dropout_rate

        if self.dropout_p > 0.0:
            import warnings
            warnings.warn(
                "SSATritonV4Attention applies dropout to output (post-V-matmul), "
                "not to attention probs. Set attention_dropout=0.0 for exact parity.",
                stacklevel=2,
            )

    def get_ssa_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current SSA parameters n and b."""
        if self.learnable_ssa:
            n = self.ssa_n_raw
            b = self.ssa_b_raw if self.learnable_b else self.ssa_b
        else:
            n, b = self.ssa_n, self.ssa_b
        return n, b

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
        Forward pass using v4 fused Triton SSA FlashAttention.

        Args:
            query: [sq, b, np, hn]  (Megatron format)
            key:   [sk, b, ng, hn]
            value: [sk, b, ng, hn]
            attention_mask: ignored (causal mask handled in kernel)

        Returns:
            context: [sq, b, hp]
        """
        assert packed_seq_params is None, (
            "Packed sequence not supported by SSATritonV4Attention."
        )
        assert attention_bias is None, "Attention bias not supported."

        sq, batch_size, num_heads_q, head_dim = query.shape
        num_heads_kv = key.shape[2]

        # Megatron [S, B, H, D] -> Triton [B, H, S, D] (zero-copy permute)
        query_t = _megatron_to_triton_view(query, batch_size, num_heads_q)
        key_t = _megatron_to_triton_view(key, batch_size, num_heads_kv)
        value_t = _megatron_to_triton_view(value, batch_size, num_heads_kv)

        ssa_n, ssa_b = self.get_ssa_params()

        is_causal = (self.attn_mask_type == AttnMaskType.causal)
        if attn_mask_type is not None:
            is_causal = (attn_mask_type == AttnMaskType.causal)

        context = ssa_flash_attention_v4(
            query_t, key_t, value_t,
            softmax_scale=self.softmax_scale,
            ssa_n=ssa_n,
            ssa_b=ssa_b,
            causal=is_causal,
            dropout_p=self.dropout_p,
            training=self.training,
        )

        # Dropout on output
        if self.training and self.dropout_p > 0:
            if not self.config.sequence_parallel:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    context = self.attention_dropout(context)
            else:
                context = self.attention_dropout(context)

        # [B, H, S, D] -> [S, B, H, D]
        context = _triton_to_megatron_view(context)

        # [S, B, H, D] -> [S, B, Hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.reshape(*new_context_shape)

        return context
