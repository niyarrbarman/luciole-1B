# Copyright (c) 2025, SSA Triton Attention v2 - Megatron-compatible module
# Drop-in replacement for SSADotProductAttention using fused Triton kernels.
#
# v2 optimizations:
#   1. No repeat_interleave — native GQA handled in Triton kernel
#   2. No permute+contiguous — zero-copy layout conversion via stride manipulation
#   3. No O(S^2) memory, single kernel launch per fwd/bwd, learnable n and b

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

from SSA.ssa_flash_attention import ssa_flash_attention


def _megatron_to_triton_view(t: Tensor, batch_size: int, num_heads: int) -> Tensor:
    """
    Convert Megatron layout [S, B, H, D] to Triton layout [B, H, S, D]
    via permute WITHOUT forcing contiguous (zero-copy).

    The Triton kernels use explicit strides, so they can handle the
    non-contiguous tensor just fine. This saves one full-tensor copy per
    Q/K/V in forward AND dQ/dK/dV in backward.

    If the tensor is already [B, H, S, D] contiguous, this is a no-op.
    """
    # Megatron gives us [S, B, H, D]
    # permute(1, 2, 0, 3) -> [B, H, S, D]
    return t.permute(1, 2, 0, 3)


def _triton_to_megatron_view(t: Tensor) -> Tensor:
    """
    Convert Triton layout [B, H, S, D] back to Megatron layout [S, B, H, D].
    Again, just a permute (zero-copy).

    NOTE: The output of the Triton kernel IS contiguous in [B,H,S,D] layout
    (it's written by the kernel), so this permute creates a non-contiguous view.
    The subsequent view() to [S, B, Hp] will call .contiguous() if needed,
    which is unavoidable regardless (we need to reshape for the linear proj).
    """
    # [B, H, S, D] -> [S, B, H, D]
    return t.permute(2, 0, 1, 3)


class SSATritonAttention(MegatronModule):
    """
    SSA attention using fused Triton FlashAttention kernels (v2).

    This is a drop-in replacement for SSADotProductAttention that fuses
    scale → causal mask → SSA transform → online softmax → V accumulation
    into a single tiled Triton kernel, eliminating:
      - O(S^2) attention matrix materialization
      - 8+ separate kernel launches
      - fp32 round-trip through HBM
      - GQA repeat_interleave memory copy (native GQA in kernel)
      - Layout permute+contiguous copies (stride-based access)

    SSA formula: softmax(n * sgn(s) * ln(1 + b|s|))
    where s = scale * Q @ K^T
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
        # SSA-specific parameters
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

        # Validate constraints
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

        # GQA ratio
        self.gqa_ratio = self.num_attention_heads_per_partition // self.num_query_groups_per_partition

        # Softmax scaling
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

        # Dropout (applied AFTER the fused kernel, on the output)
        dropout_rate = self.config.attention_dropout if attention_dropout is None else attention_dropout
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.dropout_p = dropout_rate

        if self.dropout_p > 0.0:
            import warnings
            warnings.warn(
                "SSATritonAttention applies dropout to output (post-V-matmul), "
                "unlike baseline SSA which applies to attention probs (pre-V-matmul). "
                "Set attention_dropout=0.0 for exact parity.",
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
        Forward pass using fused Triton SSA FlashAttention with native GQA.

        Args:
            query: [sq, b, np, hn]  (Megatron format, np = num_query_heads)
            key:   [sk, b, ng, hn]  (ng = num_kv_groups, no GQA expansion needed)
            value: [sk, b, ng, hn]
            attention_mask: ignored (causal mask handled in kernel)
            attn_mask_type: override
            attention_bias: not supported
            packed_seq_params: not supported

        Returns:
            context: [sq, b, hp]
        """
        assert packed_seq_params is None, (
            "Packed sequence is not supported by SSATritonAttention. "
            "Use TEDotProductAttention instead."
        )
        assert attention_bias is None, "Attention bias is not supported for SSATritonAttention."

        sq, batch_size, num_heads_q, head_dim = query.shape
        sk = key.shape[0]
        num_heads_kv = key.shape[2]

        # Convert Megatron [seq, batch, heads, dim] -> Triton [batch, heads, seq, dim]
        # via permute only (zero-copy — no .contiguous() call)
        query_t = _megatron_to_triton_view(query, batch_size, num_heads_q)     # [b, np, sq, hn]
        key_t   = _megatron_to_triton_view(key, batch_size, num_heads_kv)       # [b, ng, sk, hn]
        value_t = _megatron_to_triton_view(value, batch_size, num_heads_kv)     # [b, ng, sk, hn]

        # Get SSA params
        ssa_n, ssa_b = self.get_ssa_params()

        # Determine if causal
        is_causal = (self.attn_mask_type == AttnMaskType.causal)
        if attn_mask_type is not None:
            is_causal = (attn_mask_type == AttnMaskType.causal)

        # Fused SSA Flash Attention — native GQA, no repeat_interleave
        # K/V have num_heads_kv heads, kernel handles GQA ratio internally
        context = ssa_flash_attention(
            query_t, key_t, value_t,
            softmax_scale=self.softmax_scale,
            ssa_n=ssa_n,
            ssa_b=ssa_b,
            causal=is_causal,
            dropout_p=self.dropout_p,
            training=self.training,
        )

        # Apply dropout on output (minimal overhead, not on S^2 matrix)
        if self.training and self.dropout_p > 0:
            if not self.config.sequence_parallel:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    context = self.attention_dropout(context)
            else:
                context = self.attention_dropout(context)

        # Convert back: [b, np, sq, hn] -> [sq, b, np, hn]
        context = _triton_to_megatron_view(context)

        # Reshape: [sq, b, np, hn] -> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.reshape(*new_context_shape)

        return context
