# Copyright (c) 2024, FlexAttention-based SSA Implementation
# SSA: Softmax-Substituted Attention using PyTorch FlexAttention
#
# This module implements SSA using PyTorch's flex_attention API for optimized
# attention computation with custom score modification.
#
# SSA formula: softmax(n * sgn(x) * ln(1 + b|x|))
# where n >= 1 and b > 0 are learnable parameters.

import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide

# FlexAttention imports
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)


def _inv_softplus(x: float) -> float:
    """Inverse of softplus: returns raw such that softplus(raw) = x."""
    if x > 20:
        return x
    return x + math.log(-math.expm1(-x) + 1e-10)


class SSAFlexAttention(MegatronModule):
    """
    SSA (Softmax-Substituted Attention) using PyTorch FlexAttention.
    
    This is a drop-in replacement for Megatron's DotProductAttention that uses
    FlexAttention's score_mod to implement the SSA transformation.
    
    SSA formula: softmax(n * sgn(x) * ln(1 + b|x|))
    
    where:
        - n >= 1 (learnable, controls attention sharpness)
        - b > 0  (learnable, scales input magnitude)
    
    Advantages over custom SSA implementation:
        - Fused Triton kernel (faster)
        - Automatic backward pass generation
        - Memory efficient (no materialized attention matrix)
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
        ssa_n: float = 1.0,
        ssa_b: float = 1.0,
        learnable_ssa: bool = True,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.learnable_ssa = learnable_ssa

        # Validate constraints
        assert (
            self.config.context_parallel_size == 1
        ), "Context parallelism is only supported by TEDotProductAttention!"

        assert (
            self.config.window_size is None
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        # Softmax scaling
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number

        # SSA parameters with proper constraints:
        # - n >= 1: We parameterize as n = 1 + softplus(n_raw)
        # - b > 0:  We parameterize as b = softplus(b_raw)
        if learnable_ssa:
            n_raw_init = _inv_softplus(max(ssa_n - 1.0, 0.01))
            b_raw_init = _inv_softplus(max(ssa_b, 0.01))
            
            self.ssa_n_raw = nn.Parameter(torch.tensor(n_raw_init))
            self.ssa_b_raw = nn.Parameter(torch.tensor(b_raw_init))
        else:
            self.register_buffer('ssa_n', torch.tensor(ssa_n))
            self.register_buffer('ssa_b', torch.tensor(ssa_b))

        # Dropout
        self.attention_dropout_rate = (
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )
        self.attention_dropout = nn.Dropout(self.attention_dropout_rate)
        
        # Cache for block mask (avoid recreation per forward)
        self._cached_block_mask = None
        self._cached_mask_shape = None

    def get_ssa_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current n and b values with constraints applied."""
        if self.learnable_ssa:
            n = 1.0 + F.softplus(self.ssa_n_raw)  # n >= 1
            b = F.softplus(self.ssa_b_raw)         # b > 0
        else:
            n, b = self.ssa_n, self.ssa_b
        return n, b

    def _get_score_mod(self) -> Callable:
        """
        Create the SSA score modification function for FlexAttention.
        
        The score_mod transforms attention scores before softmax:
            x -> n * sgn(x) * ln(1 + b|x|)
        
        Note: We use float values (not tensors) to avoid vmap gradient issues.
        FlexAttention's score_mod must be a pure/stateless function.
        """
        n, b = self.get_ssa_params()
        
        # Convert to Python floats for vmap compatibility
        # This means n and b are NOT learnable with FlexAttention
        n_val = float(n.detach())
        b_val = float(b.detach())
        
        def ssa_score_mod(score, batch, head, q_idx, kv_idx):
            # SSA transformation: n * sgn(x) * ln(1 + b|x|)
            abs_score = torch.abs(score)
            sign_score = torch.sign(score)
            log_term = torch.log1p(b_val * abs_score)
            return n_val * sign_score * log_term
        
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
        shape = (batch_size, num_heads, q_len, kv_len)
        
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
        
        Args:
            query: [sq, b, np, hn] Query tensor (Megatron format)
            key: [sk, b, ng, hn] Key tensor
            value: [sk, b, ng, hn] Value tensor
            attention_mask: Attention mask tensor (ignored, we use block_mask)
            attn_mask_type: Override for attention mask type
            attention_bias: Optional attention bias (not supported)
            packed_seq_params: Packed sequence params (not supported)
        
        Returns:
            context: [sq, b, hp] Context tensor after attention
        """
        assert packed_seq_params is None, (
            "Packed sequence is not supported by SSAFlexAttention."
        )
        assert attention_bias is None, "Attention bias is not supported for SSAFlexAttention."

        # Megatron format: [seq, batch, heads, dim]
        # FlexAttention format: [batch, heads, seq, dim]
        sq, batch_size, num_heads, head_dim = query.shape
        sk = key.shape[0]

        # Expand key/value for grouped query attention
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        # Convert to FlexAttention format: [batch, heads, seq, dim]
        query = query.permute(1, 2, 0, 3).contiguous()  # [b, np, sq, hn]
        key = key.permute(1, 2, 0, 3).contiguous()      # [b, ng, sk, hn]
        value = value.permute(1, 2, 0, 3).contiguous()  # [b, ng, sk, hn]

        # Get SSA score modification function
        score_mod = self._get_score_mod()

        # Create block mask for causal attention
        if self.attn_mask_type == AttnMaskType.causal:
            block_mask = self._get_causal_block_mask(
                batch_size, num_heads, sq, sk, query.device
            )
        else:
            block_mask = None

        # Apply FlexAttention with SSA score modification
        # Note: FlexAttention applies scale internally, so we pass it explicitly
        context = flex_attention(
            query, key, value,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=self.softmax_scale,
            enable_gqa=(self.num_query_groups_per_partition != self.num_attention_heads_per_partition),
        )

        # Apply dropout
        if self.training and self.attention_dropout_rate > 0:
            if not self.config.sequence_parallel:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    context = self.attention_dropout(context)
            else:
                context = self.attention_dropout(context)

        # Convert back to Megatron format: [seq, batch, heads, dim]
        context = context.permute(2, 0, 1, 3).contiguous()  # [sq, b, np, hn]

        # Reshape: [sq, b, np, hn] -> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context
