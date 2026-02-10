# Copyright (c) 2024, Custom SSA Implementation
# SSA: Softmax-Substituted Attention
#
# This module implements a custom DotProductAttention with a modified softmax
# transformation based on the SSA formula:
#     SSA(x) = softmax(n * sgn(x) * ln(1 + b|x|))
#
# where n >= 1 and b > 0 are learnable or fixed parameters.

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import attention_mask_func, get_default_causal_mask
from megatron.core.utils import divide



class SSAScaleMaskSoftmax(torch.nn.Module):
    """
    Custom fused scale-mask-softmax with SSA transformation.
    
    SSA formula: softmax(n * sgn(x) * ln(1 + b|x|))
    
    Parameters:
        n (learnable): Controls sharpness, constrained to n >= 1
        b (learnable): Scales input magnitude, constrained to b > 0
    """

    def __init__(
        self,
        input_in_fp16: bool,
        input_in_bf16: bool,
        attn_mask_type: AttnMaskType,
        softmax_in_fp32: bool,
        scale: Optional[float],
        # SSA parameters
        ssa_n_init: float = 1.5,
        ssa_b_init: float = 0.8,
        learnable_ssa: bool = True,
        learnable_b: bool = False,  # Set to False to fix b
    ):
        super().__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        self.input_in_float16 = input_in_fp16 or input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        self.learnable_ssa = learnable_ssa
        
        # SSA transformation parameters with proper constraints:
        self.learnable_b = learnable_b
        
        if learnable_ssa:
            n_raw_init = ssa_n_init
            self.ssa_n_raw = torch.nn.Parameter(torch.tensor(n_raw_init))
            
            if learnable_b:
                b_raw_init = ssa_b_init
                self.ssa_b_raw = torch.nn.Parameter(torch.tensor(b_raw_init))
            else:
                # Fixed b as a buffer (not trained)
                self.register_buffer('ssa_b', torch.tensor(ssa_b_init))
        else:
            # Fixed parameters (not learnable)
            self.register_buffer('ssa_n', torch.tensor(ssa_n_init))
            self.register_buffer('ssa_b', torch.tensor(ssa_b_init))

    def get_ssa_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current n and b values with constraints applied."""
        if self.learnable_ssa:
            n = self.ssa_n_raw 
            if self.learnable_b:
                b = self.ssa_b_raw     
            else:
                b = self.ssa_b
        else:
            n, b = self.ssa_n, self.ssa_b
        return n, b

    def ssa_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SSA transformation: n * sgn(x) * ln(1 + b|x|)
        
        This replaces the raw attention scores before softmax.
        The function is continuous and differentiable everywhere.
        """
        # print("doing ssa transform")
        n, b = self.get_ssa_params()
        
        abs_x = torch.abs(x)
        sign_x = torch.sign(x)
        log_term = torch.log1p(b * abs_x)  # ln(1 + b|x|), numerically stable
        
        return n * sign_x * log_term

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Forward pass with SSA-transformed softmax."""
        assert input.dim() == 4, f"Expected 4D tensor, got {input.dim()}D"
        
        return self._forward_ssa_softmax(input, mask)

    def _forward_ssa_softmax(self, input: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        SSA softmax implementation.
        
        Standard attention: softmax(scale * x)
        SSA attention: softmax(n * sgn(scale * x) * ln(1 + b|scale * x|))
        """
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        # Apply scale
        if self.scale is not None:
            input = input * self.scale

        # Generate causal mask if needed
        sq, sk = input.size(2), input.size(3)
        if self.attn_mask_type == AttnMaskType.causal and mask is None and sq > 1:
            assert sq == sk, "Causal mask requires sq == sk"
            mask = get_default_causal_mask(sq)

        # Apply attention mask (sets masked positions to -inf)
        if mask is not None:
            mask_output = attention_mask_func(input, mask)
        else:
            mask_output = input

        # ============================================================
        # SSA TRANSFORMATION
        # 
        # Transform attention scores before softmax:
        #   x -> n * sgn(x) * ln(1 + b|x|)
        #
        # Note: Masked positions are -inf, and sgn(-inf) * ln(1 + b*inf) 
        # = -1 * inf = -inf, so masking is preserved.
        # ============================================================
        # Apply softmax
        probs = torch.nn.functional.softmax(self.ssa_transform(mask_output), dim=-1)

        # Cast back to input dtype if needed
        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs


class SSADotProductAttention(MegatronModule):
    """
    SSA (Softmax-Substituted Attention) DotProductAttention.
    
    This is a drop-in replacement for Megatron's DotProductAttention that uses
    a custom softmax transformation for the attention mechanism.
    
    SSA formula: softmax(n * sgn(x) * ln(1 + b|x|))
    
    where:
        - n >= 1 (learnable, controls attention sharpness)
        - b > 0  (learnable, scales input magnitude)
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

        # Validate constraints (same as native DotProductAttention)
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

        coeff = None
        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

        # SSA custom softmax (replaces FusedScaleMaskSoftmax)
        self.scale_mask_softmax = SSAScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
            ssa_n_init=ssa_n,
            ssa_b_init=ssa_b,
            learnable_ssa=learnable_ssa,
        )

        # Dropout
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

    def get_ssa_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current SSA parameters n and b."""
        return self.scale_mask_softmax.get_ssa_params()

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
        Forward pass for SSA attention.
        
        Args:
            query: [sq, b, np, hn] Query tensor
            key: [sk, b, ng, hn] Key tensor
            value: [sk, b, ng, hn] Value tensor
            attention_mask: Attention mask tensor
            attn_mask_type: Override for attention mask type
            attention_bias: Optional attention bias (not supported)
            packed_seq_params: Packed sequence params (not supported)
        
        Returns:
            context: [sq, b, hp] Context tensor after attention
        """
        assert packed_seq_params is None, (
            "Packed sequence is not supported by SSADotProductAttention. "
            "Use TEDotProductAttention instead."
        )
        assert attention_bias is None, "Attention bias is not supported for SSADotProductAttention."

        # Expand key/value for grouped query attention
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        # Compute attention scores: [b, np, sq, sk]
        output_size = (query.size(1), query.size(2), query.size(0), key.size(0))

        # Reshape for batched matmul
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # Preallocate buffer
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu"
        )

        # Q @ K^T with scaling: [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),       # [b * np, sq, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=self.softmax_scale,
        )

        # Reshape to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # Apply SSA softmax (the key difference from standard attention)
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

        # Apply dropout
        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # Compute context: attention_probs @ V
        output_size = (value.size(1), value.size(2), query.size(0), value.size(3))
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # Reshape to [b, np, sq, hn]
        context = context.view(*output_size)

        # [b, np, sq, hn] -> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] -> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context
