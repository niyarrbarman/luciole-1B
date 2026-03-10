# SSA Layer Specifications
# Custom layer specs that use SSADotProductAttention instead of the default attention

from typing import Optional

from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

# Import our custom SSA attention
import sys
from pathlib import Path

# Ensure SSA module is importable
SSA_DIR = Path(__file__).resolve().parent
TEST_DIR = SSA_DIR.parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from SSA.ssa_attention import SSADotProductAttention


def get_bias_dropout_add(training, fused_bias_dropout_add):
    """
    Return the bias_dropout_add function.
    
    This is the proper function factory expected by TransformerLayer.
    For simplicity, we return a basic implementation that just does:
    output = bias + residual + dropout(input)
    """
    import torch
    
    def bias_dropout_add(x_with_bias, residual, prob):
        """Apply bias, dropout, and residual connection."""
        x, bias = x_with_bias  # Unpack tuple
        if bias is not None:
            x = x + bias
        # During training, apply dropout
        if training and prob > 0.0:
            x = torch.nn.functional.dropout(x, p=prob, training=True)
        return x + residual
    
    return bias_dropout_add


def get_ssa_gpt_layer_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
    ssa_n: float = 1.0,
    ssa_b: float = 1.0,
    learnable_ssa: bool = True,
) -> ModuleSpec:
    """
    Get a GPT layer spec that uses SSADotProductAttention.
    
    This creates a layer specification with our custom SSA attention module
    instead of the standard DotProductAttention or TEDotProductAttention.
    
    Args:
        num_experts: Number of experts for MoE (None for dense model)
        moe_grouped_gemm: Whether to use grouped GEMM for MoE
        qk_layernorm: Whether to use QK layer normalization
        ssa_n: SSA transformation parameter n (initial value, n >= 1)
        ssa_b: SSA transformation parameter b (initial value, b > 0)
        learnable_ssa: If True, n and b are learnable parameters. If False, they are fixed.
    
    Returns:
        ModuleSpec for TransformerLayer with SSA attention
    """
    # Use native Megatron layers (not Transformer Engine) for predictable behavior
    # This ensures we use our custom softmax implementation
    mlp = _get_mlp_module_spec(num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm)
    
    # Create ModuleSpec for SSADotProductAttention with params
    # The params dict will be passed to the constructor
    ssa_core_attention_spec = ModuleSpec(
        module=SSADotProductAttention,
        params={"ssa_n": ssa_n, "ssa_b": ssa_b, "learnable_ssa": learnable_ssa},
    )
    
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=FusedLayerNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=ssa_core_attention_spec,
                    linear_proj=RowParallelLinear,
                    q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                    k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,  # Function factory, not a class
            pre_mlp_layernorm=FusedLayerNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,  # Function factory, not a class
        ),
    )


def _get_mlp_module_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
) -> ModuleSpec:
    """Get MLP module spec (handles both dense and MoE)."""
    if num_experts is None:
        # Dense MLP
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
            ),
        )
    else:
        # MoE MLP - use Megatron's MoE implementation
        from megatron.core.transformer.moe.moe_layer import MoELayer
        
        if moe_grouped_gemm:
            from megatron.core.transformer.moe.experts import GroupedMLP
            return ModuleSpec(module=MoELayer, submodules=GroupedMLP)
        else:
            from megatron.core.transformer.moe.experts import SequentialMLP
            return ModuleSpec(module=MoELayer, submodules=SequentialMLP)
