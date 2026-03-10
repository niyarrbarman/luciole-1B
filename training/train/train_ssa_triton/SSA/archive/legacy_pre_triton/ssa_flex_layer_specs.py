# SSA FlexAttention Layer Specifications
# Custom layer specs that use SSAFlexAttention instead of the default attention

import os
from typing import Optional

from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

# Import our FlexAttention-based SSA attention
import sys
from pathlib import Path

# Ensure SSA module is importable
SSA_DIR = Path(__file__).resolve().parent
TEST_DIR = SSA_DIR.parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from SSA.ssa_flex_attention import SSAFlexAttention


def get_bias_dropout_add(training, fused_bias_dropout_add):
    """
    Return the bias_dropout_add function.
    """
    import torch

    def bias_dropout_add(x_with_bias, residual, prob):
        x, bias = x_with_bias
        if bias is not None:
            x = x + bias
        if training and prob > 0.0:
            x = torch.nn.functional.dropout(x, p=prob, training=True)
        return x + residual

    return bias_dropout_add


def get_ssa_flex_gpt_layer_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
    ssa_n: float = 1.5,
    ssa_b: float = 0.8,
    learnable_ssa: bool = True,
    learnable_b: bool = False,
    use_torch_compile: Optional[bool] = None,
    torch_compile_mode: str = "max-autotune-no-cudagraphs",
    flex_backend: str = "AUTO",
    force_fp32_score_mod: bool = False,
    use_compiled_bda: Optional[bool] = None,
) -> ModuleSpec:
    """
    Get a GPT layer spec that uses SSAFlexAttention (FlexAttention-based SSA).

    This creates a layer specification with our FlexAttention-based SSA module
    which uses PyTorch's flex_attention API for optimized attention with
    custom score modification.

    Args:
        num_experts: Number of experts for MoE (None for dense model)
        moe_grouped_gemm: Whether to use grouped GEMM for MoE
        qk_layernorm: Whether to use QK layer normalization
        ssa_n: SSA transformation parameter n (initial value)
        ssa_b: SSA transformation parameter b (initial value)
        learnable_ssa: If True, n is learnable. If False, n is fixed.
        learnable_b: If True, b is learnable. If False, b stays fixed.
        use_torch_compile: If True, compile FlexAttention call path with torch.compile.
        torch_compile_mode: torch.compile mode for FlexAttention call path.
        flex_backend: FlexAttention backend hint (AUTO/TRITON/FLASH/TRITON_DECODE).
        force_fp32_score_mod: Evaluate SSA score_mod in fp32.
        use_compiled_bda: If True, use torch.compile'd bias-dropout-add.

    Returns:
        ModuleSpec for TransformerLayer with FlexAttention SSA
    """
    if use_torch_compile is None:
        use_torch_compile = os.environ.get("SSA_FLEX_TORCH_COMPILE", "1") != "0"
    if use_compiled_bda is None:
        use_compiled_bda = os.environ.get("SSA_FLEX_COMPILE_BDA", "1") != "0"

    bda_factory = get_compiled_bias_dropout_add if use_compiled_bda else get_bias_dropout_add
    mlp = _get_mlp_module_spec(num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm)

    # Create ModuleSpec for SSAFlexAttention with params
    ssa_flex_core_attention_spec = ModuleSpec(
        module=SSAFlexAttention,
        params={
            "ssa_n": ssa_n,
            "ssa_b": ssa_b,
            "learnable_ssa": learnable_ssa,
            "learnable_b": learnable_b,
            "use_torch_compile": use_torch_compile,
            "torch_compile_mode": torch_compile_mode,
            "flex_backend": flex_backend,
            "force_fp32_score_mod": force_fp32_score_mod,
        },
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
                    core_attention=ssa_flex_core_attention_spec,
                    linear_proj=RowParallelLinear,
                    q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                    k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=bda_factory,
            pre_mlp_layernorm=FusedLayerNorm,
            mlp=mlp,
            mlp_bda=bda_factory,
        ),
    )


# torch.compile'd version of bias_dropout_add for reduced kernel launch overhead
_compiled_bda_cache = {}


def get_compiled_bias_dropout_add(training, fused_bias_dropout_add):
    """
    Same as get_bias_dropout_add but returns a torch.compile'd version.
    Falls back to non-compiled if torch.compile is unavailable or fails.
    """
    import torch

    cache_key = (training,)
    if cache_key in _compiled_bda_cache:
        return _compiled_bda_cache[cache_key]

    fn = get_bias_dropout_add(training, fused_bias_dropout_add)
    try:
        compiled_fn = torch.compile(fn, mode="reduce-overhead")
        _compiled_bda_cache[cache_key] = compiled_fn
        return compiled_fn
    except Exception:
        _compiled_bda_cache[cache_key] = fn
        return fn


def _get_mlp_module_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
) -> ModuleSpec:
    """Get MLP module spec (handles both dense and MoE)."""
    if num_experts is None:
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
            ),
        )
    else:
        from megatron.core.transformer.moe.moe_layer import MoELayer

        if moe_grouped_gemm:
            from megatron.core.transformer.moe.experts import GroupedMLP
            return ModuleSpec(module=MoELayer, submodules=GroupedMLP)
        else:
            from megatron.core.transformer.moe.experts import SequentialMLP
            return ModuleSpec(module=MoELayer, submodules=SequentialMLP)
