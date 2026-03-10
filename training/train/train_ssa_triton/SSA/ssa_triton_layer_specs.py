# SSA Triton Layer Specifications
# Layer specs using SSATritonAttention (tutorial-based kernel structure)

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

import sys
from pathlib import Path

SSA_DIR = Path(__file__).resolve().parent
TEST_DIR = SSA_DIR.parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from SSA.ssa_triton_attention import SSATritonAttention


def get_bias_dropout_add(training, fused_bias_dropout_add):
    """Bias-dropout-add function factory expected by TransformerLayer."""
    import torch

    def bias_dropout_add(x_with_bias, residual, prob):
        x, bias = x_with_bias
        if bias is not None:
            x = x + bias
        if training and prob > 0.0:
            x = torch.nn.functional.dropout(x, p=prob, training=True)
        return x + residual

    return bias_dropout_add


_compiled_bda_cache = {}


def get_compiled_bias_dropout_add(training, fused_bias_dropout_add):
    """torch.compile'd version of bias_dropout_add."""
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


def get_ssa_triton_gpt_layer_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
    ssa_n: float = 1.5,
    ssa_b: float = 0.8,
    learnable_ssa: bool = True,
    learnable_b: bool = False,
    use_compiled_bda: Optional[bool] = None,
    force_contiguous_qkv: bool = False,
) -> ModuleSpec:
    """
    GPT layer spec using Triton-fused SSA attention (tutorial-based kernel).

    Args:
        num_experts: Number of experts for MoE (None for dense)
        moe_grouped_gemm: Grouped GEMM for MoE
        qk_layernorm: QK layer normalization
        ssa_n: SSA parameter n (initial value)
        ssa_b: SSA parameter b (initial value)
        learnable_ssa: If True, n is learnable
        learnable_b: If True, b is also learnable
        use_compiled_bda: If None, reads SSA_TRITON_COMPILE_BDA env (default: True)
        force_contiguous_qkv: If True, materialize Q/K/V as contiguous tensors before
            launching the Triton attention kernel.
    """
    if use_compiled_bda is None:
        use_compiled_bda = os.environ.get("SSA_TRITON_COMPILE_BDA", "1") != "0"

    bda_factory = get_compiled_bias_dropout_add if use_compiled_bda else get_bias_dropout_add

    mlp = _get_mlp_module_spec(num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm)

    ssa_triton_core_spec = ModuleSpec(
        module=SSATritonAttention,
        params={
            "ssa_n": ssa_n,
            "ssa_b": ssa_b,
            "learnable_ssa": learnable_ssa,
            "learnable_b": learnable_b,
            "force_contiguous_qkv": force_contiguous_qkv,
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
                    core_attention=ssa_triton_core_spec,
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


def _get_mlp_module_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
) -> ModuleSpec:
    """Get MLP module spec."""
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
