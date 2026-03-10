# Copyright (c) 2025, SSA Flash Attention - autograd.Function wrapper (v2)
# Wraps the Triton kernels into a differentiable PyTorch function
# with proper gradient support for Q, K, V, and SSA params n, b.
# v2: Native GQA — K/V have Hkv heads, Q has Hq heads (Hq = GQA_RATIO * Hkv)
# v3: Added support for optimized kernels via USE_OPTIMIZED_KERNEL flag

import os
import torch
from torch.autograd import Function

# Toggle between original and optimized kernels via environment variable
USE_OPTIMIZED_KERNEL = os.environ.get('SSA_USE_OPTIMIZED_KERNEL', '1') == '1'

if USE_OPTIMIZED_KERNEL:
    try:
        from SSA.ssa_triton_kernel_optimized import ssa_flash_attn_forward, ssa_flash_attn_backward
    except ImportError:
        from SSA.ssa_triton_kernel import ssa_flash_attn_forward, ssa_flash_attn_backward
        USE_OPTIMIZED_KERNEL = False
else:
    from SSA.ssa_triton_kernel import ssa_flash_attn_forward, ssa_flash_attn_backward


class SSAFlashAttnFunc(Function):
    """
    Differentiable SSA Flash Attention.

    Forward:  out = SSA_softmax(Q @ K^T * scale) @ V
    where SSA_softmax applies: softmax(n * sign(s) * log(1 + b*|s|))

    Saves minimal state for backward: Q, K, V, Out, lse, plus scalar params.
    No O(S^2) attention matrix is stored.
    """

    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, ssa_n, ssa_b, causal, dropout_p, training):
        """
        Args:
            q: [B, Hq, N, D] contiguous tensor (bf16 or fp16)
            k: [B, Hkv, N, D] contiguous tensor (Hkv <= Hq for GQA)
            v: [B, Hkv, N, D] contiguous tensor
            softmax_scale: float
            ssa_n: scalar tensor (requires_grad if learnable)
            ssa_b: scalar tensor (may or may not require grad)
            causal: bool
            dropout_p: float (dropout probability — applied after kernel)
            training: bool
        Returns:
            out: [B, Hq, N, D]
        """
        # Note: q, k, v may be non-contiguous (e.g. from permute without .contiguous()).
        # The Triton kernels handle arbitrary strides, so no copy is needed.
        # We only need contiguity for SSA scalar params.

        # Ensure SSA params are fp32 on device
        ssa_n_f32 = ssa_n.float().contiguous()
        ssa_b_f32 = ssa_b.float().contiguous()

        out, lse = ssa_flash_attn_forward(q, k, v, softmax_scale, ssa_n_f32, ssa_b_f32, causal)

        ctx.save_for_backward(q, k, v, out, lse, ssa_n_f32, ssa_b_f32)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.dropout_p = dropout_p

        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, ssa_n_f32, ssa_b_f32 = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        causal = ctx.causal

        dout = dout.contiguous()  # backward output often non-contiguous; kernel needs valid strides

        dq, dk, dv, dn, db = ssa_flash_attn_backward(
            q, k, v, out, dout, lse,
            softmax_scale, ssa_n_f32, ssa_b_f32,
            causal,
        )

        # Return gradients matching forward signature:
        # q, k, v, softmax_scale, ssa_n, ssa_b, causal, dropout_p, training
        return dq, dk, dv, None, dn, db, None, None, None


def ssa_flash_attention(
    q, k, v,
    softmax_scale,
    ssa_n,
    ssa_b,
    causal=True,
    dropout_p=0.0,
    training=True,
):
    """
    High-level API for SSA Flash Attention with native GQA.

    Args:
        q: [B, Hq, N, D] query tensor (Hq query heads)
        k: [B, Hkv, N, D] key tensor (Hkv key/value heads, no GQA expansion needed)
        v: [B, Hkv, N, D] value tensor
        softmax_scale: 1/sqrt(d) scaling factor
        ssa_n: learnable SSA parameter n (scalar tensor)
        ssa_b: SSA parameter b (scalar tensor, may be fixed)
        causal: whether to apply causal masking
        dropout_p: attention dropout probability
        training: whether in training mode

    Returns:
        out: [B, Hq, N, D]
    """
    return SSAFlashAttnFunc.apply(
        q, k, v, softmax_scale, ssa_n, ssa_b, causal, dropout_p, training
    )
