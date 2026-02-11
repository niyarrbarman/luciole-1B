# Copyright (c) 2025, SSA Flash Attention v4 - autograd.Function wrapper
# Wraps the v4 Triton kernels (tutorial-based structure) into a differentiable
# PyTorch function with gradient support for Q, K, V, n, b.

import torch
from torch.autograd import Function

from SSA.ssa_triton_v4_kernel import ssa_flash_attn_v4_forward, ssa_flash_attn_v4_backward


class SSAFlashAttnV4Func(Function):
    """
    Differentiable SSA Flash Attention (v4 — tutorial-based kernel).

    Forward:  out = SSA_norm(Q @ K^T * scale) @ V
    where SSA weights: w(s) = (1 + b|s|)^(n*sign(s))

    Saves minimal state for backward: Q, K, V, Out, L, plus scalar params.
    No O(S^2) attention matrix is stored.
    """

    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, ssa_n, ssa_b, causal, dropout_p, training):
        """
        Args:
            q: [B, Hq, N, D] tensor (bf16 or fp16)
            k: [B, Hkv, N, D] tensor (Hkv <= Hq for GQA)
            v: [B, Hkv, N, D] tensor
            softmax_scale: float
            ssa_n: scalar tensor (requires_grad if learnable)
            ssa_b: scalar tensor (may or may not require grad)
            causal: bool
            dropout_p: float (applied after kernel)
            training: bool
        Returns:
            out: [B, Hq, N, D]
        """
        ssa_n_f32 = ssa_n.float().contiguous()
        ssa_b_f32 = ssa_b.float().contiguous()

        out, lse = ssa_flash_attn_v4_forward(q, k, v, softmax_scale, ssa_n_f32, ssa_b_f32, causal)

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

        dout = dout.contiguous()

        dq, dk, dv, dn, db = ssa_flash_attn_v4_backward(
            q, k, v, out, dout, lse,
            softmax_scale, ssa_n_f32, ssa_b_f32,
            causal,
        )

        # Return gradients matching forward signature:
        # q, k, v, softmax_scale, ssa_n, ssa_b, causal, dropout_p, training
        return dq, dk, dv, None, dn, db, None, None, None


def ssa_flash_attention_v4(
    q, k, v,
    softmax_scale,
    ssa_n,
    ssa_b,
    causal=True,
    dropout_p=0.0,
    training=True,
):
    """
    High-level API for SSA Flash Attention v4 (tutorial-based kernel) with native GQA.

    Args:
        q: [B, Hq, N, D] query tensor
        k: [B, Hkv, N, D] key tensor (no GQA expansion needed)
        v: [B, Hkv, N, D] value tensor
        softmax_scale: 1/sqrt(d)
        ssa_n: learnable SSA parameter n (scalar tensor)
        ssa_b: SSA parameter b (scalar tensor)
        causal: whether to apply causal masking
        dropout_p: attention dropout probability
        training: whether in training mode

    Returns:
        out: [B, Hq, N, D]
    """
    return SSAFlashAttnV4Func.apply(
        q, k, v, softmax_scale, ssa_n, ssa_b, causal, dropout_p, training
    )
