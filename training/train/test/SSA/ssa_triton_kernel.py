# Copyright (c) 2025, SSA Triton Kernel Implementation — v2 (optimized)
# Fused SSA Flash Attention: scale -> causal mask -> SSA transform -> online softmax -> V accumulation
#
# SSA formula: softmax(n * sgn(s) * ln(1 + b|s|))   where s = scale * Q @ K^T
#
# v2 optimizations over v1:
#   1. Native GQA support — K/V have fewer heads, indexed via stride (no repeat_interleave copy)
#   2. Stride-based layout — accepts ANY layout via explicit strides (no permute/contiguous copy)
#   3. Cache warmup utility — pre-compiles all kernels to avoid JIT overhead at step 0
#
# Hardware target: NVIDIA A100-80GB (sm_80)

import torch
import triton
import triton.language as tl
import math


# ============================================================
# Forward Kernel  (GQA-aware, layout-agnostic)
# ============================================================

@triton.jit
def _ssa_attn_fwd_kernel(
    Q, K, V, Out, L,  # L = logsumexp per row (for backward)
    softmax_scale,
    ssa_n_ptr, ssa_b_ptr,  # pointers to scalar SSA params
    # Q strides: logical [B, Hq, Sq, D], flattened as B*Hq for dim0+dim1
    stride_qb, stride_qh, stride_qm, stride_qk,
    # K strides: logical [B, Hkv, Sk, D], flattened as B*Hkv
    stride_kb, stride_kh, stride_kn, stride_kk,
    # V strides
    stride_vb, stride_vh, stride_vn, stride_vk,
    # Out strides
    stride_ob, stride_oh, stride_om, stride_ok,
    # L strides: [B, Hq, Sq]
    stride_lb, stride_lh, stride_lm,
    N_CTX: tl.constexpr,
    GQA_RATIO: tl.constexpr,   # Hq / Hkv (>= 1)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Fused SSA FlashAttention forward kernel with native GQA.

    Grid: (cdiv(N_CTX, BLOCK_M), B * Hq)

    K/V are indexed via kv_head = q_head // GQA_RATIO (flattened B*Hkv),
    so no repeat_interleave copy is needed.
    """
    pid_m = tl.program_id(0)   # query block index
    pid_bh = tl.program_id(1)  # flattened batch * n_q_heads

    # Load scalar SSA params
    ssa_n = tl.load(ssa_n_ptr).to(tl.float32)
    ssa_b = tl.load(ssa_b_ptr).to(tl.float32)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    # Q/Out/L base: indexed by pid_bh (flattened B*Hq)
    q_base = Q + pid_bh * stride_qh
    o_base = Out + pid_bh * stride_oh
    l_base = L + pid_bh * stride_lh

    # K/V base: GQA — map flattened B*Hq index to flattened B*Hkv index
    kv_idx = pid_bh // GQA_RATIO
    k_base = K + kv_idx * stride_kh
    v_base = V + kv_idx * stride_vh

    # Load Q block: [BLOCK_M, BLOCK_D]
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize online softmax accumulators
    m_i = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if IS_CAUSAL:
        loop_bound = tl.minimum(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        loop_bound = N_CTX

    n_blocks = tl.cdiv(loop_bound, BLOCK_N)

    for j in range(0, n_blocks):
        start_n = j * BLOCK_N
        offs_n_j = start_n + offs_n

        # Load K block: [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + offs_n_j[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = offs_n_j[:, None] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # S = Q @ K^T * scale
        s = tl.dot(q, tl.trans(k))
        s = s * softmax_scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_j[None, :]
            s = tl.where(causal_mask, s, float('-inf'))

        kv_valid = offs_n_j[None, :] < N_CTX
        s = tl.where(kv_valid, s, float('-inf'))

        # SSA transform: f(s) = n * sign(s) * log(1 + b * |s|)
        s_fp32 = s.to(tl.float32)
        abs_s = tl.abs(s_fp32)
        sign_s = tl.where(s_fp32 > 0, 1.0, tl.where(s_fp32 < 0, -1.0, 0.0))
        log_term = tl.log(1.0 + ssa_b * abs_s)
        ssa_logits = ssa_n * sign_s * log_term

        # Online softmax
        m_ij = tl.max(ssa_logits, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(ssa_logits - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)

        # Load V block
        v_ptrs = v_base + offs_n_j[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = offs_n_j[:, None] < N_CTX
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Accumulate output
        p_v = tl.dot(p.to(v.dtype), v)
        acc = alpha[:, None] * acc + p_v.to(tl.float32)

        m_i = m_new
        l_i = l_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store logsumexp
    lse = m_i + tl.log(l_i)
    l_ptrs = l_base + offs_m * stride_lm
    l_mask = offs_m < N_CTX
    tl.store(l_ptrs, lse, mask=l_mask)

    # Store output
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = offs_m[:, None] < N_CTX
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_mask)


# ============================================================
# Backward Kernel - dQ  (GQA-aware)
# ============================================================

@triton.jit
def _ssa_attn_bwd_dq_kernel(
    Q, K, V, Out, dO, dQ, L, D,
    softmax_scale,
    ssa_n_ptr, ssa_b_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_lb, stride_lh, stride_lm,
    stride_db, stride_dh, stride_dm,
    N_CTX: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Backward kernel computing dQ.  Grid: (cdiv(N_CTX, BLOCK_M), B * Hq)"""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    ssa_n = tl.load(ssa_n_ptr).to(tl.float32)
    ssa_b = tl.load(ssa_b_ptr).to(tl.float32)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    # Q/dO/dQ/L/D base — indexed by pid_bh (flattened B*Hq)
    q_base  = Q  + pid_bh * stride_qh
    do_base = dO + pid_bh * stride_doh
    dq_base = dQ + pid_bh * stride_dqh
    l_base  = L  + pid_bh * stride_lh
    d_base  = D  + pid_bh * stride_dh

    # K/V base — GQA indexing
    kv_idx = pid_bh // GQA_RATIO
    k_base = K + kv_idx * stride_kh
    v_base = V + kv_idx * stride_vh

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
    do = tl.load(do_ptrs, mask=q_mask, other=0.0)

    l_ptrs = l_base + offs_m * stride_lm
    l_mask = offs_m < N_CTX
    lse = tl.load(l_ptrs, mask=l_mask, other=0.0)

    d_ptrs = d_base + offs_m * stride_dm
    Di = tl.load(d_ptrs, mask=l_mask, other=0.0)

    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if IS_CAUSAL:
        loop_bound = tl.minimum(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        loop_bound = N_CTX
    n_blocks = tl.cdiv(loop_bound, BLOCK_N)

    for j in range(0, n_blocks):
        start_n = j * BLOCK_N
        offs_n_j = start_n + offs_n

        k_ptrs = k_base + offs_n_j[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = offs_n_j[:, None] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        v_ptrs = v_base + offs_n_j[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # Recompute S
        s = tl.dot(q, tl.trans(k)) * softmax_scale
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_j[None, :]
            s = tl.where(causal_mask, s, float('-inf'))
        kv_valid = offs_n_j[None, :] < N_CTX
        s = tl.where(kv_valid, s, float('-inf'))

        # SSA transform (sanitize -inf -> 0 for safe backward math)
        s_fp32 = s.to(tl.float32)
        valid = s_fp32 > float('-inf')
        s_safe = tl.where(valid, s_fp32, 0.0)
        abs_s = tl.abs(s_safe)
        sign_s = tl.where(s_safe > 0, 1.0, tl.where(s_safe < 0, -1.0, 0.0))
        log_term = tl.log(1.0 + ssa_b * abs_s)
        ssa_logits = tl.where(valid, ssa_n * sign_s * log_term, float('-inf'))

        p = tl.exp(ssa_logits - lse[:, None])

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds_ssa = p * (dp - Di[:, None])

        denom = 1.0 + ssa_b * abs_s
        df_ds = ssa_n * ssa_b / denom
        ds = ds_ssa * df_ds

        dq_acc += tl.dot(ds.to(k.dtype), k).to(tl.float32) * softmax_scale

    dq_ptrs = dq_base + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqk
    dq_mask = offs_m[:, None] < N_CTX
    tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=dq_mask)


# ============================================================
# Backward Kernel - dK, dV, dn, db  (GQA-aware)
# ============================================================

@triton.jit
def _ssa_attn_bwd_dkv_kernel(
    Q, K, V, Out, dO, dK, dV, L, D,
    DN, DB,
    softmax_scale,
    ssa_n_ptr, ssa_b_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    stride_lb, stride_lh, stride_lm,
    stride_db, stride_dh, stride_dm,
    N_CTX: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    NUM_BLOCKS_N: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,  # Hq (needed for GQA iteration)
):
    """
    Backward kernel computing dK, dV and partial dn, db.
    Grid: (cdiv(N_CTX, BLOCK_N), B * Hkv)

    For each KV block, iterates over ALL GQA_RATIO Q-heads that share
    this KV head, and over all valid Q-sequence blocks.
    No atomics needed: each (kv_block, kv_head) is written by exactly one program.
    """
    pid_n = tl.program_id(0)    # KV sequence block index
    pid_bkv = tl.program_id(1)  # flattened B * Hkv

    ssa_n = tl.load(ssa_n_ptr).to(tl.float32)
    ssa_b = tl.load(ssa_b_ptr).to(tl.float32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_m = tl.arange(0, BLOCK_M)

    # K/V/dK/dV base — indexed by pid_bkv (flattened B*Hkv)
    k_base  = K  + pid_bkv * stride_kh
    v_base  = V  + pid_bkv * stride_vh
    dk_base = dK + pid_bkv * stride_dkh
    dv_base = dV + pid_bkv * stride_dvh

    # Load K, V blocks
    k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    k_mask = offs_n[:, None] < N_CTX
    k = tl.load(k_ptrs, mask=k_mask, other=0.0)

    v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    v = tl.load(v_ptrs, mask=k_mask, other=0.0)

    dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dn_acc = 0.0
    db_acc = 0.0

    if IS_CAUSAL:
        start_block = pid_n * BLOCK_N // BLOCK_M
    else:
        start_block = 0

    n_q_blocks = tl.cdiv(N_CTX, BLOCK_M)

    # Iterate over GQA_RATIO Q-heads sharing this KV head
    for g in range(0, GQA_RATIO):
        pid_bh_q = pid_bkv * GQA_RATIO + g

        q_base  = Q  + pid_bh_q * stride_qh
        do_base = dO + pid_bh_q * stride_doh
        l_base  = L  + pid_bh_q * stride_lh
        d_base  = D  + pid_bh_q * stride_dh

        for i in range(start_block, n_q_blocks):
            offs_m_i = i * BLOCK_M + offs_m

            q_ptrs = q_base + offs_m_i[:, None] * stride_qm + offs_d[None, :] * stride_qk
            q_mask = offs_m_i[:, None] < N_CTX
            q = tl.load(q_ptrs, mask=q_mask, other=0.0)

            do_ptrs = do_base + offs_m_i[:, None] * stride_dom + offs_d[None, :] * stride_dok
            do = tl.load(do_ptrs, mask=q_mask, other=0.0)

            l_ptrs = l_base + offs_m_i * stride_lm
            l_mask = offs_m_i < N_CTX
            lse = tl.load(l_ptrs, mask=l_mask, other=0.0)

            d_ptrs = d_base + offs_m_i * stride_dm
            Di = tl.load(d_ptrs, mask=l_mask, other=0.0)

            # Recompute S
            s = tl.dot(q, tl.trans(k)) * softmax_scale
            if IS_CAUSAL:
                causal_mask = offs_m_i[:, None] >= offs_n[None, :]
                s = tl.where(causal_mask, s, float('-inf'))
            kv_valid = offs_n[None, :] < N_CTX
            s = tl.where(kv_valid, s, float('-inf'))

            # SSA transform (sanitize -inf -> 0)
            s_fp32 = s.to(tl.float32)
            valid = s_fp32 > float('-inf')
            s_safe = tl.where(valid, s_fp32, 0.0)
            abs_s = tl.abs(s_safe)
            sign_s = tl.where(s_safe > 0, 1.0, tl.where(s_safe < 0, -1.0, 0.0))
            log_term = tl.log(1.0 + ssa_b * abs_s)
            ssa_logits = tl.where(valid, ssa_n * sign_s * log_term, float('-inf'))

            p = tl.exp(ssa_logits - lse[:, None])

            # dV
            dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do).to(tl.float32)

            # dP, dSSA
            dp = tl.dot(do, tl.trans(v)).to(tl.float32)
            ds_ssa = p * (dp - Di[:, None])

            # SSA backward
            denom = 1.0 + ssa_b * abs_s
            df_ds = ssa_n * ssa_b / denom
            ds = ds_ssa * df_ds

            dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q).to(tl.float32) * softmax_scale

            # Partial dn, db
            dn_acc += tl.sum(ds_ssa * sign_s * log_term)
            db_acc += tl.sum(ds_ssa * ssa_n * abs_s / denom)

    # Store dK, dV
    dk_ptrs = dk_base + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
    dk_mask = offs_n[:, None] < N_CTX
    tl.store(dk_ptrs, dk_acc.to(dK.dtype.element_ty), mask=dk_mask)

    dv_ptrs = dv_base + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk
    tl.store(dv_ptrs, dv_acc.to(dV.dtype.element_ty), mask=dk_mask)

    # Store partial dn, db — shape: [B*Hkv, num_kv_blocks]
    dn_ptr = DN + pid_bkv * NUM_BLOCKS_N + pid_n
    db_ptr = DB + pid_bkv * NUM_BLOCKS_N + pid_n
    tl.store(dn_ptr, dn_acc)
    tl.store(db_ptr, db_acc)


# ============================================================
# D precompute kernel: D_i = rowsum(dO_i * O_i)
# ============================================================

@triton.jit
def _ssa_attn_bwd_preprocess_kernel(
    Out, dO, D,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_db, stride_dh, stride_dm,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Precompute D_i = sum_d(O_i[d] * dO_i[d]) per row."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    o_base = Out + pid_bh * stride_oh
    do_base = dO + pid_bh * stride_doh
    d_base = D + pid_bh * stride_dh

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
    mask = offs_m[:, None] < N_CTX

    o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)

    d = tl.sum(o * do, axis=1)

    d_ptrs = d_base + offs_m * stride_dm
    d_mask = offs_m < N_CTX
    tl.store(d_ptrs, d, mask=d_mask)


# ============================================================
# Python Wrapper Functions
# ============================================================

def _get_block_sizes(D):
    """Select tile sizes tuned for A100 (sm_80)."""
    BLOCK_D = triton.next_power_of_2(D)
    if BLOCK_D <= 32:
        BLOCK_M, BLOCK_N = 128, 128
    elif BLOCK_D <= 64:
        BLOCK_M, BLOCK_N = 128, 64
    else:
        BLOCK_M, BLOCK_N = 64, 64
    return BLOCK_D, BLOCK_M, BLOCK_N


def ssa_flash_attn_forward(q, k, v, softmax_scale, ssa_n, ssa_b, causal=True):
    """
    Forward pass wrapper with native GQA.

    Args:
        q: [B, Hq, N, D] — Q tensor (Hq query heads)
        k: [B, Hkv, N, D] — K tensor (Hkv <= Hq key/value heads)
        v: [B, Hkv, N, D] — V tensor
        softmax_scale: float
        ssa_n, ssa_b: scalar tensors (float32, on device)
        causal: bool

    Returns:
        out: [B, Hq, N, D]
        lse: [B, Hq, N]
    """
    B, Hq, N, D = q.shape
    Hkv = k.shape[1]
    GQA_RATIO = Hq // Hkv
    assert Hq % Hkv == 0, f"Hq={Hq} must be divisible by Hkv={Hkv}"

    if ssa_n.dim() == 0:
        ssa_n = ssa_n.contiguous()
    if ssa_b.dim() == 0:
        ssa_b = ssa_b.contiguous()

    out = torch.empty_like(q)
    lse = torch.empty((B, Hq, N), device=q.device, dtype=torch.float32)

    BLOCK_D, BLOCK_M, BLOCK_N = _get_block_sizes(D)
    grid = (triton.cdiv(N, BLOCK_M), B * Hq)

    _ssa_attn_fwd_kernel[grid](
        q, k, v, out, lse,
        softmax_scale,
        ssa_n, ssa_b,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        N_CTX=N,
        GQA_RATIO=GQA_RATIO,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
    )

    return out, lse


def ssa_flash_attn_backward(q, k, v, out, dout, lse, softmax_scale, ssa_n, ssa_b, causal=True):
    """
    Backward pass wrapper with native GQA.

    Args:
        q: [B, Hq, N, D]
        k: [B, Hkv, N, D]
        v: [B, Hkv, N, D]
        out: [B, Hq, N, D]
        dout: [B, Hq, N, D]
        lse: [B, Hq, N]

    Returns:
        dq: [B, Hq, N, D]
        dk: [B, Hkv, N, D]
        dv: [B, Hkv, N, D]
        dn, db: scalar tensors
    """
    B, Hq, N, D = q.shape
    Hkv = k.shape[1]
    GQA_RATIO = Hq // Hkv

    BLOCK_D, BLOCK_M, BLOCK_N = _get_block_sizes(D)

    # Precompute D = rowsum(dO * O) — per Q-head
    D_vec = torch.empty((B, Hq, N), device=q.device, dtype=torch.float32)
    grid_pre = (triton.cdiv(N, BLOCK_M), B * Hq)

    _ssa_attn_bwd_preprocess_kernel[grid_pre](
        out, dout, D_vec,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        D_vec.stride(0), D_vec.stride(1), D_vec.stride(2),
        N_CTX=N,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )

    # dQ kernel — grid over Q-heads
    dq = torch.zeros_like(q)
    grid_dq = (triton.cdiv(N, BLOCK_M), B * Hq)

    _ssa_attn_bwd_dq_kernel[grid_dq](
        q, k, v, out, dout, dq, lse, D_vec,
        softmax_scale,
        ssa_n, ssa_b,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        D_vec.stride(0), D_vec.stride(1), D_vec.stride(2),
        N_CTX=N,
        GQA_RATIO=GQA_RATIO,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
    )

    # dK, dV kernel — grid over KV-heads
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    num_kv_blocks = triton.cdiv(N, BLOCK_N)
    dn_partial = torch.zeros((B * Hkv, num_kv_blocks), device=q.device, dtype=torch.float32)
    db_partial = torch.zeros((B * Hkv, num_kv_blocks), device=q.device, dtype=torch.float32)

    grid_dkv = (num_kv_blocks, B * Hkv)

    _ssa_attn_bwd_dkv_kernel[grid_dkv](
        q, k, v, out, dout, dk, dv, lse, D_vec,
        dn_partial, db_partial,
        softmax_scale,
        ssa_n, ssa_b,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        D_vec.stride(0), D_vec.stride(1), D_vec.stride(2),
        N_CTX=N,
        GQA_RATIO=GQA_RATIO,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
        NUM_BLOCKS_N=num_kv_blocks,
        NUM_Q_HEADS=Hq,
    )

    dn = dn_partial.sum()
    db = db_partial.sum()

    return dq, dk, dv, dn, db


# ============================================================
# Cache Warmup Utility
# ============================================================

def warmup_triton_kernels(
    B: int = 2,
    Hq: int = 24,
    Hkv: int = 8,
    N: int = 128,
    D: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    Pre-compile all Triton kernels by running a tiny dummy forward+backward.
    Call this once before training to avoid JIT compilation overhead at step 0.

    Uses small tensors to minimize GPU time (~1-2 seconds).
    """
    q = torch.randn(B, Hq, N, D, dtype=dtype, device=device)
    k = torch.randn(B, Hkv, N, D, dtype=dtype, device=device)
    v = torch.randn(B, Hkv, N, D, dtype=dtype, device=device)
    ssa_n = torch.tensor(1.5, dtype=torch.float32, device=device)
    ssa_b = torch.tensor(0.8, dtype=torch.float32, device=device)
    scale = 1.0 / (D ** 0.5)

    # Forward — compiles fwd kernel
    out, lse = ssa_flash_attn_forward(q, k, v, scale, ssa_n, ssa_b, causal=True)

    # Backward — compiles preprocess, dQ, dKV kernels
    dout = torch.randn_like(out)
    dq, dk, dv, dn, db = ssa_flash_attn_backward(
        q, k, v, out, dout, lse, scale, ssa_n, ssa_b, causal=True,
    )

    # Sync to ensure compilation is complete
    torch.cuda.synchronize()
