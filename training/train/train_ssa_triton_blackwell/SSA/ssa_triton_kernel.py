# Copyright (c) 2025, SSA Triton Kernel
# Based on Triton tutorial 06-fused-attention (Flash Attention v2 structure)
# with SSA (Softmax-Substituted Attention) integrated.
#
# Key design choices from the tutorial that we adopt:
#   1. Stage-based causal handling (off-band vs on-band) — avoids mask check on majority of blocks
#   2. BLK_SLICE_FACTOR — smaller sub-blocks on the causal diagonal to reduce wasted compute
#   3. Separate dQ and dKV backward kernels with different block sizes
#   4. Autotuning over BLOCK_M, BLOCK_N, num_warps, num_stages
#   5. Config pruning for invalid combinations
#
# Key SSA differences from standard softmax:
#   - Replaces raw logits s with transformed logits:
#       t(s) = n * sign(s) * log(1 + b|s|)
#   - Keeps tutorial-style online normalization (running max + running sum)
#     for numerical stability in BF16/FP16.
#   - Uses tutorial-style base-2 accumulation (exp2/log2) for close parity
#     with Triton tutorial numerics.
#   - Uses chain rule factor dt/ds = n*b/(1+b|s|) in backward.
#   - Learnable parameter gradients dn, db with Kahan summation.
#   - Stores M = log2(sum_j exp2(t_j / ln(2))) for backward.
#
# SSA formula:
#   t(s) = n * sign(s) * log(1 + b|s|)
#   P = softmax(t)
#   out = P @ V
#
# Hardware target: NVIDIA A100/H100 (sm_80+)

import os
import torch
import triton
import triton.language as tl
import math


# ============================================================
# Forward Inner Loop
# ============================================================

@triton.jit
def _ssa_attn_fwd_inner(
    acc, l_i, m_i, q,
    K, V,
    stride_kn, stride_kk,
    stride_vn, stride_vk,
    k_base_offset, v_base_offset,
    softmax_scale,
    ssa_n, ssa_b,
    start_m, offs_m, offs_n,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    MATMUL_PRECISION: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    NEG_LARGE: tl.constexpr = -1.0e6

    # Range of KV positions handled by this stage.
    if STAGE == 1:
        # Off-band: all blocks before the diagonal.
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # On-band: the diagonal block.
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        # Non-causal: all blocks.
        lo, hi = 0, N_CTX

    offs_d = tl.arange(0, HEAD_DIM)

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_j = start_n + offs_n

        # Load K block: [BLOCK_N, HEAD_DIM].
        k_ptrs = K + k_base_offset + offs_n_j[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = offs_n_j[:, None] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # S = Q @ K^T * scale.
        s = tl.dot(q, tl.trans(k)) * softmax_scale

        # Apply causal mask on diagonal stage.
        if STAGE == 2:
            mask = offs_m[:, None] >= offs_n_j[None, :]
            s = tl.where(mask, s, float('-inf'))

        # Boundary mask.
        kv_valid = offs_n_j[None, :] < N_CTX
        s = tl.where(kv_valid, s, float('-inf'))

        # --- SSA transformed logits ---
        s_fp32 = s.to(tl.float32)
        valid = s_fp32 > float('-inf')
        s_safe = tl.where(valid, s_fp32, 0.0)
        abs_s = tl.abs(s_safe)
        sign_s = tl.where(s_safe > 0, 1.0, tl.where(s_safe < 0, -1.0, 0.0))
        u = ssa_b * abs_s
        one_plus_bs = 1.0 + u
        # log1p precision: Taylor expansion for very small u.
        log_opbs = tl.where(u < 1e-4, u - 0.5 * u * u, tl.log(one_plus_bs))
        t = ssa_n * sign_s * log_opbs
        # Tutorial numerics: accumulate with exp2/log2 over base-2 logits.
        t2 = tl.where(valid, t * RCP_LN2, NEG_LARGE)

        # Tutorial-style online normalization on transformed logits.
        m_ij = tl.maximum(m_i, tl.max(t2, axis=1))
        p = tl.math.exp2(t2 - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Load V block: [BLOCK_N, HEAD_DIM].
        v_ptrs = V + v_base_offset + offs_n_j[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = offs_n_j[:, None] < N_CTX
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(MATMUL_PRECISION), v).to(tl.float32)
        m_i = m_ij

    return acc, l_i, m_i


# ============================================================
# Forward Kernel (GQA-aware, stage-based causal)
# ============================================================

_fwd_configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64]
    for s in [2, 3, 4]
    for w in [4, 8]
]


def _fwd_prune_configs(configs, named_args, **kwargs):
    N_CTX = kwargs.get("N_CTX", 4096)
    STAGE = kwargs.get("STAGE", 1)
    return [
        c for c in configs
        if c.kwargs["BLOCK_M"] <= N_CTX
        and (c.kwargs["BLOCK_M"] >= c.kwargs["BLOCK_N"] or STAGE == 1)
    ]


@triton.autotune(
    configs=_fwd_configs,
    key=["N_CTX", "HEAD_DIM", "GQA_RATIO"],
    prune_configs_by={"early_config_prune": _fwd_prune_configs},
)
@triton.jit
def _ssa_attn_fwd(
    Q, K, V, Out, L,
    softmax_scale,
    ssa_n_ptr, ssa_b_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_lz, stride_lh, stride_lm,
    Z, H_Q, N_CTX,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    MATMUL_PRECISION: tl.constexpr,
):
    """
    SSA FlashAttention forward kernel.

    Grid: (cdiv(N_CTX, BLOCK_M), Z * H_Q)

    STAGE=3 for causal (runs stages 1 then 2), STAGE=1 for non-causal (runs stage 3).
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Load SSA params once.
    ssa_n = tl.load(ssa_n_ptr).to(tl.float32)
    ssa_b = tl.load(ssa_b_ptr).to(tl.float32)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Tutorial-style explicit z/h decomposition (no flattened stride assumptions).
    off_z = off_hz // H_Q
    off_h_q = off_hz % H_Q
    off_h_kv = off_h_q // GQA_RATIO

    q_base = Q + off_z * stride_qz + off_h_q * stride_qh
    o_base = Out + off_z * stride_oz + off_h_q * stride_oh
    l_base = L + off_z * stride_lz + off_h_q * stride_lh
    k_base_offset = off_z * stride_kz + off_h_kv * stride_kh
    v_base_offset = off_z * stride_vz + off_h_kv * stride_vh

    # Load Q block.
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Tutorial-style online-softmax state.
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], value=1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Stage 1: off-band (no mask needed)
    # For causal: STAGE=3, so STAGE & 1 = True -> run with inner STAGE = 4-3 = 1
    # For non-causal: STAGE=1, so STAGE & 1 = True -> run with inner STAGE = 4-1 = 3
    if STAGE & 1:
        acc, l_i, m_i = _ssa_attn_fwd_inner(
            acc, l_i, m_i, q,
            K, V,
            stride_kn, stride_kk,
            stride_vn, stride_vk,
            k_base_offset, v_base_offset,
            softmax_scale,
            ssa_n, ssa_b,
            start_m, offs_m, offs_n,
            N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM,
            4 - STAGE,  # 1 for causal off-band, 3 for non-causal all
            MATMUL_PRECISION,
        )

    # Stage 2: on-band (diagonal, masked)
    # For causal: STAGE=3, so STAGE & 2 = True -> run with inner STAGE = 2
    if STAGE & 2:
        acc, l_i, m_i = _ssa_attn_fwd_inner(
            acc, l_i, m_i, q,
            K, V,
            stride_kn, stride_kk,
            stride_vn, stride_vk,
            k_base_offset, v_base_offset,
            softmax_scale,
            ssa_n, ssa_b,
            start_m, offs_m, offs_n,
            N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM,
            2,
            MATMUL_PRECISION,
        )

    # Epilogue: normalize and store.
    l_i_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_i_safe[:, None]
    m = m_i + tl.math.log2(l_i_safe)

    # Store M = log2-normalizer for backward.
    l_ptrs = l_base + offs_m * stride_lm
    l_mask = offs_m < N_CTX
    tl.store(l_ptrs, m, mask=l_mask)

    # Store output
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = offs_m[:, None] < N_CTX
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_mask)


# ============================================================
# Backward Preprocess: D_i = rowsum(O * dO)
# ============================================================

@triton.jit
def _ssa_attn_bwd_preprocess(
    Out, dO, Delta,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dz, stride_dh, stride_dm,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    o_base = Out + off_hz * stride_oh
    do_base = dO + off_hz * stride_doh
    d_base = Delta + off_hz * stride_dh

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
    mask = offs_m[:, None] < N_CTX

    o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    d_ptrs = d_base + offs_m * stride_dm
    d_mask = offs_m < N_CTX
    tl.store(d_ptrs, delta, mask=d_mask)


# ============================================================
# Backward: dK, dV inner loop (processes Q blocks for a fixed KV block)
# ============================================================

@triton.jit
def _ssa_attn_bwd_dkdv(
    dk, dv,
    dn_acc, dn_comp, db_acc, db_comp,
    Q, k, v, softmax_scale,
    dO, L, Delta,
    ssa_n, ssa_b, ssa_nb,
    stride_qm, stride_qk,
    stride_dom, stride_dok,
    stride_lm, stride_dm,
    q_base, do_base, l_base, d_base,
    offs_n,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    MATMUL_PRECISION: tl.constexpr,
    start_n, start_m, num_steps,
    MASK: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    NEG_LARGE: tl.constexpr = -1.0e6

    offs_d = tl.arange(0, HEAD_DIM)

    curr_m = start_m
    for blk_idx in range(num_steps):
        offs_m = curr_m + tl.arange(0, BLOCK_M1)

        # Load Q, dO blocks
        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q_mask = offs_m[:, None] < N_CTX
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
        do = tl.load(do_ptrs, mask=q_mask, other=0.0)

        # Load saved M = log2-normalizer and Delta.
        l_mask = offs_m < N_CTX
        m_i = tl.load(l_base + offs_m * stride_lm, mask=l_mask, other=0.0)
        Di = tl.load(d_base + offs_m * stride_dm, mask=l_mask, other=0.0)

        # Recompute S = Q @ K^T * scale
        s = tl.dot(q, tl.trans(k)) * softmax_scale

        # Apply causal mask on diagonal blocks
        if MASK:
            mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(mask, s, float('-inf'))

        # Boundary mask
        kv_valid = offs_n[None, :] < N_CTX
        s = tl.where(kv_valid, s, float('-inf'))

        # Recompute transformed logits and probabilities.
        s_fp32 = s.to(tl.float32)
        valid = s_fp32 > float('-inf')
        s_safe = tl.where(valid, s_fp32, 0.0)
        abs_s = tl.abs(s_safe)
        sign_s = tl.where(s_safe > 0, 1.0, tl.where(s_safe < 0, -1.0, 0.0))
        u = ssa_b * abs_s
        one_plus_bs = 1.0 + u
        log_opbs = tl.where(u < 1e-4, u - 0.5 * u * u, tl.log(one_plus_bs))
        t2 = tl.where(valid, ssa_n * sign_s * log_opbs * RCP_LN2, NEG_LARGE)
        p = tl.where(valid, tl.math.exp2(t2 - m_i[:, None]), 0.0)

        # dV = P^T @ dO
        dv += tl.dot(tl.trans(p.to(MATMUL_PRECISION)), do).to(tl.float32)

        # dp = dO @ V^T
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)

        # ds_ssa = P * (dp - Di) — same form as softmax backward
        ds_ssa = p * (dp - Di[:, None])

        # SSA chain rule: ds = ds_ssa * n*b / (1 + b|s|)
        ds = ds_ssa * (ssa_nb / one_plus_bs)

        # dK += ds^T @ Q * scale
        dk += tl.dot(tl.trans(ds.to(MATMUL_PRECISION)), q).to(tl.float32) * softmax_scale

        # Accumulate dn, db with Kahan compensated summation
        block_dn = tl.sum(ds_ssa * sign_s * log_opbs)
        y_dn = block_dn - dn_comp
        t_dn = dn_acc + y_dn
        dn_comp = (t_dn - dn_acc) - y_dn
        dn_acc = t_dn

        block_db = tl.sum(ds_ssa * ssa_n * sign_s * abs_s / one_plus_bs)
        y_db = block_db - db_comp
        t_db = db_acc + y_db
        db_comp = (t_db - db_acc) - y_db
        db_acc = t_db

        curr_m += BLOCK_M1

    return dk, dv, dn_acc, dn_comp, db_acc, db_comp


# ============================================================
# Backward: dQ inner loop (processes KV blocks for a fixed Q block)
# ============================================================

@triton.jit
def _ssa_attn_bwd_dq(
    dq, q, K, V,
    do, L_row, Di,
    softmax_scale,
    ssa_n, ssa_b, ssa_nb,
    stride_kn, stride_kk,
    stride_vn, stride_vk,
    k_base_offset, v_base_offset,
    offs_m,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    MATMUL_PRECISION: tl.constexpr,
    start_m, start_n, num_steps,
    MASK: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    NEG_LARGE: tl.constexpr = -1.0e6

    offs_d = tl.arange(0, HEAD_DIM)

    curr_n = start_n
    for blk_idx in range(num_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N2)

        # Load K, V blocks.
        k_ptrs = K + k_base_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = offs_n[:, None] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        v_ptrs = V + v_base_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # Recompute S = Q @ K^T * scale
        s = tl.dot(q, tl.trans(k)) * softmax_scale

        # Apply causal mask
        if MASK:
            mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(mask, s, float('-inf'))

        # Boundary mask
        kv_valid = offs_n[None, :] < N_CTX
        s = tl.where(kv_valid, s, float('-inf'))

        # Recompute transformed logits and probabilities.
        s_fp32 = s.to(tl.float32)
        valid = s_fp32 > float('-inf')
        s_safe = tl.where(valid, s_fp32, 0.0)
        abs_s = tl.abs(s_safe)
        sign_s = tl.where(s_safe > 0, 1.0, tl.where(s_safe < 0, -1.0, 0.0))
        u = ssa_b * abs_s
        one_plus_bs = 1.0 + u
        log_opbs = tl.where(u < 1e-4, u - 0.5 * u * u, tl.log(one_plus_bs))
        t2 = tl.where(valid, ssa_n * sign_s * log_opbs * RCP_LN2, NEG_LARGE)
        p = tl.where(valid, tl.math.exp2(t2 - L_row[:, None]), 0.0)

        # dp = dO @ V^T
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)

        # ds_ssa = P * (dp - Di)
        ds_ssa = p * (dp - Di[:, None])

        # SSA chain rule
        ds = ds_ssa * (ssa_nb / one_plus_bs)

        # dQ += ds @ K * scale
        dq += tl.dot(ds.to(MATMUL_PRECISION), k).to(tl.float32) * softmax_scale

        curr_n += BLOCK_N2

    return dq


# ============================================================
# Backward Outer Kernel (combines dK/dV and dQ like the tutorial)
# ============================================================

@triton.jit
def _ssa_attn_bwd(
    Q, K, V,
    Out, dO,
    dQ, dK, dV,
    L, Delta,
    DN, DB,
    softmax_scale,
    ssa_n_ptr, ssa_b_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_lz, stride_lh, stride_lm,
    stride_dez, stride_deh, stride_dem,
    Z, H_Q, N_CTX,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    MATMUL_PRECISION: tl.constexpr,
    CAUSAL: tl.constexpr,
    NUM_BLOCKS_N: tl.constexpr,
):
    """
    SSA backward kernel computing dK, dV, dQ, dn, db.

    Grid: (N_CTX // BLOCK_N1, 1, Z * H_kv)

    For each KV block:
      1. Compute dK, dV by iterating over all Q-heads (GQA) and Q-blocks
      2. Compute dQ by iterating over KV blocks (from the same program)

    This follows the tutorial's structure but adapts for GQA and SSA.
    """
    pid = tl.program_id(0)
    off_bkv = tl.program_id(2)  # flattened Z * H_kv

    # Load SSA params
    ssa_n = tl.load(ssa_n_ptr).to(tl.float32)
    ssa_b = tl.load(ssa_b_ptr).to(tl.float32)
    ssa_nb = ssa_n * ssa_b

    offs_d = tl.arange(0, HEAD_DIM)

    # Tutorial-style explicit z/h decomposition.
    H_KV = H_Q // GQA_RATIO
    off_z = off_bkv // H_KV
    off_h_kv = off_bkv % H_KV

    # ========== Part 1: dK, dV ==========
    start_n = pid * BLOCK_N1
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    # K/V/dK/dV base.
    k_base = K + off_z * stride_kz + off_h_kv * stride_kh
    v_base = V + off_z * stride_vz + off_h_kv * stride_vh
    dk_base = dK + off_z * stride_dkz + off_h_kv * stride_dkh
    dv_base = dV + off_z * stride_dvz + off_h_kv * stride_dvh

    # Load K, V for this block (stay in SRAM throughout)
    k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    k_mask = offs_n[:, None] < N_CTX
    k = tl.load(k_ptrs, mask=k_mask, other=0.0)

    v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    v = tl.load(v_ptrs, mask=k_mask, other=0.0)

    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dn_acc = 0.0
    dn_comp = 0.0
    db_acc = 0.0
    db_comp = 0.0

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR

    # Iterate over all GQA_RATIO Q-heads sharing this KV head
    for g in range(0, GQA_RATIO):
        off_h_q = off_h_kv * GQA_RATIO + g

        q_base = Q + off_z * stride_qz + off_h_q * stride_qh
        do_base = dO + off_z * stride_doz + off_h_q * stride_doh
        l_base = L + off_z * stride_lz + off_h_q * stride_lh
        d_base = Delta + off_z * stride_dez + off_h_q * stride_deh

        start_m = 0
        if CAUSAL:
            # On the diagonal: use smaller sub-blocks (BLK_SLICE_FACTOR)
            start_m = start_n
            num_steps = BLOCK_N1 // MASK_BLOCK_M1
            dk, dv, dn_acc, dn_comp, db_acc, db_comp = _ssa_attn_bwd_dkdv(
                dk, dv,
                dn_acc, dn_comp, db_acc, db_comp,
                Q, k, v, softmax_scale,
                dO, L, Delta,
                ssa_n, ssa_b, ssa_nb,
                stride_qm, stride_qk,
                stride_dom, stride_dok,
                stride_lm, stride_dem,
                q_base, do_base, l_base, d_base,
                offs_n,
                N_CTX, HEAD_DIM,
                MASK_BLOCK_M1, BLOCK_N1,
                MATMUL_PRECISION,
                start_n, start_m, num_steps,
                MASK=True,
            )
            start_m += num_steps * MASK_BLOCK_M1

        # Off-diagonal: full blocks, no mask
        num_steps = (N_CTX - start_m) // BLOCK_M1
        dk, dv, dn_acc, dn_comp, db_acc, db_comp = _ssa_attn_bwd_dkdv(
            dk, dv,
            dn_acc, dn_comp, db_acc, db_comp,
            Q, k, v, softmax_scale,
            dO, L, Delta,
            ssa_n, ssa_b, ssa_nb,
            stride_qm, stride_qk,
            stride_dom, stride_dok,
            stride_lm, stride_dem,
            q_base, do_base, l_base, d_base,
            offs_n,
            N_CTX, HEAD_DIM,
            BLOCK_M1, BLOCK_N1,
            MATMUL_PRECISION,
            start_n, start_m, num_steps,
            MASK=False,
        )

    # Store dK, dV
    dk_ptrs = dk_base + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
    tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=k_mask)

    dv_ptrs = dv_base + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk
    tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=k_mask)

    # Store partial dn, db
    dn_ptr = DN + off_bkv * NUM_BLOCKS_N + pid
    db_ptr = DB + off_bkv * NUM_BLOCKS_N + pid
    tl.store(dn_ptr, dn_acc)
    tl.store(db_ptr, db_acc)

    # ========== Part 2: dQ ==========
    # Each program also handles one Q-block per Q-head
    # Iterate over all GQA Q-heads for this KV index
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR

    for g in range(0, GQA_RATIO):
        off_h_q = off_h_kv * GQA_RATIO + g

        start_m_q = pid * BLOCK_M2
        offs_m_q = start_m_q + tl.arange(0, BLOCK_M2)

        q_base = Q + off_z * stride_qz + off_h_q * stride_qh
        do_base = dO + off_z * stride_doz + off_h_q * stride_doh
        l_base = L + off_z * stride_lz + off_h_q * stride_lh
        d_base = Delta + off_z * stride_dez + off_h_q * stride_deh
        dq_base = dQ + off_z * stride_dqz + off_h_q * stride_dqh

        # K/V base for dQ iteration uses the same KV head
        k_base_offset = off_z * stride_kz + off_h_kv * stride_kh
        v_base_offset = off_z * stride_vz + off_h_kv * stride_vh

        # Load Q, dO
        q_ptrs = q_base + offs_m_q[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q_mask = offs_m_q[:, None] < N_CTX
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        do_ptrs = do_base + offs_m_q[:, None] * stride_dom + offs_d[None, :] * stride_dok
        do = tl.load(do_ptrs, mask=q_mask, other=0.0)

        m_mask = offs_m_q < N_CTX
        L_row = tl.load(l_base + offs_m_q * stride_lm, mask=m_mask, other=0.0)
        Di = tl.load(d_base + offs_m_q * stride_dem, mask=m_mask, other=0.0)

        dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)

        start_n_q = 0
        num_steps = N_CTX // BLOCK_N2

        if CAUSAL:
            # Masked diagonal blocks (right to left)
            end_n = start_m_q + BLOCK_M2
            num_steps_mask = BLOCK_M2 // MASK_BLOCK_N2
            dq = _ssa_attn_bwd_dq(
                dq, q, K, V,
                do, L_row, Di,
                softmax_scale,
                ssa_n, ssa_b, ssa_nb,
                stride_kn, stride_kk,
                stride_vn, stride_vk,
                k_base_offset, v_base_offset,
                offs_m_q,
                N_CTX, HEAD_DIM,
                BLOCK_M2, MASK_BLOCK_N2,
                MATMUL_PRECISION,
                start_m_q, end_n - num_steps_mask * MASK_BLOCK_N2, num_steps_mask,
                MASK=True,
            )
            end_n -= num_steps_mask * MASK_BLOCK_N2
            num_steps = end_n // BLOCK_N2
            start_n_q = end_n - num_steps * BLOCK_N2

        # Non-masked blocks
        dq = _ssa_attn_bwd_dq(
            dq, q, K, V,
            do, L_row, Di,
            softmax_scale,
            ssa_n, ssa_b, ssa_nb,
            stride_kn, stride_kk,
            stride_vn, stride_vk,
            k_base_offset, v_base_offset,
            offs_m_q,
            N_CTX, HEAD_DIM,
            BLOCK_M2, BLOCK_N2,
            MATMUL_PRECISION,
            start_m_q, start_n_q, num_steps,
            MASK=False,
        )

        # Store dQ — each (Q-head, Q-block) is written by exactly one program,
        # so no atomic needed. The dQ inner loop already iterates over ALL KV blocks.
        dq_ptrs = dq_base + offs_m_q[:, None] * stride_dqm + offs_d[None, :] * stride_dqk
        tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=q_mask)


# ============================================================
# Python Wrapper Functions
# ============================================================

def _get_block_sizes(D):
    """Select tile sizes tuned for A100 (sm_80)."""
    BLOCK_D = triton.next_power_of_2(D)
    return BLOCK_D


def ssa_flash_attn_forward(q, k, v, softmax_scale, ssa_n, ssa_b, causal=True):
    """
    Forward pass wrapper.

    Args:
        q: [B, Hq, N, D]
        k: [B, Hkv, N, D]
        v: [B, Hkv, N, D]
        softmax_scale: float
        ssa_n, ssa_b: scalar tensors (float32, on device)
        causal: bool

    Returns:
        out: [B, Hq, N, D]
        m: [B, Hq, N] (log2-normalizer of transformed SSA logits)
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
    m = torch.empty((B, Hq, N), device=q.device, dtype=torch.float32)

    HEAD_DIM = _get_block_sizes(D)
    assert D == HEAD_DIM, (
        f"Head dim D={D} must be power-of-two for triton tutorial kernel; got padded HEAD_DIM={HEAD_DIM}."
    )
    stage = 3 if causal else 1

    if q.dtype == torch.float16:
        MATMUL_PRECISION = tl.float16
    elif q.dtype == torch.bfloat16:
        MATMUL_PRECISION = tl.bfloat16
    else:
        MATMUL_PRECISION = tl.float32

    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * Hq)

    _ssa_attn_fwd[grid](
        q, k, v, out, m,
        softmax_scale,
        ssa_n, ssa_b,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        B, Hq, N,
        HEAD_DIM=HEAD_DIM,
        GQA_RATIO=GQA_RATIO,
        STAGE=stage,
        MATMUL_PRECISION=MATMUL_PRECISION,
    )

    return out, m


def ssa_flash_attn_backward(q, k, v, out, dout, m, softmax_scale, ssa_n, ssa_b, causal=True):
    """
    Backward pass wrapper.

    Returns:
        dq, dk, dv, dn, db
    """
    B, Hq, N, D = q.shape
    Hkv = k.shape[1]
    GQA_RATIO = Hq // Hkv

    HEAD_DIM = _get_block_sizes(D)
    assert D == HEAD_DIM, (
        f"Head dim D={D} must be power-of-two for triton tutorial kernel; got padded HEAD_DIM={HEAD_DIM}."
    )

    # Block sizes for backward (following tutorial's choices)
    BLOCK_M1 = 32   # Q block size for dKV computation
    BLOCK_N1 = 128  # KV block size (main iteration)
    BLOCK_M2 = 128  # Q block size for dQ computation
    BLOCK_N2 = 32   # KV block size for dQ computation
    BLK_SLICE_FACTOR = 2
    # Blackwell (sm_100) backward tuning — SAFE subset: only warp count and pipeline
    # depth are tunable; BLOCK_* sizes stay frozen so the causal-masking math is
    # byte-identical (num_stages = no numerical change; num_warps = last-bit reduction
    # order only). Sweep per run without editing code, e.g.:
    #   SSA_BWD_NUM_STAGES=3 SSA_BWD_NUM_WARPS=8 sbatch train_ssa_triton.sh
    NUM_WARPS = int(os.environ.get("SSA_BWD_NUM_WARPS", "4"))
    NUM_STAGES = int(os.environ.get("SSA_BWD_NUM_STAGES", "2"))

    if q.dtype == torch.float16:
        MATMUL_PRECISION = tl.float16
    elif q.dtype == torch.bfloat16:
        MATMUL_PRECISION = tl.bfloat16
    else:
        MATMUL_PRECISION = tl.float32

    # Precompute Delta = rowsum(O * dO)
    delta = torch.empty((B, Hq, N), device=q.device, dtype=torch.float32)
    PRE_BLOCK = 128
    pre_grid = (triton.cdiv(N, PRE_BLOCK), B * Hq)

    _ssa_attn_bwd_preprocess[pre_grid](
        out, dout, delta,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_CTX=N,
        BLOCK_M=PRE_BLOCK,
        HEAD_DIM=HEAD_DIM,
    )

    # Allocate gradient tensors
    # dQ uses regular store (each Q-block written by exactly one program)
    # dK, dV likewise (each KV-block written by exactly one program)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    num_kv_blocks = N // BLOCK_N1
    dn_partial = torch.zeros((B * Hkv, num_kv_blocks), device=q.device, dtype=torch.float32)
    db_partial = torch.zeros((B * Hkv, num_kv_blocks), device=q.device, dtype=torch.float32)

    assert N % BLOCK_N1 == 0, f"N_CTX={N} must be divisible by BLOCK_N1={BLOCK_N1}"
    assert N % BLOCK_M2 == 0, f"N_CTX={N} must be divisible by BLOCK_M2={BLOCK_M2}"
    assert BLOCK_N1 == BLOCK_M2, (
        f"BLOCK_N1={BLOCK_N1} must equal BLOCK_M2={BLOCK_M2} so that the same grid "
        "dimension handles both dKV (one KV-block per program) and dQ (one Q-block per program)"
    )

    grid = (N // BLOCK_N1, 1, B * Hkv)

    _ssa_attn_bwd[grid](
        q, k, v,
        out, dout,
        dq, dk, dv,
        m, delta,
        dn_partial, db_partial,
        softmax_scale,
        ssa_n, ssa_b,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        B, Hq, N,
        HEAD_DIM=HEAD_DIM,
        GQA_RATIO=GQA_RATIO,
        BLOCK_M1=BLOCK_M1,
        BLOCK_N1=BLOCK_N1,
        BLOCK_M2=BLOCK_M2,
        BLOCK_N2=BLOCK_N2,
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
        MATMUL_PRECISION=MATMUL_PRECISION,
        CAUSAL=causal,
        NUM_BLOCKS_N=num_kv_blocks,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    dn = dn_partial.sum()
    db = db_partial.sum()

    return dq, dk, dv, dn, db


# ============================================================
# Cache Warmup
# ============================================================

def warmup_ssa_triton_kernels(
    B: int = 2,
    Hq: int = 24,
    Hkv: int = 8,
    N: int = 128,
    D: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Pre-compile all Triton kernels by running a tiny dummy forward+backward."""
    q = torch.randn(B, Hq, N, D, dtype=dtype, device=device)
    k = torch.randn(B, Hkv, N, D, dtype=dtype, device=device)
    v = torch.randn(B, Hkv, N, D, dtype=dtype, device=device)
    ssa_n = torch.tensor(1.5, dtype=torch.float32, device=device)
    ssa_b = torch.tensor(0.8, dtype=torch.float32, device=device)
    scale = 1.0 / (D ** 0.5)

    out, m = ssa_flash_attn_forward(q, k, v, scale, ssa_n, ssa_b, causal=True)

    dout = torch.randn_like(out)
    dq, dk, dv, dn, db = ssa_flash_attn_backward(
        q, k, v, out, dout, m, scale, ssa_n, ssa_b, causal=True,
    )

    torch.cuda.synchronize()
