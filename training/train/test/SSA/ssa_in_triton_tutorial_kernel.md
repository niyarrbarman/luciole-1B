# Integrating SSA into the Triton Tutorial Fused Attention Kernel

## 1. The Two Kernels Side by Side

### What the Tutorial Does (Standard Softmax)

The Triton tutorial (`06-fused-attention.py`) implements Flash Attention v2 with a **base-2 online softmax** trick. The pipeline for each query row is:

```
s_j = scale * dot(q, k_j)                    # raw score
u_j = s_j / ln(2) = s_j * 1.44269504        # convert to base-2
m   = max_j(u_j)                              # running max (base-2)
p_j = 2^(u_j - m)                            # unnormalized weight
l   = sum_j(p_j)                              # running normalizer
out = sum_j(p_j * v_j) / l                   # output
M   = m + log2(l) = log2(sum_j exp(s_j))     # stored for backward
```

The exp2/log2 formulation is a pure performance optimization -- `exp2` and `log2` are faster than `exp`/`log` on NVIDIA hardware. The result is mathematically identical to standard softmax.

### What SSA Does

Your SSA kernel replaces the softmax normalization with a different weight function:

```
s_j = scale * dot(q, k_j)                    # raw score (same)
w_j = exp(n * sign(s_j) * log(1 + b*|s_j|))  # SSA weight
    = (1 + b*|s_j|)^(n * sign(s_j))          # equivalent power form
l   = sum_j(w_j)                              # normalizer
out = sum_j(w_j * v_j) / l                   # output (same)
```

The key difference: **the score-to-weight mapping** is `s -> exp(n*sign(s)*log(1+b|s|))` instead of `s -> exp(s)`.

---

## 2. The Fundamental Question: Can SSA Use Online Softmax?

The tutorial's online softmax trick relies on a critical mathematical property of `exp`:

```
exp(x - m_new) = exp(x - m_old) * exp(m_old - m_new)
```

This means when the running max changes from `m_old` to `m_new`, you can **rescale** all previously accumulated values by a single scalar `alpha = exp(m_old - m_new)`. This is what makes Flash Attention work -- you never need to go back and recompute old blocks.

**SSA's weight function does NOT have this property.** The SSA weight is:

```
w(s) = (1 + b|s|)^(n * sign(s))
```

There is no global "max" you can subtract and later correct for. The weight of each score depends only on that score -- there's no shared base to factor out. SSA weights are not log-linear in the scores the way `exp(s)` is.

### What your current SSA kernel does instead

Your kernel uses **simple accumulate-then-divide**:

```python
# Forward inner loop:
w_sum_i = 0.0    # no running max needed
acc = 0.0

for each KV block:
    s = Q @ K^T * scale
    # SSA weight computation (no max subtraction needed)
    ssa_w = exp(n * sign(s) * log(1 + b*|s|))
    acc += ssa_w @ V
    w_sum_i += sum(ssa_w)

# Single division at the end
acc /= w_sum_i
```

This is simpler than the tutorial's online approach but **just as correct** because:
1. SSA weights are always positive (exp of anything is positive)
2. SSA weights are bounded -- unlike `exp(s)` which can overflow for large `s`, `(1+b|s|)^n` grows polynomially, not exponentially
3. No numerical stability issues from large exponentials -- `log(1+x)` compresses large values

**This is actually an advantage of SSA** -- you don't need the max-subtraction trick at all.

---

## 3. Exact Integration: Forward Kernel

Here's how to modify `_attn_fwd_inner` from the tutorial to use SSA. I'll show both the original and modified versions with annotations.

### Tutorial's `_attn_fwd_inner` (original)

```python
# BEFORE loop:
m_i = zeros - inf     # running max
l_i = zeros + 1.0     # running normalizer (init 1.0 for stability)
acc = zeros

# INSIDE loop:
qk = tl.dot(q, k)
m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)     # [1] update max
qk = qk * qk_scale - m_ij[:, None]                     # [2] shift by max
p = tl.math.exp2(qk)                                    # [3] compute weights
alpha = tl.math.exp2(m_i - m_ij)                        # [4] correction factor
acc = acc * alpha[:, None]                               # [5] rescale accumulator
acc = tl.dot(p, v, acc)                                  # [6] accumulate P @ V
l_i = l_i * alpha + l_ij                                 # [7] update normalizer
m_i = m_ij

# AFTER loop (epilogue):
m_i += tl.math.log2(l_i)                                # [8] fuse into M
acc = acc / l_i[:, None]                                 # [9] normalize
```

### SSA-Modified `_attn_fwd_inner`

```python
# BEFORE loop:
# No m_i needed -- SSA weights don't need max subtraction
w_sum_i = tl.zeros([BLOCK_M], dtype=tl.float32)    # just a sum
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

# Load SSA params (once, before loop)
ssa_n = tl.load(ssa_n_ptr).to(tl.float32)
ssa_b = tl.load(ssa_b_ptr).to(tl.float32)

# INSIDE loop:
k = load_k_block(...)
s = tl.dot(q, k)                                        # [1] raw QK^T
s = s * softmax_scale                                    # [2] apply scale

if IS_CAUSAL:
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    s = tl.where(mask, s, float('-inf'))

# --- SSA TRANSFORMATION (replaces steps [1]-[5] of tutorial) ---
s_fp32 = s.to(tl.float32)
valid = s_fp32 > float('-inf')
s_safe = tl.where(valid, s_fp32, 0.0)

abs_s = tl.abs(s_safe)
sign_s = tl.where(s_safe > 0, 1.0,
            tl.where(s_safe < 0, -1.0, 0.0))

one_plus_bs = 1.0 + ssa_b * abs_s
log_opbs = tl.log(one_plus_bs)                           # log(1 + b|s|)
ssa_w = tl.where(valid,
            tl.exp(ssa_n * sign_s * log_opbs),           # the SSA weight
            0.0)
# ---------------------------------------------------------------

v = load_v_block(...)
acc += tl.dot(ssa_w.to(v.dtype), v).to(tl.float32)      # [6] accumulate
w_sum_i += tl.sum(ssa_w, axis=1)                         # [7] accumulate normalizer

# AFTER loop (epilogue):
w_sum_safe = tl.where(w_sum_i > 0.0, w_sum_i, 1.0)
acc = acc / w_sum_safe[:, None]                           # [9] normalize

# Store L = w_sum_i (for backward), NOT log2-encoded
tl.store(L_ptrs, w_sum_i)
tl.store(Out_ptrs, acc)
```

### What Changed and Why

| Tutorial Step | What It Does | SSA Equivalent | Why |
|---|---|---|---|
| `m_i` (running max) | Numerical stability for exp | **Removed** | SSA weights don't overflow -- polynomial growth not exponential |
| `qk_scale = sm_scale / ln(2)` | Convert to base-2 | **Just sm_scale** | No base-2 trick needed |
| `exp2(qk - m)` | Compute softmax weight | `exp(n * sign(s) * log(1+b*|s|))` | Different weight function entirely |
| `alpha = exp2(m_old - m_new)` | Rescale old accumulator | **Removed** | No running max means no rescaling |
| `m_i += log2(l_i)` | Encode LSE for backward | **Store raw w_sum** | Backward recomputes SSA weights directly |

### What Stays Identical

- `Q @ K^T` computation
- Causal masking
- `P @ V` accumulation (dot product with values)
- Final division by normalizer
- Grid structure: `(cdiv(N_CTX, BLOCK_M), B * H)`
- Block pointer management

---

## 4. Exact Integration: Backward Kernels

The backward pass is where the real complexity lies. The tutorial has three backward kernels:
1. `_attn_bwd_preprocess` -- compute `Di = rowsum(O * dO)`
2. `_attn_bwd_dkdv` -- compute dK, dV
3. `_attn_bwd_dq` -- compute dQ

Plus SSA adds:
4. Gradients for `n` and `b` (which the tutorial doesn't have)

### 4.1 Preprocess Kernel: UNCHANGED

```python
# Di = sum_d(O[i,d] * dO[i,d])  -- identical for both
# This is because Di comes from the chain rule d(P@V)/dP,
# and the P@V step is identical in both formulations.
```

This kernel is the same regardless of whether you use softmax or SSA.

### 4.2 Backward Chain Rule Comparison

For standard softmax:
```
P_j = exp(z_j) / sum exp(z_k)
dL/dz_j = P_j * (dp_j - Di)              where dp = dO @ V^T, Di = dO . O
```

For SSA:
```
f(s) = n * sign(s) * log(1 + b|s|)       # SSA transform
w(s) = exp(f(s))                          # unnormalized weight
P_j  = w(s_j) / sum_k w(s_k)             # normalized probability

# The gradient through SSA normalization + SSA transform:
dL/ds_ssa_j = P_j * (dp_j - Di)          # gradient through normalization (SAME form as softmax!)
df/ds_j = n * b / (1 + b|s_j|)           # derivative of SSA transform w.r.t. raw score
dL/ds_j = dL/ds_ssa_j * df/ds_j          # chain rule
```

**Critical insight**: The gradient through the normalization step `w/sum(w)` has the **same algebraic form** as the softmax gradient: `P * (dp - Di)`. This is because any function of the form `f_j / sum(f_k)` has this gradient structure. The only difference is what `P` actually equals.

The extra factor `df/ds = n*b / (1+b|s|)` is the chain rule through the SSA score transformation that softmax doesn't have.

### 4.3 dK, dV Kernel

**Tutorial version:**
```python
# K is pre-scaled: k' = K * sm_scale * RCP_LN2
qkT = tl.dot(k, qT)                      # = z_j / ln(2)
pT = tl.math.exp2(qkT - m[None, :])      # recompute P from M
# ...
dpT = tl.dot(v, tl.trans(do))
dsT = pT * (dpT - Di[None, :])            # softmax gradient
dk += tl.dot(dsT, tl.trans(qT))           # accumulate dK
dv += tl.dot(ppT, do)                     # accumulate dV
```

**SSA version:**
```python
# No pre-scaling of K needed.
for each Q block:
    s = tl.dot(q, tl.trans(k)) * softmax_scale
    # Apply causal mask
    
    # --- Recompute SSA weights (same code as forward) ---
    s_fp32 = s.to(tl.float32)
    valid = s_fp32 > float('-inf')
    s_safe = tl.where(valid, s_fp32, 0.0)
    abs_s = tl.abs(s_safe)
    sign_s = tl.where(s_safe > 0, 1.0, tl.where(s_safe < 0, -1.0, 0.0))
    one_plus_bs = 1.0 + ssa_b * abs_s
    log_opbs = tl.log(one_plus_bs)
    ssa_w = tl.where(valid, tl.exp(ssa_n * sign_s * log_opbs), 0.0)
    
    # Normalize to get P
    row_sum_w = tl.load(L + ...)       # stored from forward
    p = ssa_w / row_sum_w[:, None]
    # -------------------------------------------------------
    
    # dV (identical structure)
    dv += tl.dot(tl.trans(p), do)
    
    # dp and ds_ssa (identical structure to softmax gradient)
    dp = tl.dot(do, tl.trans(v))
    ds_ssa = p * (dp - Di[:, None])
    
    # --- SSA-SPECIFIC: chain rule through f(s) ---
    df_ds = ssa_n * ssa_b / one_plus_bs    # n*b / (1+b|s|)
    ds = ds_ssa * df_ds
    # -----------------------------------------------
    
    dk += tl.dot(tl.trans(ds), q) * softmax_scale
    
    # --- SSA-SPECIFIC: accumulate dn, db ---
    block_dn = tl.sum(ds_ssa * sign_s * log_opbs)
    block_db = tl.sum(ds_ssa * ssa_n * sign_s * abs_s / one_plus_bs)
    dn_acc += block_dn   # (with Kahan summation ideally)
    db_acc += block_db
    # ------------------------------------------
```

### 4.4 dQ Kernel

**Tutorial version:**
```python
qk = tl.dot(q, kT)               # kT pre-scaled by sm_scale/ln2
p = tl.math.exp2(qk - m)         # recompute P
dp = tl.dot(do, vT)
ds = p * (dp - Di[:, None])       # softmax gradient
dq += tl.dot(ds, tl.trans(kT))   # accumulate dQ
# Final: dq *= LN2               # base-2 correction
```

**SSA version:**
```python
for each KV block:
    s = tl.dot(q, tl.trans(k)) * softmax_scale
    # Apply causal mask
    
    # --- Recompute SSA weights ---
    # (same SSA recomputation code as in dKV kernel)
    p = ssa_w / row_sum_w[:, None]
    
    dp = tl.dot(do, tl.trans(v))
    ds_ssa = p * (dp - Di[:, None])
    
    # Chain rule through SSA transform
    df_ds = ssa_n * ssa_b / one_plus_bs
    ds = ds_ssa * df_ds
    
    dq += tl.dot(ds, k) * softmax_scale   # note: k is NOT pre-scaled
    
# No LN2 correction needed -- SSA doesn't use base-2 trick
```

### 4.5 Why No LN2 Correction for SSA

The tutorial's LN2 correction exists because:
1. The tutorial uses `exp2` instead of `exp`
2. The derivative of `exp2(x)` is `ln(2) * exp2(x)`, not `exp2(x)`
3. The code omits `ln(2)` from the per-element gradient to save FLOPs
4. Then corrects `dq` by multiplying by `ln(2)` once at the end

SSA uses natural `exp` and `log` throughout. There is no base-2 optimization, so no correction is needed. (You could optionally convert SSA to base-2, but the SSA weight computation already requires `tl.log` and `tl.exp`, so the benefit would be marginal.)

---

## 5. The Tutorial's "Stages" and How SSA Maps to Them

The tutorial splits causal attention into stages:
- **Stage 1** (STAGE=1): Process all KV blocks **before** the diagonal (fully unmasked)
- **Stage 2** (STAGE=2): Process the **diagonal** block (partially masked)
- **Stage 3** (STAGE=3, non-causal): Process all blocks (no masking)

For SSA, stages still make sense:
- **Off-diagonal blocks**: No causal mask check needed, SSA weights computed normally
- **Diagonal block**: Need `tl.where(mask, s, -inf)` before SSA weight computation

Your current kernel handles this with a simple `if IS_CAUSAL` check inside the loop. The tutorial's stage approach is more optimized because it avoids the mask check entirely for the majority of blocks. You could adopt the same two-stage approach for SSA.

### Mapping

```
Stage 1 (off-band, lo=0, hi=start_m*BLOCK_M):
    # No mask check needed
    s = Q @ K^T * scale
    # SSA weight (no causal masking)
    ssa_w = exp(n * sign(s) * log(1 + b|s|))

Stage 2 (on-band, lo=start_m*BLOCK_M, hi=(start_m+1)*BLOCK_M):
    s = Q @ K^T * scale
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    s = tl.where(mask, s, float('-inf'))
    # SSA weight (masked positions get w=0 via the valid check)
    ssa_w = tl.where(valid, exp(n * sign(s) * log(1 + b|s|)), 0.0)
```

---

## 6. What the Tutorial Has That Your Kernel Doesn't (Potential Upgrades)

### 6.1 TensorDescriptor API (Hopper/Blackwell)

The tutorial uses `TensorDescriptor` for hardware-accelerated memory access on SM90+:
```python
desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
k = desc_k.load([offset_y, 0]).T
```

This replaces manual pointer arithmetic with hardware TMA (Tensor Memory Accelerator) units. Your kernel uses explicit pointer computation. On Hopper/Blackwell, TMA can significantly improve memory throughput.

**To adopt**: Replace your `tl.load(k_ptrs, mask=k_mask, other=0.0)` calls with `desc_k.load([offset, 0])` when running on SM90+. Requires contiguous layout.

### 6.2 Warp Specialization

The tutorial supports `warp_specialize=True` which assigns different warps to different tasks (e.g., some warps handle data loading while others handle computation). This is a Blackwell/Hopper optimization.

```python
for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
```

**To adopt**: Add `warp_specialize` as a constexpr parameter. The SSA weight computation (log, exp, sign, abs) is compute-heavy enough that overlapping it with memory loads could be beneficial.

### 6.3 FP8 Support

The tutorial supports `float8_e5m2` for keys/values. This halves memory bandwidth requirements. However, SSA's weight computation requires fp32 precision (especially the log and exp), so the benefit is limited to the Q@K^T and P@V matmuls.

### 6.4 Autotuning with Config Pruning

The tutorial does aggressive autotuning:
```python
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
    for BM in [64, 128] for BN in [32, 64, 128] for s in [2,3,4] for w in [4, 8]
]
```

Your v3 kernel only autotunes `num_warps` and `num_stages` with fixed `BLOCK_M/BLOCK_N`. You could autotune block sizes too, but ensure `BLOCK_M >= BLOCK_N` for causal (as the tutorial does via `prune_invalid_configs`).

### 6.5 The BLK_SLICE_FACTOR Trick

The tutorial's backward uses `BLK_SLICE_FACTOR=2`, which means the diagonal block processing uses smaller sub-blocks (`MASK_BLOCK_M1 = BLOCK_M1 // 2`). This reduces wasted computation on the partially-masked diagonal. Your kernel doesn't have this optimization.

---

## 7. What Your Kernel Has That the Tutorial Doesn't

### 7.1 GQA Support

Your kernel maps Q-heads to KV-heads via `kv_idx = pid_bh // GQA_RATIO`. The tutorial assumes `H_q == H_kv`. This is a significant practical advantage.

### 7.2 Learnable Parameter Gradients (dn, db)

The tutorial has no learnable parameters in the attention mechanism. Your kernel computes gradients for `n` and `b` inside the backward kernel with Kahan compensated summation.

### 7.3 Stride-Based Layout

Your kernel accepts arbitrary strides, allowing non-contiguous tensors. The tutorial requires contiguous layout (or uses TensorDescriptors).

### 7.4 No O(N^2) Storage

Both kernels are O(N) memory. But your kernel stores `L = w_sum_i` (raw sum) while the tutorial stores `M = log2(LSE)` (log-encoded). Both serve the same purpose -- enabling weight recomputation in the backward.

---

## 8. Concrete Integration Recipe

If you want to take the tutorial kernel and add SSA to it, here is the step-by-step recipe:

### Step 1: Forward Kernel Modifications

In `_attn_fwd`:

1. **Add SSA parameters**: `ssa_n_ptr, ssa_b_ptr` as kernel arguments
2. **Remove base-2 scaling**: Delete `qk_scale *= 1.44269504` (keep just `sm_scale`)
3. **Change accumulators**: Replace `m_i = -inf, l_i = 1.0` with `w_sum_i = 0.0`
4. **Change epilogue**: Replace `m_i += log2(l_i); acc /= l_i` with `acc /= w_sum_i`; store `w_sum_i` instead of `m_i`

In `_attn_fwd_inner`:

5. **Replace weight computation** (the 5 lines from `m_ij = ...` through `p = exp2(...)` with:
```python
s = tl.dot(q, k) * softmax_scale       # scale here, not via qk_scale
if STAGE == 2:  # causal diagonal
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    s = tl.where(mask, s, float('-inf'))

# SSA weights
s_fp32 = s.to(tl.float32)
valid = s_fp32 > float('-inf')
s_safe = tl.where(valid, s_fp32, 0.0)
abs_s = tl.abs(s_safe)
sign_s = tl.where(s_safe > 0, 1.0, tl.where(s_safe < 0, -1.0, 0.0))
one_plus_bs = 1.0 + ssa_b * abs_s
ssa_w = tl.where(valid, tl.exp(ssa_n * sign_s * tl.log(one_plus_bs)), 0.0)
```

6. **Remove online rescaling**: Delete `alpha = exp2(m_i - m_ij)` and `acc *= alpha[:, None]` and `l_i = l_i * alpha + l_ij`. Replace with:
```python
acc = tl.dot(ssa_w.to(v.dtype), v, acc)
w_sum_i += tl.sum(ssa_w, axis=1)
```

### Step 2: Backward Preprocess

7. **No changes needed** -- `Di = rowsum(O * dO)` is identical.

### Step 3: dK, dV Kernel

In `_attn_bwd_dkdv`:

8. **Remove K pre-scaling**: In the Python wrapper, remove `arg_k = k * (sm_scale * RCP_LN2)`. Pass `k` directly.
9. **Replace weight recomputation**: Replace `pT = exp2(qkT - m[None, :])` with SSA weight recomputation (same as forward + normalize by L).
10. **Add SSA chain rule**: After computing `dsT = pT * (dpT - Di)`, multiply by `df/ds`:
```python
df_ds = ssa_n * ssa_b / one_plus_bs
dsT = dsT * df_ds
```
11. **Add dn, db accumulation**:
```python
dn_acc += tl.sum(ds_ssa * sign_s * log_opbs)
db_acc += tl.sum(ds_ssa * ssa_n * sign_s * abs_s / one_plus_bs)
```
12. **Fix dk scaling**: Replace `dk *= sm_scale` at the end with `dk *= softmax_scale` (multiply by scale inside the loop instead, or keep it at the end -- just don't use RCP_LN2).

### Step 4: dQ Kernel

In `_attn_bwd_dq`:

13. **Replace weight recomputation**: Same as dKV -- recompute SSA weights and normalize by L.
14. **Add SSA chain rule**: Same `ds *= df_ds` multiplication.
15. **Remove LN2 correction**: Delete the final `dq *= LN2` line. SSA doesn't use base-2.
16. **Fix scaling**: Use `softmax_scale` directly (not divided by ln2).

### Step 5: Autograd Wrapper

In the `_attention` class:

17. **Add ssa_n, ssa_b to forward signature**: Pass as additional arguments.
18. **Save ssa_n, ssa_b for backward**: Add to `ctx.save_for_backward(...)`.
19. **Allocate dn_partial, db_partial in backward**: Same as your current kernel -- shape `[B*Hkv, num_kv_blocks]`, sum at the end.
20. **Return gradients for ssa_n, ssa_b**: Add to the backward return tuple.

---

## 9. The Online Normalization Question: Is It Actually Needed?

Your current SSA kernel uses **accumulate-then-divide** (no online rescaling). This is simpler than the tutorial's online softmax. The natural question: should you adopt the online approach?

**No, and here's why:**

For standard softmax, `exp(s)` can overflow fp32 for `s > 88`. The max-subtraction trick is **essential** to prevent this. Without it, `exp(1000) = inf` and the kernel crashes.

For SSA, the weight is `(1 + b|s|)^(n*sign(s))`. For typical values:
- `s` is in the range `[-20, +20]` (after Q@K^T scaling)
- `b = 0.8`, `n = 1.5`
- Maximum weight: `(1 + 0.8 * 20)^1.5 = 17^1.5 = 70`
- Minimum weight: `(1 + 0.8 * 20)^(-1.5) = 17^(-1.5) = 0.014`

These are well within fp32 range. No overflow, no underflow. The accumulate-then-divide approach is numerically safe.

An online version *is* possible (using a running-max-of-log-weights trick), but it would add complexity for no numerical benefit and would slow down the kernel with extra rescaling operations.

---

## 10. Performance Considerations

### SSA vs Softmax Compute Cost per Block

| Operation | Softmax (tutorial) | SSA |
|---|---|---|
| Q @ K^T | 1 matmul | 1 matmul (same) |
| Weight computation | 1 exp2 | 1 abs + 1 sign + 1 fma + 1 log + 1 mul + 1 exp (6 ops) |
| Online rescaling | 1 exp2 + 1 mul + 1 add | None (0 ops) |
| P @ V | 1 matmul | 1 matmul (same) |
| **Net extra ops** | - | +5 transcendentals, -1 exp2 - 1 mul |

SSA is compute-heavier per block due to the log/exp chain, but avoids the online rescaling overhead. The matmul operations dominate for large HEAD_DIM, so the extra SSA arithmetic is a small fraction of total compute.

### Backward: SSA Has Additional Work

| Operation | Softmax | SSA |
|---|---|---|
| Recompute P | 1 exp2 | 6 ops (same as forward) |
| ds = P*(dp-Di) | 2 ops | 2 ops (same) |
| Chain rule df/ds | 0 | 1 div + 1 mul (extra) |
| dn, db accumulation | 0 | ~6 ops per block (extra) |
| K pre-scaling | 1 (amortized) | 0 |

The backward is where SSA pays the most vs softmax -- the SSA weight recomputation + chain rule + parameter gradient accumulation adds meaningful compute. This is consistent with your profiling showing the backward kernels dominate GPU time.

---

## 11. Summary: What to Do

**If you're building a clean SSA kernel from the tutorial template:**

1. **Keep**: Grid structure, block pointer management, tiled Q/K/V loading, separate dQ and dKV kernels, BLK_SLICE_FACTOR diagonal optimization, autotuning framework
2. **Modify**: Weight computation (SSA instead of exp2), remove online rescaling, remove base-2 tricks, add ssa_n/ssa_b parameters, add dn/db gradients
3. **Consider adding**: TensorDescriptor support for Hopper+, warp specialization, stage-based causal handling (instead of per-element mask check)

**If you're keeping your current kernel and just want to steal ideas from the tutorial:**

1. **BLK_SLICE_FACTOR**: Process the causal diagonal with smaller sub-blocks to reduce waste
2. **Stage-based causal**: Split the KV iteration into off-diagonal (no mask check) and diagonal (with mask) -- avoids branching in the majority of iterations
3. **TensorDescriptor**: For Hopper/Blackwell targets
4. **Wider autotuning**: Autotune BLOCK_M and BLOCK_N in addition to warps/stages

Your current kernel is already well-structured. The main inefficiency compared to the tutorial is the per-iteration `if IS_CAUSAL` check on every block (vs the tutorial's pre-split stages). The tutorial's approach can save ~5-10% on causal attention by avoiding the mask check on off-diagonal blocks.
