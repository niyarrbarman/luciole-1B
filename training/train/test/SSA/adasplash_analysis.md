# AdaSplash Analysis: Relevance to SSA Attention

## 1. What Is AdaSplash?

AdaSplash ("Adaptive Sparse Flash Attention") is a Triton-based implementation of **alpha-entmax attention** -- a sparse alternative to softmax. Published at ICML 2025 by Goncalves, Treviso, and Martins (Deep-Spin group, Instituto de Telecomunicacoes / IST).

Paper: https://arxiv.org/abs/2502.12082

Repository: https://github.com/deep-spin/adasplash (MIT license)

### Core Idea

Standard attention uses softmax to produce a **dense** probability distribution over keys. AdaSplash replaces softmax with **alpha-entmax**, which produces **sparse** attention weights -- many entries are exactly zero. The sparsity is data-dependent and controlled by the alpha hyperparameter.

---

## 2. Mathematical Comparison: Entmax vs SSA

### AdaSplash (alpha-entmax)

Given raw attention scores `z = Q @ K^T`:

```
p* = argmax_p { p^T z - H_alpha(p) }
   where H_alpha(p) = (1/(alpha*(alpha-1))) * sum_j(p_j - p_j^alpha)
```

The solution is:
```
p_j = [ (z_j - tau)_+ ]^{1/(alpha-1)}    (then normalized to sum to 1)
```

Where `tau` is found by bisection (or Halley's method) such that `sum(p_j) = 1`.

- **alpha = 1** --> softmax (no sparsity)
- **alpha = 1.5** (default) --> 1.5-entmax, moderate sparsity
- **alpha = 2** --> sparsemax, maximum sparsity

**Key properties**:
- Produces **exact zeros** in the output distribution
- Sparsity is **adaptive** (data-dependent) -- different queries produce different sparsity patterns
- alpha is a **fixed hyperparameter** in AdaSplash (NOT learned)

### SSA (Softmax-Substituted Attention)

Given raw attention scores `s = scale * Q @ K^T`:

```
f(s) = n * sign(s) * log(1 + b * |s|)
p = softmax(f(s))
```

- **n >= 1**: learnable sharpness parameter
- **b > 0**: learnable magnitude scaling
- Both are **learnable nn.Parameters** with gradients computed inside the Triton kernel

**Key properties**:
- Output is **dense** (softmax always produces non-zero values)
- The log-linear transformation **compresses** large scores and **preserves** small ones
- Parameters n and b are **learned during training** and their gradients are fused into the backward kernel

### Summary Table

| Property | AdaSplash (entmax) | SSA |
|---|---|---|
| **Output distribution** | Sparse (exact zeros) | Dense (all positive) |
| **Transformation** | Replace softmax entirely with entmax | Transform scores, then apply softmax |
| **Learnable params** | None (alpha is a hyperparameter) | n, b (scalar, learned end-to-end) |
| **Sparsity** | Yes, adaptive | No |
| **Key math** | `[z - tau]_+^{1/(alpha-1)}` | `softmax(n * sign(s) * log(1 + b*|s|))` |
| **Backward complexity** | Entmax JVP with `gppr = (a-1) * p^{2-a}` | SSA chain rule: `df/ds = n*b / (1+b*|s|)` |

---

## 3. Do They Rewrite Attention? Yes.

AdaSplash **completely replaces** the softmax normalization step in attention with alpha-entmax. This is not just a pre/post-processing of scores -- it fundamentally changes the normalization function.

SSA, by contrast, **wraps** softmax: it applies a nonlinear score transformation `f(s)` and then uses standard softmax on the transformed scores.

Both approaches:
- Keep Q @ K^T and the final P @ V matmuls unchanged
- Only modify what happens between computing raw scores and producing attention weights

---

## 4. Learnable Parameters

### AdaSplash: NO learnable attention parameters

Alpha is a **fixed hyperparameter** set at initialization (default 1.5). It is passed as a `tl.constexpr` to the Triton kernels. The backward pass computes gradients for Q, K, V only -- never for alpha.

```python
# AdaSplash forward signature
def adasplash_attention(q, k, v, is_causal=False, alpha=1.5)
```

The alpha parameter cannot be learned because:
1. It appears in the **exponent** of the normalization (`p^{1/(alpha-1)}`)
2. The entmax threshold `tau` depends on alpha through an iterative solver
3. Making alpha differentiable would require differentiating through 40 bisection iterations

### SSA: YES, learnable n and b

```python
# SSA parameters
self.ssa_n_raw = nn.Parameter(torch.tensor(1.5))  # learnable
self.ssa_b_raw = nn.Parameter(torch.tensor(0.8))  # optionally learnable
```

Gradients for n and b are computed **inside the Triton backward kernel** using Kahan compensated summation:
```
dn = sum_ij( ds_ssa_ij * sign(s_ij) * log(1 + b*|s_ij|) )
db = sum_ij( ds_ssa_ij * n * sign(s_ij) * |s_ij| / (1 + b*|s_ij|) )
```

This is a significant advantage of SSA: the attention behavior adapts during training through gradient descent on n and b.

---

## 5. Triton Kernel Architecture Comparison

### AdaSplash Triton Kernels

**File structure**: 3 files
- `triton_entmax.py` -- standalone entmax activation (bisection-based)
- `adasplash_block_mask.py` -- fused attention with FlexAttention-style block masks
- `adasplash_no_block_mask.py` -- fused attention without block masks

**Forward kernel** (`adasplash_no_block_mask.py`):
```
Grid: (cdiv(N_CTX, BLOCK_M), Z * H)

For each Q-block:
  For each KV-block:
    s = Q @ K^T
    Apply causal mask
    m_ij = max(m_i, max(s))           # running max for numerical stability
    p = max(s - m_ij, 0)^{1/(a-1)}   # entmax transformation
    l_ij = sum(p)                      # normalizer
    Rescale old accumulator by (m_i - m_ij)^{1/(a-1)}  # "online" entmax trick
    acc += p @ V
    Store p to Scores tensor          # <-- CRITICAL: materializes full NxN scores!
```

**Backward kernel**:
```
Grid: (cdiv(N_CTX, BLOCK_N), Z * H)

For each KV-block:
  For each Q-block:
    Load saved scores p
    p = p / l_i                        # normalize
    dv += p^T @ dO
    dp = dO @ V^T
    # Entmax JVP (Jacobian-vector product):
    gppr = (alpha-1) * p^{2-alpha}     # entmax derivative
    ds = dp*gppr - gppr * (sum(dp*gppr) / sum(gppr))
    dk += ds^T @ Q
    dq = ds @ K^T
```

### SSA Triton Kernels

**File structure**: 4 files
- `ssa_triton_kernel.py` -- v2: forward + backward with GQA
- `ssa_triton_kernel_optimized.py` -- v3: + autotuning
- `ssa_flash_attention.py` -- autograd.Function wrapper
- `ssa_triton_attention.py` -- Megatron module wrapper

**Forward kernel**:
```
Grid: (cdiv(N_CTX, BLOCK_M), B * Hq)

For each Q-block:
  For each KV-block:
    s = Q @ K^T * scale
    Apply causal mask
    abs_s = |s|, sign_s = sign(s)
    one_plus_bs = 1 + b * abs_s
    ssa_w = exp(n * sign_s * log(one_plus_bs))  # SSA weight
    acc += ssa_w @ V
    w_sum_i += sum(ssa_w)              # running normalizer
  acc /= w_sum_i                       # single normalization at end
  Store w_sum_i to L                   # <-- O(N) storage only
```

**Backward kernels** (separate dQ and dKV):
```
dQ kernel: Grid (cdiv(N_CTX, BLOCK_M), B * Hq)
  Recomputes SSA weights from Q, K (no stored scores)
  dq = sum_j( ds_ssa * df/ds @ K ) * scale

dKV kernel: Grid (cdiv(N_CTX, BLOCK_N), B * Hkv)
  Iterates over all GQA_RATIO Q-heads
  Recomputes SSA weights
  dk, dv = standard flash-attn backward
  dn, db = Kahan-compensated accumulation
```

### Key Architectural Differences

| Aspect | AdaSplash | SSA |
|---|---|---|
| **Memory: forward scores** | Materializes full `(Z, H, N, N)` scores tensor | O(N) -- only stores row-wise normalizer L |
| **Memory scaling** | O(N^2) per head | O(N) per head |
| **Backward strategy** | Loads saved scores from memory | Recomputes scores from Q, K (flash-attn style) |
| **GQA support** | None (assumes H_q = H_kv) | Native (maps Q-head to KV-head via stride) |
| **Autotuning** | None | Triton autotuning on warps/stages (v3) |
| **Numerical tricks** | None special | Kahan compensated summation for dn, db |
| **Block mask support** | Yes (FlexAttention-style) | No |
| **Online normalization** | Online entmax with running max + rescaling | Simple accumulate-then-divide |
| **Separate dQ/dKV kernels** | No (single backward kernel) | Yes (enables different grid sizing) |
| **Warmup utility** | No | Yes (pre-compiles kernels) |

---

## 6. Critical Issue: AdaSplash's O(N^2) Memory

The most important architectural difference: **AdaSplash materializes the full N x N attention score matrix**.

```python
# adasplash_no_block_mask.py, line in _AdaSplashAttentionNoMask.forward:
scores = torch.zeros(Z, H, N_CTX, N_CTX, device=q.device, dtype=q.dtype)
```

This means for a sequence length of 2048 with 16 heads and batch 4:
- AdaSplash: `4 * 16 * 2048 * 2048 * 2 bytes = 1 GB` (bf16)
- SSA: `4 * 16 * 2048 * 4 bytes = 0.5 MB` (fp32 L tensor)

**AdaSplash cannot scale to long sequences.** The SSA kernel is a true flash-attention implementation with O(N) memory, while AdaSplash is not. This is likely because entmax's iterative solver and the complex rescaling make true online/recomputation-based backward harder to implement.

The block mask variant partially mitigates this by only storing scores for non-masked blocks, but still materializes O(T_r * T_c) per block.

---

## 7. The Online Entmax Problem

AdaSplash's forward pass uses an "online" entmax approach inspired by online softmax:

```python
# Rescale old accumulator when max changes:
old_scale = pow(max(m_i - m_ij, 0), 1/(alpha-1))
alpha_tmp = old_scale * l_i
acc = acc * alpha_tmp[:, None]
l_i = alpha_tmp + l_ij
```

This is mathematically correct but has subtleties:
1. When `m_i < m_ij` (new max is larger), `m_i - m_ij < 0`, so `max(m_i - m_ij, 0) = 0`, and `old_scale = 0^{1/(a-1)} = 0`. This **zeros out** the previous accumulator, which is correct because all previous entries `(z - old_tau)_+` become zero under the new (larger) threshold.
2. When `m_i >= m_ij`, the old accumulator is rescaled by the ratio of (old max - new max)^{1/(a-1)}.

This is a clever adaptation of the online softmax trick to entmax. However, it only works for the forward pass. The backward pass cannot use the same trick easily, which is why they store the full scores.

SSA doesn't have this problem because its weights are based on `exp(...)` which trivially supports the standard online softmax algorithm.

---

## 8. Relevance to Your Use Case

### High relevance (ideas to borrow):

1. **Standalone entmax module**: AdaSplash's `triton_entmax.py` is a clean, standalone Triton entmax that could be useful as a comparison or building block. If you ever wanted to experiment with sparse attention distributions, this is a ready-made component.

2. **Block mask integration**: The block mask variant shows how to interface with PyTorch's `create_block_mask` from `torch.nn.attention.flex_attention`. This could be useful if you want to add structured sparsity patterns on top of SSA.

3. **Entmax backward formula**: The entmax JVP (Jacobian-vector product) is:
   ```
   gppr = (alpha-1) * p^{2-alpha}
   ds = dp*gppr - gppr * (sum(dp*gppr) / (sum(gppr) + eps))
   ```
   This is the backward analog to softmax's `ds = p * (dp - sum(dp*p))`. It's interesting to see how different normalization functions affect the gradient structure.

4. **The paper's benchmarks**: The paper shows entmax attention can approach FlashAttention-2 speed in some settings. This validates the general approach of fusing custom attention functions into Triton kernels.

### Medium relevance (conceptual similarities):

5. **Both modify the attention normalization step**: Both projects share the philosophy that the softmax step in attention is replaceable. SSA wraps it with a score transformation; AdaSplash replaces it entirely.

6. **Both use Triton for fused kernels**: The kernel architecture is similar (tiled Q/K/V, block pointers, row-wise normalization). Your SSA kernels are actually more sophisticated (GQA, autotuning, Kahan summation, recomputation-based backward).

7. **Both handle causal masking inside kernels**: Same approach -- constexpr IS_CAUSAL flag, mask check `offs_m >= offs_n`.

### Low relevance (not applicable to SSA):

8. **Sparse attention patterns**: SSA produces dense attention weights (softmax always outputs non-zero values). Entmax's key advantage -- exact sparsity -- doesn't apply to SSA. If you wanted sparsity, you'd need to change the SSA formula fundamentally.

9. **Alpha as a fixed hyperparameter**: AdaSplash doesn't learn alpha, while SSA learns n and b. The approaches to parameterization are orthogonal.

10. **O(N^2) memory usage**: AdaSplash's approach of storing full scores is a significant limitation that SSA already avoids.

---

## 9. Can You Use Triton in a Similar Way?

**You already do, and you do it better.**

Your SSA Triton kernels already implement the same "fused flash attention" pattern that AdaSplash uses, with several important advantages:

| Feature | Your SSA Kernels | AdaSplash |
|---|---|---|
| Memory | O(N) -- true flash attention | O(N^2) -- materializes scores |
| GQA | Native in kernel | Not supported |
| Learnable params | Yes (n, b with gradients in kernel) | No |
| Autotuning | Yes (v3) | No |
| Numerical stability | Kahan summation for param gradients | None |
| Backward kernels | Split dQ / dKV (better parallelism) | Single combined kernel |
| Warmup | Yes | No |
| Megatron integration | Full layer specs | Standalone only |

The main thing AdaSplash does that you don't is the **block mask** support for structured sparsity and the **standalone entmax activation**. If you wanted to add block-level sparsity to SSA (skip entire KV blocks where attention is known to be negligible), the block mask approach from AdaSplash could be adapted.

---

## 10. Could You Combine SSA + Entmax?

A speculative idea: instead of `softmax(f(s))` where `f(s) = n * sign(s) * log(1 + b|s|)`, you could use:

```
p = entmax_alpha(f(s))
```

This would give you:
- SSA's learnable score transformation (n, b parameters)
- Entmax's adaptive sparsity (exact zeros in attention)
- Best of both worlds?

However, this would make the kernel significantly more complex:
1. You'd need the iterative tau-finding (bisection/Halley) inside the attention kernel
2. The backward pass would need to chain the entmax JVP with the SSA derivative
3. Online normalization becomes the entmax online trick instead of standard online softmax

The implementation difficulty is high, and it's unclear whether the combination would provide benefits beyond what each method achieves alone.

---

## 11. Summary

**AdaSplash is moderately relevant to your work.** It validates the approach of fusing custom attention functions into Triton kernels and provides a clean reference for entmax attention. However:

1. Your SSA kernels are architecturally superior (O(N) memory vs O(N^2), GQA, autotuning, learnable params)
2. AdaSplash solves a different problem (sparsity) than SSA (score transformation)
3. The learnable parameter story is opposite: AdaSplash has none, SSA has n and b
4. AdaSplash cannot scale to long sequences due to O(N^2) score storage

The most useful takeaway is that the general Triton kernel pattern (tiled Q/K/V blocks, row-wise normalization, fused forward/backward) is shared and well-validated across both approaches. Your implementation is more production-ready.
