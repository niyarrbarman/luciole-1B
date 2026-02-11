# SSA Triton Divergence Analysis

## Problem
The Triton-fused SSA FlashAttention kernel gets ~2x speedup but **loss diverges after ~step 2000** (goes UP), while the original PyTorch SSA converges fine.

## Loss Comparison (same data, same hyperparams, from scratch)

| Step | Original SSA | Triton SSA | Delta |
|------|-------------|------------|-------|
| 0    | 10.89       | 10.88      | OK    |
| 100  | 7.91        | 8.11       | +0.2  |
| 500  | 5.61        | 6.29       | +0.7  |
| 1000 | 4.67        | 5.96       | +1.3  |
| 2000 | 3.88        | 5.70       | +1.8  |
| 2500 | 3.74        | 6.17       | DIVERGING |
| 3000 | 3.60        | 6.89       | DIVERGING |
| 3300 | 3.57        | 7.19       | DIVERGING |

Source runs: `tr_bbyluc_ssa_76444.out` (original), `tr_bbyluc_ssa_triton_76676.out` (Triton)

## SSA n Parameter Evolution

**Original (step 3000):** n grows significantly, per-layer differentiation
```
Layer 0: 1.797, Layer 2: 1.922, Layer 3: 1.945, Layer 6: 1.570, Layer 11: 1.688
```

**Triton (step 3000):** n barely moves, weaker per-layer differentiation
```
Layer 0: 1.648, Layer 2: 1.547, Layer 3: 1.563, Layer 6: 1.609, Layer 11: 1.727
```

This is evidence that `dn` gradients in the Triton kernel are different (smaller magnitude) from PyTorch autograd.

---

## Files Analyzed

### Original (working) path
- `SSA/ssa_attention.py` — `SSADotProductAttention` + `SSAScaleMaskSoftmax`
- `SSA/ssa_layer_specs.py` — `get_ssa_gpt_layer_spec`
- `train_ssa.py` — training harness

### Triton (broken) path
- `SSA/ssa_triton_attention.py` — `SSATritonAttention`
- `SSA/ssa_triton_kernel_optimized.py` — forward/backward Triton kernels
- `SSA/ssa_flash_attention.py` — `SSAFlashAttnFunc` autograd wrapper
- `SSA/ssa_triton_layer_specs.py` — Triton layer specs
- `train_ssa_triton.sh` — SLURM script

---

## Confirmed Differences Between Implementations

### 2. `scale/coeff` handling — POTENTIAL DOUBLE-SCALING BUG

**Original** (`ssa_attention.py:222-233`):
```python
coeff = None
if self.config.apply_query_key_layer_scaling:
    coeff = self.layer_number
    self.softmax_scale /= coeff  # scale = 1/(sqrt(d) * layer_num)

self.scale_mask_softmax = SSAScaleMaskSoftmax(
    scale=coeff,  # Passed to softmax module
    ...
)
```
Then in `SSAScaleMaskSoftmax._forward_ssa_softmax`:
```python
if self.scale is not None:
    input = input * self.scale  # Re-multiply by layer_number
```
**Net effect**: The `/ layer_number` and `* layer_number` cancel. SSA sees `Q@K^T / sqrt(d)`.

**Triton** (`ssa_triton_attention.py`):
```python
if self.config.apply_query_key_layer_scaling:
    self.softmax_scale /= self.layer_number
```
No second multiplication. SSA sees `Q@K^T / (sqrt(d) * layer_number)`.

**If `apply_query_key_layer_scaling=True`**: Later layers (layer 12) would see scores 12x smaller in Triton vs original. The SSA transform `log(1 + b|s|)` would behave nearly linearly for small s (since `log(1+x) ≈ x` for small x), making it basically standard attention with reduced magnitude. This could absolutely cause the observed divergence pattern.

**TODO**: Check if `apply_query_key_layer_scaling` is True for the baby_luciole architecture.

### 3. `db` gradient bug in backward kernel (minor, since b isn't learnable)

In `ssa_triton_kernel_optimized.py`, the dKV backward kernel:
```python
# WRONG:
db_acc += tl.sum(ds_ssa * ssa_n * abs_s / one_plus_bs)
# CORRECT:
db_acc += tl.sum(ds_ssa * ssa_n * sign_s * abs_s / one_plus_bs)
```
Missing `sign_s` factor. Doesn't matter currently since b is fixed, but would be wrong if `learnable_b=True`.

### 4. `torch.log1p` vs `tl.log(1 + x)` numerical stability

**Original**: `torch.log1p(b * abs_x)` — uses hardware log1p, accurate for small arguments.

**Triton**: `tl.log(1.0 + ssa_b * abs_s)` — standard log, loses precision when `ssa_b * abs_s` is very small (catastrophic cancellation in `1.0 + tiny`).

**Impact**: Likely minor for typical score ranges, but could accumulate over many layers and steps.

---

## Investigation TODO (when SSH is back)

### High Priority — likely root cause
1. **Check `apply_query_key_layer_scaling`**: Run `python -c "from recipes.recipe_utils import get_recipe; r = get_recipe('baby_luciole', ...); print(r.model.config.apply_query_key_layer_scaling)"` to see if this flag is True. If yes, this is the scaling bug.

2. **Verify the actual server `ssa_attention.py`**: Confirm softplus is truly removed — need to see the actual `get_ssa_params()` method on the server version.

3. **Write a unit test comparing forward outputs**: Run both implementations on the same small input and compare the attention output numerically (max abs diff, cosine similarity). If forward outputs diverge, the bug is in the forward. If they match, the bug is in the backward.

4. **Write a unit test comparing backward gradients**: Same test but compare dQ, dK, dV, dn, db against PyTorch autograd. This will pinpoint exactly which gradient is wrong.

### Medium Priority

6. **Replace `tl.log(1+x)` with a log1p equivalent**: `tl.log(1.0 + x)` could be replaced with a custom implementation for better numerical stability.

### Lower Priority
7. **Profile gradient norms**: Log per-parameter gradient norms to see if any are exploding in the Triton version.

8. **Check if the non-contiguous stride access causes precision issues**: Compare Triton kernel output with vs without `.contiguous()` calls on Q/K/V before passing to kernel.

---

## Recommended Fix Order

1. **Fix the `scale/coeff` handling** (if `apply_query_key_layer_scaling=True`)
   - Add the second `* coeff` multiplication inside the kernel after computing `s`
   - OR just set `softmax_scale = 1/sqrt(d)` without dividing by layer_number

2. **Make both `n` and `b` learnable** (matching original behavior)

3. **Fix the `db` gradient** (add `sign_s`)

4. **Write forward/backward numerical tests** to validate equivalence

---

## Key Insight

The divergence pattern (loss initially drops then reverses after ~2000 steps) is characteristic of **backward gradient corruption**, not a forward-only bug. If the forward were completely wrong, the initial loss would differ. The matching initial loss (10.88/10.89) strongly suggests the forward is approximately correct. The gradual divergence points to:
- Subtle gradient errors that accumulate over steps
- Or a scaling issue that becomes problematic as the model's internal representations evolve during training

The `scale/coeff` issue (difference #2) is the strongest candidate because it causes a systematic, layer-dependent bias in the attention scores that would compound across layers during training.
