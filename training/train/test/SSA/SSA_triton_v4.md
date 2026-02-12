# SSA Triton v4 Implementation Notes

This document records the full set of changes made to build and stabilize the `v4` SSA Triton path based on the Triton fused-attention tutorial structure (`06-fused-attention`), plus integration and validation changes around it.

## 1) Target and design intent

Goal:
- Keep Triton tutorial execution structure as close as possible (stage-based causal scheduling, online normalization, split backward loops, autotuned forward).
- Replace standard softmax logits with SSA-transformed logits:
  - `t(s) = n * sign(s) * log(1 + b*|s|)`
- Keep `n` learnable and `b` fixed for training (`n=1.5`, `b=0.8`).

Files involved:
- `test/SSA/ssa_triton_v4_kernel.py`
- `test/SSA/ssa_flash_attention_v4.py`
- `test/SSA/ssa_triton_v4_attention.py`
- `test/SSA/ssa_triton_v4_layer_specs.py`
- `test/train_ssa_triton.py`
- `test/train_ssa_triton.sh`
- `test/check_ssa_triton_v4_parity.py`
- `test/check_ssa_triton_v4_parity.slurm`

## 2) Kernel architecture changes (`ssa_triton_v4_kernel.py`)

### 2.1 Forward changed to tutorial-style online normalization

Forward inner loop (`_ssa_attn_fwd_inner`) was updated to:
- Compute transformed logits `t(s)` from SSA formula.
- Use tutorial-style online running state (`m_i`, `l_i`, `acc`) instead of accumulate-then-divide on raw SSA weights.
- Use base-2 normalization math (`exp2/log2`) to align numerically with tutorial style:
  - convert to base-2 domain with `RCP_LN2`
  - update with `p = exp2(t2 - m_ij)`
  - update scale with `alpha = exp2(m_i - m_ij)`
- Keep stage split behavior:
  - off-band
  - on-band (diagonal masked)
  - non-causal full band

### 2.2 Stored normalization state changed from row-sum to tutorial M

Forward epilogue now stores:
- `M = m_i + log2(l_i)` (tutorial-style log2 normalizer)
instead of storing row-wise SSA weight sums.

This is the buffer used by backward probability recomputation.

### 2.3 Explicit z/h indexing (remove flattened-head assumptions)

Forward and backward pointer base computation now explicitly decomposes program ids:
- `off_z`, `off_h_q`, `off_h_kv`
and uses `stride_*z + stride_*h` paths.

This removes dependency on flattened head assumptions and keeps behavior closer to tutorial-like explicit indexing.

### 2.4 Backward probability recomputation aligned with forward M

Backward inner loops:
- `_ssa_attn_bwd_dkdv`
- `_ssa_attn_bwd_dq`

were changed to recompute probabilities from transformed logits and stored `M`:
- `p = exp2(t2 - M_row)`

Then gradients follow SSA chain rule:
- `ds = ds_ssa * (n*b / (1 + b*|s|))`

`dn` and `db` remain Kahan-compensated accumulations.

### 2.5 Forward autotune prune callback adjusted

`_fwd_prune_configs` uses `kwargs` inputs for `N_CTX`/`STAGE` during autotune pruning, matching the expected Triton autotune callback calling pattern in this runtime.

### 2.6 Runtime compatibility fix for Triton API

Some Triton builds used here do not expose `tl.ones`.
Fix:
- replaced `tl.ones([BLOCK_M], dtype=tl.float32)` with
- `tl.full([BLOCK_M], value=1.0, dtype=tl.float32)`

This resolved the compile-time error:
- `AttributeError: module 'triton.language' has no attribute 'ones'`

### 2.7 Wrapper-level constraints retained/added

`ssa_flash_attn_v4_forward` and `ssa_flash_attn_v4_backward` enforce:
- head dim must be power-of-two (`D == next_power_of_2(D)` path).

Backward grid assumptions remain:
- `N_CTX` divisible by backward tile sizes (`BLOCK_N1=128`, etc.).

## 3) Autograd wrapper updates (`ssa_flash_attention_v4.py`)

Updated naming/state semantics to reflect stored tutorial normalizer:
- save `M` buffer from forward (not row-sum `L`).
- backward consumes `M`.

No public API change for model callsites (`ssa_flash_attention_v4(...)` unchanged).

## 4) Model integration (`ssa_triton_v4_attention.py`, `ssa_triton_v4_layer_specs.py`)

`SSATritonV4Attention` uses:
- Megatron format conversion `[S,B,H,D] <-> [B,H,S,D]`
- `ssa_flash_attention_v4(...)` call
- output dropout after fused attention output (same module behavior retained)

Layer spec wiring in `ssa_triton_v4_layer_specs.py`:
- `SSATritonV4Attention` used as core attention.
- optional compiled BDA path remains controlled by `use_compiled_bda` / env.

## 5) Training launcher behavior (`train_ssa_triton.py`, `train_ssa_triton.sh`)

### 5.1 v4-only launcher policy

Training launcher is pinned to `v4`:
- rejects non-`v4` `SSA_KERNEL_VERSION`.

### 5.2 SSA parameter policy enforcement

Requested policy enforced:
- `n` learnable and initialized to `1.5`
- `b` fixed at `0.8` (non-learnable)

Implementation:
- any CLI override for `ssa_n/ssa_b` is reset to `1.5/0.8`
- `learnable_b` is forced `False`
- layer spec constructed with `learnable_ssa=True`, `learnable_b=False`

Shell launcher also pins:
- `SSA_N=1.5`
- `SSA_B=0.8`
- `SSA_KERNEL_VERSION=v4`

## 6) Parity tooling updates (`check_ssa_triton_v4_parity.py`)

### 6.1 Internal-normalizer comparison handling

Because `v4` internal stored normalizer semantics differ from prior kernel/reference internals:
- parity comparator now supports `include_l` toggle.
- default pairwise pass/fail excludes internal `l/m` buffer from strict pass criteria.

Compared tensors/scalars for pass/fail:
- `out`, `dq`, `dk`, `dv`, `dn`, `db`

### 6.2 Suites in parity script

Script covers:
- kernel-level v4 vs v3 (+ optional reference)
- determinism suite
- module-level parity (Megatron module wrappers)
- short training drift suite

## 7) Validation results summary observed

After fixes:
- determinism suite: pass
- module suite: pass
- short training drift: pass
- 20-step distributed training canary: pass (`max_steps=20`, no NaN, checkpoint saved)

Known remaining parity discrepancy:
- one bf16 kernel parity case with `Hq=24, Hkv=8, N=128, causal=True` failed on `db` tolerance only.
- `out/dq/dk/dv/dn` passed in that case.
- since production policy uses fixed `b`, `db` is not used in optimization.

## 8) Operational notes

### 8.1 `DISABLE_COMPILED_BDA`

This flag controls only bias-dropout-add helper execution mode:
- `1`: eager BDA (more conservative/stable)
- `0`: `torch.compile` BDA (can be slightly faster, depends on runtime stability)

It does not change SSA Triton v4 kernel math.

### 8.2 Warmup behavior

Launcher warmup executes a tiny forward+backward to trigger Triton compile/autotune.
RNG state is saved/restored around warmup to keep init reproducibility vs non-warmup runs.

## 9) Current constraints/assumptions

- Head dimension must be power-of-two for the current kernel path.
- Backward schedule expects sequence lengths aligned to backward tile divisibility (`N % 128 == 0` under current block config).
- Training policy is fixed-`b`; learnable-`b` was intentionally disabled for production runs.

## 10) Suggested next cleanup (optional)

- In parity script, optionally gate `db` strictness when `b` is fixed policy.
- If needed later, improve `db` numerical parity for the `Hq/Hkv=24/8` causal bf16 corner case.
- Add a dedicated long-run (few hundred to thousand steps) monitoring report for gradient norm drift and throughput.
