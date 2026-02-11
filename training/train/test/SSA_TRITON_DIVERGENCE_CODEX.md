# SSA Triton Divergence (Codex Notes)

## Scope
This note captures:
- What I investigated in the local codebase and logs
- What I think is most likely causing divergence
- What I implemented for remote verification (Python + SLURM checks)

Date of analysis: 2026-02-11  
Workspace: `training/train`

---

## Executive Summary
Most likely root cause is a **scale semantics mismatch** between original SSA and Triton SSA when `apply_query_key_layer_scaling=True`.

In the original path, layer scaling is effectively canceled out before SSA transform.  
In the Triton path, layer scaling is not canceled, so deeper layers get smaller logits (`~1/layer`), which likely weakens SSA gradients (especially `dn`) and causes late-stage optimization drift/divergence.

Secondary differences that may contribute:
- Dropout placement mismatch (probability matrix vs attention output)
- `db` gradient bug in the optimized kernel (minor for current `learnable_b=False`)
- `log1p` vs `log(1+x)` numerical behavior

---

## What I Inspected

### Core SSA (baseline)
- `test/SSA/ssa_attention.py`
- `test/SSA/ssa_layer_specs.py`
- `test/train_ssa.py`

### Triton SSA path
- `test/SSA/ssa_triton_attention.py`
- `test/SSA/ssa_flash_attention.py`
- `test/SSA/ssa_triton_kernel.py`
- `test/SSA/ssa_triton_kernel_optimized.py`
- `test/SSA/ssa_triton_layer_specs.py`
- `test/train_ssa_triton.py`

### Logs and debug artifacts
- `test/slurm/tr_bbyluc_ssa_76444.out`
- `test/slurm/tr_bbyluc_ssa_triton_76676.out`
- `test/slurm/tr_bbyluc_ssa_triton_76631.out` (another triton run)
- `test/slurm/ssa_debug_*.out`
- `test/slurm/ssa_kernel_bench_*.out`

---

## Main Findings

### 1) Scale semantics mismatch (highest confidence)

#### Baseline SSA path (effective scale is `1/sqrt(d)`)
In `test/SSA/ssa_attention.py`:
- `self.softmax_scale /= layer_number` when query-key layer scaling is enabled
- `scale=coeff` is passed to `SSAScaleMaskSoftmax`
- then `input = input * self.scale` happens inside SSA softmax module

So effective pre-SSA multiplier becomes:
- `(1/sqrt(d)/layer) * layer = 1/sqrt(d)`

#### Triton path (effective scale appears `1/(sqrt(d)*layer)`)
In `test/SSA/ssa_triton_attention.py`:
- `self.softmax_scale /= self.layer_number` if enabled
- no compensating multiply before kernel call

So effective pre-SSA multiplier becomes:
- `1/(sqrt(d)*layer)`

If `apply_query_key_layer_scaling=True`, this is a systematic layer-dependent mismatch (e.g., layer 12 logits about 12x smaller than baseline).

This aligns with your symptom pattern:
- Early training looks plausible
- Later loss degrades
- `ssa_n` dynamics weaker/less differentiated in Triton run

---

### 2) Dropout placement mismatch (medium confidence)

Baseline:
- Dropout applied on `attention_probs` (after softmax/SSA, before `@V`)
- `test/SSA/ssa_attention.py` (attention_probs dropout block)

Triton:
- Kernel computes attention output first
- Dropout is applied on final context output
- `test/SSA/ssa_triton_attention.py` (dropout on `context`)

This changes training noise characteristics and gradient pathways.

---

### 3) Historical optimized-kernel gradient bug for `dn/db` (now likely fixed in your current file)

From debug/bench logs:
- Earlier optimized kernel runs showed severe `dn/db` mismatch (`ssa_debug_76618.out`, `ssa_kernel_bench_76620.out`)
- Later runs show parity restored (`ssa_debug_76622.out`, `ssa_kernel_bench_76626.out`)

So this issue existed at some point in history, but your current optimized file appears improved.

Note:
- `db` expression in `test/SSA/ssa_triton_kernel_optimized.py` still lacks `sign_s` in one place, but with `learnable_b=False` it is less likely to drive current divergence.

---

### 4) `log1p` vs `log(1+x)` numeric difference (low confidence)

Baseline uses `torch.log1p`, Triton kernels use `tl.log(1 + x)`.
Likely secondary unless logits are consistently tiny.

---

## Important Log Observations

### Baseline converges
- `test/slurm/tr_bbyluc_ssa_76444.out`
  - step 1000: `4.671`
  - step 2000: `3.901`
  - step 2500: `3.711`
  - step 3000: `3.596`

### Triton diverges in your cited run
- `test/slurm/tr_bbyluc_ssa_triton_76676.out`
  - step 1000: `5.947`
  - step 2000: `5.690`
  - step 2500: `6.218`
  - step 3000: `6.810`

### Another Triton run diverges harder
- `test/slurm/tr_bbyluc_ssa_triton_76631.out`
  - step 1000: `6.075`
  - step 2000: `6.338`
  - step 2500: `7.461`
  - step 3000: `7.303`

Also in that run, scheduler warmup differed (`warmup_steps=2000`), which can further affect loss shape.

---

## What I Implemented (Remote Checks)

I added these files:

1. `test/check_ssa_recipe_flags.py`
2. `test/check_ssa_recipe_flags.slurm`
3. `test/check_ssa_triton_vs_original.py`
4. `test/check_ssa_triton_vs_original.slurm`

### A) Recipe/flag/scale inspector
`check_ssa_recipe_flags.py` prints:
- `apply_query_key_layer_scaling`
- `attention_softmax_in_fp32`
- head dims / heads / query groups
- implied original vs triton effective scale by layer

Use to confirm whether scaling mismatch is actually active in `baby_luciole`.

### B) Direct parity check (forward + backward)
`check_ssa_triton_vs_original.py`:
- Instantiates original SSA and Triton SSA attention for chosen layer(s)
- Runs same random Q/K/V through both
- Compares forward outputs and gradients (`dQ/dK/dV/dn`)
- Supports `--compensate_triton_scaling` flag to emulate baseline effective scale in Triton without editing source

The paired SLURM (`check_ssa_triton_vs_original.slurm`) runs:
- `[1/2]` current behavior
- `[2/2]` compensated scaling behavior

This is the quickest way to validate the scaling hypothesis before changing training code.

---

## How To Run (Remote)

From `training/train/test`:

```bash
sbatch check_ssa_recipe_flags.slurm
sbatch check_ssa_triton_vs_original.slurm
```

Useful overrides:

```bash
ARCH=baby_luciole LAYERS=1,12 sbatch check_ssa_recipe_flags.slurm
ARCH=baby_luciole LAYERS=1,12 SEQ_LENGTH=128 BATCH_SIZE=2 DTYPE=bf16 sbatch check_ssa_triton_vs_original.slurm
FORCE_APPLY_QK_LAYER_SCALING=1 sbatch check_ssa_triton_vs_original.slurm
FORCE_APPLY_QK_LAYER_SCALING=0 sbatch check_ssa_triton_vs_original.slurm
```

---

## Results From Your Run (2026-02-11)

### 1) `ssa_cfg_check_76773.out`

Key output:
- `apply_query_key_layer_scaling: False`
- `attention_softmax_in_fp32: False`
- `attention_dropout: 0.0`
- head dim resolved to `32`

Implication:
- For this `baby_luciole` recipe/config, the earlier "layer scaling mismatch" hypothesis is **not active**.
- So divergence is likely due to other differences (backward math mismatch, dropout semantics mismatch, or related numerical issues).

### 2) `ssa_triton_parity_76774.out`

Status:
- Check failed before parity math due to Megatron parallel state not initialized:
  - `AssertionError: tensor model parallel group is not initialized`

This was a tooling issue in the parity script, not a model conclusion.

---

## Post-Run Tooling Fix Applied

I updated:
- `test/check_ssa_triton_vs_original.py`
- `test/check_ssa_triton_vs_original.slurm`

Changes:
1. Parity script now initializes and cleans up single-rank Megatron parallel state via:
   - `init_single_gpu_parallel_state(...)`
   - `cleanup_parallel_state()`
2. SLURM launcher now exports single-rank distributed env vars:
   - `MASTER_ADDR`, `MASTER_PORT`, `RANK=0`, `WORLD_SIZE=1`

This should unblock the parity check execution.

---

## Interpretation Guide

If `apply_query_key_layer_scaling=True` and compensated run significantly improves:
- forward diff metrics
- and especially `dn` relative diff

then scaling mismatch is strongly confirmed as primary issue.

If scaling compensation does not materially improve parity, next suspects are:
1. dropout placement difference
2. remaining backward math mismatch in Triton kernels

---

## What I Did Not Change Yet

I did **not** patch core training or kernel logic yet.  
I only added verification tooling so you can run controlled checks on the remote environment first.
