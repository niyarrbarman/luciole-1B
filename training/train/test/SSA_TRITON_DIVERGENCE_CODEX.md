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
Initial highest-confidence hypothesis was scale semantics mismatch, but your remote checks falsified that for the actual `baby_luciole` config.

Current status after remote checks:
- `apply_query_key_layer_scaling=False`, so no layer-scale mismatch is active.
- Direct parity at `seq=128` is very close (forward/backward, including `dn`).
- Divergence is likely from a **shape/regime-dependent issue** (e.g. `seq=1024`) or an integration-level difference in the Triton path.

Most likely remaining suspects:
- Optimized Triton backward path (`SSA_USE_OPTIMIZED_KERNEL=1`) at training shape.
- `torch.compile`d bias-dropout-add path used only in Triton layer specs.
- Smaller training setup mismatches (e.g., different trainer horizon affects LR/data index mapping), though this alone is unlikely to explain strong divergence.

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

### 1) Scale semantics mismatch (historical hypothesis, not active for current run)

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

Your `ssa_cfg_check_76773.out` confirms:
- `apply_query_key_layer_scaling: False`

So this mismatch is not the cause for your current `baby_luciole` divergence run.

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

### 3) `ssa_triton_parity_76775.out` (after fix)

Status:
- Parity run completed successfully.

Key results (`layer=1,12`, `seq=128`, `bf16`, optimized kernel):
- Forward cosine: `~0.999993`
- Forward max abs diff: `1.56e-02`
- `dQ/dK/dV` max abs diffs: `~1e-7` to `1e-6`
- `dn` rel diff: `~3.5e-3` to `5.0e-3`
- Compensated-scaling run is identical (expected since QK layer scaling is disabled in recipe)

Implication:
- No obvious math bug at this small shape.
- Next checks must target **training shape/regime** (`seq=1024`, training batch/micro-batch characteristics, kernel mode variants).

---

## Post-Run Tooling Fix Applied

I updated:
- `test/check_ssa_triton_vs_original.py`
- `test/check_ssa_triton_vs_original.slurm`
- `test/check_ssa_triton_kernel_modes.slurm`
- `test/SSA/ssa_triton_layer_specs.py`
- `test/train_ssa_triton.py`
- `test/train_ssa_triton.sh`

Changes:
1. Parity script now initializes and cleans up single-rank Megatron parallel state via:
   - `init_single_gpu_parallel_state(...)`
   - `cleanup_parallel_state()`
2. SLURM launcher now exports single-rank distributed env vars:
   - `MASTER_ADDR`, `MASTER_PORT`, `RANK=0`, `WORLD_SIZE=1`
3. Parity script now prints which Triton kernel path is active:
   - `ssa_use_optimized_kernel`
4. Added kernel-mode parity matrix launcher:
   - `check_ssa_triton_kernel_modes.slurm`
   - runs parity across `SSA_USE_OPTIMIZED_KERNEL={1,0}` and configurable seq lengths (default `128,1024`)
5. Added Triton BDA compilation toggle:
   - `SSA_TRITON_COMPILE_BDA` env and `use_compiled_bda` path in layer specs
6. Added Triton trainer toggles:
   - `--disable_compiled_bda`
   - `--skip_triton_warmup`
   - `--warmup_steps`
7. Warmup robustness improvements:
   - warmup now sets per-rank CUDA device via `LOCAL_RANK`
   - RNG snapshot/restore retained so warmup does not perturb initialization

---

## Interpretation Guide

Current recommended isolate order:
1. Run parity at `seq=1024` for both kernel modes (`optimized` vs `reference`):
   - if optimized fails but reference passes, divergence is likely in optimized backward/autotune path.
2. Run short training A/B with same launch params:
   - `SSA_USE_OPTIMIZED_KERNEL=1` vs `0`
3. If both kernel modes still diverge, disable compiled BDA:
   - `DISABLE_COMPILED_BDA=1` or `SSA_TRITON_COMPILE_BDA=0`
4. Keep trainer horizon aligned between baseline and triton when comparing losses:
   - same `trainer.max_steps` to avoid different dataset index mappings and slightly different LR decay trajectories.

---

## What Is Still Unresolved

Root cause is not fully pinned yet, but the highest-probability remaining branch is now:
- **optimized Triton path behavior at training shape (`seq=1024`)**

The new checks and toggles above are meant to confirm this quickly without long retrains.
