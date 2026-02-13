# SSA Triton v4: Q/K/V Contiguous Fix

## Summary

When running SSA Triton v4, training quality was significantly below the baseline SSA path.
Enabling `Q/K/V` contiguity before the Triton attention call appears to fix (or strongly reduce) the issue in practice.

This note explains what changed and why it likely works.

## Problem Pattern

- Baseline SSA (`tr_bbyluc_ssa_76444.out`) converges normally.
- Triton v4 runs (`tr_bbyluc_ssa_triton_76956.out`, and similar) stayed in a worse loss regime.
- Parity checks showed layout-sensitive failures for model-relevant head config (`Hq=24`, `Hkv=8`), especially in strided cases.

## Change Introduced

New optional flag:

- Python launcher: `--force_contiguous_qkv`
- SLURM env var: `FORCE_CONTIGUOUS_QKV=1`

Wiring:

- `test/train_ssa_triton.py`: adds `--force_contiguous_qkv` and forwards to layer spec.
- `test/SSA/ssa_triton_v4_layer_specs.py`: forwards `force_contiguous_qkv` to attention module.
- `test/SSA/ssa_triton_v4_attention.py`: if enabled, runs:
  - `query_t = query_t.contiguous()`
  - `key_t = key_t.contiguous()`
  - `value_t = value_t.contiguous()`

## Why This Likely Worked

`permute` creates a view with non-trivial strides. The Triton kernel supports strided tensors, but observed failures are layout-sensitive in parity and training.

Making `Q/K/V` contiguous:

- removes strided-layout variability before the kernel,
- gives a regular memory layout for reads,
- avoids potential indexing/stride corner-case behavior in the fused path.

So this acts as a strong isolation lever: if quality improves with this flag, the issue is likely in stride/layout handling rather than high-level training recipe settings.

## How To Run

```bash
cd training/train/test
FORCE_CONTIGUOUS_QKV=1 sbatch train_ssa_triton.sh
```

You can combine with other diagnostics, but for clean attribution test one change at a time.

## Tradeoffs

- Extra copy cost each forward pass (`Q/K/V` materialization).
- Higher memory bandwidth usage.
- Potentially lower throughput.

This is acceptable for debugging or temporary stability, but should be benchmarked before making it default for production training.

## Recommended Follow-up

1. Run A/B with identical config and seed:
   - A: default (no contiguous)
   - B: `FORCE_CONTIGUOUS_QKV=1`
2. Compare:
   - loss at steps `1000`, `1300`, `2000`
   - tokens/s and step time
3. If B remains clearly better, treat stride/layout handling as the primary kernel debugging target.
