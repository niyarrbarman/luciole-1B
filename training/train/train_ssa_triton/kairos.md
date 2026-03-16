# Kairos — Nemotron-1B SSA Triton Production Training Plan

## Model
- Architecture: Nemotron-1B (24 layers, 32 heads, 8 KV groups, hidden=2048, ffn=8192)
- Attention: SSA Triton fused kernel (n=1.5 learnable, b=0.8 fixed)
- Tokenizer: `/work/m24047/m24047brmn/Luciole-23B-Base`

## Data
- Dataset: Luciole Phase 1 (73 datasets, original weights)
- Total tokens: ~3T (3,002,228,277,248)
- Path: `/work/shares/IA-Datasets/private/tokens_luciole_phase1`

## Hardware

### Test Server (current)
| Param | Value |
|---|---|
| GPUs | 12x A100 |
| Nodes | 6 |
| GPUs/node | 2 |
| SBATCH time | 24:00:00 |

### Production Server (Kairos — weekends)
| Param | Value |
|---|---|
| GPUs | 64x H200 (200GB) |
| Nodes | 16 |
| GPUs/node | 4 |
| SBATCH time | 48:00:00 |

## Hyperparameters

| Parameter | Test (12 A100s) | Prod (64 H200s) |
|---|---|---|
| seq_length | 2048 | 4096 |
| GBS | 384 | 1024 |
| mbs | 2 | 16 |
| TP / PP / CP | 1/1/1 | 1/1/1 |
| GLOBAL_MAX_STEPS | 3,817,000 | 716,000 |
| THIS_RUN_MAX_STEPS | 0 (disabled) | 0 (disabled) |
| save_every_n_steps | 6,000 | 10,000 |
| LR_WARMUP_STEPS | 2,000 | 2,000 |
| log_every_n_steps (slurm) | 200 | 200 |
| log_ssa_every_n_steps | 1,000 | 1,000 |
| SSA n | 1.5 (learnable) | 1.5 (learnable) |
| SSA b | 0.8 (fixed) | 0.8 (fixed) |

## Step Math (Production)
- Tokens per step: 4096 x 1024 = 4,194,304
- Steps for 1 epoch: 3,002,228,277,248 / 4,194,304 = ~716,000
- Grad accumulation: 1024 / (64 x 16) = 1 (none)
- Total checkpoints: ~72

## Step Math (Test)
- Tokens per step: 2048 x 384 = 786,432
- Steps for 1 epoch: 3,002,228,277,248 / 786,432 = ~3,817,000
- Grad accumulation: 384 / (12 x 2) = 16

## Time Estimates (Production)
- H200 step time: ~0.5-1s (1B model, no grad accum)
- Steps per weekend (48h): ~170K-345K
- Weekends to complete 1 epoch: 2-4

## Resume Strategy
- `StatelessTimer` handles graceful shutdown before walltime
- `AutoResume(resume_if_exists=True)` picks up latest checkpoint on resubmit
- `save_last=True` ensures checkpoint on every clean exit
- Timer resets each job (stateless) — fresh walltime window per submission
- Just `sbatch` again each weekend, no manual intervention needed
