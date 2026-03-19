#!/bin/bash
# Submit index build + training as a two-stage SLURM pipeline.
# Training starts only after index build succeeds.
#
# Config: MBS=8, BS=1024, 32 GPUs (8 nodes × 4), accum=4
# Usage: ./submit_pipeline.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting index build job..."
JOB1=$(sbatch --parsable "${SCRIPT_DIR}/build_index.sh")
echo "  Index build job: $JOB1"

echo "Submitting training job (depends on $JOB1)..."
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} --exclude=kairosgh8,kairosgh15 "${SCRIPT_DIR}/train_ssa_triton.sh")
echo "  Training job: $JOB2"

echo ""
echo "Pipeline submitted:"
echo "  1. build_index  ($JOB1) — 6h, 1 GPU"
echo "  2. train         ($JOB2) — 24h, 32 GPUs (starts after $JOB1 completes)"
echo ""
echo "Monitor with: squeue -u \$USER"
