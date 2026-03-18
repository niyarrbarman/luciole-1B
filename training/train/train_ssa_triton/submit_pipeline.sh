#!/bin/bash
# Submit index build + training as a two-stage SLURM pipeline.
# Training starts only after index build succeeds.
#
# Usage:
#   ./submit_pipeline.sh              # submit all 4 batch sizes
#   BATCH_SIZES="640 1280" ./submit_pipeline.sh  # submit specific sizes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# MBS=4, 20 GPUs => accum 4/8/16/32
BATCH_SIZES=(${BATCH_SIZES:-320 640 1280 2560})

for BS in "${BATCH_SIZES[@]}"; do
  echo "--- Batch size ${BS} ---"

  echo "  Submitting index build..."
  JOB1=$(BATCH_SIZE=${BS} sbatch --parsable --export=ALL,BATCH_SIZE=${BS} "${SCRIPT_DIR}/build_index.sh")
  echo "  Index build job: $JOB1"

  echo "  Submitting training job (depends on $JOB1)..."
  JOB2=$(BATCH_SIZE=${BS} sbatch --parsable --dependency=afterok:${JOB1} --export=ALL,BATCH_SIZE=${BS} "${SCRIPT_DIR}/train_ssa_triton.sh")
  echo "  Training job: $JOB2"
  echo ""
done

echo "All pipelines submitted."
echo "Monitor with: squeue -u \$USER"
