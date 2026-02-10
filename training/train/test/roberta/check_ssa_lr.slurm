#!/bin/bash
#SBATCH -J check_ssa
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=00:05:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

echo "Checking SSA parameter groups and learning rates..."

apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind /tmpdir,/work \
    /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python3 /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/test/roberta/check_ssa_lr.py

echo "Done with status $?"
