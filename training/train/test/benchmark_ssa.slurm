#!/bin/bash
#SBATCH -J ssa_kernel_bench
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

echo "=========================================="
echo "SSA Triton Kernel Benchmark"
echo "=========================================="

cd /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/test

apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --env "TRITON_CACHE_DIR=/tmpdir/m24047brmn/triton_cache" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python SSA/benchmark_ssa_kernel.py \
        --batch 8 \
        --hq 24 \
        --hkv 8 \
        --seq 1024 \
        --dim 32 \
        --warmup 10 \
        --iters 50

echo "=========================================="
echo "Benchmark complete"
echo "=========================================="
