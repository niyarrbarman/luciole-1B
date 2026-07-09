#!/bin/bash
#SBATCH -J ssa_cfg_check
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=00:20:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
mkdir -p slurm

ARCH=${ARCH:-baby_luciole}
LAYERS=${LAYERS:-1,2,6,12}
PROJECT_ROOT=${PROJECT_ROOT:-/work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training}
SCRIPT_PATH="${PROJECT_ROOT}/training/train/train_ssa_triton/check_ssa_recipe_flags.py"

echo "=========================================="
echo "SSA Recipe Flag Check"
echo "Arch:        ${ARCH}"
echo "Layers:      ${LAYERS}"
echo "Script:      ${SCRIPT_PATH}"
echo "=========================================="

srun apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python3 "${SCRIPT_PATH}" \
    --arch "${ARCH}" \
    --layers "${LAYERS}"

echo "=========================================="
echo "SSA Recipe Flag Check finished."
echo "=========================================="
