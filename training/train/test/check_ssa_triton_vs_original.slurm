#!/bin/bash
#SBATCH -J ssa_triton_parity
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=00:45:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
mkdir -p slurm

ARCH=${ARCH:-baby_luciole}
LAYERS=${LAYERS:-1,12}
SEQ_LENGTH=${SEQ_LENGTH:-128}
BATCH_SIZE=${BATCH_SIZE:-2}
DTYPE=${DTYPE:-bf16}
SEED=${SEED:-1234}
FORCE_APPLY_QK_LAYER_SCALING=${FORCE_APPLY_QK_LAYER_SCALING:-""}  # "1" or "0" or empty (use recipe default)

PROJECT_ROOT=${PROJECT_ROOT:-/work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training}
SCRIPT_PATH="${PROJECT_ROOT}/training/train/test/check_ssa_triton_vs_original.py"

# Single-process distributed init vars used by Megatron parallel_state.
export MASTER_ADDR=${MASTER_ADDR:-$(hostname --ip-address)}
export MASTER_PORT=${MASTER_PORT:-$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)}
export RANK=0
export WORLD_SIZE=1

EXTRA_FLAGS=()
if [[ "${FORCE_APPLY_QK_LAYER_SCALING}" == "1" ]]; then
    EXTRA_FLAGS+=(--force_apply_query_key_layer_scaling)
elif [[ "${FORCE_APPLY_QK_LAYER_SCALING}" == "0" ]]; then
    EXTRA_FLAGS+=(--force_disable_query_key_layer_scaling)
fi

echo "=========================================="
echo "SSA Triton vs Original Parity Check"
echo "Arch:        ${ARCH}"
echo "Layers:      ${LAYERS}"
echo "Seq/Bs:      ${SEQ_LENGTH}/${BATCH_SIZE}"
echo "Dtype:       ${DTYPE}"
echo "Seed:        ${SEED}"
echo "Force QK layer scaling: ${FORCE_APPLY_QK_LAYER_SCALING:-<recipe-default>}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "Script:      ${SCRIPT_PATH}"
echo "=========================================="

echo
echo "[1/2] Current Triton scaling behavior"
srun apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python3 "${SCRIPT_PATH}" \
    --arch "${ARCH}" \
    --layers "${LAYERS}" \
    --seq_length "${SEQ_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --seed "${SEED}" \
    "${EXTRA_FLAGS[@]}"

echo
echo "[2/2] Triton with compensated scaling (emulates baseline effective scale)"
srun apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python3 "${SCRIPT_PATH}" \
    --arch "${ARCH}" \
    --layers "${LAYERS}" \
    --seq_length "${SEQ_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --seed "${SEED}" \
    --compensate_triton_scaling \
    "${EXTRA_FLAGS[@]}"

echo "=========================================="
echo "SSA Triton parity checks finished."
echo "=========================================="
