#!/bin/bash
#SBATCH -J ssa_triton_v4_parity
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=03:30:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
mkdir -p slurm

# Core settings
ARCH=${ARCH:-baby_luciole}
SEED=${SEED:-1234}
SSA_N=${SSA_N:-1.5}
SSA_B=${SSA_B:-0.8}
QUICK=${QUICK:-0}                         # 1 -> reduced matrix
FAIL_FAST=${FAIL_FAST:-0}                 # 1 -> stop at first failure

# Kernel matrix settings
DTYPES=${DTYPES:-bf16,fp16}
KERNEL_BATCH_SIZES=${KERNEL_BATCH_SIZES:-1,2}
KERNEL_SEQ_LENGTHS=${KERNEL_SEQ_LENGTHS:-128,512,1024}
KERNEL_HEAD_PAIRS=${KERNEL_HEAD_PAIRS:-8x8,16x8,24x8}
KERNEL_HEAD_DIMS=${KERNEL_HEAD_DIMS:-32}
KERNEL_CAUSAL_MODES=${KERNEL_CAUSAL_MODES:-1,0}
KERNEL_LAYOUTS=${KERNEL_LAYOUTS:-contig,strided}
KERNEL_USE_REFERENCE=${KERNEL_USE_REFERENCE:-1}
REFERENCE_MAX_SEQ=${REFERENCE_MAX_SEQ:-512}

# Optional suite toggles
SKIP_KERNEL=${SKIP_KERNEL:-0}
SKIP_DETERMINISM=${SKIP_DETERMINISM:-0}
SKIP_MODULE=${SKIP_MODULE:-0}
SKIP_TRAINING=${SKIP_TRAINING:-0}

# Module settings
MODULE_DTYPE=${MODULE_DTYPE:-bf16}
MODULE_LAYERS=${MODULE_LAYERS:-1,6,12}
MODULE_BATCH_SIZE=${MODULE_BATCH_SIZE:-2}
MODULE_SEQ_LENGTH=${MODULE_SEQ_LENGTH:-256}
MODULE_LEARNABLE_B=${MODULE_LEARNABLE_B:-0}

# Training drift settings
TRAIN_DTYPE=${TRAIN_DTYPE:-bf16}
TRAIN_STEPS=${TRAIN_STEPS:-80}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
TRAIN_SEQ_LENGTH=${TRAIN_SEQ_LENGTH:-128}
TRAIN_HQ=${TRAIN_HQ:-24}
TRAIN_HKV=${TRAIN_HKV:-8}
TRAIN_HEAD_DIM=${TRAIN_HEAD_DIM:-32}
TRAIN_LR=${TRAIN_LR:-1e-3}
TRAIN_CAUSAL=${TRAIN_CAUSAL:-1}
TRAIN_LEARNABLE_B=${TRAIN_LEARNABLE_B:-0}

TRAIN_LOSS_TOL_V4_REF=${TRAIN_LOSS_TOL_V4_REF:-6e-3}
TRAIN_LOSS_TOL_V4_V3=${TRAIN_LOSS_TOL_V4_V3:-2e-3}
TRAIN_PARAM_TOL_V4_REF=${TRAIN_PARAM_TOL_V4_REF:-3e-2}
TRAIN_PARAM_TOL_V4_V3=${TRAIN_PARAM_TOL_V4_V3:-1e-2}

PROJECT_ROOT=${PROJECT_ROOT:-/work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training}
SCRIPT_PATH="${PROJECT_ROOT}/training/train/test/check_ssa_triton_v4_parity.py"
REPORT_DIR=${REPORT_DIR:-"${PROJECT_ROOT}/training/train/test/parity_reports"}
REPORT_PATH="${REPORT_DIR}/ssa_triton_v4_parity_${SLURM_JOB_ID}.json"

# Single-process distributed env used by Megatron parallel_state init
export MASTER_ADDR=${MASTER_ADDR:-$(hostname --ip-address)}
export MASTER_PORT=${MASTER_PORT:-$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)}
export RANK=0
export WORLD_SIZE=1

# Triton cache
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmpdir/m24047brmn/triton_cache}
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${REPORT_DIR}"

EXTRA_ARGS=()
if [[ "${QUICK}" == "1" ]]; then
    EXTRA_ARGS+=(--quick)
fi
if [[ "${FAIL_FAST}" == "1" ]]; then
    EXTRA_ARGS+=(--fail_fast)
fi
if [[ "${SKIP_KERNEL}" == "1" ]]; then
    EXTRA_ARGS+=(--skip_kernel)
fi
if [[ "${SKIP_DETERMINISM}" == "1" ]]; then
    EXTRA_ARGS+=(--skip_determinism)
fi
if [[ "${SKIP_MODULE}" == "1" ]]; then
    EXTRA_ARGS+=(--skip_module)
fi
if [[ "${SKIP_TRAINING}" == "1" ]]; then
    EXTRA_ARGS+=(--skip_training)
fi
echo "=========================================="
echo "SSA Triton v4 Comprehensive Parity Checks"
echo "Arch:            ${ARCH}"
echo "Seed:            ${SEED}"
echo "SSA n/b:         ${SSA_N} / ${SSA_B}"
echo "Quick:           ${QUICK}"
echo "Fail-fast:       ${FAIL_FAST}"
echo "Kernel matrix:"
echo "  dtypes:        ${DTYPES}"
echo "  B:             ${KERNEL_BATCH_SIZES}"
echo "  N:             ${KERNEL_SEQ_LENGTHS}"
echo "  heads:         ${KERNEL_HEAD_PAIRS}"
echo "  D:             ${KERNEL_HEAD_DIMS}"
echo "  causal modes:  ${KERNEL_CAUSAL_MODES}"
echo "  layouts:       ${KERNEL_LAYOUTS}"
echo "  use ref:       ${KERNEL_USE_REFERENCE}"
echo "  ref max seq:   ${REFERENCE_MAX_SEQ}"
echo "Module: dtype=${MODULE_DTYPE}, layers=${MODULE_LAYERS}, B=${MODULE_BATCH_SIZE}, N=${MODULE_SEQ_LENGTH}"
echo "Training drift: dtype=${TRAIN_DTYPE}, steps=${TRAIN_STEPS}, B=${TRAIN_BATCH_SIZE}, N=${TRAIN_SEQ_LENGTH}, learnable_b=${TRAIN_LEARNABLE_B}"
echo "Script:          ${SCRIPT_PATH}"
echo "JSON report:     ${REPORT_PATH}"
echo "MASTER_ADDR:     ${MASTER_ADDR}"
echo "MASTER_PORT:     ${MASTER_PORT}"
echo "TRITON_CACHE:    ${TRITON_CACHE_DIR}"
echo "=========================================="

srun apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --env "MASTER_ADDR=${MASTER_ADDR}" \
    --env "MASTER_PORT=${MASTER_PORT}" \
    --env "RANK=${RANK}" \
    --env "WORLD_SIZE=${WORLD_SIZE}" \
    --env "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}" \
    --env "SSA_KERNEL_VERSION=v4" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python3 "${SCRIPT_PATH}" \
    --arch "${ARCH}" \
    --output_dir "/tmp/ssa_v4_parity" \
    --name "ssa-v4-parity" \
    --seed "${SEED}" \
    --ssa_n "${SSA_N}" \
    --ssa_b "${SSA_B}" \
    --dtypes "${DTYPES}" \
    --kernel_batch_sizes "${KERNEL_BATCH_SIZES}" \
    --kernel_seq_lengths "${KERNEL_SEQ_LENGTHS}" \
    --kernel_head_pairs "${KERNEL_HEAD_PAIRS}" \
    --kernel_head_dims "${KERNEL_HEAD_DIMS}" \
    --kernel_causal_modes "${KERNEL_CAUSAL_MODES}" \
    --kernel_layouts "${KERNEL_LAYOUTS}" \
    --kernel_use_reference "${KERNEL_USE_REFERENCE}" \
    --reference_max_seq "${REFERENCE_MAX_SEQ}" \
    --module_dtype "${MODULE_DTYPE}" \
    --module_layers "${MODULE_LAYERS}" \
    --module_batch_size "${MODULE_BATCH_SIZE}" \
    --module_seq_length "${MODULE_SEQ_LENGTH}" \
    --module_learnable_b "${MODULE_LEARNABLE_B}" \
    --train_dtype "${TRAIN_DTYPE}" \
    --train_steps "${TRAIN_STEPS}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --train_seq_length "${TRAIN_SEQ_LENGTH}" \
    --train_hq "${TRAIN_HQ}" \
    --train_hkv "${TRAIN_HKV}" \
    --train_head_dim "${TRAIN_HEAD_DIM}" \
    --train_lr "${TRAIN_LR}" \
    --train_causal "${TRAIN_CAUSAL}" \
    --train_learnable_b "${TRAIN_LEARNABLE_B}" \
    --train_loss_tol_v4_ref "${TRAIN_LOSS_TOL_V4_REF}" \
    --train_loss_tol_v4_v3 "${TRAIN_LOSS_TOL_V4_V3}" \
    --train_param_tol_v4_ref "${TRAIN_PARAM_TOL_V4_REF}" \
    --train_param_tol_v4_v3 "${TRAIN_PARAM_TOL_V4_V3}" \
    --json_report "${REPORT_PATH}" \
    "${EXTRA_ARGS[@]}"

status=$?
echo "=========================================="
echo "SSA Triton v4 parity checks finished with status ${status}"
echo "JSON report: ${REPORT_PATH}"
echo "=========================================="
exit ${status}
