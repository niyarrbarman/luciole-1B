#!/bin/bash
#SBATCH -J tr_bbyluc_ssa_triton_wandb_test
#SBATCH -N 6
#SBATCH -n 6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -p small
#SBATCH --time=08:00:00
#SBATCH --output=slurm/%x_%j.out

set -eo pipefail

mkdir -p slurm

# Defaults (same data/model setup as train_ssa_triton.sh)
DATAMIX=${DATAMIX:-"/tmpdir/m24047brmn/nemo_1b/data_fwe_50k/datamix_fineweb_edu_50k.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"/tmpdir/m24047brmn/nemo_1b/output"}
RUN_STAMP=${RUN_STAMP:-"$(date +%Y%m%d_%H%M%S)"}
NAME=${NAME:-"baby_luciole-ssa-triton-v4-wandb-test-${SLURM_JOB_ID:-nojob}-${RUN_STAMP}"}
SEED=${SEED:-1234}

# W&B config
WANDB_PROJECT=${WANDB_PROJECT:-"luciole_ssa"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_GROUP=${WANDB_GROUP:-"${NAME}"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${NAME}"}
WANDB_TAGS=${WANDB_TAGS:-"ssa,triton,wandb,test"}
WANDB_NOTES=${WANDB_NOTES:-"200-step SSA Triton W&B test run"}
WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-"train"}
WANDB_MODE=${WANDB_MODE:-"offline"}
WANDB_DIR=${WANDB_DIR:-"${OUTPUT_DIR}/${NAME}/wandb"}
WANDB_RUN_ID=${WANDB_RUN_ID:-""}
WANDB_RESUME=${WANDB_RESUME:-"never"}
WANDB_LOG_MODEL=${WANDB_LOG_MODEL:-0}

# Allow `api_key=... sbatch ...` as requested, while supporting WANDB_API_KEY too.
WANDB_API_KEY=${WANDB_API_KEY:-${api_key:-""}}

# SSA hyperparameters / kernel settings (same as train_ssa_triton.sh)
SSA_N=1.5
SSA_B=0.8
SSA_KERNEL_VERSION=${SSA_KERNEL_VERSION:-v4}
SSA_TRITON_COMPILE_BDA=${SSA_TRITON_COMPILE_BDA:-1}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-500}
SKIP_TRITON_WARMUP=${SKIP_TRITON_WARMUP:-0}
DISABLE_COMPILED_BDA=${DISABLE_COMPILED_BDA:-0}
FORCE_CONTIGUOUS_QKV=${FORCE_CONTIGUOUS_QKV:-1}

# Test-run limits
GLOBAL_MAX_STEPS=${GLOBAL_MAX_STEPS:-200}
THIS_RUN_MAX_STEPS=${THIS_RUN_MAX_STEPS:-200}

if [[ "${SSA_KERNEL_VERSION}" != "v4" ]]; then
    echo "ERROR: SSA_KERNEL_VERSION must be 'v4' (got '${SSA_KERNEL_VERSION}')."
    exit 2
fi

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY}" ]]; then
    echo "ERROR: WANDB_MODE=online but no API key provided."
    echo "Use: WANDB_API_KEY='...' sbatch train_ssa_triton_wandb_test.sh"
    echo "or:  api_key='...' sbatch train_ssa_triton_wandb_test.sh"
    exit 2
fi

# Safety: prevent resume/overwrite by refusing to use an existing experiment directory.
if [[ -d "${OUTPUT_DIR}/${NAME}" ]]; then
    echo "ERROR: ${OUTPUT_DIR}/${NAME} already exists."
    echo "Pick a new NAME (or RUN_STAMP) to keep this run from resuming/overwriting."
    exit 2
fi

# Multi-node coordination
export MASTER_PORT
MASTER_PORT=$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR
MASTER_ADDR=$(hostname --ip-address)

# Convert SBATCH time to DD:HH:MM:SS for StatelessTimer
SBATCH_TIME=$(grep -E '^#SBATCH --time=' "$0" | head -n1 | sed -E 's/^#SBATCH --time=//')
if [[ "$SBATCH_TIME" == *-* ]]; then
    SLURM_DURATION=$(echo "$SBATCH_TIME" | awk -F'[-:]' '{printf "%02d:%02d:%02d:%02d", $1, $2, $3, $4}')
else
    SLURM_DURATION=$(echo "$SBATCH_TIME" | awk -F':' '{printf "00:%02d:%02d:%02d", $1, $2, $3}')
fi

echo "=========================================="
echo "Starting SSA Triton W&B test run"
echo "Datamix:      ${DATAMIX}"
echo "Output:       ${OUTPUT_DIR}"
echo "Name:         ${NAME}"
echo "Nodes:        ${SLURM_NNODES}"
echo "Duration:     ${SLURM_DURATION}"
echo "Max steps:    ${GLOBAL_MAX_STEPS}"
echo "This-run max: ${THIS_RUN_MAX_STEPS}"
echo "W&B project:  ${WANDB_PROJECT}"
echo "W&B mode:     ${WANDB_MODE}"
echo "=========================================="

EXTRA_ARGS=(
    --this_run_max_steps "${THIS_RUN_MAX_STEPS}"
    --wandb
    --wandb_project "${WANDB_PROJECT}"
    --wandb_group "${WANDB_GROUP}"
    --wandb_run_name "${WANDB_RUN_NAME}"
    --wandb_job_type "${WANDB_JOB_TYPE}"
    --wandb_mode "${WANDB_MODE}"
    --wandb_resume "${WANDB_RESUME}"
    --wandb_dir "${WANDB_DIR}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
    EXTRA_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_TAGS}" ]]; then
    EXTRA_ARGS+=(--wandb_tags "${WANDB_TAGS}")
fi
if [[ -n "${WANDB_NOTES}" ]]; then
    EXTRA_ARGS+=(--wandb_notes "${WANDB_NOTES}")
fi
if [[ -n "${WANDB_RUN_ID}" ]]; then
    EXTRA_ARGS+=(--wandb_id "${WANDB_RUN_ID}")
fi
if [[ "${WANDB_LOG_MODEL}" == "1" ]]; then
    EXTRA_ARGS+=(--wandb_log_model)
fi
if [[ "${SKIP_TRITON_WARMUP}" == "1" ]]; then
    EXTRA_ARGS+=(--skip_triton_warmup)
fi
if [[ "${DISABLE_COMPILED_BDA}" == "1" ]]; then
    EXTRA_ARGS+=(--disable_compiled_bda)
fi
if [[ "${FORCE_CONTIGUOUS_QKV}" == "1" ]]; then
    EXTRA_ARGS+=(--force_contiguous_qkv)
fi

export TRITON_CACHE_DIR="/tmpdir/m24047brmn/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

WANDB_ENV_ARGS=(
    --env "WANDB_MODE=${WANDB_MODE}"
    --env "WANDB_DIR=${WANDB_DIR}"
    --env "WANDB_START_METHOD=${WANDB_START_METHOD:-thread}"
)
if [[ -n "${WANDB_API_KEY}" ]]; then
    WANDB_ENV_ARGS+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")
fi

srun apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --env "MASTER_ADDR=${MASTER_ADDR}" \
    --env "MASTER_PORT=${MASTER_PORT}" \
    --env "SLURM_NNODES=${SLURM_NNODES}" \
    --env "NVTE_DEBUG=1" \
    --env "NVTE_DEBUG_LEVEL=2" \
    --env "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}" \
    --env "SSA_KERNEL_VERSION=${SSA_KERNEL_VERSION}" \
    --env "SSA_TRITON_COMPILE_BDA=${SSA_TRITON_COMPILE_BDA}" \
    "${WANDB_ENV_ARGS[@]}" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    torchrun \
        --nnodes=${SLURM_NNODES} \
        --nproc_per_node=2 \
        --rdzv_id=${SLURM_JOB_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/test/train_ssa_triton.py \
        --datamix "${DATAMIX}" \
        --output_dir "${OUTPUT_DIR}" \
        --name "${NAME}" \
        --arch baby_luciole \
        --max_steps "${GLOBAL_MAX_STEPS}" \
        --seq_length 1024 \
        --batch_size 768 \
        --micro_batch_size 8 \
        --num_nodes "${SLURM_NNODES}" \
        --gpus_per_node 2 \
        --tensor_parallelism 1 \
        --pipeline_parallelism 1 \
        --context_parallelism 1 \
        --duration "${SLURM_DURATION}" \
        --save_every_n_steps 200 \
        --log_ssa_every_n_steps 50 \
        --ssa_n "${SSA_N}" \
        --ssa_b "${SSA_B}" \
        --warmup_steps "${LR_WARMUP_STEPS}" \
        "${EXTRA_ARGS[@]}" \
        --seed "${SEED}"

status=$?
echo "=========================================="
echo "SSA Triton W&B test run finished with status ${status}"
echo "Run name: ${NAME}"
echo "=========================================="
exit ${status}
