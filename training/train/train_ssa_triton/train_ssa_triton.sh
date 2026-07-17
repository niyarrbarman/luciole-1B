#!/bin/bash
#SBATCH -J tr_p2_nemo1b_ssa
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/%x_%j.log
#SBATCH --error=slurm/%x_%j.log
#SBATCH --cpus-per-task=144
#SBATCH --exclude=dalianvl[02-05]

module purge
module load gcc slurm

mkdir -p slurm

# Defaults
DATAMIX=${DATAMIX:-"/lustre/work/pdl17996/udl62d273/luciole-1B/training/train/train_ssa_triton/luciole_phase2.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"/lustre/work/pdl17996/udl62d273/phase2_outputs/outputs"}
mkdir -p "$OUTPUT_DIR"
BATCH_SIZE=${BATCH_SIZE:-1024}
NAME=${NAME:-"nemotron-1B-SSA-Triton-phase2-bs${BATCH_SIZE}"}
SEED=${SEED:-1234}

export HF_HOME="${HF_HOME:-/lustre/work/pdl17996/udl62d273/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# W&B logging (optional)
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-"luciole_ssa"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_GROUP=${WANDB_GROUP:-"${NAME}"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${NAME}"}
WANDB_TAGS=${WANDB_TAGS:-"ssa,triton,wandb,test"}
WANDB_NOTES=${WANDB_NOTES:-"Nemotron-1B SSA Triton phase2 training"}
WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-"train"}
WANDB_MODE=${WANDB_MODE:-"offline"}
WANDB_DIR=${WANDB_DIR:-"${OUTPUT_DIR}/${NAME}/wandb"}
WANDB_RUN_ID=${WANDB_RUN_ID:-"nemo1b-ssa-triton-v1"}
WANDB_RESUME=${WANDB_RESUME:-"allow"}
WANDB_LOG_MODEL=${WANDB_LOG_MODEL:-0}
# Allow `api_key=... sbatch ...` while supporting WANDB_API_KEY.
WANDB_API_KEY=${WANDB_API_KEY:-${api_key:-""}}

# SSA hyperparameters
SSA_N=1.5 # fixed
SSA_B=0.8 # fixed
SSA_KERNEL_VERSION=${SSA_KERNEL_VERSION:-triton}
SSA_TRITON_COMPILE_BDA=${SSA_TRITON_COMPILE_BDA:-1}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
SKIP_TRITON_WARMUP=${SKIP_TRITON_WARMUP:-1}
DISABLE_COMPILED_BDA=${DISABLE_COMPILED_BDA:-0}
FORCE_CONTIGUOUS_QKV=${FORCE_CONTIGUOUS_QKV:-1}
# GLOBAL_MAX_STEPS=${GLOBAL_MAX_STEPS:-3817000}
GLOBAL_MAX_STEPS=${GLOBAL_MAX_STEPS:-20}
# Backward-compatible alias: if THIS_RUN_MAX_STEPS is unset, use legacy MAX_STEPS when provided.
THIS_RUN_MAX_STEPS=${THIS_RUN_MAX_STEPS:-0}

if [[ "${SSA_KERNEL_VERSION}" != "triton" ]]; then
  echo "ERROR: SSA_KERNEL_VERSION must be 'triton' (got '${SSA_KERNEL_VERSION}')."
  exit 2
fi

if [[ "${USE_WANDB}" == "1" && "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY}" ]]; then
  echo "ERROR: USE_WANDB=1 and WANDB_MODE=online but no API key provided."
  echo "Use: WANDB_API_KEY='...' sbatch train_ssa_triton.sh"
  echo "or:  api_key='...' sbatch train_ssa_triton.sh"
  exit 2
fi

# Multi-node coordination
export MASTER_PORT=$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(hostname --ip-address)

# Convert actual SLURM wall time to DD:HH:MM:SS for StatelessTimer
if [[ -n "${SLURM_JOB_END_TIME:-}" && -n "${SLURM_JOB_START_TIME:-}" ]]; then
  WALL_SECS=$((SLURM_JOB_END_TIME - SLURM_JOB_START_TIME))
  SLURM_DURATION=$(printf "%02d:%02d:%02d:%02d" $((WALL_SECS / 86400)) $(((WALL_SECS % 86400) / 3600)) $(((WALL_SECS % 3600) / 60)) $((WALL_SECS % 60)))
else
  # Fallback: parse from script header
  SBATCH_TIME=$(grep -E '^#SBATCH --time=' "$0" | head -n1 | sed -E 's/^#SBATCH --time=//')
  if [[ "$SBATCH_TIME" == *-* ]]; then
    SLURM_DURATION=$(echo "$SBATCH_TIME" | awk -F'[-:]' '{printf "%02d:%02d:%02d:%02d", $1, $2, $3, $4}')
  else
    SLURM_DURATION=$(echo "$SBATCH_TIME" | awk -F':' '{printf "00:%02d:%02d:%02d", $1, $2, $3}')
  fi
fi

echo "=========================================="
echo "Starting SSA Triton Training"
echo "Datamix:     $DATAMIX"
echo "Output:      $OUTPUT_DIR"
echo "Name:        $NAME"
echo "Batch size:  $BATCH_SIZE"
echo "Nodes:       $SLURM_NNODES"
echo "Duration:    ${SLURM_DURATION}"
echo "SSA n:       $SSA_N"
echo "SSA b:       $SSA_B"
echo "Kernel ver:  $SSA_KERNEL_VERSION"
echo "Compile BDA: $SSA_TRITON_COMPILE_BDA"
echo "Warmup step: $LR_WARMUP_STEPS"
echo "Skip warmup: $SKIP_TRITON_WARMUP"
echo "Contig QKV:  $FORCE_CONTIGUOUS_QKV"
echo "Global max:  $GLOBAL_MAX_STEPS"
echo "This-run max:${THIS_RUN_MAX_STEPS}"
echo "W&B:         $USE_WANDB (mode=${WANDB_MODE})"
echo "=========================================="

EXTRA_ARGS=()
if [[ "${SKIP_TRITON_WARMUP}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_triton_warmup)
fi
if [[ "${DISABLE_COMPILED_BDA}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_compiled_bda)
fi
if [[ "${FORCE_CONTIGUOUS_QKV}" == "1" ]]; then
  EXTRA_ARGS+=(--force_contiguous_qkv)
fi
if [[ "${THIS_RUN_MAX_STEPS}" != "0" ]]; then
  EXTRA_ARGS+=(--this_run_max_steps "${THIS_RUN_MAX_STEPS}")
fi
if [[ "${USE_WANDB}" == "1" ]]; then
  EXTRA_ARGS+=(--wandb --wandb_mode "${WANDB_MODE}" --wandb_dir "${WANDB_DIR}" --wandb_job_type "${WANDB_JOB_TYPE}" --wandb_resume "${WANDB_RESUME}")
  if [[ -n "${WANDB_PROJECT}" ]]; then
    EXTRA_ARGS+=(--wandb_project "${WANDB_PROJECT}")
  fi
  if [[ -n "${WANDB_ENTITY}" ]]; then
    EXTRA_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_GROUP}" ]]; then
    EXTRA_ARGS+=(--wandb_group "${WANDB_GROUP}")
  else
    EXTRA_ARGS+=(--wandb_group "${NAME}")
  fi
  if [[ -n "${WANDB_RUN_NAME}" ]]; then
    EXTRA_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
  else
    EXTRA_ARGS+=(--wandb_run_name "${NAME}")
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
fi

# Pre-compile Triton kernels (warmup) — avoids JIT overhead at step 0
# Triton caches compiled kernels in ~/.triton/cache, so this only helps first run
export TRITON_CACHE_DIR="/lustre/work/pdl17996/udl62d273/phase2_outputs/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

WANDB_ENV_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ENV_ARGS+=(--env "WANDB_MODE=${WANDB_MODE}")
  WANDB_ENV_ARGS+=(--env "WANDB_DIR=${WANDB_DIR}")
  WANDB_ENV_ARGS+=(--env "WANDB_START_METHOD=${WANDB_START_METHOD:-thread}")
  if [[ -n "${WANDB_API_KEY}" ]]; then
    WANDB_ENV_ARGS+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")
  fi
fi

srun apptainer exec \
  --env "PYTHONUSERBASE=/lustre/work/pdl17996/udl62d273/nemo" \
  --env "MASTER_ADDR=${MASTER_ADDR}" \
  --env "MASTER_PORT=${MASTER_PORT}" \
  --env "SLURM_NNODES=${SLURM_NNODES}" \
  --env "NVTE_DEBUG=1" \
  --env "NVTE_DEBUG_LEVEL=2" \
  --env "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}" \
  --env "SSA_KERNEL_VERSION=${SSA_KERNEL_VERSION}" \
  --env "SSA_TRITON_COMPILE_BDA=${SSA_TRITON_COMPILE_BDA}" \
  --env "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  "${WANDB_ENV_ARGS[@]}" \
  --bind "$WORK:$WORK,/lustre/work/pdl17996/shared:/lustre/work/pdl17996/shared" \
  --nv /lustre/work/pdl17996/shared/containers/nemo_25.04.03_arm.sif \
  torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  /lustre/work/pdl17996/udl62d273/luciole-1B/training/train/train_ssa_triton/launcher.py \
  /lustre/work/pdl17996/udl62d273/luciole-1B/training/train/train_ssa_triton/train_ssa_triton.py \
  --datamix "$DATAMIX" \
  --output_dir "$OUTPUT_DIR" \
  --name "$NAME" \
  --arch nemotron1b \
  --tokenizer OpenLLM-BPI/tokenizer_128k-arab-regional_v2 \
  --max_steps ${GLOBAL_MAX_STEPS} \
  --seq_length 4096 \
  --batch_size ${BATCH_SIZE} \
  --micro_batch_size ${MICRO_BATCH_SIZE:-8} \
  --num_nodes ${SLURM_NNODES} \
  --gpus_per_node 4 \
  --base_checkpoint /lustre/work/pdl17996/udl62d273/checkpoint_phase1/nemotron-1B-SSA-Triton-bs1024-step=0715787-last \
  --tensor_parallelism 1 \
  --pipeline_parallelism 1 \
  --context_parallelism 1 \
  --duration "${SLURM_DURATION}" \
  --save_every_n_steps 5 \
  --log_ssa_every_n_steps 2 \
  --ssa_n $SSA_N \
  --ssa_b $SSA_B \
  --warmup_steps ${LR_WARMUP_STEPS} \
  "${EXTRA_ARGS[@]}" \
  --seed $SEED

status=$?
echo "=========================================="
echo "SSA Triton Training finished with status $status"
echo "=========================================="
exit $status
