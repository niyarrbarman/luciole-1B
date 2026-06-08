#!/bin/bash
#SBATCH -J diag_ssa_triton
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=70
#SBATCH --gres=gpu:4
#SBATCH -p gpu
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out
##SBATCH --mail-user=niyar-r.barman@utoulouse.fr
##SBATCH --mail-type=ALL

mkdir -p slurm

# Defaults
DATAMIX=${DATAMIX:-"/work/p26037/barman/luciole-1B/training/train/train_ssa_triton/datamix_luciole_phase1.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"/tmpdir/barman/bs1024_diag/outputs"}
mkdir -p "$OUTPUT_DIR"
BATCH_SIZE=${BATCH_SIZE:-1024}
NAME=${NAME:-"diag-SSA-Triton-bs${BATCH_SIZE}"}
SEED=${SEED:-1234}

# W&B disabled for diag
USE_WANDB=0

# SSA hyperparameters
SSA_N=1.5 # fixed
SSA_B=0.8 # fixed
SSA_KERNEL_VERSION=${SSA_KERNEL_VERSION:-triton}
SSA_TRITON_COMPILE_BDA=${SSA_TRITON_COMPILE_BDA:-1}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-2000}
SKIP_TRITON_WARMUP=${SKIP_TRITON_WARMUP:-1}
DISABLE_COMPILED_BDA=${DISABLE_COMPILED_BDA:-0}
FORCE_CONTIGUOUS_QKV=${FORCE_CONTIGUOUS_QKV:-1}
GLOBAL_MAX_STEPS=${GLOBAL_MAX_STEPS:-715788}
THIS_RUN_MAX_STEPS=${THIS_RUN_MAX_STEPS:-10}

if [[ "${SSA_KERNEL_VERSION}" != "triton" ]]; then
  echo "ERROR: SSA_KERNEL_VERSION must be 'triton' (got '${SSA_KERNEL_VERSION}')."
  exit 2
fi

# Multi-node coordination
export MASTER_PORT=$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(hostname --ip-address)

# Convert actual SLURM wall time to DD:HH:MM:SS for StatelessTimer
if [[ -n "${SLURM_JOB_END_TIME:-}" && -n "${SLURM_JOB_START_TIME:-}" ]]; then
  WALL_SECS=$(( SLURM_JOB_END_TIME - SLURM_JOB_START_TIME ))
  SLURM_DURATION=$(printf "%02d:%02d:%02d:%02d" $((WALL_SECS/86400)) $(((WALL_SECS%86400)/3600)) $(((WALL_SECS%3600)/60)) $((WALL_SECS%60)))
else
  SBATCH_TIME=$(grep -E '^#SBATCH --time=' "$0" | head -n1 | sed -E 's/^#SBATCH --time=//')
  if [[ "$SBATCH_TIME" == *-* ]]; then
    SLURM_DURATION=$(echo "$SBATCH_TIME" | awk -F'[-:]' '{printf "%02d:%02d:%02d:%02d", $1, $2, $3, $4}')
  else
    SLURM_DURATION=$(echo "$SBATCH_TIME" | awk -F':' '{printf "00:%02d:%02d:%02d", $1, $2, $3}')
  fi
fi

echo "=========================================="
echo "DIAGNOSTIC SSA Triton Training"
echo "Node:        $(hostname)"
echo "Datamix:     $DATAMIX"
echo "Output:      $OUTPUT_DIR"
echo "Name:        $NAME"
echo "Batch size:  $BATCH_SIZE"
echo "Nodes:       $SLURM_NNODES"
echo "Duration:    ${SLURM_DURATION}"
echo "This-run max:${THIS_RUN_MAX_STEPS}"
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

export TRITON_CACHE_DIR="/tmpdir/barman/bs1024_diag/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

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
  --env "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  --env "OMP_NUM_THREADS=18" \
  --bind /tmpdir,/work --nv /work/conteneurs/shared/AI/nemo_25.04.03_arm.sif \
  torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  /work/p26037/barman/luciole-1B/training/train/train_ssa_triton/train_ssa_triton.py \
  --datamix "$DATAMIX" \
  --output_dir "$OUTPUT_DIR" \
  --name "$NAME" \
  --arch nemotron1b \
  --tokenizer /work/p26037/barman/tokenizer_128k-arab-regional_v2 \
  --max_steps ${GLOBAL_MAX_STEPS} \
  --seq_length 4096 \
  --batch_size ${BATCH_SIZE} \
  --micro_batch_size ${MICRO_BATCH_SIZE:-4} \
  --num_nodes ${SLURM_NNODES} \
  --gpus_per_node 4 \
  --tensor_parallelism 1 \
  --pipeline_parallelism 1 \
  --context_parallelism 1 \
  --duration "${SLURM_DURATION}" \
  --save_every_n_steps 500 \
  --log_ssa_every_n_steps 500 \
  --ssa_n $SSA_N \
  --ssa_b $SSA_B \
  --warmup_steps ${LR_WARMUP_STEPS} \
  "${EXTRA_ARGS[@]}" \
  --seed $SEED

status=$?
echo "=========================================="
echo "DIAGNOSTIC SSA Triton Training finished with status $status"
echo "=========================================="
exit $status
