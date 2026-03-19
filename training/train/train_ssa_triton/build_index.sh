#!/bin/bash
#SBATCH -J build_index_1b
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=70
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --time=06:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

# Must match training parameters exactly so index files are reusable
DATAMIX=${DATAMIX:-"/work/p26037/barman/luciole-1B/training/train/train_ssa_triton/datamix_luciole_phase1.json"}
# OUTPUT_DIR=${OUTPUT_DIR:-"/tmpdir/m24047brmn/nemo_1b/output_1b"}
OUTPUT_DIR=${OUTPUT_DIR:-"/tmpdir/barman/32bs_ssa_run/outputs"}
BATCH_SIZE=${BATCH_SIZE:-1024}
NAME=${NAME:-"nemotron-1B-SSA-Triton-bs${BATCH_SIZE}"}
SEED=${SEED:-1234}
# GLOBAL_MAX_STEPS=${GLOBAL_MAX_STEPS:-3817000}
GLOBAL_MAX_STEPS=${GLOBAL_MAX_STEPS:-715788}

# export OMP_NUM_THREADS=${1}
export MASTER_PORT=$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(hostname --ip-address)

echo "=========================================="
echo "Building Index Mappings"
echo "Datamix:     $DATAMIX"
echo "Output:      $OUTPUT_DIR"
echo "Batch size:  $BATCH_SIZE"
echo "Max steps:   $GLOBAL_MAX_STEPS"
echo "=========================================="

srun apptainer exec \
  --env "PYTHONUSERBASE=${MYENVS}/nemo" \
  --env "MASTER_ADDR=${MASTER_ADDR}" \
  --env "MASTER_PORT=${MASTER_PORT}" \
  --env "SLURM_NNODES=1" \
  --env "SSA_KERNEL_VERSION=triton" \
  --env "SSA_TRITON_COMPILE_BDA=1" \
  --bind /tmpdir,/work --nv /work/conteneurs/shared/AI/nemo_25.04.03_arm.sif \
  torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
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
  --micro_batch_size 1 \
  --num_nodes 1 \
  --gpus_per_node 1 \
  --tensor_parallelism 1 \
  --pipeline_parallelism 1 \
  --context_parallelism 1 \
  --duration "00:06:00:00" \
  --save_every_n_steps 999999 \
  --this_run_max_steps 1 \
  --skip_triton_warmup \
  --force_contiguous_qkv \
  --seed $SEED

status=$?
echo "=========================================="
echo "Index build finished with status $status"
echo "=========================================="
exit $status
