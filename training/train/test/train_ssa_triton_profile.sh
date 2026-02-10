#!/bin/bash
#SBATCH -J tr_bbyluc_ssa_triton_prof
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -p small
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

# Defaults
DATAMIX=${DATAMIX:-"/tmpdir/m24047brmn/nemo_1b/data_fwe_50k/datamix_fineweb_edu_50k.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"/tmpdir/m24047brmn/nemo_1b/output_prof"}
NAME=${NAME:-"baby_luciole-ssa-triton-prof"}
SEED=${SEED:-1234}

# SSA hyperparameters
SSA_N=${SSA_N:-1.5}
SSA_B=${SSA_B:-0.8}

# Multi-node coordination
export MASTER_PORT=$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

# Convert SBATCH time to DD:HH:MM:SS for StatelessTimer
SBATCH_TIME=$(grep -E '^#SBATCH --time=' "$0" | head -n1 | sed -E 's/^#SBATCH --time=//')
if [[ "$SBATCH_TIME" == *-* ]]; then
    SLURM_DURATION=$(echo "$SBATCH_TIME" | awk -F'[-:]' '{printf "%02d:%02d:%02d:%02d", $1, $2, $3, $4}')
else
    SLURM_DURATION=$(echo "$SBATCH_TIME" | awk -F':' '{printf "00:%02d:%02d:%02d", $1, $2, $3}')
fi

echo "=========================================="
echo "Starting SSA Triton Profiling"
echo "Datamix:     $DATAMIX"
echo "Output:      $OUTPUT_DIR"
echo "Name:        $NAME"
echo "Nodes:       $SLURM_NNODES"
echo "Duration:    ${SLURM_DURATION}"
echo "SSA n:       $SSA_N"
echo "SSA b:       $SSA_B"
echo "=========================================="

# Triton cache for JIT kernels
export TRITON_CACHE_DIR="/tmpdir/m24047brmn/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

# Nsight Systems output directory (shared path)
NSYS_DIR="/work/m24047/m24047brmn/nsys"
mkdir -p "$NSYS_DIR"
# Use per-process output to avoid file clobbering in multi-rank runs
NSYS_OUT="$NSYS_DIR/${NAME}_${SLURM_JOB_ID}_%p"

srun apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --env "MASTER_ADDR=${MASTER_ADDR}" \
    --env "MASTER_PORT=${MASTER_PORT}" \
    --env "SLURM_NNODES=${SLURM_NNODES}" \
    --env "NVTE_DEBUG=1" \
    --env "NVTE_DEBUG_LEVEL=2" \
    --env "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    nsys profile \
        --output "$NSYS_OUT" \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cuda-event-trace=false \
        --stats=true \
        --duration=180 \
        torchrun \
            --nnodes=${SLURM_NNODES} \
            --nproc_per_node=2 \
            --rdzv_id=${SLURM_JOB_ID} \
            --rdzv_backend=c10d \
            --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
            /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/test/train_ssa_triton.py \
            --datamix "$DATAMIX" \
            --output_dir "$OUTPUT_DIR" \
            --name "$NAME" \
            --arch baby_luciole \
            --max_steps 10 \
            --seq_length 1024 \
            --batch_size 768 \
            --micro_batch_size 8 \
            --num_nodes ${SLURM_NNODES} \
            --gpus_per_node 2 \
            --tensor_parallelism 1 \
            --pipeline_parallelism 1 \
            --context_parallelism 1 \
            --duration "${SLURM_DURATION}" \
            --global_max_steps 60000 \
            --save_every_n_steps 10000 \
            --log_ssa_every_n_steps 1000 \
            --ssa_n $SSA_N \
            --ssa_b $SSA_B \
            --seed $SEED

status=$?
echo "=========================================="
echo "SSA Triton Profiling finished with status $status"
echo "Nsight Systems report: ${NSYS_OUT}.nsys-rep"
echo "=========================================="
exit $status
