#!/bin/bash
#SBATCH -J tr_ssa_fa
#SBATCH -N 5
#SBATCH -n 5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

# Defaults
DATAMIX=${DATAMIX:-"/tmpdir/m24047brmn/nemo_1b/data/nemo1b_mock_datamix.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"/tmpdir/m24047brmn/nemo_1b/output"}
NAME=${NAME:-"nemotron1b-ssa-flex-test"}
SEED=${SEED:-1234}

# SSA parameters (fixed, not learnable with FlexAttention)
SSA_N=${SSA_N:-1.5}
SSA_B=${SSA_B:-0.8}

# Multi-node coordination
export MASTER_PORT=$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(hostname --ip-address)

# Convert SBATCH time to DD:HH:MM:SS for StatelessTimer
SBATCH_TIME=$(grep -E '^#SBATCH --time=' "$0" | head -n1 | sed -E 's/^#SBATCH --time=//')
if [[ "$SBATCH_TIME" == *-* ]]; then
    SLURM_DURATION=$(echo "$SBATCH_TIME" | awk -F'[-:]' '{printf "%02d:%02d:%02d:%02d", $1, $2, $3, $4}')
else
    SLURM_DURATION=$(echo "$SBATCH_TIME" | awk -F':' '{printf "00:%02d:%02d:%02d", $1, $2, $3}')
fi

echo "==========================================="
echo "Starting SSA FlexAttention Training"
echo "Datamix:     $DATAMIX"
echo "Output:      $OUTPUT_DIR"
echo "Nodes:       $SLURM_NNODES"
echo "Duration:    ${SLURM_DURATION}"
echo "SSA n:       $SSA_N"
echo "SSA b:       $SSA_B"
echo "==========================================="

srun apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --env "MASTER_ADDR=${MASTER_ADDR}" \
    --env "MASTER_PORT=${MASTER_PORT}" \
    --env "SLURM_NNODES=${SLURM_NNODES}" \
    --env "NVTE_DEBUG=1" \
    --env "NVTE_DEBUG_LEVEL=2" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    torchrun \
        --nnodes=${SLURM_NNODES} \
        --nproc_per_node=2 \
        --rdzv_id=${SLURM_JOB_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/test/train_ssa_fa.py \
        --datamix "$DATAMIX" \
        --output_dir "$OUTPUT_DIR" \
        --name "$NAME" \
        --arch nemotron1b \
        --max_steps 10 \
        --batch_size 1280 \
        --num_nodes ${SLURM_NNODES} \
        --gpus_per_node 2 \
        --tensor_parallelism 1 \
        --pipeline_parallelism 1 \
        --context_parallelism 1 \
        --duration "${SLURM_DURATION}" \
        --global_max_steps 1000 \
        --save_every_n_steps 10 \
        --ssa_n $SSA_N \
        --ssa_b $SSA_B \
        --ssa_fixed \
        --seed $SEED

status=$?
echo "==========================================="
echo "SSA FlexAttention Training finished with status $status"
echo "==========================================="
exit $status
