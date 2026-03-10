#!/bin/bash
#SBATCH -J load_ckpt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%j.out

# Ensure output directory for logs exists
mkdir -p slurm

# Checkpoint path from job 73515
CHECKPOINT="/tmpdir/m24047brmn/nemo_1b/output/nemotron1b-1layer-test/2026-01-11_19-11-56/checkpoints/model_name=0--val_loss=0.00-step=3627-consumed_samples=464384.0-last"

echo "=========================================="
echo "Loading Nemotron 1B checkpoint"
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="

apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.11_arm.sif \
    python /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/train_ssa_triton/load_model.py \
        --checkpoint "$CHECKPOINT" \
        --num_layers 24 \
        --prompt "<|startoftext|> Question: In a historical reenactment fair at Boone County, Illinois, each designer scarves container can hold 3 designer scarves. There are 22 children and 8 adults, each child has 4 designer scarves, and each adult has 10 designer scarves. If all the designer scarves containers are full, how many designer scarves containers are there? Answer: In a historical reenactment fair at Boone County, Illinois, there are 88 " \
        --max_new_tokens 128 \
        --temperature 0.7

status=$?
echo "=========================================="
echo "Inference finished with status $status"
echo "=========================================="
exit $status
