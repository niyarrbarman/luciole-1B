#!/bin/bash
#SBATCH -J load_nemotron1l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/%x_%j.out

# Ensure output directory exists
mkdir -p slurm

# Checkpoint path for the 1-layer test model
CHECKPOINT_PATH=${CHECKPOINT:-"/tmpdir/m24047brmn/nemo_1b/output/nemotron1b-1layer-test/2026-01-10_03-36-53/checkpoints/model_name=0--val_loss=0.00-step=881-consumed_samples=112896.0-last"}

# Generic test prompt
DEMO_PROMPT=${PROMPT:-"<|startoftext|> Question: Juana Saquic is constructing a collectibles in a"}
TOKENIZER_PATH=${TOKENIZER:-"/work/m24047/m24047brmn/tokenizers/minitron-4b"}
NUM_LAYERS=${NUM_LAYERS:-1}

echo "=========================================="
echo "Loading ${NUM_LAYERS}-layer Nemotron model (NeMo)"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Prompt: $DEMO_PROMPT"
echo "=========================================="

# Run with apptainer using the NeMo ARM container with user env for additional packages
apptainer exec --env "PYTHONUSERBASE=${MYENVS}/nemo" --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.11_arm.sif \
bash -lc "cd /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/train_ssa_triton && \
    export PYTHONPATH=${MYENVS}/nemo/lib/python3.12/site-packages:/usr/local/lib/python3.12/dist-packages:/opt/venv/lib/python3.12/site-packages:/opt/nemo:/opt/NeMo:/opt/NeMo/examples:/opt/megatron-lm:${PYTHONPATH} && \
    python3 ./load_model.py \
        --checkpoint '$CHECKPOINT_PATH' \
        --tokenizer '$TOKENIZER_PATH' \
        --prompt '$DEMO_PROMPT' \
        --num_layers $NUM_LAYERS \
        --max_new_tokens 64 \
        --temperature 0.7 \
        --top_k 50 \
        --top_p 0.9"

status=$?
echo "=========================================="
echo "Inference finished with status $status"
echo "=========================================="
exit $status
