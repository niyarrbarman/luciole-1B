#!/bin/bash
#SBATCH -J benchmark_nemotron
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

CHECKPOINT_PATH=${CHECKPOINT:-"/tmpdir/m24047brmn/nemo_1b/output/nemotron1b-1layer-test/2026-01-11_19-11-56/checkpoints/model_name=0--val_loss=0.00-step=3627-consumed_samples=464384.0-last"}
TOKENIZER_PATH=${TOKENIZER:-"/work/m24047/m24047brmn/tokenizers/minitron-4b"}
TASKS=${TASKS:-"arc_easy"}
NUM_LAYERS=${NUM_LAYERS:-24}
BATCH_SIZE=${BATCH_SIZE:-1}
LIMIT=${LIMIT:-""}
NUM_FEWSHOT=${NUM_FEWSHOT:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"benchmark_results"}

mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/results_${TASKS//,/_}_${TIMESTAMP}.json"

echo "==========================================="
echo "LM Evaluation Harness - Nemotron Model"
echo "==========================================="
echo "Checkpoint:  $CHECKPOINT_PATH"
echo "Tokenizer:   $TOKENIZER_PATH"
echo "Tasks:       $TASKS"
echo "Num Layers:  $NUM_LAYERS"
echo "Batch Size:  $BATCH_SIZE"
echo "Limit:       ${LIMIT:-'None (full dataset)'}"
echo "Num Fewshot: ${NUM_FEWSHOT:-'Task default'}"
echo "Output:      $OUTPUT_FILE"
echo "==========================================="

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

FEWSHOT_ARG=""
if [ -n "$NUM_FEWSHOT" ]; then
    FEWSHOT_ARG="--num_fewshot $NUM_FEWSHOT"
fi

apptainer exec \
    --env "HF_HOME=/work/m24047/m24047brmn/lm_eval" \
    --env "HF_DATASETS_OFFLINE=1" \
    --bind /tmpdir,/work \
    --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    bash -lc "cd /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/train_ssa_triton && \
        export PYTHONPATH=/usr/lib/python3.12:/usr/local/lib/python3.12/dist-packages:/usr/local/lib/python3.12/dist-packages/lightning_utilities-0.14.0-py3.12.egg:/opt/venv/lib/python3.12/site-packages:/opt/nemo:/opt/NeMo:/opt/NeMo/examples:/opt/megatron-lm:${MYENVS}/nemo/lib/python3.12/site-packages:\${PYTHONPATH} && \
        python3 run_benchmarks.py \
            --checkpoint '$CHECKPOINT_PATH' \
            --tokenizer '$TOKENIZER_PATH' \
            --tasks '$TASKS' \
            --num_layers $NUM_LAYERS \
            --batch_size $BATCH_SIZE \
            --output '$OUTPUT_FILE' \
            $LIMIT_ARG \
            $FEWSHOT_ARG"

status=$?
echo "==========================================="
echo "Benchmark finished with status $status"
if [ -f "$OUTPUT_FILE" ]; then
    echo "Results saved to: $OUTPUT_FILE"
fi
echo "==========================================="
exit $status
