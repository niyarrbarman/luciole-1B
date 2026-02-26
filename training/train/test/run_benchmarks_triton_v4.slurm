#!/bin/bash
#SBATCH -J benchmark_bbyluc_triton_v4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

CHECKPOINT_PATH=${CHECKPOINT:-"/tmpdir/m24047brmn/nemo_1b/output/baby_luciole-ssa-triton-v4/checkpoints/baby_luciole-ssa-triton-v4-step=0029999-last"}
TOKENIZER_PATH=${TOKENIZER:-"/work/m24047/m24047brmn/tokenizers/luciole_50k"}
TASKS=${TASKS:-"all"}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-2048}
LIMIT=${LIMIT:-""}
NUM_FEWSHOT=${NUM_FEWSHOT:-5}
COMPILED_BDA=${COMPILED_BDA:-0}
FORCE_CONTIGUOUS_QKV=${FORCE_CONTIGUOUS_QKV:-1}
HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"benchmark_results"}

mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/results_triton_v4_${TASKS//,/_}_${TIMESTAMP}.json"

echo "==========================================="
echo "LM Evaluation Harness - Baby Luciole SSA Triton-v4"
echo "==========================================="
echo "Checkpoint:      $CHECKPOINT_PATH"
echo "Tokenizer:       $TOKENIZER_PATH"
echo "Tasks:           $TASKS"
echo "Batch Size:      $BATCH_SIZE"
echo "Max Length:      $MAX_LENGTH"
echo "Limit:           ${LIMIT:-'None (full dataset)'}"
echo "Num Fewshot:     ${NUM_FEWSHOT:-'Task default'}"
echo "Compiled BDA:    $COMPILED_BDA"
echo "Contiguous QKV:  $FORCE_CONTIGUOUS_QKV"
echo "HF Offline:      $HF_DATASETS_OFFLINE"
echo "Output:          $OUTPUT_FILE"
echo "==========================================="

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

FEWSHOT_ARG=""
if [ -n "$NUM_FEWSHOT" ]; then
    FEWSHOT_ARG="--num_fewshot $NUM_FEWSHOT"
fi

COMPILED_BDA_ARG="--no-compiled-bda"
if [ "$COMPILED_BDA" = "1" ]; then
    COMPILED_BDA_ARG="--compiled-bda"
fi

CONTIG_QKV_ARG="--no-force-contiguous-qkv"
if [ "$FORCE_CONTIGUOUS_QKV" = "1" ]; then
    CONTIG_QKV_ARG="--force-contiguous-qkv"
fi

apptainer exec \
    --env "HF_HOME=/work/m24047/m24047brmn/lm_eval" \
    --env "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}" \
    --bind /tmpdir,/work \
    --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    bash -lc "cd /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/test && \
        export PYTHONPATH=/usr/lib/python3.12:/usr/local/lib/python3.12/dist-packages:/usr/local/lib/python3.12/dist-packages/lightning_utilities-0.14.0-py3.12.egg:/opt/venv/lib/python3.12/site-packages:/opt/nemo:/opt/NeMo:/opt/NeMo/examples:/opt/megatron-lm:${MYENVS}/nemo/lib/python3.12/site-packages:\${PYTHONPATH} && \
        python3 run_benchmarks_triton_v4.py \
            --checkpoint '$CHECKPOINT_PATH' \
            --tokenizer '$TOKENIZER_PATH' \
            --tasks '$TASKS' \
            --batch_size $BATCH_SIZE \
            --max_length $MAX_LENGTH \
            --output '$OUTPUT_FILE' \
            $COMPILED_BDA_ARG \
            $CONTIG_QKV_ARG \
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
