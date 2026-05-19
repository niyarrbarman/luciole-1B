#!/bin/bash
#SBATCH -J benchmark_random_baseline
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

# Random baseline evaluation - no model loading, just random predictions
TOKENIZER_PATH=${TOKENIZER:-"/tmpdir/m24047brmn/kairos/tokenizer_128k-arab-regional_v2"}
TASKS=${TASKS:-"french_bench"} # e.g. "copa" or "french_bench"
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_LENGTH=${MAX_LENGTH:-2048}
LIMIT=${LIMIT:-""}
GSM8K_LIMIT=${GSM8K_LIMIT:-"-1"}
GSM8K_RANDOM_SEED=${GSM8K_RANDOM_SEED:-42}
NUM_FEWSHOT=${NUM_FEWSHOT:-5}
HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
HF_HOME=${HF_HOME:-"/work/m24047/m24047flhg/.cache/huggingface"}
HF_HOME_LEGACY=${HF_HOME_LEGACY:-"/work/m24047/m24047brmn/lm_eval"}
HF_CACHE_MERGE_MODE=${HF_CACHE_MERGE_MODE:-"symlink"} # symlink|copy|none
OUTPUT_DIR=${OUTPUT_DIR:-"benchmark_results_random"}

mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/random_baseline_${TASKS//,/_}_${TIMESTAMP}.json"

echo "==========================================="
echo "Random Baseline Evaluation"
echo "==========================================="
echo "Tokenizer:       $TOKENIZER_PATH"
echo "Tasks:           $TASKS"
echo "Batch Size:      $BATCH_SIZE"
echo "Max Length:      $MAX_LENGTH"
echo "Limit:           ${LIMIT:-'None (full dataset)'}"
echo "GSM8K Limit:     ${GSM8K_LIMIT:-'None (full dataset)'}"
echo "GSM8K Seed:      ${GSM8K_RANDOM_SEED}"
echo "Num Fewshot:     ${NUM_FEWSHOT:-'Task default'}"
echo "HF Offline:      $HF_DATASETS_OFFLINE"
echo "HF Home:         $HF_HOME"
echo "Output:          $OUTPUT_FILE"
echo "==========================================="

prepare_hf_cache() {
    mkdir -p "$HF_HOME/datasets" "$HF_HOME/hub"

    if [ "$HF_CACHE_MERGE_MODE" = "none" ]; then
        echo "HF cache merge disabled"
        return
    fi

    if [ ! -d "$HF_HOME_LEGACY" ]; then
        echo "HF legacy cache not found: $HF_HOME_LEGACY (continuing)"
        return
    fi

    echo "Preparing HF cache (mode=$HF_CACHE_MERGE_MODE) from $HF_HOME_LEGACY"

    if [ "$HF_CACHE_MERGE_MODE" = "copy" ]; then
        if command -v rsync >/dev/null 2>&1; then
            if [ -d "$HF_HOME_LEGACY/datasets" ]; then
                rsync -a --ignore-existing "$HF_HOME_LEGACY/datasets/" "$HF_HOME/datasets/"
            fi
            if [ -d "$HF_HOME_LEGACY/hub" ]; then
                rsync -a --ignore-existing "$HF_HOME_LEGACY/hub/" "$HF_HOME/hub/"
            fi
        else
            if [ -d "$HF_HOME_LEGACY/datasets" ]; then
                cp -an "$HF_HOME_LEGACY/datasets/." "$HF_HOME/datasets/"
            fi
            if [ -d "$HF_HOME_LEGACY/hub" ]; then
                cp -an "$HF_HOME_LEGACY/hub/." "$HF_HOME/hub/"
            fi
        fi
    elif [ "$HF_CACHE_MERGE_MODE" = "symlink" ]; then
        if [ -d "$HF_HOME_LEGACY/datasets" ]; then
            for src in "$HF_HOME_LEGACY"/datasets/*; do
                [ -e "$src" ] || continue
                base=$(basename "$src")
                dst="$HF_HOME/datasets/$base"
                if [ ! -e "$dst" ]; then
                    ln -s "$src" "$dst"
                fi
            done
        fi
        if [ -d "$HF_HOME_LEGACY/hub" ]; then
            for src in "$HF_HOME_LEGACY"/hub/*; do
                [ -e "$src" ] || continue
                base=$(basename "$src")
                dst="$HF_HOME/hub/$base"
                if [ ! -e "$dst" ]; then
                    ln -s "$src" "$dst"
                fi
            done
        fi
    else
        echo "WARNING: Unknown HF_CACHE_MERGE_MODE='$HF_CACHE_MERGE_MODE' (expected symlink|copy|none)."
    fi
}

prepare_hf_cache

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

FEWSHOT_ARG=""
if [ -n "$NUM_FEWSHOT" ]; then
    FEWSHOT_ARG="--num_fewshot $NUM_FEWSHOT"
fi

echo "-------------------------------------------"
echo "Running random baseline evaluation"
echo "-------------------------------------------"

apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/luciole_bench" \
    --env "HF_HOME=${HF_HOME}" \
    --env "HF_DATASETS_CACHE=${HF_HOME}/datasets" \
    --env "HF_HUB_CACHE=${HF_HOME}/hub" \
    --env "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}" \
    --bind /tmpdir,/work \
    --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    bash -c "cd /users/m24047/m24047flhg/luciole-eval/luciole-1B/training/train/train_ssa_triton && \
        export PYTHONPATH=/usr/lib/python3.12:/usr/local/lib/python3.12/dist-packages:/usr/local/lib/python3.12/dist-packages/lightning_utilities-0.14.0-py3.12.egg:/opt/venv/lib/python3.12/site-packages:/opt/nemo:/opt/NeMo:/opt/NeMo/examples:/opt/megatron-lm:${MYENVS}/nemo/lib/python3.12/site-packages:\${PYTHONPATH} && \
        python3 run_benchmarks_random.py \
            --tokenizer '$TOKENIZER_PATH' \
            --tasks '$TASKS' \
            --batch_size $BATCH_SIZE \
            --max_length $MAX_LENGTH \
            --gsm8k-limit $GSM8K_LIMIT \
            --gsm8k-random-seed $GSM8K_RANDOM_SEED \
            --output '$OUTPUT_FILE' \
            $LIMIT_ARG \
            $FEWSHOT_ARG"

status=$?

echo "==========================================="
echo "Random baseline evaluation finished with status $status"
echo "Results saved to: $OUTPUT_FILE"
echo "==========================================="
exit $status
