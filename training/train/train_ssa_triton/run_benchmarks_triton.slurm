#!/bin/bash
#SBATCH -J benchmark_luc1b_triton_99ksteps_ssa
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

# example of how it should be, you need to give the folder not a .pth or .ckpt
# CHECKPOINT=${SOFTMAX_CHECKPOINT:-"/tmpdir/m24047brmn/nemo_1b/output/baby_luciole-softmax-test/checkpoints/baby_luciole-softmax-test-step=0020998-last"}

SSA_CHECKPOINT=${SSA_CHECKPOINT:-"/tmpdir/m24047brmn/kairos/bs1024_fullrun/outputs/nemotron-1B-SSA-Triton-bs1024/checkpoints/nemotron-1B-SSA-Triton-bs1024-step=0011999-last"}
SSA_CHECKPOINT=${SSA_CHECKPOINT:-"/tmpdir/m24047flhg/nemotron-1B-SSA-Triton-bs1024-step=0099999"}
# SOFTMAX_CHECKPOINT=${SOFTMAX_CHECKPOINT:-"/tmpdir/m24047flhg/luciole-phase1-step0015000"}
SOFTMAX_CHECKPOINT=${SOFTMAX_CHECKPOINT:-"/tmpdir/m24047flhg/luciole-phase1-last"}


SSA_MODEL_NAME=${SSA_MODEL_NAME:-"ssa_luciole_10k"}
SOFTMAX_MODEL_NAME=${SOFTMAX_MODEL_NAME:-"luciole_10k"}
MODEL_SELECTION=${MODEL_SELECTION:-"ssa"} # both|ssa|softmax
TOKENIZER_PATH=${TOKENIZER:-"/tmpdir/m24047brmn/kairos/tokenizer_128k-arab-regional_v2"}
TASKS=${TASKS:-"all"}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_LENGTH=${MAX_LENGTH:-2048}
LIMIT=${LIMIT:-""}
GSM8K_LIMIT=${GSM8K_LIMIT:-"1000"}
GSM8K_RANDOM_SEED=${GSM8K_RANDOM_SEED:-42}
NUM_FEWSHOT=${NUM_FEWSHOT:-5}
COMPILED_BDA=${COMPILED_BDA:-0}
FORCE_CONTIGUOUS_QKV=${FORCE_CONTIGUOUS_QKV:-1}
HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"benchmark_results_99ksteps_ssa"}

mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PREFIX="${OUTPUT_DIR}/results_luc1b_${MODEL_SELECTION}_${TASKS//,/_}_${TIMESTAMP}"

echo "==========================================="
echo "LM Evaluation Harness - Baby Luciole"
echo "==========================================="
echo "Model selection:  $MODEL_SELECTION"
echo "SSA checkpoint:  $SSA_CHECKPOINT"
echo "Softmax ckpt:    $SOFTMAX_CHECKPOINT"
echo "Tokenizer:       $TOKENIZER_PATH"
echo "Tasks:           $TASKS"
echo "Batch Size:      $BATCH_SIZE"
echo "Max Length:      $MAX_LENGTH"
echo "Limit:           ${LIMIT:-'None (full dataset)'}"
echo "GSM8K Limit:     ${GSM8K_LIMIT:-'None (full dataset)'}"
echo "GSM8K Seed:      ${GSM8K_RANDOM_SEED}"
echo "Num Fewshot:     ${NUM_FEWSHOT:-'Task default'}"
echo "Compiled BDA:    $COMPILED_BDA"
echo "Contiguous QKV:  $FORCE_CONTIGUOUS_QKV"
echo "HF Offline:      $HF_DATASETS_OFFLINE"
echo "Output prefix:   $OUTPUT_PREFIX"
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

run_single_model() {
    local model_name="$1"
    local model_type="$2"
    local checkpoint="$3"
    local output_file="${OUTPUT_PREFIX}_${model_name}.json"

    echo "-------------------------------------------"
    echo "Running model:   $model_name ($model_type)"
    echo "Checkpoint:      $checkpoint"
    echo "Output:          $output_file"
    echo "-------------------------------------------"

    apptainer exec \
        --env "PYTHONUSERBASE=${MYENVS}/luciole_bench" \
        --env "HF_HOME=/work/m24047/m24047brmn/lm_eval" \
        --env "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}" \
        --bind /tmpdir,/work \
        --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
        bash -c "cd /users/m24047/m24047flhg/luciole-eval/luciole-1B/training/train/train_ssa_triton && \
            export PYTHONPATH=/usr/lib/python3.12:/usr/local/lib/python3.12/dist-packages:/usr/local/lib/python3.12/dist-packages/lightning_utilities-0.14.0-py3.12.egg:/opt/venv/lib/python3.12/site-packages:/opt/nemo:/opt/NeMo:/opt/NeMo/examples:/opt/megatron-lm:${MYENVS}/nemo/lib/python3.12/site-packages:\${PYTHONPATH} && \
            python3 run_benchmarks_triton.py \
                --checkpoint '$checkpoint' \
                --model-type '$model_type' \
                --model-name '$model_name' \
                --tokenizer '$TOKENIZER_PATH' \
                --tasks '$TASKS' \
                --batch_size $BATCH_SIZE \
                --max_length $MAX_LENGTH \
                --gsm8k-limit $GSM8K_LIMIT \
                --gsm8k-random-seed $GSM8K_RANDOM_SEED \
                --output '$output_file' \
                $COMPILED_BDA_ARG \
                $CONTIG_QKV_ARG \
                $LIMIT_ARG \
                $FEWSHOT_ARG"

    local model_status=$?
    echo "Model '$model_name' finished with status $model_status"
    return $model_status
}

status=0
case "$MODEL_SELECTION" in
    both)
        run_single_model "$SSA_MODEL_NAME" "ssa_triton" "$SSA_CHECKPOINT" || status=$?
        run_single_model "$SOFTMAX_MODEL_NAME" "softmax" "$SOFTMAX_CHECKPOINT" || status=$?
        ;;
    ssa)
        run_single_model "$SSA_MODEL_NAME" "ssa_triton" "$SSA_CHECKPOINT" || status=$?
        ;;
    softmax)
        run_single_model "$SOFTMAX_MODEL_NAME" "softmax" "$SOFTMAX_CHECKPOINT" || status=$?
        ;;
    *)
        echo "ERROR: Invalid MODEL_SELECTION='$MODEL_SELECTION'. Use one of: both, ssa, softmax."
        status=2
        ;;
esac

echo "==========================================="
echo "Benchmark finished with status $status (selection=$MODEL_SELECTION)"
if ls "${OUTPUT_PREFIX}"_*.json >/dev/null 2>&1; then
    echo "Results saved to:"
    ls -1 "${OUTPUT_PREFIX}"_*.json
fi
echo "==========================================="
exit $status
