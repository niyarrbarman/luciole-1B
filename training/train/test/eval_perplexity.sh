#!/bin/bash
#SBATCH -J eval_bbyluc_ppl
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out

# Ensure output directory exists
mkdir -p slurm

# Checkpoint path - use latest checkpoint from SSA training
# CHECKPOINT_PATH=${CHECKPOINT:-"/tmpdir/m24047brmn/nemo_1b/output/baby_luciole-ssa-test/checkpoints/baby_luciole-ssa-test-step=0020498-last"}
CHECKPOINT_PATH=${CHECKPOINT:-"/tmpdir/m24047brmn/nemo_1b/output/baby_luciole-softmax-test/checkpoints/baby_luciole-softmax-test-step=0020998-last"}


# Data path - preprocessed FineWeb data
# DATA_PATH=${DATA_PATH:-"/tmpdir/m24047brmn/nemo_1b/data_fwe_50k/fineweb_edu_text_document"}
DATA_PATH=${DATA_PATH:-"/tmpdir/m24047brmn/nemo_1b/data_wiki/wikipedia_en_text_document"}


# Tokenizer
TOKENIZER_PATH=${TOKENIZER:-"/work/m24047/m24047brmn/tokenizers/luciole_50k"}

# Evaluation parameters
NUM_SAMPLES=${NUM_SAMPLES:-1000}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
BATCH_SIZE=${BATCH_SIZE:-8}
SEED=${SEED:-42}

# Output file for results
OUTPUT_FILE=${OUTPUT:-"/tmpdir/m24047brmn/nemo_1b/output/perplexity_results_${SLURM_JOB_ID}.json"}

echo "=========================================="
echo "Evaluating Perplexity: Baby Luciole SSA"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data:       $DATA_PATH"
echo "Tokenizer:  $TOKENIZER_PATH"
echo "Samples:    $NUM_SAMPLES"
echo "Seq Length: $SEQ_LENGTH"
echo "Batch Size: $BATCH_SIZE"
echo "Output:     $OUTPUT_FILE"
echo "=========================================="

# Run evaluation with apptainer
apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python3 /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/test/eval_perplexity.py \
        --checkpoint "$CHECKPOINT_PATH" \
        --tokenizer "$TOKENIZER_PATH" \
        --data_path "$DATA_PATH" \
        --num_samples $NUM_SAMPLES \
        --seq_length $SEQ_LENGTH \
        --batch_size $BATCH_SIZE \
        --no_ssa \
        --seed $SEED \
        --output "$OUTPUT_FILE"

status=$?
echo "=========================================="
echo "Perplexity evaluation finished with status $status"
echo "=========================================="
exit $status
