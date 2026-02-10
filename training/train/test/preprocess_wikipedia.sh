#!/bin/bash
#SBATCH -J prep_wiki
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

# Parameters
TOKENIZER=${TOKENIZER:-"/work/m24047/m24047brmn/tokenizers/luciole_50k"}
INPUT_DIR=${INPUT_DIR:-"/work/shares/IA-Datasets/Wikipedia-EN/wikipedia/data/20220301.en"}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"/tmpdir/m24047brmn/nemo_1b/data_wiki/wikipedia_en_text_document"}
MAX_TOKENS=${MAX_TOKENS:-10000000}

echo "=========================================="
echo "Preprocessing Wikipedia-EN"
echo "Tokenizer:  $TOKENIZER"
echo "Input:      $INPUT_DIR"
echo "Output:     $OUTPUT_PREFIX"
echo "Max Tokens: $MAX_TOKENS"
echo "=========================================="

apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python3 /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/test/preprocess_wikipedia.py \
        --tokenizer "$TOKENIZER" \
        --input_dir "$INPUT_DIR" \
        --output_prefix "$OUTPUT_PREFIX" \
        --max_tokens $MAX_TOKENS

status=$?
echo "=========================================="
echo "Preprocessing finished with status $status"
echo "=========================================="
exit $status
