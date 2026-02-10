#!/bin/bash
#SBATCH -J prep_fwe
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p small
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x_%j.out

# Ensure output directory for logs exists
mkdir -p slurm

# Configuration
TOKENIZER=${TOKENIZER:-"/work/m24047/m24047brmn/tokenizers/minitron-4b"}
INPUT_DIR=${INPUT_DIR:-"/work/shares/IA-Datasets/fineweb_edu_10bt_shuffled"}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"/tmpdir/m24047brmn/nemo_1b/data_fwe/fineweb_edu_text_document"}
MAX_TOKENS=${MAX_TOKENS:-1000000000}  # 1B tokens

echo "=========================================="
echo "Preprocessing FineWeb-Edu to indexed dataset"
echo "Tokenizer:   $TOKENIZER"
echo "Input:       $INPUT_DIR"
echo "Output:      $OUTPUT_PREFIX.{bin,idx}"
echo "Max tokens:  $MAX_TOKENS"
echo "=========================================="

# Create output directory
mkdir -p $(dirname $OUTPUT_PREFIX)

# Run preprocessing
apptainer exec --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind /tmpdir,/work \
    --nv /work/conteneurs/calmip/nemo_25.11_arm.sif \
    bash -lc "cd /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/nemo1b/data_fwe && \
    export PYTHONPATH=/usr/local/lib/python3.12/dist-packages:/opt/venv/lib/python3.12/site-packages:/opt/nemo:/opt/NeMo:/opt/NeMo/examples:/opt/megatron-lm:\${PYTHONPATH} && \
    python3 preprocess_fineweb_edu.py \
        --tokenizer $TOKENIZER \
        --input_dir $INPUT_DIR \
        --output_prefix $OUTPUT_PREFIX \
        --max_tokens $MAX_TOKENS"

status=$?
echo "=========================================="
echo "Preprocessing finished with status $status"
echo "=========================================="
exit $status
