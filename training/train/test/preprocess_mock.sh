#!/bin/bash
#SBATCH -J prep_nemo1l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%j.out

# Ensure output directory for logs exists
mkdir -p slurm

INPUT_TXT=${INPUT_TXT:-"/tmpdir/m24047brmn/nemo_1b/data/train_data.txt"}
TOKENIZER=${TOKENIZER:-"/work/m24047/m24047brmn/tokenizers/minitron-4b"}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"/tmpdir/m24047brmn/nemo_1b/data/train_text_document"}
MAX_TOKENS=${MAX_TOKENS:-1000000}

echo "=========================================="
echo "Preprocessing text to indexed dataset"
echo "Input:      $INPUT_TXT"
echo "Tokenizer:  $TOKENIZER"
echo "Output:     $OUTPUT_PREFIX.{bin,idx}"
echo "Max tokens: $MAX_TOKENS"
echo "=========================================="

apptainer exec --env "PYTHONUSERBASE=${MYENVS}/nemo" --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.11_arm.sif \
bash -lc "cd /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/test && \
    export PYTHONPATH=/usr/local/lib/python3.12/dist-packages:/opt/venv/lib/python3.12/site-packages:/opt/nemo:/opt/NeMo:/opt/NeMo/examples:/opt/megatron-lm:${PYTHONPATH} && \
    python3 preprocess_mock.py \
        --input_txt $INPUT_TXT \
        --tokenizer $TOKENIZER \
        --output_prefix $OUTPUT_PREFIX \
        --max_tokens $MAX_TOKENS"

status=$?
echo "=========================================="
echo "Preprocessing finished with status $status"
echo "=========================================="
exit $status
