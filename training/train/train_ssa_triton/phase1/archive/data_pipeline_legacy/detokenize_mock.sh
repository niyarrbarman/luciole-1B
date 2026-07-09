#!/bin/bash
#SBATCH -J detok_nemo
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%j.out

# Ensure output directory for logs exists
mkdir -p slurm

INPUT_PREFIX=${INPUT_PREFIX:-"/tmpdir/m24047brmn/nemo_1b/data_fwe_50k/fineweb_edu_text_document"}
TOKENIZER=${TOKENIZER:-"/work/m24047/m24047brmn/tokenizers/luciole_50k"}
NUM_TOKENS=${NUM_TOKENS:-10000}

echo "=========================================="
echo "Detokenizing indexed dataset"
echo "Input:      $INPUT_PREFIX.{bin,idx}"
echo "Tokenizer:  $TOKENIZER"
echo "Num tokens: $NUM_TOKENS"
echo "=========================================="

apptainer exec --env "PYTHONUSERBASE=${MYENVS}/nemo" --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.11_arm.sif \
bash -lc "cd /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/train_ssa_triton && \
    export PYTHONPATH=/usr/local/lib/python3.12/dist-packages:/opt/venv/lib/python3.12/site-packages:/opt/nemo:/opt/NeMo:/opt/NeMo/examples:/opt/megatron-lm:${PYTHONPATH} && \
    python3 detokenize_mock.py \
        --input_prefix $INPUT_PREFIX \
        --tokenizer $TOKENIZER \
        --num_tokens $NUM_TOKENS"

status=$?
echo "=========================================="
echo "Detokenization finished with status $status"
echo "=========================================="
exit $status
