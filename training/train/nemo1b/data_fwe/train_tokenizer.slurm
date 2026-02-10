#!/bin/bash
#SBATCH -J train_tok
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=80
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

# Configuration
INPUT_DIR=${INPUT_DIR:-"/work/shares/IA-Datasets/fineweb_edu_10bt_shuffled"}
OUTPUT_DIR=${OUTPUT_DIR:-"/work/m24047/m24047brmn/tokenizers/luciole_50k"}
VOCAB_SIZE=${VOCAB_SIZE:-50256}
MAX_GB=${MAX_GB:-50}
NUM_THREADS=${NUM_THREADS:-80}

echo "=========================================="
echo "Training SentencePiece Tokenizer"
echo "Input:       $INPUT_DIR"
echo "Output:      $OUTPUT_DIR"
echo "Vocab size:  $VOCAB_SIZE"
echo "Max GB:      $MAX_GB"
echo "Threads:     $NUM_THREADS"
echo "=========================================="

# Run with apptainer (NeMo container has sentencepiece)
srun apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/nemo1b/data_fwe/train_tokenizer.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --vocab_size $VOCAB_SIZE \
        --max_gb $MAX_GB \
        --num_threads $NUM_THREADS \
        --file_pattern "*.chunk.*.jsonl"

status=$?
echo "=========================================="
echo "Tokenizer training finished with status $status"
echo "=========================================="
echo ""
echo "Output files:"
echo "  $OUTPUT_DIR/tokenizer.model"
echo "  $OUTPUT_DIR/tokenizer.vocab"
echo "  $OUTPUT_DIR/tokenizer_config.json"
echo ""
echo "To use with NeMo, set tokenizer path to: $OUTPUT_DIR"
exit $status
