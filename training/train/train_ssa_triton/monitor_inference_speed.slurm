#!/bin/bash
#SBATCH -J monitor_infer_speed
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
mkdir -p slurm benchmark_results

SSA_TRITON_CHECKPOINT=${SSA_TRITON_CHECKPOINT:-/tmpdir/m24047brmn/kairos/bs1024_fullrun/outputs/nemotron-1B-SSA-Triton-bs1024/checkpoints/nemotron-1B-SSA-Triton-bs1024-step=0011999-last}
SSA_CHECKPOINT=${SSA_CHECKPOINT:-$SSA_TRITON_CHECKPOINT}
SOFTMAX_CHECKPOINT=${SOFTMAX_CHECKPOINT:-/tmpdir/m24047flhg/luciole-phase1-step0015000}
TOKENIZER_PATH=${TOKENIZER:-/tmpdir/m24047brmn/kairos/tokenizer_128k-arab-regional_v2}
DEVICE=${DEVICE:-cuda}

SEQ_LENS=${SEQ_LENS:-2048}
BATCH_SIZES=${BATCH_SIZES:-32}
WARMUP_ITERS=${WARMUP_ITERS:-5}
MEASURE_ITERS=${MEASURE_ITERS:-20}
AUTO_FIND_MAX_BATCH=${AUTO_FIND_MAX_BATCH:-0}
START_BATCH=${START_BATCH:-1}
MAX_BATCH_CAP=${MAX_BATCH_CAP:-1024}
FORCE_BF16=${FORCE_BF16:-1}
COMPILED_BDA=${COMPILED_BDA:-0}
FORCE_CONTIGUOUS_QKV=${FORCE_CONTIGUOUS_QKV:-1}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PREFIX=${OUTPUT_PREFIX:-benchmark_results/infer_speed_compare_${TIMESTAMP}}

SSA_TRITON_OUTPUT_JSON="${OUTPUT_PREFIX}_ssa_triton.json"
SSA_OUTPUT_JSON="${OUTPUT_PREFIX}_ssa.json"
SOFTMAX_OUTPUT_JSON="${OUTPUT_PREFIX}_softmax.json"

AUTO_FIND_ARG=""
if [ "$AUTO_FIND_MAX_BATCH" = "1" ]; then
  AUTO_FIND_ARG="--auto-find-max-batch"
fi

FORCE_BF16_ARG=""
if [ "$FORCE_BF16" = "1" ]; then
  FORCE_BF16_ARG="--force-bf16"
fi

COMPILED_BDA_ARG="--no-compiled-bda"
if [ "$COMPILED_BDA" = "1" ]; then
  COMPILED_BDA_ARG="--compiled-bda"
fi

CONTIG_QKV_ARG="--no-force-contiguous-qkv"
if [ "$FORCE_CONTIGUOUS_QKV" = "1" ]; then
  CONTIG_QKV_ARG="--force-contiguous-qkv"
fi

echo "==========================================="
echo "Inference Speed Monitor (3-way comparison)"
echo "==========================================="
echo "SSA Triton checkpoint: $SSA_TRITON_CHECKPOINT"
echo "SSA checkpoint:        $SSA_CHECKPOINT"
echo "Softmax checkpoint:    $SOFTMAX_CHECKPOINT"
echo "Tokenizer:             $TOKENIZER_PATH"
echo "Device:                $DEVICE"
echo "Seq lens:              $SEQ_LENS"
echo "Batch sizes:           $BATCH_SIZES"
echo "Warmup iters:          $WARMUP_ITERS"
echo "Measure iters:         $MEASURE_ITERS"
echo "Auto find max batch:   $AUTO_FIND_MAX_BATCH"
echo "Start batch:           $START_BATCH"
echo "Max batch cap:         $MAX_BATCH_CAP"
echo "Force bf16:            $FORCE_BF16"
echo "Compiled BDA:          $COMPILED_BDA"
echo "Contiguous QKV:        $FORCE_CONTIGUOUS_QKV"
echo "Output prefix:         $OUTPUT_PREFIX"
echo "==========================================="

run_one_model() {
  local model_type="$1"
  local checkpoint="$2"
  local output_json="$3"

  echo "-------------------------------------------"
  echo "Running model_type=$model_type"
  echo "Checkpoint:            $checkpoint"
  echo "Output JSON:           $output_json"
  echo "-------------------------------------------"

  apptainer exec \
  --env "PYTHONUSERBASE=${MYENVS}/luciole_bench" \
  --env "HF_HOME=/work/m24047/m24047brmn/lm_eval" \
  --env "HF_DATASETS_OFFLINE=1" \
  --env "TOKENIZERS_PARALLELISM=false" \
  --bind /tmpdir,/work \
  --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
  bash -c "cd /users/m24047/m24047flhg/luciole-eval/luciole-1B/training/train/train_ssa_triton && \
    export PYTHONPATH=/usr/lib/python3.12:/usr/local/lib/python3.12/dist-packages:/usr/local/lib/python3.12/dist-packages/lightning_utilities-0.14.0-py3.12.egg:/opt/venv/lib/python3.12/site-packages:/opt/nemo:/opt/NeMo:/opt/NeMo/examples:/opt/megatron-lm:${MYENVS}/nemo/lib/python3.12/site-packages:\${PYTHONPATH} && \
    python3 monitor_inference_speed.py \
    --model-type '$model_type' \
    --checkpoint '$checkpoint' \
      --tokenizer '$TOKENIZER_PATH' \
      --device '$DEVICE' \
      --seq-lens '$SEQ_LENS' \
      --batch-sizes '$BATCH_SIZES' \
      --warmup-iters $WARMUP_ITERS \
      --measure-iters $MEASURE_ITERS \
      --start-batch $START_BATCH \
      --max-batch-cap $MAX_BATCH_CAP \
    --output '$output_json' \
      $AUTO_FIND_ARG \
      $FORCE_BF16_ARG \
      $COMPILED_BDA_ARG \
      $CONTIG_QKV_ARG"
}

status=0

run_one_model "ssa_triton" "$SSA_TRITON_CHECKPOINT" "$SSA_TRITON_OUTPUT_JSON" || status=1
# run_one_model "ssa" "$SSA_CHECKPOINT" "$SSA_OUTPUT_JSON" || status=1
run_one_model "softmax" "$SOFTMAX_CHECKPOINT" "$SOFTMAX_OUTPUT_JSON" || status=1

python3 - <<PY
import json
import os

files = {
  "ssa_triton": "${SSA_TRITON_OUTPUT_JSON}",
  # "ssa": "${SSA_OUTPUT_JSON}",
  "softmax": "${SOFTMAX_OUTPUT_JSON}",
}

print("\n===========================================")
print("THROUGHPUT SUMMARY (tokens/s)")
print("===========================================")

for model_name, path in files.items():
  if not os.path.exists(path):
    print(f"{model_name:<12} missing report: {path}")
    continue

  with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)

  measurements = [m for m in payload.get("measurements", []) if "tokens_per_second" in m]
  if not measurements:
    print(f"{model_name:<12} no valid measurements")
    continue

  best = max(measurements, key=lambda m: m.get("tokens_per_second", 0.0))
  print(
    f"{model_name:<12} batch={best.get('batch_size')} seq={best.get('seq_len')} "
    f"toks/s={best.get('tokens_per_second', 0.0):.1f} "
    f"lat={best.get('avg_latency_s', 0.0):.4f}s"
  )

print("===========================================")
PY

echo "Done. Reports:"
echo "  $SSA_TRITON_OUTPUT_JSON"
# echo "  $SSA_OUTPUT_JSON"
echo "  $SOFTMAX_OUTPUT_JSON"
exit $status
