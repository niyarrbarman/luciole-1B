#!/bin/bash
#SBATCH -J gen_bbyluc_triton
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=00:45:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

CHECKPOINT_PATH=${CHECKPOINT:-"/tmpdir/m24047brmn/nemo_1b/output/baby_luciole-ssa-triton/checkpoints/baby_luciole-ssa-triton-step=0019999-last"}
TOKENIZER_PATH=${TOKENIZER:-"/work/m24047/m24047brmn/tokenizers/luciole_50k"}

CONTEXT=${CONTEXT:-"<|startoftext|> Continue each prompt clearly and concisely."}

PROMPT_1=${PROMPT_1:-"The water cycle begins when the sun heats the surface of the ocean. As the water warms,"}
PROMPT_2=${PROMPT_2:-"Photosynthesis is the process by which plants convert sunlight into energy. During this process,"}
PROMPT_3=${PROMPT_3:-"The French Revolution started in 1789 when the people of France"}
PROMPT_4=${PROMPT_4:-"There are eight planets in our solar system. The closest planet to the sun is Mercury, and"}
PROMPT_5=${PROMPT_5:-"Addition and subtraction are two basic operations in mathematics. For example, if you have 5 apples and"}

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_K=${TOP_K:-50}
TOP_P=${TOP_P:-0.9}
SEED=${SEED:-1234}

COMPILED_BDA=${COMPILED_BDA:-0}
FORCE_CONTIGUOUS_QKV=${FORCE_CONTIGUOUS_QKV:-1}
GREEDY=${GREEDY:-0}

OUTPUT_JSON=${OUTPUT:-"/tmpdir/m24047brmn/nemo_1b/output/generation_triton_${SLURM_JOB_ID}.json"}

echo "=========================================="
echo "Generating Text: Baby Luciole SSA Triton"
echo "Checkpoint:    $CHECKPOINT_PATH"
echo "Tokenizer:     $TOKENIZER_PATH"
echo "Context:       $CONTEXT"
echo "Max new toks:  $MAX_NEW_TOKENS"
echo "Temperature:   $TEMPERATURE"
echo "Top-k:         $TOP_K"
echo "Top-p:         $TOP_P"
echo "Seed:          $SEED"
echo "Compiled BDA:  $COMPILED_BDA"
echo "Contig QKV:    $FORCE_CONTIGUOUS_QKV"
echo "Greedy:        $GREEDY"
echo "Output JSON:   $OUTPUT_JSON"
echo "Prompt 1:      $PROMPT_1"
echo "Prompt 2:      $PROMPT_2"
echo "Prompt 3:      $PROMPT_3"
echo "Prompt 4:      $PROMPT_4"
echo "Prompt 5:      $PROMPT_5"
echo "=========================================="

EXTRA_ARGS=()

if [[ "${COMPILED_BDA}" == "1" ]]; then
    EXTRA_ARGS+=(--compiled-bda)
fi

if [[ "${FORCE_CONTIGUOUS_QKV}" == "0" ]]; then
    EXTRA_ARGS+=(--no-force-contiguous-qkv)
fi

if [[ "${GREEDY}" == "1" ]]; then
    EXTRA_ARGS+=(--greedy)
fi

apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind /tmpdir,/work --nv /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python3 /work/m24047/m24047brmn/nemo/OpenLLM-BPI-Training/training/train/train_ssa_triton/generate_triton.py \
        --checkpoint "$CHECKPOINT_PATH" \
        --tokenizer "$TOKENIZER_PATH" \
        --context "$CONTEXT" \
        --prompt "$PROMPT_1" \
        --prompt "$PROMPT_2" \
        --prompt "$PROMPT_3" \
        --prompt "$PROMPT_4" \
        --prompt "$PROMPT_5" \
        --num-prompts 5 \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --top_k $TOP_K \
        --top_p $TOP_P \
        --seed $SEED \
        --output "$OUTPUT_JSON" \
        "${EXTRA_ARGS[@]}"

status=$?
echo "=========================================="
echo "Generation finished with status $status"
echo "=========================================="
exit $status
