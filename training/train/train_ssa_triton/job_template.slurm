#!/bin/bash
#SBATCH -J job_name
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

# Environment variables
HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
HF_HOME=${HF_HOME:-"/work/m24047/m24047flhg/.cache/huggingface"}
OUTPUT_DIR=${OUTPUT_DIR:-"output"}
PYTHON_ENV_NAME=${PYTHON_ENV_NAME:-"luciole_bench"}
SIF_PATH=${SIF_PATH:-"/work/conteneurs/calmip/nemo_25.04.03_arm.sif"}

mkdir -p "$OUTPUT_DIR"

echo "==========================================="
echo "Job:             $JOB_NAME"
echo "==========================================="
echo "HF Offline:      $HF_DATASETS_OFFLINE"
echo "HF Home:         $HF_HOME"
echo "Output prefix:   $OUTPUT_PREFIX"
echo "SIF Path:        $SIF_PATH"
echo "Python env:      $PYTHON_ENV_NAME"
echo "Output dir:      $OUTPUT_DIR"
echo "Timestamp:       $TIMESTAMP"
echo "==========================================="

COMMAND="python3 test_script.py"


# Run the command in the container
apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/${PYTHON_ENV_NAME}" \
    --env "HF_HOME=${HF_HOME}" \
    --env "HF_DATASETS_CACHE=${HF_HOME}/datasets" \
    --env "HF_HUB_CACHE=${HF_HOME}/hub" \
    --env "HF_DATASETS_OFFLINE=0" \
    --bind /tmpdir,/work \
    --nv ${SIF_PATH} \
    bash -c "cd /users/m24047/m24047flhg/luciole-eval/luciole-1B/training/train/train_ssa_triton && \
        export PYTHONPATH=/usr/lib/python3.12:/usr/local/lib/python3.12/dist-packages:/usr/local/lib/python3.12/dist-packages/lightning_utilities-0.14.0-py3.12.egg:/opt/venv/lib/python3.12/site-packages:/opt/nemo:/opt/NeMo:/opt/NeMo/examples:/opt/megatron-lm:${MYENVS}/nemo/lib/python3.12/site-packages:\${PYTHONPATH} && \
        $COMMAND "

status=$?
echo "Job '$JOB_NAME' finished with status $status"

exit $status
