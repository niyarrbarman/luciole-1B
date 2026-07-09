#!/bin/bash
#SBATCH -J verify_n_state
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p full-gpu
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%j.out
#
# Read-only inspection of SSA n parameter + AdamW optimizer state across known
# job-boundary dips. Runs three comparisons:
#   1) CONTROL: normal training pair (no crash) — should look clean.
#   2) DIP 1:   across the layer-21 dip near step 314.5k.
#   3) DIP 2:   across the layer-21 dip near step 327.5k.
#
# Read-only: only torch.load + analysis. No writes to checkpoint dirs.
#
# Run:    sbatch verify_n_state.sh
# Output: slurm/verify_n_state_<jobid>.out

set -euo pipefail
mkdir -p slurm

CKPT_DIR=/scratch/barman/bs1024_fullrun/outputs/nemotron-1B-SSA-Triton-bs1024/checkpoints
PFX=nemotron-1B-SSA-Triton-bs1024-step

MYENVS=${MYENVS:-/work/p26037/barman/envs}
SCRIPT=/work/p26037/barman/luciole-1B/training/train/train_ssa_triton/loss_analysis/verify_n_state.py
SIF=/work/conteneurs/shared/AI/nemo_25.04.03_arm.sif

run_pair() {
  local label=$1
  local s1=$2
  local s2=$3
  local c1="${CKPT_DIR}/${PFX}=${s1}"
  local c2="${CKPT_DIR}/${PFX}=${s2}"

  echo
  echo "=================================================================="
  echo "  ${label}"
  echo "  before: ${c1}"
  echo "  after:  ${c2}"
  echo "=================================================================="

  if [[ ! -d "$c1" ]]; then echo "MISSING: $c1"; return; fi
  if [[ ! -d "$c2" ]]; then echo "MISSING: $c2"; return; fi

  srun apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind /scratch,/tmpdir,/work --nv \
    "${SIF}" \
    python "${SCRIPT}" "${c1}" --compare "${c2}"
}

# 1) Control: same job, ~36 min apart, no crash
run_pair "CONTROL — normal training (no crash)" "0313499" "0313999"

# 2) Dip 1: crash + resume near step 314.5k (Apr 24 08:49 → 11:27, 2h38m gap)
run_pair "DIP 1 — crash near step 314.5k" "0314499" "0314999"

# 3) Dip 2: crash + resume near step 327.5k (Apr 28 08:30 → 12:21, 4h gap)
run_pair "DIP 2 — crash near step 327.5k" "0327499" "0327999"

echo
echo "Done. Look for: step counter not advancing by 500 across pairs,"
echo "or exp_avg_sq much higher in DIP pairs than CONTROL."
