#!/bin/bash
set -euo pipefail

# ── Torchrun env ───────────────────────────────────────────────────────────────
LOCAL_RANK=${LOCAL_RANK:-0}
LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-1}

# ── GH200 topology ─────────────────────────────────────────────────────────────
TOTAL_CORES=288
CORES_PER_RANK=$((TOTAL_CORES / LOCAL_WORLD_SIZE))

CORE_START=$((LOCAL_RANK * CORES_PER_RANK))
CORE_END=$((CORE_START + CORES_PER_RANK - 1))

# ── Threading (important) ─────────────────────────────────────────────────────
export OMP_NUM_THREADS=$CORES_PER_RANK

# ── Debug info ────────────────────────────────────────────────────────────────
echo "[wrapper] host=$(hostname) rank=${LOCAL_RANK}/${LOCAL_WORLD_SIZE} cores=${CORE_START}-${CORE_END} OMP=${OMP_NUM_THREADS}" >&2

# ── Execute with CPU binding ONLY ─────────────────────────────────────────────
exec numactl \
    --physcpubind=${CORE_START}-${CORE_END} \
    "$@"
