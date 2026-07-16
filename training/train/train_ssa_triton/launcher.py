#!/usr/bin/env python3
import os
import sys

# ── Torchrun environment ───────────────────────────────────────────────────────
local_rank = int(os.environ.get("LOCAL_RANK", 0))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

# ── CPU topology (GB200: 144 cores / node) ─────────────────────────────────────
TOTAL_CORES = 144
cores_per_rank = TOTAL_CORES // local_world_size

core_start = local_rank * cores_per_rank
core_end = core_start + cores_per_rank - 1

# ── IMPORTANT: do NOT touch CUDA_VISIBLE_DEVICES ───────────────────────────────
# torchrun already handles GPU assignment correctly
# Instead, explicitly set device inside Python runtime

os.environ["LOCAL_RANK"] = str(local_rank)

# Optional: control threading (prevents oversubscription)
os.environ["OMP_NUM_THREADS"] = str(cores_per_rank)

# ── Build command (SAFE binding) ───────────────────────────────────────────────
cmd = (
    [
        "numactl",
        f"--physcpubind={core_start}-{core_end}",  # CPU pinning ONLY
        # DO NOT use membind / cpunodebind → can break NIC access
        sys.executable,
    ]
    + sys.argv[1:]
)

# ── Logging ────────────────────────────────────────────────────────────────────
print(
    f"[launcher] rank={local_rank}/{local_world_size - 1} | "
    f"cores={core_start}-{core_end} | "
    f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']} | "
    f"script={os.path.basename(sys.argv[1])}",
    flush=True,
)

# ── Exec ───────────────────────────────────────────────────────────────────────
os.execvp(cmd[0], cmd)
