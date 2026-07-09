#!/usr/bin/env python3
"""Verify SSA n parameter and AdamW optimizer state across checkpoints.

Use to test the hypothesis that AdamW moment buffers (exp_avg, exp_avg_sq) and
the per-param `step` counter for SSA `n` are not correctly preserved across
checkpoint save/load — which would cause oversized updates on the first step
after a SLURM resume and the observed dips at job boundaries.

Run inside the NeMo training environment so dist_checkpointing imports work.

Usage:
    python verify_n_state.py <ckpt_dir>
    python verify_n_state.py <ckpt_dir_before_dip> --compare <ckpt_dir_after_dip>

What it reports:
    - The `n` parameter values per layer (model state)
    - The AdamW state for each: step, |exp_avg|, |exp_avg_sq|
    - Diff between two checkpoints (if --compare passed)

Interpretation:
    BUG CONFIRMED if `step` is 0 / 1 / much smaller than global_step.
    BUG CONFIRMED if exp_avg_sq is exactly 0 after a known resume.
    BUG ELSEWHERE if model state diff is large but optimizer state looks fine.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def _init_single_rank_dist():
    """Initialize a 1-rank gloo process group so Megatron dist_checkpointing
    can run on a single CPU process without a real distributed launcher."""
    if not dist.is_available():
        return
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _try_dist_checkpointing(path: Path):
    """Try loading via Megatron dist_checkpointing.load_plain_tensors."""
    from megatron.core import dist_checkpointing
    return dist_checkpointing.load_plain_tensors(str(path))


def _try_dcp(path: Path):
    """Try loading via torch.distributed.checkpoint."""
    import tempfile
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    out_pt = Path(tempfile.gettempdir()) / f"{path.parent.name}_{path.name}.inspect.pt"
    if not out_pt.exists():
        dcp_to_torch_save(str(path), str(out_pt))
    return torch.load(out_pt, map_location="cpu", weights_only=False)


def load_checkpoint(ckpt_path: Path) -> dict:
    """Load a NeMo-style checkpoint. Tries `weights/` and `optimizer/` subfolders
    (NeMo 2.x stores the real distributed-checkpoint shards there) and merges them.
    """
    if ckpt_path.is_file():
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if not ckpt_path.is_dir():
        raise FileNotFoundError(ckpt_path)

    # Candidate sub-paths to try as distributed checkpoints
    candidates = []
    for sub in ("weights", "optimizer", "model_weights", "model_optim_rng"):
        if (ckpt_path / sub).is_dir():
            candidates.append(("nemo:" + sub, ckpt_path / sub))
    # Also try the top-level dir itself (some layouts store the dist ckpt there)
    candidates.append(("root", ckpt_path))

    merged: dict = {}
    loaded_any = False
    for label, p in candidates:
        # Try megatron.core.dist_checkpointing first
        try:
            d = _try_dist_checkpointing(p)
            print(f"  [ok]   loaded via megatron dist_checkpointing: {label} ({p.name})")
            merged[label] = d
            loaded_any = True
            continue
        except Exception as e:
            print(f"  [info] megatron dist_checkpointing on {label}: {type(e).__name__}: {e}")

        # Try DCP if .metadata present
        if (p / ".metadata").exists():
            try:
                d = _try_dcp(p)
                print(f"  [ok]   loaded via DCP: {label}")
                merged[label] = d
                loaded_any = True
                continue
            except Exception as e:
                print(f"  [info] DCP on {label}: {type(e).__name__}: {e}")

        # Fallback: load any .pt files we can find in this subdir
        pts = list(p.glob("*.pt"))
        for pt in pts:
            try:
                d = torch.load(pt, map_location="cpu", weights_only=False)
                merged[f"{label}:{pt.name}"] = d
                loaded_any = True
            except Exception as e:
                print(f"  [warn] failed {pt}: {e}")

    if not loaded_any:
        raise RuntimeError(f"Could not load any tensors from {ckpt_path}")
    return merged


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
_N_TOKEN = re.compile(r"(^|\.)n(\.|$|\.exp_avg|\.exp_avg_sq|\.step|\.fp32_param)")


def is_n_key(key: str) -> bool:
    k = key.lower()
    # Match  *.n  /  *.n.exp_avg  /  *.n.exp_avg_sq  /  *.n.step  /  *.n.fp32_param
    # but not unrelated tokens like "norm" or "linear"
    if "ssa_n" in k:
        return True
    if ".self_attention.n" in k or ".attention.n" in k:
        return True
    return bool(_N_TOKEN.search(k))


def walk_tensors(state, prefix=""):
    """Yield (full_key, tensor) for every tensor in a nested dict/list."""
    if isinstance(state, dict):
        for k, v in state.items():
            yield from walk_tensors(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(state, (list, tuple)):
        for i, v in enumerate(state):
            yield from walk_tensors(v, f"{prefix}[{i}]")
    elif isinstance(state, torch.Tensor):
        yield prefix, state


def extract_n_params(state) -> dict:
    """Find all tensors that look like SSA n parameters (model weights)."""
    out = {}
    for k, t in walk_tensors(state):
        if is_n_key(k) and "exp_avg" not in k and "step" not in k:
            out[k] = t
    return out


def extract_n_optim_state(state) -> dict:
    """Find optimizer state (exp_avg, exp_avg_sq, step) associated with n params.
    Returns dict[param_key] = {'step': ..., 'exp_avg': tensor, 'exp_avg_sq': tensor}.
    """
    out = {}
    # Different optimizers/checkpoint formats use different keys. Search broadly.
    for k, t in walk_tensors(state):
        if not is_n_key(k):
            continue
        # Group by approximate parent key
        parent = k.rsplit(".", 1)[0]
        slot = out.setdefault(parent, {})
        kl = k.lower()
        if "exp_avg_sq" in kl or "second_moment" in kl or "v" == k.split(".")[-1]:
            slot["exp_avg_sq"] = t
        elif "exp_avg" in kl or "first_moment" in kl or "m" == k.split(".")[-1]:
            slot["exp_avg"] = t
        elif kl.endswith(".step") or kl.endswith("step_count"):
            slot["step"] = t
        else:
            slot.setdefault("param", t)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(label: str, state: dict):
    print(f"\n=== {label} ===")
    print(f"  Top-level keys: {list(state.keys())[:20] if isinstance(state, dict) else type(state)}")

    params = extract_n_params(state)
    if not params:
        print("  ⚠ No SSA n parameters auto-detected. Dumping likely candidate keys:")
        for k, t in walk_tensors(state):
            if isinstance(t, torch.Tensor) and t.numel() < 64 and t.ndim <= 1:
                print(f"    {k}: shape={tuple(t.shape)} val={t.flatten()[:8].tolist()}")
        return None

    print(f"  Found {len(params)} candidate n parameters:")
    for k in sorted(params.keys()):
        v = params[k]
        flat = v.flatten()[:4].tolist() if v.numel() else []
        print(f"    {k}: shape={tuple(v.shape)} val={flat}")

    optim = extract_n_optim_state(state)
    if optim:
        print(f"\n  Optimizer state for n parameters:")
        for k in sorted(optim.keys()):
            slot = optim[k]
            step = slot.get("step")
            ea = slot.get("exp_avg")
            ev = slot.get("exp_avg_sq")
            print(
                f"    {k}: "
                f"step={step.item() if isinstance(step, torch.Tensor) and step.numel()==1 else step} "
                f"|exp_avg|={ea.abs().mean().item():.3e} " if isinstance(ea, torch.Tensor) else f"    {k}: step={step} exp_avg=NA "
            )
            if isinstance(ev, torch.Tensor):
                print(f"        |exp_avg_sq|={ev.abs().mean().item():.3e}, "
                      f"max={ev.abs().max().item():.3e}, "
                      f"is_zero={(ev == 0).all().item()}")
    else:
        print("  ⚠ Did not find optimizer state slots. Try inspecting raw state structure.")
    return params


def diff_states(p1: dict, p2: dict):
    print("\n=== Diff (after - before) ===")
    keys = sorted(set(p1) | set(p2))
    for k in keys:
        if k not in p1:
            print(f"    {k}: only in AFTER")
            continue
        if k not in p2:
            print(f"    {k}: only in BEFORE")
            continue
        a, b = p1[k], p2[k]
        if a.shape != b.shape:
            print(f"    {k}: shape changed {tuple(a.shape)} -> {tuple(b.shape)}")
            continue
        delta = (b - a).abs()
        print(f"    {k}: |Δ|.mean={delta.mean().item():.4e}  "
              f"|Δ|.max={delta.max().item():.4e}  "
              f"before={a.flatten()[:3].tolist()}  after={b.flatten()[:3].tolist()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path, help="Checkpoint dir or .pt file")
    p.add_argument("--compare", type=Path, default=None,
                   help="Second checkpoint to compare against (e.g. one before vs after a dip)")
    p.add_argument("--dump-keys", action="store_true",
                   help="Just print all top-level keys of the loaded state and exit")
    args = p.parse_args()

    _init_single_rank_dist()

    print(f"Loading {args.ckpt} ...")
    s1 = load_checkpoint(args.ckpt)
    if args.dump_keys:
        for k, t in walk_tensors(s1):
            shape = tuple(t.shape) if isinstance(t, torch.Tensor) else None
            print(f"  {k}: {shape}")
        return

    p1 = summarize(str(args.ckpt), s1)

    if args.compare:
        print(f"\nLoading {args.compare} ...")
        s2 = load_checkpoint(args.compare)
        p2 = summarize(str(args.compare), s2)
        if p1 and p2:
            diff_states(p1, p2)


if __name__ == "__main__":
    main()
