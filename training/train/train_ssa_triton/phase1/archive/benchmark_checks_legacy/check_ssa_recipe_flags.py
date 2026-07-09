#!/usr/bin/env python3
"""Print key SSA/Triton config flags and implied scale math."""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipes.recipe_utils import get_recipe  # noqa: E402


def _parse_layers(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _resolve_head_dim(cfg) -> int | None:
    head_dim = getattr(cfg, "kv_channels", None)
    if head_dim is not None:
        return int(head_dim)
    hidden = getattr(cfg, "hidden_size", None)
    heads = getattr(cfg, "num_attention_heads", None)
    if hidden is None or heads is None:
        return None
    return int(hidden) // int(heads)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SSA recipe flags and scaling math")
    parser.add_argument("--arch", default="baby_luciole", type=str)
    parser.add_argument("--output_dir", default="/tmp/ssa_check", type=str)
    parser.add_argument("--name", default="ssa-check", type=str)
    parser.add_argument("--layers", default="1,2,6,12", type=str)
    args = parser.parse_args()

    recipe = get_recipe(
        arch=args.arch,
        recipe_args=dict(dir=args.output_dir, name=args.name, num_nodes=1, num_gpus_per_node=1),
        performance_mode_if_possible=False,
    )
    cfg = recipe.model.config

    head_dim = _resolve_head_dim(cfg)
    apply_qk_layer_scaling = bool(getattr(cfg, "apply_query_key_layer_scaling", False))
    attn_softmax_fp32 = bool(getattr(cfg, "attention_softmax_in_fp32", False))
    attention_dropout = float(getattr(cfg, "attention_dropout", 0.0))
    num_heads = getattr(cfg, "num_attention_heads", None)
    num_q_groups = getattr(cfg, "num_query_groups", None)
    kv_channels = getattr(cfg, "kv_channels", None)

    print("=== Recipe Flags ===")
    print(f"arch: {args.arch}")
    print(f"num_attention_heads: {num_heads}")
    print(f"num_query_groups: {num_q_groups}")
    print(f"hidden_size: {getattr(cfg, 'hidden_size', None)}")
    print(f"kv_channels (raw): {kv_channels}")
    print(f"resolved_head_dim: {head_dim}")
    print(f"apply_query_key_layer_scaling: {apply_qk_layer_scaling}")
    print(f"attention_softmax_in_fp32: {attn_softmax_fp32}")
    print(f"attention_dropout: {attention_dropout}")
    print(f"masked_softmax_fusion: {getattr(cfg, 'masked_softmax_fusion', None)}")
    print()

    if head_dim is None:
        print("Could not resolve head_dim; skipping scale math.")
        return

    base = 1.0 / math.sqrt(head_dim)
    print("=== Implied Scale Math ===")
    print(f"base_scale (1/sqrt(d)): {base:.10f}")
    print("Assuming current code:")
    print("- Original SSA effective pre-SSA multiplier: (base/layer) * layer = base")
    print("- Triton SSA effective pre-SSA multiplier:    (base/layer)")
    print()
    print("layer | original_effective | triton_effective | triton/original")
    print("----- | ------------------ | ---------------- | ---------------")
    for layer in _parse_layers(args.layers):
        if apply_qk_layer_scaling:
            original_effective = base
            triton_effective = base / layer
        else:
            original_effective = base
            triton_effective = base
        ratio = triton_effective / original_effective if original_effective != 0 else float("nan")
        print(
            f"{layer:>5} | {original_effective:>18.10f} | "
            f"{triton_effective:>16.10f} | {ratio:>15.10f}"
        )


if __name__ == "__main__":
    main()
