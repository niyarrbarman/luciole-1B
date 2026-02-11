#!/usr/bin/env python3
"""Compare original SSA attention vs Triton SSA attention on identical inputs."""

import argparse
import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipes.recipe_utils import get_recipe  # noqa: E402
from SSA.ssa_attention import SSADotProductAttention  # noqa: E402
from SSA.ssa_triton_attention import SSATritonAttention  # noqa: E402
from SSA.ssa_flash_attention import USE_OPTIMIZED_KERNEL  # noqa: E402
from megatron.core.transformer.enums import AttnMaskType  # noqa: E402
from load_model import init_single_gpu_parallel_state, cleanup_parallel_state  # noqa: E402


def _parse_layers(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _resolve_head_dim(cfg) -> int:
    head_dim = getattr(cfg, "kv_channels", None)
    if head_dim is not None:
        return int(head_dim)
    hidden = int(getattr(cfg, "hidden_size"))
    heads = int(getattr(cfg, "num_attention_heads"))
    return hidden // heads


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _tensor_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().reshape(-1)
    b_flat = b.float().reshape(-1)
    return float(F.cosine_similarity(a_flat, b_flat, dim=0).item())


def _grad_stats(ga: torch.Tensor, gb: torch.Tensor) -> tuple[float, float]:
    max_abs = float((ga - gb).abs().max().item())
    denom = float(ga.abs().max().item()) + 1e-12
    rel = max_abs / denom
    return max_abs, rel


def _run_one_layer(
    base_cfg,
    layer_number: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    ssa_n: float,
    ssa_b: float,
    compensate_triton_scaling: bool,
):
    cfg_o = copy.deepcopy(base_cfg)
    cfg_t = copy.deepcopy(base_cfg)

    original = SSADotProductAttention(
        config=cfg_o,
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        ssa_n=ssa_n,
        ssa_b=ssa_b,
        learnable_ssa=True,
    ).to(device)
    triton = SSATritonAttention(
        config=cfg_t,
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        ssa_n=ssa_n,
        ssa_b=ssa_b,
        learnable_ssa=True,
        learnable_b=False,
    ).to(device)

    if compensate_triton_scaling and bool(getattr(cfg_t, "apply_query_key_layer_scaling", False)):
        # Emulates the original SSA effective scaling without changing source code.
        triton.softmax_scale *= triton.layer_number

    original.eval()
    triton.eval()

    head_dim = _resolve_head_dim(base_cfg)
    num_heads_q = int(getattr(base_cfg, "num_attention_heads"))
    num_heads_kv = int(getattr(base_cfg, "num_query_groups"))

    q_ref = torch.randn(seq_len, batch_size, num_heads_q, head_dim, device=device, dtype=dtype)
    k_ref = torch.randn(seq_len, batch_size, num_heads_kv, head_dim, device=device, dtype=dtype)
    v_ref = torch.randn(seq_len, batch_size, num_heads_kv, head_dim, device=device, dtype=dtype)

    q_o = q_ref.detach().clone().requires_grad_(True)
    k_o = k_ref.detach().clone().requires_grad_(True)
    v_o = v_ref.detach().clone().requires_grad_(True)
    q_t = q_ref.detach().clone().requires_grad_(True)
    k_t = k_ref.detach().clone().requires_grad_(True)
    v_t = v_ref.detach().clone().requires_grad_(True)

    out_o = original(q_o, k_o, v_o, attention_mask=None)
    out_t = triton(q_t, k_t, v_t, attention_mask=None)

    out_max_abs = float((out_o - out_t).abs().max().item())
    out_mean_abs = float((out_o - out_t).abs().mean().item())
    out_cos = _tensor_cosine(out_o, out_t)

    loss_o = out_o.float().pow(2).mean()
    loss_t = out_t.float().pow(2).mean()

    original.zero_grad(set_to_none=True)
    triton.zero_grad(set_to_none=True)
    loss_o.backward()
    loss_t.backward()

    dq_max_abs, dq_rel = _grad_stats(q_o.grad, q_t.grad)
    dk_max_abs, dk_rel = _grad_stats(k_o.grad, k_t.grad)
    dv_max_abs, dv_rel = _grad_stats(v_o.grad, v_t.grad)

    dn_o = float(original.scale_mask_softmax.ssa_n_raw.grad.item())
    dn_t = float(triton.ssa_n_raw.grad.item())
    dn_abs = abs(dn_o - dn_t)
    dn_rel = dn_abs / (abs(dn_o) + 1e-12)

    coeff = original.scale_mask_softmax.scale
    coeff_val = float(coeff) if coeff is not None else None
    original_effective = float(original.softmax_scale * (coeff_val if coeff_val is not None else 1.0))
    triton_effective = float(triton.softmax_scale)

    return {
        "layer": layer_number,
        "original_softmax_scale": float(original.softmax_scale),
        "original_scale_coeff": coeff_val,
        "original_effective_pre_ssa_scale": original_effective,
        "triton_effective_pre_ssa_scale": triton_effective,
        "forward": {
            "max_abs": out_max_abs,
            "mean_abs": out_mean_abs,
            "cosine": out_cos,
        },
        "backward": {
            "dq_max_abs": dq_max_abs,
            "dq_rel": dq_rel,
            "dk_max_abs": dk_max_abs,
            "dk_rel": dk_rel,
            "dv_max_abs": dv_max_abs,
            "dv_rel": dv_rel,
            "dn_original": dn_o,
            "dn_triton": dn_t,
            "dn_abs_diff": dn_abs,
            "dn_rel_diff": dn_rel,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original SSA attention vs Triton SSA attention")
    parser.add_argument("--arch", default="baby_luciole", type=str)
    parser.add_argument("--output_dir", default="/tmp/ssa_check", type=str)
    parser.add_argument("--name", default="ssa-parity-check", type=str)
    parser.add_argument("--layers", default="12", type=str, help="Comma-separated layer numbers")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--seq_length", default=128, type=int)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--ssa_n", default=1.5, type=float)
    parser.add_argument("--ssa_b", default=0.8, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--compensate_triton_scaling", action="store_true", default=False)
    parser.add_argument("--force_apply_query_key_layer_scaling", action="store_true", default=False)
    parser.add_argument("--force_disable_query_key_layer_scaling", action="store_true", default=False)
    args = parser.parse_args()

    if args.force_apply_query_key_layer_scaling and args.force_disable_query_key_layer_scaling:
        raise ValueError("Use at most one of force_apply/force_disable query key layer scaling flags.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton attention checks.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    init_single_gpu_parallel_state(seed=args.seed, device="cuda")

    try:
        recipe = get_recipe(
            arch=args.arch,
            recipe_args=dict(dir=args.output_dir, name=args.name, num_nodes=1, num_gpus_per_node=1),
            performance_mode_if_possible=False,
        )
        cfg = recipe.model.config
        cfg.attention_dropout = 0.0
        cfg.masked_softmax_fusion = False
        if args.force_apply_query_key_layer_scaling:
            cfg.apply_query_key_layer_scaling = True
        if args.force_disable_query_key_layer_scaling:
            cfg.apply_query_key_layer_scaling = False

        # Keep config explicitly coherent for these checks.
        cfg.fp16 = args.dtype == "fp16"
        cfg.bf16 = args.dtype == "bf16"
        head_dim = _resolve_head_dim(cfg)
        cfg.kv_channels = head_dim

        device = torch.device("cuda")
        dtype = _dtype_from_name(args.dtype)
        layers = _parse_layers(args.layers)

        print("=== Run Setup ===")
        print(f"arch: {args.arch}")
        print(f"layers: {layers}")
        print(f"batch_size: {args.batch_size}")
        print(f"seq_length: {args.seq_length}")
        print(f"dtype: {args.dtype}")
        print(f"num_attention_heads: {cfg.num_attention_heads}")
        print(f"num_query_groups: {cfg.num_query_groups}")
        print(f"head_dim: {head_dim}")
        print(f"apply_query_key_layer_scaling: {cfg.apply_query_key_layer_scaling}")
        print(f"compensate_triton_scaling: {args.compensate_triton_scaling}")
        print(f"ssa_use_optimized_kernel: {USE_OPTIMIZED_KERNEL}")
        print()

        results = []
        for layer in layers:
            result = _run_one_layer(
                base_cfg=cfg,
                layer_number=layer,
                batch_size=args.batch_size,
                seq_len=args.seq_length,
                dtype=dtype,
                device=device,
                ssa_n=args.ssa_n,
                ssa_b=args.ssa_b,
                compensate_triton_scaling=args.compensate_triton_scaling,
            )
            results.append(result)

            print(f"=== Layer {layer} ===")
            print(
                "scale_effective: "
                f"original={result['original_effective_pre_ssa_scale']:.10f}, "
                f"triton={result['triton_effective_pre_ssa_scale']:.10f}"
            )
            print(
                "forward: "
                f"max_abs={result['forward']['max_abs']:.6e}, "
                f"mean_abs={result['forward']['mean_abs']:.6e}, "
                f"cosine={result['forward']['cosine']:.8f}"
            )
            print(
                "backward dQ/dK/dV max_abs: "
                f"{result['backward']['dq_max_abs']:.6e}, "
                f"{result['backward']['dk_max_abs']:.6e}, "
                f"{result['backward']['dv_max_abs']:.6e}"
            )
            print(
                "backward dn: "
                f"orig={result['backward']['dn_original']:.6e}, "
                f"triton={result['backward']['dn_triton']:.6e}, "
                f"abs_diff={result['backward']['dn_abs_diff']:.6e}, "
                f"rel_diff={result['backward']['dn_rel_diff']:.6e}"
            )
            print()

        # Print compact summary at end for copy/paste into notes.
        print("=== Compact Summary ===")
        for r in results:
            print(
                f"layer={r['layer']} "
                f"scale_o={r['original_effective_pre_ssa_scale']:.8e} "
                f"scale_t={r['triton_effective_pre_ssa_scale']:.8e} "
                f"fwd_max={r['forward']['max_abs']:.3e} "
                f"dq_max={r['backward']['dq_max_abs']:.3e} "
                f"dn_rel={r['backward']['dn_rel_diff']:.3e}"
            )
    finally:
        cleanup_parallel_state()


if __name__ == "__main__":
    main()
