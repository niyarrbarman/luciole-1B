#!/usr/bin/env python3
"""Comprehensive parity checks for SSA Triton kernels and integration paths."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_model import cleanup_parallel_state, init_single_gpu_parallel_state  # noqa: E402
from megatron.core.transformer.enums import AttnMaskType  # noqa: E402
from recipes.recipe_utils import get_recipe  # noqa: E402
from SSA.archive.legacy_pre_triton.ssa_flash_attention import USE_OPTIMIZED_KERNEL, ssa_flash_attention as ssa_flash_attention_v3  # noqa: E402
from SSA.ssa_flash_attention import ssa_flash_attention  # noqa: E402
from SSA.archive.legacy_pre_triton.ssa_triton_attention import SSATritonAttention as SSATritonV3Attention  # noqa: E402
from SSA.ssa_triton_attention import SSATritonAttention  # noqa: E402
from SSA.ssa_triton_kernel import (  # noqa: E402
    ssa_flash_attn_backward,
    ssa_flash_attn_forward,
)

if USE_OPTIMIZED_KERNEL:
    try:
        from SSA.archive.legacy_pre_triton.ssa_triton_kernel_optimized import (  # noqa: E402
            ssa_flash_attn_backward as ssa_flash_attn_v3_backward,
        )
        from SSA.archive.legacy_pre_triton.ssa_triton_kernel_optimized import (  # noqa: E402
            ssa_flash_attn_forward as ssa_flash_attn_v3_forward,
        )
        V3_KERNEL_IMPL = "optimized"
    except ImportError:
        from SSA.archive.legacy_pre_triton.ssa_triton_kernel import (  # noqa: E402
            ssa_flash_attn_backward as ssa_flash_attn_v3_backward,
        )
        from SSA.archive.legacy_pre_triton.ssa_triton_kernel import (  # noqa: E402
            ssa_flash_attn_forward as ssa_flash_attn_v3_forward,
        )
        V3_KERNEL_IMPL = "reference-fallback"
else:
    from SSA.archive.legacy_pre_triton.ssa_triton_kernel import (  # noqa: E402
        ssa_flash_attn_backward as ssa_flash_attn_v3_backward,
    )
    from SSA.archive.legacy_pre_triton.ssa_triton_kernel import (  # noqa: E402
        ssa_flash_attn_forward as ssa_flash_attn_v3_forward,
    )
    V3_KERNEL_IMPL = "reference"


@dataclass
class PairTolerances:
    atol: float
    rtol: float
    scalar_atol: float
    scalar_rtol: float


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _resolve_head_dim(cfg: Any) -> int:
    head_dim = getattr(cfg, "kv_channels", None)
    if head_dim is not None:
        return int(head_dim)
    hidden = int(getattr(cfg, "hidden_size"))
    heads = int(getattr(cfg, "num_attention_heads"))
    return hidden // heads


def _parse_int_csv(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_head_pairs(s: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for raw in _parse_str_csv(s):
        if "x" not in raw:
            raise ValueError(f"Invalid head pair '{raw}', expected HQxHKV")
        hq_s, hkv_s = raw.lower().split("x", 1)
        hq, hkv = int(hq_s), int(hkv_s)
        if hq <= 0 or hkv <= 0 or hq % hkv != 0:
            raise ValueError(f"Invalid head pair '{raw}', require HQ>0, HKV>0, HQ%HKV==0")
        out.append((hq, hkv))
    return out


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _tensor_stats(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> dict[str, Any]:
    a32 = a.float()
    b32 = b.float()
    d = a32 - b32
    max_abs = float(d.abs().max().item())
    mean_abs = float(d.abs().mean().item())
    rmse = float(torch.sqrt((d * d).mean()).item())
    ref_rmse = float(torch.sqrt((b32 * b32).mean()).item())
    rel_rmse = rmse / (ref_rmse + 1e-12)
    a_flat = a32.reshape(-1)
    b_flat = b32.reshape(-1)
    a_norm = float(torch.linalg.vector_norm(a_flat).item())
    b_norm = float(torch.linalg.vector_norm(b_flat).item())
    if a_norm == 0.0 and b_norm == 0.0:
        cosine = 1.0
    else:
        cosine = float(F.cosine_similarity(a_flat, b_flat, dim=0).item())
    allclose = bool(torch.allclose(a32, b32, atol=atol, rtol=rtol))
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "cosine": cosine,
        "allclose": allclose,
    }


def _scalar_stats(a: float, b: float, atol: float, rtol: float) -> dict[str, Any]:
    diff = abs(a - b)
    limit = atol + rtol * abs(b)
    return {
        "a": float(a),
        "b": float(b),
        "abs_diff": float(diff),
        "limit": float(limit),
        "allclose": bool(diff <= limit),
    }


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite values detected in {name}")


def _ssa_reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    ssa_n: torch.Tensor,
    ssa_b: torch.Tensor,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, hq, n_ctx, _ = q.shape
    hkv = k.shape[1]
    if hq % hkv != 0:
        raise ValueError(f"HQ={hq} must be divisible by HKV={hkv}")

    gqa_ratio = hq // hkv
    k_exp = k.repeat_interleave(gqa_ratio, dim=1)
    v_exp = v.repeat_interleave(gqa_ratio, dim=1)

    s = torch.matmul(q.float(), k_exp.float().transpose(-1, -2)) * float(softmax_scale)
    if causal:
        causal_mask = torch.triu(
            torch.ones((n_ctx, n_ctx), device=s.device, dtype=torch.bool),
            diagonal=1,
        )
        s = s.masked_fill(causal_mask.view(1, 1, n_ctx, n_ctx), float("-inf"))

    valid = torch.isfinite(s)
    s_safe = torch.where(valid, s, torch.zeros_like(s))
    abs_s = s_safe.abs()
    sign_s = torch.sign(s_safe)
    n_val = ssa_n.float()
    b_val = ssa_b.float()
    log_term = torch.log1p(b_val * abs_s)
    w = torch.where(valid, torch.exp(n_val * sign_s * log_term), torch.zeros_like(s))
    l = w.sum(dim=-1)
    p = w / l.clamp_min(1e-12).unsqueeze(-1)
    out = torch.matmul(p, v_exp.float())
    return out.to(q.dtype), l.float()


def _run_reference_grads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    ssa_n: torch.Tensor,
    ssa_b: torch.Tensor,
    dout: torch.Tensor,
    causal: bool,
) -> dict[str, Any]:
    q_r = q.detach().clone().requires_grad_(True)
    k_r = k.detach().clone().requires_grad_(True)
    v_r = v.detach().clone().requires_grad_(True)
    n_r = ssa_n.detach().clone().requires_grad_(True)
    b_r = ssa_b.detach().clone().requires_grad_(True)

    out_r, l_r = _ssa_reference_forward(q_r, k_r, v_r, softmax_scale, n_r, b_r, causal)
    loss_r = torch.sum(out_r.float() * dout.float())
    loss_r.backward()

    return {
        "out": out_r.detach(),
        "l": l_r.detach(),
        "dq": q_r.grad.detach(),
        "dk": k_r.grad.detach(),
        "dv": v_r.grad.detach(),
        "dn": float(n_r.grad.detach().item()),
        "db": float(b_r.grad.detach().item()),
    }


def _tol_for_dtype(dtype_name: str, args: argparse.Namespace) -> PairTolerances:
    if dtype_name == "bf16":
        return PairTolerances(args.atol_bf16, args.rtol_bf16, args.scalar_atol, args.scalar_rtol)
    if dtype_name == "fp16":
        return PairTolerances(args.atol_fp16, args.rtol_fp16, args.scalar_atol, args.scalar_rtol)
    return PairTolerances(args.atol_fp32, args.rtol_fp32, args.scalar_atol_fp32, args.scalar_rtol_fp32)


def _make_qkv(
    bsz: int,
    hq: int,
    hkv: int,
    n_ctx: int,
    d_head: int,
    dtype: torch.dtype,
    device: torch.device,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if layout == "contig":
        q = torch.randn(bsz, hq, n_ctx, d_head, device=device, dtype=dtype)
        k = torch.randn(bsz, hkv, n_ctx, d_head, device=device, dtype=dtype)
        v = torch.randn(bsz, hkv, n_ctx, d_head, device=device, dtype=dtype)
    elif layout == "strided":
        q = torch.randn(n_ctx, bsz, hq, d_head, device=device, dtype=dtype).permute(1, 2, 0, 3)
        k = torch.randn(n_ctx, bsz, hkv, d_head, device=device, dtype=dtype).permute(1, 2, 0, 3)
        v = torch.randn(n_ctx, bsz, hkv, d_head, device=device, dtype=dtype).permute(1, 2, 0, 3)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    return q, k, v


def _compare_kernel_outputs(
    lhs: dict[str, Any],
    rhs: dict[str, Any],
    tol: PairTolerances,
    include_l: bool = False,
) -> dict[str, Any]:
    stats = {
        "out": _tensor_stats(lhs["out"], rhs["out"], tol.atol, tol.rtol),
        "dq": _tensor_stats(lhs["dq"], rhs["dq"], tol.atol, tol.rtol),
        "dk": _tensor_stats(lhs["dk"], rhs["dk"], tol.atol, tol.rtol),
        "dv": _tensor_stats(lhs["dv"], rhs["dv"], tol.atol, tol.rtol),
        "dn": _scalar_stats(float(lhs["dn"]), float(rhs["dn"]), tol.scalar_atol, tol.scalar_rtol),
        "db": _scalar_stats(float(lhs["db"]), float(rhs["db"]), tol.scalar_atol, tol.scalar_rtol),
    }
    if include_l:
        stats["l"] = _tensor_stats(lhs["l"], rhs["l"], tol.atol, tol.rtol)
        check_keys = ("out", "l", "dq", "dk", "dv", "dn", "db")
    else:
        check_keys = ("out", "dq", "dk", "dv", "dn", "db")
    passed = all(stats[name]["allclose"] for name in check_keys)
    stats["passed"] = passed
    return stats


def _run_kernel_case(
    *,
    case_id: int,
    total_cases: int,
    dtype_name: str,
    bsz: int,
    hq: int,
    hkv: int,
    n_ctx: int,
    d_head: int,
    causal: bool,
    layout: str,
    ssa_n_val: float,
    ssa_b_val: float,
    compare_ref: bool,
    tol: PairTolerances,
    device: torch.device,
) -> dict[str, Any]:
    print(
        f"[Kernel {case_id}/{total_cases}] dtype={dtype_name} "
        f"B={bsz} Hq={hq} Hkv={hkv} N={n_ctx} D={d_head} causal={int(causal)} layout={layout}"
    )
    dtype = _dtype_from_name(dtype_name)
    q, k, v = _make_qkv(bsz, hq, hkv, n_ctx, d_head, dtype, device, layout)
    scale = 1.0 / math.sqrt(float(d_head))
    ssa_n = torch.tensor(ssa_n_val, device=device, dtype=torch.float32)
    ssa_b = torch.tensor(ssa_b_val, device=device, dtype=torch.float32)

    out4, l4 = ssa_flash_attn_forward(q, k, v, scale, ssa_n, ssa_b, causal=causal)
    dout = torch.randn_like(out4)
    dq4, dk4, dv4, dn4, db4 = ssa_flash_attn_backward(
        q, k, v, out4, dout, l4, scale, ssa_n, ssa_b, causal=causal
    )
    _check_finite("triton.out", out4)
    _check_finite("triton.l", l4)
    _check_finite("triton.dq", dq4)
    _check_finite("triton.dk", dk4)
    _check_finite("triton.dv", dv4)

    out3, l3 = ssa_flash_attn_v3_forward(q, k, v, scale, ssa_n, ssa_b, causal=causal)
    dq3, dk3, dv3, dn3, db3 = ssa_flash_attn_v3_backward(
        q, k, v, out3, dout, l3, scale, ssa_n, ssa_b, causal=causal
    )
    _check_finite("v3.out", out3)
    _check_finite("v3.l", l3)
    _check_finite("v3.dq", dq3)
    _check_finite("v3.dk", dk3)
    _check_finite("v3.dv", dv3)

    triton_bundle = {
        "out": out4.detach(),
        "l": l4.detach(),
        "dq": dq4.detach(),
        "dk": dk4.detach(),
        "dv": dv4.detach(),
        "dn": float(dn4.detach().item() if torch.is_tensor(dn4) else dn4),
        "db": float(db4.detach().item() if torch.is_tensor(db4) else db4),
    }
    v3_bundle = {
        "out": out3.detach(),
        "l": l3.detach(),
        "dq": dq3.detach(),
        "dk": dk3.detach(),
        "dv": dv3.detach(),
        "dn": float(dn3.detach().item() if torch.is_tensor(dn3) else dn3),
        "db": float(db3.detach().item() if torch.is_tensor(db3) else db3),
    }

    triton_vs_v3 = _compare_kernel_outputs(triton_bundle, v3_bundle, tol, include_l=False)
    result: dict[str, Any] = {
        "case": {
            "dtype": dtype_name,
            "B": bsz,
            "Hq": hq,
            "Hkv": hkv,
            "N": n_ctx,
            "D": d_head,
            "causal": causal,
            "layout": layout,
        },
        "triton_vs_v3": triton_vs_v3,
        "passed": bool(triton_vs_v3["passed"]),
        "reference_enabled": compare_ref,
    }

    if compare_ref:
        ref_bundle = _run_reference_grads(q, k, v, scale, ssa_n, ssa_b, dout, causal)
        triton_vs_ref = _compare_kernel_outputs(triton_bundle, ref_bundle, tol, include_l=False)
        v3_vs_ref = _compare_kernel_outputs(v3_bundle, ref_bundle, tol, include_l=False)
        result["triton_vs_ref"] = triton_vs_ref
        result["v3_vs_ref"] = v3_vs_ref
        result["passed"] = bool(result["passed"] and triton_vs_ref["passed"])

    return result


def _run_kernel_suite(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    dtypes = _parse_str_csv(args.dtypes)
    batch_sizes = _parse_int_csv(args.kernel_batch_sizes)
    seq_lengths = _parse_int_csv(args.kernel_seq_lengths)
    head_dims = _parse_int_csv(args.kernel_head_dims)
    head_pairs = _parse_head_pairs(args.kernel_head_pairs)
    causal_modes = [bool(int(x)) for x in _parse_str_csv(args.kernel_causal_modes)]
    layouts = _parse_str_csv(args.kernel_layouts)

    cases: list[tuple[str, int, int, int, int, int, bool, str]] = []
    for dtype_name in dtypes:
        for bsz in batch_sizes:
            for (hq, hkv) in head_pairs:
                for n_ctx in seq_lengths:
                    for d_head in head_dims:
                        for causal in causal_modes:
                            for layout in layouts:
                                cases.append((dtype_name, bsz, hq, hkv, n_ctx, d_head, causal, layout))

    results: list[dict[str, Any]] = []
    total_cases = len(cases)
    for idx, case in enumerate(cases, start=1):
        dtype_name, bsz, hq, hkv, n_ctx, d_head, causal, layout = case
        tol = _tol_for_dtype(dtype_name, args)
        if not _is_power_of_2(d_head):
            res = {
                "case": {
                    "dtype": dtype_name,
                    "B": bsz,
                    "Hq": hq,
                    "Hkv": hkv,
                    "N": n_ctx,
                    "D": d_head,
                    "causal": causal,
                    "layout": layout,
                },
                "passed": False,
                "error": "Head dimension must be power-of-two for current triton kernel.",
            }
            results.append(res)
            print(f"[Kernel {idx}/{total_cases}] SKIP-FAIL D={d_head} is not power-of-two")
            if args.fail_fast:
                break
            continue

        if n_ctx % 128 != 0:
            res = {
                "case": {
                    "dtype": dtype_name,
                    "B": bsz,
                    "Hq": hq,
                    "Hkv": hkv,
                    "N": n_ctx,
                    "D": d_head,
                    "causal": causal,
                    "layout": layout,
                },
                "passed": False,
                "error": "N must be divisible by 128 for current triton backward kernel.",
            }
            results.append(res)
            print(f"[Kernel {idx}/{total_cases}] SKIP-FAIL N={n_ctx} not divisible by 128")
            if args.fail_fast:
                break
            continue

        compare_ref = bool(args.kernel_use_reference) and (n_ctx <= args.reference_max_seq)
        try:
            res = _run_kernel_case(
                case_id=idx,
                total_cases=total_cases,
                dtype_name=dtype_name,
                bsz=bsz,
                hq=hq,
                hkv=hkv,
                n_ctx=n_ctx,
                d_head=d_head,
                causal=causal,
                layout=layout,
                ssa_n_val=args.ssa_n,
                ssa_b_val=args.ssa_b,
                compare_ref=compare_ref,
                tol=tol,
                device=device,
            )
            results.append(res)
            if not res["passed"] and args.fail_fast:
                break
        except Exception as exc:
            results.append(
                {
                    "case": {
                        "dtype": dtype_name,
                        "B": bsz,
                        "Hq": hq,
                        "Hkv": hkv,
                        "N": n_ctx,
                        "D": d_head,
                        "causal": causal,
                        "layout": layout,
                    },
                    "passed": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[Kernel {idx}/{total_cases}] ERROR: {exc}")
            if args.fail_fast:
                break
        finally:
            torch.cuda.empty_cache()

    passed = all(r.get("passed", False) for r in results)
    return {"name": "kernel_suite", "passed": passed, "results": results}


def _run_determinism_case(
    dtype_name: str,
    bsz: int,
    hq: int,
    hkv: int,
    n_ctx: int,
    d_head: int,
    causal: bool,
    layout: str,
    ssa_n: float,
    ssa_b: float,
    det_atol: float,
    device: torch.device,
) -> dict[str, Any]:
    dtype = _dtype_from_name(dtype_name)
    q, k, v = _make_qkv(bsz, hq, hkv, n_ctx, d_head, dtype, device, layout)
    scale = 1.0 / math.sqrt(float(d_head))
    n = torch.tensor(ssa_n, device=device, dtype=torch.float32)
    b = torch.tensor(ssa_b, device=device, dtype=torch.float32)

    out1, l1 = ssa_flash_attn_forward(q, k, v, scale, n, b, causal=causal)
    dout = torch.randn_like(out1)
    dq1, dk1, dv1, dn1, db1 = ssa_flash_attn_backward(
        q, k, v, out1, dout, l1, scale, n, b, causal=causal
    )

    out2, l2 = ssa_flash_attn_forward(q, k, v, scale, n, b, causal=causal)
    dq2, dk2, dv2, dn2, db2 = ssa_flash_attn_backward(
        q, k, v, out2, dout, l2, scale, n, b, causal=causal
    )

    comps = {
        "out": float((out1.float() - out2.float()).abs().max().item()),
        "l": float((l1.float() - l2.float()).abs().max().item()),
        "dq": float((dq1.float() - dq2.float()).abs().max().item()),
        "dk": float((dk1.float() - dk2.float()).abs().max().item()),
        "dv": float((dv1.float() - dv2.float()).abs().max().item()),
        "dn": float(abs(float(dn1) - float(dn2))),
        "db": float(abs(float(db1) - float(db2))),
    }
    passed = all(v <= det_atol for v in comps.values())
    return {
        "case": {
            "dtype": dtype_name,
            "B": bsz,
            "Hq": hq,
            "Hkv": hkv,
            "N": n_ctx,
            "D": d_head,
            "causal": causal,
            "layout": layout,
        },
        "max_abs_diffs": comps,
        "determinism_atol": det_atol,
        "passed": passed,
    }


def _run_determinism_suite(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    dtypes = _parse_str_csv(args.dtypes)
    head_pairs = _parse_head_pairs(args.kernel_head_pairs)
    seq_lengths = _parse_int_csv(args.determinism_seq_lengths)
    layouts = _parse_str_csv(args.determinism_layouts)
    causal_modes = [bool(int(x)) for x in _parse_str_csv(args.kernel_causal_modes)]
    d_head = _parse_int_csv(args.kernel_head_dims)[0]
    bsz = _parse_int_csv(args.kernel_batch_sizes)[0]

    cases: list[dict[str, Any]] = []
    for dtype_name in dtypes:
        for n_ctx in seq_lengths:
            for (hq, hkv) in head_pairs[:1]:
                for causal in causal_modes:
                    for layout in layouts:
                        cases.append(
                            {
                                "dtype_name": dtype_name,
                                "bsz": bsz,
                                "hq": hq,
                                "hkv": hkv,
                                "n_ctx": n_ctx,
                                "d_head": d_head,
                                "causal": causal,
                                "layout": layout,
                            }
                        )

    out: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[Determinism {idx}/{len(cases)}] {case}")
        if case["n_ctx"] % 128 != 0:
            out.append({"case": case, "passed": False, "error": "N must be divisible by 128 for triton backward."})
            if args.fail_fast:
                break
            continue
        try:
            res = _run_determinism_case(
                dtype_name=case["dtype_name"],
                bsz=case["bsz"],
                hq=case["hq"],
                hkv=case["hkv"],
                n_ctx=case["n_ctx"],
                d_head=case["d_head"],
                causal=case["causal"],
                layout=case["layout"],
                ssa_n=args.ssa_n,
                ssa_b=args.ssa_b,
                det_atol=args.determinism_atol,
                device=device,
            )
            out.append(res)
            if not res["passed"] and args.fail_fast:
                break
        except Exception as exc:
            out.append({"case": case, "passed": False, "error": str(exc), "traceback": traceback.format_exc()})
            if args.fail_fast:
                break

    return {"name": "determinism_suite", "passed": all(r.get("passed", False) for r in out), "results": out}


def _module_pair_stats(
    lhs: dict[str, torch.Tensor],
    rhs: dict[str, torch.Tensor],
    tol: PairTolerances,
) -> dict[str, Any]:
    stats = {
        "out": _tensor_stats(lhs["out"], rhs["out"], tol.atol, tol.rtol),
        "dq": _tensor_stats(lhs["dq"], rhs["dq"], tol.atol, tol.rtol),
        "dk": _tensor_stats(lhs["dk"], rhs["dk"], tol.atol, tol.rtol),
        "dv": _tensor_stats(lhs["dv"], rhs["dv"], tol.atol, tol.rtol),
        "dn": _scalar_stats(float(lhs["dn"]), float(rhs["dn"]), tol.scalar_atol, tol.scalar_rtol),
    }
    if "db" in lhs and "db" in rhs:
        stats["db"] = _scalar_stats(float(lhs["db"]), float(rhs["db"]), tol.scalar_atol, tol.scalar_rtol)
        passed = all(stats[n]["allclose"] for n in ("out", "dq", "dk", "dv", "dn", "db"))
    else:
        passed = all(stats[n]["allclose"] for n in ("out", "dq", "dk", "dv", "dn"))
    stats["passed"] = passed
    return stats


def _run_module_suite(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    dtype_name = args.module_dtype
    dtype = _dtype_from_name(dtype_name)
    tol = _tol_for_dtype(dtype_name, args)
    layers = _parse_int_csv(args.module_layers)
    if args.module_seq_length % 128 != 0:
        raise ValueError("module_seq_length must be divisible by 128 for triton backward kernel.")

    init_single_gpu_parallel_state(seed=args.seed, device="cuda")
    out_results: list[dict[str, Any]] = []
    try:
        recipe = get_recipe(
            arch=args.arch,
            recipe_args=dict(dir=args.output_dir, name=args.name, num_nodes=1, num_gpus_per_node=1),
            performance_mode_if_possible=False,
        )
        cfg = copy.deepcopy(recipe.model.config)
        cfg.attention_dropout = 0.0
        cfg.masked_softmax_fusion = False
        cfg.sequence_parallel = False
        cfg.fp16 = dtype == torch.float16
        cfg.bf16 = dtype == torch.bfloat16
        head_dim = _resolve_head_dim(cfg)
        cfg.kv_channels = head_dim
        hq = int(cfg.num_attention_heads)
        hkv = int(cfg.num_query_groups)

        for idx, layer in enumerate(layers, start=1):
            print(f"[Module {idx}/{len(layers)}] layer={layer} dtype={dtype_name}")
            cfg_v3 = copy.deepcopy(cfg)
            cfg_triton = copy.deepcopy(cfg)
            m3 = SSATritonV3Attention(
                config=cfg_v3,
                layer_number=layer,
                attn_mask_type=AttnMaskType.causal,
                ssa_n=args.ssa_n,
                ssa_b=args.ssa_b,
                learnable_ssa=True,
                learnable_b=bool(args.module_learnable_b),
            ).to(device)
            m4 = SSATritonAttention(
                config=cfg_triton,
                layer_number=layer,
                attn_mask_type=AttnMaskType.causal,
                ssa_n=args.ssa_n,
                ssa_b=args.ssa_b,
                learnable_ssa=True,
                learnable_b=bool(args.module_learnable_b),
            ).to(device)
            m3.eval()
            m4.eval()

            q_ref = torch.randn(args.module_seq_length, args.module_batch_size, hq, head_dim, device=device, dtype=dtype)
            k_ref = torch.randn(args.module_seq_length, args.module_batch_size, hkv, head_dim, device=device, dtype=dtype)
            v_ref = torch.randn(args.module_seq_length, args.module_batch_size, hkv, head_dim, device=device, dtype=dtype)
            grad_out = torch.randn(
                args.module_seq_length,
                args.module_batch_size,
                hq * head_dim,
                device=device,
                dtype=dtype,
            )

            q3 = q_ref.detach().clone().requires_grad_(True)
            k3 = k_ref.detach().clone().requires_grad_(True)
            v3 = v_ref.detach().clone().requires_grad_(True)
            q4 = q_ref.detach().clone().requires_grad_(True)
            k4 = k_ref.detach().clone().requires_grad_(True)
            triton = v_ref.detach().clone().requires_grad_(True)

            out3 = m3(q3, k3, v3, attention_mask=None)
            out4 = m4(q4, k4, triton, attention_mask=None)
            m3.zero_grad(set_to_none=True)
            m4.zero_grad(set_to_none=True)
            loss3 = torch.sum(out3.float() * grad_out.float())
            loss4 = torch.sum(out4.float() * grad_out.float())
            loss3.backward()
            loss4.backward()

            bundle3: dict[str, Any] = {
                "out": out3.detach(),
                "dq": q3.grad.detach(),
                "dk": k3.grad.detach(),
                "dv": v3.grad.detach(),
                "dn": float(m3.ssa_n_raw.grad.detach().item()),
            }
            bundle4: dict[str, Any] = {
                "out": out4.detach(),
                "dq": q4.grad.detach(),
                "dk": k4.grad.detach(),
                "dv": triton.grad.detach(),
                "dn": float(m4.ssa_n_raw.grad.detach().item()),
            }
            if bool(args.module_learnable_b):
                bundle3["db"] = float(m3.ssa_b_raw.grad.detach().item())
                bundle4["db"] = float(m4.ssa_b_raw.grad.detach().item())

            pair = _module_pair_stats(bundle4, bundle3, tol)
            out_results.append(
                {
                    "layer": layer,
                    "dtype": dtype_name,
                    "pair": pair,
                    "passed": pair["passed"],
                }
            )
            if args.fail_fast and not pair["passed"]:
                break
    finally:
        cleanup_parallel_state()

    return {"name": "module_suite", "passed": all(r.get("passed", False) for r in out_results), "results": out_results}


def _engine_forward(
    engine: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n: torch.Tensor,
    b: torch.Tensor,
    dtype: torch.dtype,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    qd = q.to(dtype)
    kd = k.to(dtype)
    vd = v.to(dtype)
    if engine == "reference":
        out, _ = _ssa_reference_forward(qd, kd, vd, scale, n, b, causal)
        return out
    if engine == "v3":
        return ssa_flash_attention_v3(qd, kd, vd, scale, n, b, causal=causal, dropout_p=0.0, training=True)
    if engine == "triton":
        return ssa_flash_attention(qd, kd, vd, scale, n, b, causal=causal, dropout_p=0.0, training=True)
    raise ValueError(f"Unknown engine: {engine}")


def _run_training_drift_suite(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    dtype = _dtype_from_name(args.train_dtype)
    bsz = args.train_batch_size
    hq = args.train_hq
    hkv = args.train_hkv
    n_ctx = args.train_seq_length
    d_head = args.train_head_dim
    if hq % hkv != 0:
        raise ValueError("train_hq must be divisible by train_hkv")
    if n_ctx % 128 != 0:
        raise ValueError("train_seq_length must be divisible by 128 for triton backward kernel.")
    if not _is_power_of_2(d_head):
        raise ValueError("train_head_dim must be power-of-two for current triton kernel.")

    scale = 1.0 / math.sqrt(float(d_head))
    params_ref = {
        "q": torch.nn.Parameter(torch.randn(bsz, hq, n_ctx, d_head, device=device, dtype=torch.float32) * 0.05),
        "k": torch.nn.Parameter(torch.randn(bsz, hkv, n_ctx, d_head, device=device, dtype=torch.float32) * 0.05),
        "v": torch.nn.Parameter(torch.randn(bsz, hkv, n_ctx, d_head, device=device, dtype=torch.float32) * 0.05),
        "n": torch.nn.Parameter(torch.tensor(float(args.ssa_n), device=device, dtype=torch.float32)),
        "b": torch.tensor(float(args.ssa_b), device=device, dtype=torch.float32),
    }
    params_v3 = {
        "q": torch.nn.Parameter(params_ref["q"].detach().clone()),
        "k": torch.nn.Parameter(params_ref["k"].detach().clone()),
        "v": torch.nn.Parameter(params_ref["v"].detach().clone()),
        "n": torch.nn.Parameter(params_ref["n"].detach().clone()),
        "b": params_ref["b"].detach().clone(),
    }
    params_triton = {
        "q": torch.nn.Parameter(params_ref["q"].detach().clone()),
        "k": torch.nn.Parameter(params_ref["k"].detach().clone()),
        "v": torch.nn.Parameter(params_ref["v"].detach().clone()),
        "n": torch.nn.Parameter(params_ref["n"].detach().clone()),
        "b": params_ref["b"].detach().clone(),
    }
    if bool(args.train_learnable_b):
        params_ref["b"] = torch.nn.Parameter(params_ref["b"].detach().clone())
        params_v3["b"] = torch.nn.Parameter(params_v3["b"].detach().clone())
        params_triton["b"] = torch.nn.Parameter(params_triton["b"].detach().clone())
    target = torch.randn(bsz, hq, n_ctx, d_head, device=device, dtype=torch.float32)

    ref_optim_params = [params_ref["q"], params_ref["k"], params_ref["v"], params_ref["n"]]
    v3_optim_params = [params_v3["q"], params_v3["k"], params_v3["v"], params_v3["n"]]
    triton_optim_params = [params_triton["q"], params_triton["k"], params_triton["v"], params_triton["n"]]
    if bool(args.train_learnable_b):
        ref_optim_params.append(params_ref["b"])
        v3_optim_params.append(params_v3["b"])
        triton_optim_params.append(params_triton["b"])
    opt_ref = torch.optim.AdamW(ref_optim_params, lr=args.train_lr, betas=(0.9, 0.99), weight_decay=0.0)
    opt_v3 = torch.optim.AdamW(v3_optim_params, lr=args.train_lr, betas=(0.9, 0.99), weight_decay=0.0)
    opt_triton = torch.optim.AdamW(triton_optim_params, lr=args.train_lr, betas=(0.9, 0.99), weight_decay=0.0)

    worst_loss_triton_ref = 0.0
    worst_loss_triton_v3 = 0.0
    worst_param_triton_ref = 0.0
    worst_param_triton_v3 = 0.0
    history: list[dict[str, float]] = []

    for step in range(1, args.train_steps + 1):
        for opt in (opt_ref, opt_v3, opt_triton):
            opt.zero_grad(set_to_none=True)

        out_ref = _engine_forward(
            "reference",
            params_ref["q"],
            params_ref["k"],
            params_ref["v"],
            params_ref["n"],
            params_ref["b"],
            dtype=dtype,
            scale=scale,
            causal=bool(args.train_causal),
        )
        out_v3 = _engine_forward(
            "v3",
            params_v3["q"],
            params_v3["k"],
            params_v3["v"],
            params_v3["n"],
            params_v3["b"],
            dtype=dtype,
            scale=scale,
            causal=bool(args.train_causal),
        )
        out_triton = _engine_forward(
            "triton",
            params_triton["q"],
            params_triton["k"],
            params_triton["v"],
            params_triton["n"],
            params_triton["b"],
            dtype=dtype,
            scale=scale,
            causal=bool(args.train_causal),
        )

        loss_ref = F.mse_loss(out_ref.float(), target)
        loss_v3 = F.mse_loss(out_v3.float(), target)
        loss_triton = F.mse_loss(out_triton.float(), target)
        for name, loss in (("reference", loss_ref), ("v3", loss_v3), ("triton", loss_triton)):
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {name} at step {step}")

        loss_ref.backward()
        loss_v3.backward()
        loss_triton.backward()
        opt_ref.step()
        opt_v3.step()
        opt_triton.step()

        loss_diff_triton_ref = abs(float(loss_triton.item()) - float(loss_ref.item()))
        loss_diff_triton_v3 = abs(float(loss_triton.item()) - float(loss_v3.item()))

        param_diff_triton_ref = 0.0
        param_diff_triton_v3 = 0.0
        compare_names = ("q", "k", "v", "n", "b") if bool(args.train_learnable_b) else ("q", "k", "v", "n")
        for name in compare_names:
            param_diff_triton_ref = max(
                param_diff_triton_ref,
                float((params_triton[name].detach() - params_ref[name].detach()).abs().max().item()),
            )
            param_diff_triton_v3 = max(
                param_diff_triton_v3,
                float((params_triton[name].detach() - params_v3[name].detach()).abs().max().item()),
            )

        worst_loss_triton_ref = max(worst_loss_triton_ref, loss_diff_triton_ref)
        worst_loss_triton_v3 = max(worst_loss_triton_v3, loss_diff_triton_v3)
        worst_param_triton_ref = max(worst_param_triton_ref, param_diff_triton_ref)
        worst_param_triton_v3 = max(worst_param_triton_v3, param_diff_triton_v3)
        history.append(
            {
                "step": float(step),
                "loss_ref": float(loss_ref.item()),
                "loss_v3": float(loss_v3.item()),
                "loss_triton": float(loss_triton.item()),
                "loss_diff_triton_ref": loss_diff_triton_ref,
                "loss_diff_triton_v3": loss_diff_triton_v3,
                "param_diff_triton_ref": param_diff_triton_ref,
                "param_diff_triton_v3": param_diff_triton_v3,
            }
        )
        if step % max(1, args.train_steps // 10) == 0 or step == 1:
            print(
                f"[Train {step:04d}/{args.train_steps}] "
                f"loss_diff(triton,ref)={loss_diff_triton_ref:.3e} "
                f"loss_diff(triton,v3)={loss_diff_triton_v3:.3e} "
                f"param_diff(triton,ref)={param_diff_triton_ref:.3e} "
                f"param_diff(triton,v3)={param_diff_triton_v3:.3e}"
            )

    passed = (
        worst_loss_triton_ref <= args.train_loss_tol_triton_ref
        and worst_loss_triton_v3 <= args.train_loss_tol_triton_v3
        and worst_param_triton_ref <= args.train_param_tol_triton_ref
        and worst_param_triton_v3 <= args.train_param_tol_triton_v3
    )
    return {
        "name": "training_drift_suite",
        "passed": passed,
        "worst": {
            "loss_diff_triton_ref": worst_loss_triton_ref,
            "loss_diff_triton_v3": worst_loss_triton_v3,
            "param_diff_triton_ref": worst_param_triton_ref,
            "param_diff_triton_v3": worst_param_triton_v3,
        },
        "thresholds": {
            "train_loss_tol_triton_ref": args.train_loss_tol_triton_ref,
            "train_loss_tol_triton_v3": args.train_loss_tol_triton_v3,
            "train_param_tol_triton_ref": args.train_param_tol_triton_ref,
            "train_param_tol_triton_v3": args.train_param_tol_triton_v3,
        },
        "history": history,
    }


def _summarize_suite(suite: dict[str, Any]) -> None:
    name = suite["name"]
    passed = suite["passed"]
    print(f"\n=== {name} :: {'PASS' if passed else 'FAIL'} ===")
    if name == "training_drift_suite":
        worst = suite["worst"]
        print(
            "worst drift: "
            f"loss(triton,ref)={worst['loss_diff_triton_ref']:.3e}, "
            f"loss(triton,v3)={worst['loss_diff_triton_v3']:.3e}, "
            f"param(triton,ref)={worst['param_diff_triton_ref']:.3e}, "
            f"param(triton,v3)={worst['param_diff_triton_v3']:.3e}"
        )
        return

    results = suite.get("results", [])
    total = len(results)
    failed = [r for r in results if not r.get("passed", False)]
    print(f"cases={total}, failed={len(failed)}")
    if failed:
        first = failed[0]
        print(f"first failure case: {first.get('case', {})}")
        if "error" in first:
            print(f"error: {first['error']}")


def _apply_quick_mode(args: argparse.Namespace) -> None:
    args.dtypes = "bf16"
    args.kernel_batch_sizes = "1"
    args.kernel_seq_lengths = "128,512"
    args.kernel_head_pairs = "8x8,24x8"
    args.kernel_head_dims = "32"
    args.kernel_layouts = "contig"
    args.reference_max_seq = 512
    args.module_layers = "1,12"
    args.module_seq_length = 128
    args.train_steps = 20
    args.train_seq_length = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive SSA Triton parity checks")
    parser.add_argument("--arch", default="baby_luciole", type=str)
    parser.add_argument("--output_dir", default="/tmp/ssa_triton_parity", type=str)
    parser.add_argument("--name", default="ssa-triton-parity", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--ssa_n", default=1.5, type=float)
    parser.add_argument("--ssa_b", default=0.8, type=float)

    parser.add_argument("--quick", action="store_true", default=False, help="Run a reduced matrix.")
    parser.add_argument("--fail_fast", action="store_true", default=False, help="Stop on first failing case.")

    parser.add_argument("--skip_kernel", action="store_true", default=False)
    parser.add_argument("--skip_determinism", action="store_true", default=False)
    parser.add_argument("--skip_module", action="store_true", default=False)
    parser.add_argument("--skip_training", action="store_true", default=False)

    # Kernel matrix
    parser.add_argument("--dtypes", default="bf16,fp16", type=str)
    parser.add_argument("--kernel_batch_sizes", default="1,2", type=str)
    parser.add_argument("--kernel_seq_lengths", default="128,512,1024", type=str)
    parser.add_argument("--kernel_head_pairs", default="8x8,16x8,24x8", type=str)
    parser.add_argument("--kernel_head_dims", default="32", type=str)
    parser.add_argument("--kernel_causal_modes", default="1,0", type=str, help="Comma list of 1/0")
    parser.add_argument("--kernel_layouts", default="contig,strided", type=str)
    parser.add_argument("--kernel_use_reference", default=1, type=int, choices=[0, 1])
    parser.add_argument("--reference_max_seq", default=512, type=int)

    # Determinism
    parser.add_argument("--determinism_seq_lengths", default="128,1024", type=str)
    parser.add_argument("--determinism_layouts", default="contig,strided", type=str)
    parser.add_argument("--determinism_atol", default=0.0, type=float)

    # Module parity
    parser.add_argument("--module_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--module_layers", default="1,6,12", type=str)
    parser.add_argument("--module_batch_size", default=2, type=int)
    parser.add_argument("--module_seq_length", default=256, type=int)
    parser.add_argument("--module_learnable_b", default=0, type=int, choices=[0, 1])

    # Training drift
    parser.add_argument("--train_dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--train_steps", default=80, type=int)
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--train_seq_length", default=128, type=int)
    parser.add_argument("--train_hq", default=24, type=int)
    parser.add_argument("--train_hkv", default=8, type=int)
    parser.add_argument("--train_head_dim", default=32, type=int)
    parser.add_argument("--train_lr", default=1e-3, type=float)
    parser.add_argument("--train_causal", default=1, type=int, help="1 for causal, 0 for non-causal")
    parser.add_argument("--train_learnable_b", default=0, type=int, choices=[0, 1])
    parser.add_argument("--train_loss_tol_triton_ref", default=6e-3, type=float)
    parser.add_argument("--train_loss_tol_triton_v3", default=2e-3, type=float)
    parser.add_argument("--train_param_tol_triton_ref", default=3e-2, type=float)
    parser.add_argument("--train_param_tol_triton_v3", default=1e-2, type=float)

    # Numeric tolerances
    parser.add_argument("--atol_bf16", default=2e-2, type=float)
    parser.add_argument("--rtol_bf16", default=5e-2, type=float)
    parser.add_argument("--atol_fp16", default=2e-2, type=float)
    parser.add_argument("--rtol_fp16", default=5e-2, type=float)
    parser.add_argument("--atol_fp32", default=5e-4, type=float)
    parser.add_argument("--rtol_fp32", default=1e-3, type=float)
    parser.add_argument("--scalar_atol", default=2e-3, type=float)
    parser.add_argument("--scalar_rtol", default=2e-2, type=float)
    parser.add_argument("--scalar_atol_fp32", default=5e-5, type=float)
    parser.add_argument("--scalar_rtol_fp32", default=1e-4, type=float)

    parser.add_argument("--json_report", default="", type=str, help="Optional path to write JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        _apply_quick_mode(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SSA Triton parity checks.")

    _seed_everything(args.seed)
    torch.cuda.set_device(0)

    print("=== SSA Triton Parity Check ===")
    print(f"seed={args.seed}")
    print(f"v3_kernel_impl={V3_KERNEL_IMPL}")
    print(f"USE_OPTIMIZED_KERNEL={USE_OPTIMIZED_KERNEL}")
    print(f"quick_mode={args.quick}")
    print(f"arch={args.arch}")
    print(f"skip_kernel={args.skip_kernel} skip_determinism={args.skip_determinism} "
          f"skip_module={args.skip_module} skip_training={args.skip_training}")

    device = torch.device("cuda")
    suites: list[dict[str, Any]] = []

    if not args.skip_kernel:
        suite = _run_kernel_suite(args, device)
        suites.append(suite)
        _summarize_suite(suite)

    if not args.skip_determinism:
        suite = _run_determinism_suite(args, device)
        suites.append(suite)
        _summarize_suite(suite)

    if not args.skip_module:
        suite = _run_module_suite(args, device)
        suites.append(suite)
        _summarize_suite(suite)

    if not args.skip_training:
        suite = _run_training_drift_suite(args, device)
        suites.append(suite)
        _summarize_suite(suite)

    report = {
        "meta": {
            "seed": args.seed,
            "arch": args.arch,
            "v3_kernel_impl": V3_KERNEL_IMPL,
            "use_optimized_kernel_env": USE_OPTIMIZED_KERNEL,
            "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "args": vars(args),
        "suites": suites,
    }
    overall_passed = all(s.get("passed", False) for s in suites) if suites else False
    report["overall_passed"] = overall_passed

    if args.json_report:
        report_path = Path(args.json_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report written to: {report_path}")

    print(f"\n=== OVERALL: {'PASS' if overall_passed else 'FAIL'} ===")
    if not overall_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
