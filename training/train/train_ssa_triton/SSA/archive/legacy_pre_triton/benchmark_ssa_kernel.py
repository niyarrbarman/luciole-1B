#!/usr/bin/env python3
"""
Benchmark script to compare original vs optimized SSA Triton kernels.

Usage:
    python benchmark_ssa_kernel.py

This script:
1. Tests correctness (gradients match between original and optimized)
2. Measures performance (forward + backward time)
3. Reports speedup
"""

import torch
import time
import argparse

# Import both kernel implementations
from SSA.ssa_triton_kernel import (
    ssa_flash_attn_forward as fwd_original,
    ssa_flash_attn_backward as bwd_original,
)
from SSA.ssa_triton_kernel_optimized import (
    ssa_flash_attn_forward as fwd_optimized,
    ssa_flash_attn_backward as bwd_optimized,
)


def benchmark_kernel(fwd_fn, bwd_fn, q, k, v, scale, ssa_n, ssa_b, warmup=5, iters=20):
    """Run forward + backward and measure time."""
    # Warmup
    for _ in range(warmup):
        out, lse = fwd_fn(q, k, v, scale, ssa_n, ssa_b, causal=True)
        dout = torch.randn_like(out)
        dq, dk, dv, dn, db = bwd_fn(q, k, v, out, dout, lse, scale, ssa_n, ssa_b, causal=True)
    torch.cuda.synchronize()

    # Timed iterations
    start = time.perf_counter()
    for _ in range(iters):
        out, lse = fwd_fn(q, k, v, scale, ssa_n, ssa_b, causal=True)
        dout = torch.randn_like(out)
        dq, dk, dv, dn, db = bwd_fn(q, k, v, out, dout, lse, scale, ssa_n, ssa_b, causal=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters * 1000  # ms per iteration


def test_correctness(q, k, v, scale, ssa_n, ssa_b, atol=1e-2, rtol=1e-2):
    """Check that optimized kernel produces same results as original."""
    # Forward
    out_orig, lse_orig = fwd_original(q, k, v, scale, ssa_n, ssa_b, causal=True)
    out_opt, lse_opt = fwd_optimized(q, k, v, scale, ssa_n, ssa_b, causal=True)

    fwd_out_match = torch.allclose(out_orig, out_opt, atol=atol, rtol=rtol)
    fwd_lse_match = torch.allclose(lse_orig, lse_opt, atol=atol, rtol=rtol)

    # Backward
    dout = torch.randn_like(out_orig)
    dq_orig, dk_orig, dv_orig, dn_orig, db_orig = bwd_original(
        q, k, v, out_orig, dout, lse_orig, scale, ssa_n, ssa_b, causal=True
    )
    dq_opt, dk_opt, dv_opt, dn_opt, db_opt = bwd_optimized(
        q, k, v, out_opt, dout, lse_opt, scale, ssa_n, ssa_b, causal=True
    )

    bwd_dq_match = torch.allclose(dq_orig, dq_opt, atol=atol, rtol=rtol)
    bwd_dk_match = torch.allclose(dk_orig, dk_opt, atol=atol, rtol=rtol)
    bwd_dv_match = torch.allclose(dv_orig, dv_opt, atol=atol, rtol=rtol)
    bwd_dn_match = torch.allclose(dn_orig, dn_opt, atol=atol, rtol=rtol)
    bwd_db_match = torch.allclose(db_orig, db_opt, atol=atol, rtol=rtol)

    all_match = all([
        fwd_out_match, fwd_lse_match,
        bwd_dq_match, bwd_dk_match, bwd_dv_match, bwd_dn_match, bwd_db_match
    ])

    return {
        'all_match': all_match,
        'fwd_out': fwd_out_match,
        'fwd_lse': fwd_lse_match,
        'bwd_dq': bwd_dq_match,
        'bwd_dk': bwd_dk_match,
        'bwd_dv': bwd_dv_match,
        'bwd_dn': bwd_dn_match,
        'bwd_db': bwd_db_match,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark SSA Triton kernels')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--hq', type=int, default=24, help='Number of Q heads')
    parser.add_argument('--hkv', type=int, default=8, help='Number of KV heads')
    parser.add_argument('--seq', type=int, default=1024, help='Sequence length')
    parser.add_argument('--dim', type=int, default=32, help='Head dimension')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=50, help='Benchmark iterations')
    parser.add_argument('--skip-correctness', action='store_true', help='Skip correctness check')
    args = parser.parse_args()

    device = 'cuda'
    dtype = torch.bfloat16

    print("=" * 60)
    print("SSA Triton Kernel Benchmark")
    print("=" * 60)
    print(f"Config: B={args.batch}, Hq={args.hq}, Hkv={args.hkv}, N={args.seq}, D={args.dim}")
    print(f"GQA ratio: {args.hq // args.hkv}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print("=" * 60)

    # Create test tensors
    q = torch.randn(args.batch, args.hq, args.seq, args.dim, dtype=dtype, device=device)
    k = torch.randn(args.batch, args.hkv, args.seq, args.dim, dtype=dtype, device=device)
    v = torch.randn(args.batch, args.hkv, args.seq, args.dim, dtype=dtype, device=device)
    ssa_n = torch.tensor(1.5, dtype=torch.float32, device=device)
    ssa_b = torch.tensor(0.8, dtype=torch.float32, device=device)
    scale = 1.0 / (args.dim ** 0.5)

    # Correctness check
    if not args.skip_correctness:
        print("\n[1] Correctness Check")
        print("-" * 40)
        results = test_correctness(q, k, v, scale, ssa_n, ssa_b)
        for key, val in results.items():
            status = "✓" if val else "✗"
            print(f"  {key}: {status}")
        if not results['all_match']:
            print("\n⚠️  WARNING: Some outputs differ! Check tolerances or kernel logic.")
        else:
            print("\n✓ All outputs match within tolerance.")

    # Performance benchmark
    print("\n[2] Performance Benchmark")
    print("-" * 40)

    time_original = benchmark_kernel(
        fwd_original, bwd_original, q, k, v, scale, ssa_n, ssa_b,
        warmup=args.warmup, iters=args.iters
    )
    print(f"  Original kernel:  {time_original:.2f} ms/iter")

    time_optimized = benchmark_kernel(
        fwd_optimized, bwd_optimized, q, k, v, scale, ssa_n, ssa_b,
        warmup=args.warmup, iters=args.iters
    )
    print(f"  Optimized kernel: {time_optimized:.2f} ms/iter")

    speedup = time_original / time_optimized
    print(f"\n  Speedup: {speedup:.2f}x")

    if speedup > 1.0:
        print(f"  ✓ Optimized kernel is {(speedup - 1) * 100:.1f}% faster")
    else:
        print(f"  ⚠️  Optimized kernel is {(1 - speedup) * 100:.1f}% slower")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
