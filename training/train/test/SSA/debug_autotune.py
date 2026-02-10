#!/usr/bin/env python3
"""Debug autotuning config selection."""

import torch
import triton
from SSA.ssa_triton_kernel_optimized import (
    ssa_flash_attn_forward as fwd_optimized,
    ssa_flash_attn_backward as bwd_optimized,
    _ssa_attn_bwd_dkv_kernel,
)

device = 'cuda'
dtype = torch.bfloat16

# Benchmark config
B, Hq, Hkv, N, D = 8, 24, 8, 1024, 32
q = torch.randn(B, Hq, N, D, dtype=dtype, device=device)
k = torch.randn(B, Hkv, N, D, dtype=dtype, device=device)
v = torch.randn(B, Hkv, N, D, dtype=dtype, device=device)
ssa_n = torch.tensor(1.5, dtype=torch.float32, device=device)
ssa_b = torch.tensor(0.8, dtype=torch.float32, device=device)
scale = 1.0 / (D ** 0.5)

print("Running forward...")
out, lse = fwd_optimized(q, k, v, scale, ssa_n, ssa_b, causal=True)

print("Running backward...")
dout = torch.randn_like(out)
dq, dk, dv, dn, db = bwd_optimized(q, k, v, out, dout, lse, scale, ssa_n, ssa_b, causal=True)

# Check what config was selected
print("\nAutotuned configs:")
print(f"dKV kernel best config: {_ssa_attn_bwd_dkv_kernel.best_config}")
