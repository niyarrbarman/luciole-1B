#!/usr/bin/env python3
"""Quick debug script to check dn/db differences."""

import torch
from SSA.ssa_triton_kernel import (
    ssa_flash_attn_forward as fwd_original,
    ssa_flash_attn_backward as bwd_original,
)
from SSA.ssa_triton_kernel_optimized import (
    ssa_flash_attn_forward as fwd_optimized,
    ssa_flash_attn_backward as bwd_optimized,
)

device = 'cuda'
dtype = torch.bfloat16

B, Hq, Hkv, N, D = 8, 24, 8, 1024, 32
q = torch.randn(B, Hq, N, D, dtype=dtype, device=device)
k = torch.randn(B, Hkv, N, D, dtype=dtype, device=device)
v = torch.randn(B, Hkv, N, D, dtype=dtype, device=device)
ssa_n = torch.tensor(1.5, dtype=torch.float32, device=device)
ssa_b = torch.tensor(0.8, dtype=torch.float32, device=device)
scale = 1.0 / (D ** 0.5)

# Forward
out_orig, lse_orig = fwd_original(q, k, v, scale, ssa_n, ssa_b, causal=True)
out_opt, lse_opt = fwd_optimized(q, k, v, scale, ssa_n, ssa_b, causal=True)

# Backward
dout = torch.randn_like(out_orig)
dq_orig, dk_orig, dv_orig, dn_orig, db_orig = bwd_original(
    q, k, v, out_orig, dout, lse_orig, scale, ssa_n, ssa_b, causal=True
)
dq_opt, dk_opt, dv_opt, dn_opt, db_opt = bwd_optimized(
    q, k, v, out_opt, dout, lse_opt, scale, ssa_n, ssa_b, causal=True
)

print("dn comparison:")
print(f"  Original: {dn_orig.item():.6f}")
print(f"  Optimized: {dn_opt.item():.6f}")
print(f"  Diff: {abs(dn_orig.item() - dn_opt.item()):.6f}")
print(f"  Relative diff: {abs(dn_orig.item() - dn_opt.item()) / (abs(dn_orig.item()) + 1e-8) * 100:.4f}%")

print("\ndb comparison:")
print(f"  Original: {db_orig.item():.6f}")
print(f"  Optimized: {db_opt.item():.6f}")
print(f"  Diff: {abs(db_orig.item() - db_opt.item()):.6f}")
print(f"  Relative diff: {abs(db_orig.item() - db_opt.item()) / (abs(db_orig.item()) + 1e-8) * 100:.4f}%")

print("\ndQ max diff:", (dq_orig - dq_opt).abs().max().item())
print("dK max diff:", (dk_orig - dk_opt).abs().max().item())
print("dV max diff:", (dv_orig - dv_opt).abs().max().item())
