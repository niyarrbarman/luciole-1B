#!/usr/bin/env python3
"""Deep debug of dn/db computation."""

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

# Small config for debugging
B, Hq, Hkv, N, D = 8, 24, 8, 1024, 32  # Same as benchmark
q = torch.randn(B, Hq, N, D, dtype=dtype, device=device)
k = torch.randn(B, Hkv, N, D, dtype=dtype, device=device)
v = torch.randn(B, Hkv, N, D, dtype=dtype, device=device)
ssa_n = torch.tensor(1.5, dtype=torch.float32, device=device)
ssa_b = torch.tensor(0.8, dtype=torch.float32, device=device)
scale = 1.0 / (D ** 0.5)

print("=" * 60)
print("FORWARD COMPARISON")
print("=" * 60)

# Forward passes
out_orig, lse_orig = fwd_original(q, k, v, scale, ssa_n, ssa_b, causal=True)
out_opt, lse_opt = fwd_optimized(q, k, v, scale, ssa_n, ssa_b, causal=True)

print(f"out_orig shape: {out_orig.shape}")
print(f"lse_orig shape: {lse_orig.shape}")
print(f"out max diff: {(out_orig - out_opt).abs().max().item():.6f}")
print(f"lse max diff: {(lse_orig - lse_opt).abs().max().item():.6f}")

# Make backward inputs IDENTICAL
print("\n" + "=" * 60)
print("BACKWARD COMPARISON (identical inputs)")
print("=" * 60)

dout = torch.randn_like(out_orig)

# Use SAME inputs for both backward passes
dq_orig, dk_orig, dv_orig, dn_orig, db_orig = bwd_original(
    q, k, v, out_orig, dout, lse_orig, scale, ssa_n, ssa_b, causal=True
)
dq_opt, dk_opt, dv_opt, dn_opt, db_opt = bwd_optimized(
    q, k, v, out_orig, dout, lse_orig, scale, ssa_n, ssa_b, causal=True  # SAME out/lse
)

print("dQ max diff:", (dq_orig - dq_opt).abs().max().item())
print("dK max diff:", (dk_orig - dk_opt).abs().max().item())
print("dV max diff:", (dv_orig - dv_opt).abs().max().item())

print(f"\ndn_orig = {dn_orig.item():.6f}")
print(f"dn_opt  = {dn_opt.item():.6f}")
print(f"dn diff = {abs(dn_orig.item() - dn_opt.item()):.6f}")

print(f"\ndb_orig = {db_orig.item():.6f}")
print(f"db_opt  = {db_opt.item():.6f}")
print(f"db diff = {abs(db_orig.item() - db_opt.item()):.6f}")

# Test with NON-CAUSAL to narrow down
print("\n" + "=" * 60)
print("NON-CAUSAL test")
print("=" * 60)

out_orig_nc, lse_orig_nc = fwd_original(q, k, v, scale, ssa_n, ssa_b, causal=False)
dq_orig_nc, dk_orig_nc, dv_orig_nc, dn_orig_nc, db_orig_nc = bwd_original(
    q, k, v, out_orig_nc, dout, lse_orig_nc, scale, ssa_n, ssa_b, causal=False
)
dq_opt_nc, dk_opt_nc, dv_opt_nc, dn_opt_nc, db_opt_nc = bwd_optimized(
    q, k, v, out_orig_nc, dout, lse_orig_nc, scale, ssa_n, ssa_b, causal=False
)

print(f"NON-CAUSAL dn_orig = {dn_orig_nc.item():.6f}")
print(f"NON-CAUSAL dn_opt  = {dn_opt_nc.item():.6f}")
print(f"NON-CAUSAL dn diff = {abs(dn_orig_nc.item() - dn_opt_nc.item()):.6f}")

print(f"\nNON-CAUSAL db_orig = {db_orig_nc.item():.6f}")
print(f"NON-CAUSAL db_opt  = {db_opt_nc.item():.6f}")
print(f"NON-CAUSAL db diff = {abs(db_orig_nc.item() - db_opt_nc.item()):.6f}")
