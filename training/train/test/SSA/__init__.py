# SSA: Softmax-Substituted Attention
# Implementations:
#   ssa_attention.py          - Original unfused (PyTorch ops)
#   ssa_flex_attention.py     - FlexAttention-based (PyTorch flex_attention)
#   ssa_triton_attention.py   - Fused Triton FlashAttention (fastest)
