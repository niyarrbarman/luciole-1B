# SSA: Softmax-Substituted Attention
# Implementations:
#   ssa_attention.py              - Original unfused (PyTorch ops)
#   ssa_flex_attention.py         - FlexAttention-based (PyTorch flex_attention)
#   ssa_triton_attention.py       - Fused Triton FlashAttention v2/v3
#   ssa_triton_v4_attention.py    - Fused Triton FlashAttention v4 (tutorial-based structure)
