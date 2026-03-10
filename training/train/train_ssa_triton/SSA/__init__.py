# SSA: Softmax-Substituted Attention
# Implementations:
#   archive/legacy_pre_triton/ssa_attention.py         - Original unfused (PyTorch ops)
#   archive/legacy_pre_triton/ssa_flex_attention.py    - FlexAttention-based (PyTorch flex_attention)
#   archive/legacy_pre_triton/ssa_triton_attention.py  - Fused Triton FlashAttention v2/v3
#   ssa_triton_attention.py                         - Fused Triton FlashAttention (current main)
