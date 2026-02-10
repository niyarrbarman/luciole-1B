"""Standalone test for SSA parameter groups and learning rates."""
import torch
from torch.optim import AdamW
from transformers.models.roberta import RobertaConfig, RobertaForMaskedLM

# Create a minimal model
config = RobertaConfig(
    vocab_size=1000,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    attn_implementation="eager",
)
model = RobertaForMaskedLM(config)

# Check SSA parameters exist
print("=" * 60)
print("SSA Parameters in Model:")
print("=" * 60)
ssa_params = []
other_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if 'ssa_n_raw' in name or 'ssa_b_raw' in name:
            ssa_params.append(param)
            print(f"  [SSA] {name}: shape={param.shape}, value={param.item():.4f}")
        else:
            other_params.append(param)

print(f"\nTotal SSA params: {len(ssa_params)}")
print(f"Total other params: {len(other_params)}")

# Create optimizer with different LRs
base_lr = 1e-4
ssa_lr = base_lr * 10  # 10x multiplier

optimizer = AdamW([
    {'params': other_params, 'lr': base_lr, 'weight_decay': 0.01},
    {'params': ssa_params, 'lr': ssa_lr, 'weight_decay': 0.0},
])

# Check optimizer param groups
print("\n" + "=" * 60)
print("Optimizer Parameter Groups:")
print("=" * 60)
for i, group in enumerate(optimizer.param_groups):
    label = "SSA" if i == 1 else "Other"
    print(f"Group {i} ({label}): lr={group['lr']}, weight_decay={group['weight_decay']}, num_params={len(group['params'])}")

print("\n" + "=" * 60)
print("✓ Verification complete!")
print("=" * 60)
