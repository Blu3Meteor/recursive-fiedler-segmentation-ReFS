#!/usr/bin/env python3
"""Debug DINOv2 output structure."""

import torch
from PIL import Image

print("Loading DINOv2 model...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model.eval()

print(f"Model: {model.__class__.__name__}")
print(f"Patch size: {model.patch_size}")

# Create a test input (divisible by patch size)
print("\nCreating test input...")
test_input = torch.randn(1, 3, 224, 224)  # 224 is divisible by 14

print("Running forward_features...")
with torch.no_grad():
    output = model.forward_features(test_input)

print(f"\nOutput type: {type(output)}")
print(f"Output keys: {output.keys() if isinstance(output, dict) else 'not a dict'}")

if isinstance(output, dict):
    for key, val in output.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {type(val)}")
