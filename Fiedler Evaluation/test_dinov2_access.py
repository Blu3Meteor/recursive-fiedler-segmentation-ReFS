#!/usr/bin/env python3
"""Test if DINOv2 models are accessible."""

import torch
import sys

print("=" * 80)
print("Testing DINOv2 Model Access")
print("=" * 80)

# Test loading small model
print("\n1. Attempting to load dinov2_vits14 from torch.hub...")
try:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    print(f"   SUCCESS! Model loaded: {model.__class__.__name__}")
    print(f"   Embed dim: {model.embed_dim}")
    print(f"   Patch size: {model.patch_size}")
except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

print("\n2. Attempting to load dinov2_vitb14 from torch.hub...")
try:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    print(f"   SUCCESS! Model loaded: {model.__class__.__name__}")
    print(f"   Embed dim: {model.embed_dim}")
    print(f"   Patch size: {model.patch_size}")
except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("DINOv2 models are ACCESSIBLE!")
print("=" * 80)
