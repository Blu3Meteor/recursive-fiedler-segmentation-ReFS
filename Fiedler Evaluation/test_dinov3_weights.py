#!/usr/bin/env python3
"""Debug DINOv3 model weight loading and feature quality."""

import sys
from pathlib import Path
import torch
from PIL import Image

# Add paths
sys.path.insert(0, "d:/Coding/Recursive Segmentation Master's Project/dinov3_repo")
sys.path.insert(0, "d:/Coding/Recursive Segmentation Master's Project/CutLER Implementation/TFSS-CutLER/maskcut")

import dinov3_feat

print("=" * 80)
print("Testing DINOv3 Model Weight Loading")
print("=" * 80)

# Load model
print("\n1. Loading DINOv3 model...")
feat_extractor = dinov3_feat.DINOv3Feat(
    "dinov3_vits16plus_pretrain_lvd1689m",
    feat_type="k",
    device="cuda"
)

# Check model weights
print("\n2. Inspecting model weights...")
model = feat_extractor.model
state_dict = model.state_dict()

# Get first few weight values to check if they look like pretrained or random
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check patch embedding weight (should be different from random init)
patch_embed_key = "patch_embed.proj.weight"
if patch_embed_key in state_dict:
    w = state_dict[patch_embed_key]
    print(f"   Patch embed shape: {w.shape}")
    print(f"   Mean: {w.mean().item():.6f}, Std: {w.std().item():.6f}")
    print(f"   Min: {w.min().item():.6f}, Max: {w.max().item():.6f}")
    print(f"   First 5 values: {w.view(-1)[:5].tolist()}")
else:
    print(f"   WARNING: {patch_embed_key} not found in state_dict")

# Test feature extraction
print("\n3. Extracting features from test image...")
test_image_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2010_002639.jpg")
if test_image_path.exists():
    img = Image.open(test_image_path).convert("RGB")
    print(f"   Image size: {img.size}")

    features, grid = feat_extractor.extract_features(img, debiased=False)
    print(f"   Features shape: {features.shape}")
    print(f"   Features mean: {features.mean().item():.6f}")
    print(f"   Features std: {features.std().item():.6f}")
    print(f"   Features min: {features.min().item():.6f}")
    print(f"   Features max: {features.max().item():.6f}")

    # Check if features look like they come from a real model
    # Pretrained DINO features typically have much larger magnitude than random
    print(f"   Feature L2 norm (per patch): {torch.norm(features, dim=1).mean().item():.6f}")

    # Now test with debiasing
    features_debiased, _ = feat_extractor.extract_features(img, debiased=True)
    print(f"\n4. After positional debiasing:")
    print(f"   Features shape: {features_debiased.shape}")
    print(f"   Features mean: {features_debiased.mean().item():.6f}")
    print(f"   Features std: {features_debiased.std().item():.6f}")
    print(f"   Features min: {features_debiased.min().item():.6f}")
    print(f"   Features max: {features_debiased.max().item():.6f}")
    print(f"   Feature L2 norm (per patch): {torch.norm(features_debiased, dim=1).mean().item():.6f}")

    # Compare
    diff = (features - features_debiased).abs().mean()
    print(f"   Mean absolute change from debiasing: {diff.item():.6f}")

else:
    print(f"   ERROR: Test image not found at {test_image_path}")

print("\n5. Checking positional basis...")
if feat_extractor.positional_basis is not None:
    B = feat_extractor.positional_basis
    print(f"   Positional basis shape: {B.shape}")
    print(f"   Basis mean: {B.mean().item():.6f}")
    print(f"   Basis std: {B.std().item():.6f}")
else:
    print(f"   Positional basis is None (debiasing will not work)")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

# Diagnostic checks
patch_embed = state_dict.get(patch_embed_key)
if patch_embed is not None:
    # Random weights from N(0,1) have std ~= 1
    # Pretrained weights are usually more structured
    w_std = patch_embed.std().item()
    if w_std < 0.1:
        print("❌ PROBLEM: Patch embedding has very small std - likely random initialization")
    elif w_std > 0.5:
        print("✓ Patch embedding std looks reasonable for pretrained weights")
    else:
        print("? Patch embedding std is borderline")

if features.shape[0] > 0:
    feat_norm = torch.norm(features, dim=1).mean().item()
    if feat_norm < 0.5:
        print("❌ PROBLEM: Feature norms are very small - features may be mostly noise")
    else:
        print("✓ Feature norms look reasonable")

    if diff.item() > 0.5:
        print("⚠ WARNING: Debiasing is making large changes to features")

print("=" * 80)
