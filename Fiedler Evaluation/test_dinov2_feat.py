#!/usr/bin/env python3
"""Test DINOv2 feature extraction."""

import sys
from pathlib import Path

sys.path.insert(0, "d:/Coding/Recursive Segmentation Master's Project/CutLER Implementation/TFSS-CutLER/maskcut")

import dinov2_feat
from PIL import Image

print("Testing DINOv2Feat class...")

# Load model
print("\n1. Loading DINOv2 small model...")
try:
    feat_extractor = dinov2_feat.DINOv2Feat("dinov2_vits14", feat_type="k", device="cuda")
    print(f"   SUCCESS: {feat_extractor.__class__.__name__}")
    print(f"   Feature dim: {feat_extractor.feat_dim}")
    print(f"   Patch size: {feat_extractor.patch_size}")
except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test feature extraction
print("\n2. Extracting features from test image...")
test_image_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2010_002639.jpg")
if test_image_path.exists():
    try:
        img = Image.open(test_image_path).convert("RGB")
        print(f"   Image size: {img.size}")

        features, grid = feat_extractor.extract_features(img)
        print(f"   Features shape: {features.shape}")
        print(f"   Grid info: {grid}")
        print(f"   Features mean: {features.mean().item():.6f}")
        print(f"   Features std: {features.std().item():.6f}")
        print(f"   SUCCESS!")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print(f"   Test image not found")
    sys.exit(1)

print("\nDINOv2Feat is working correctly!")
