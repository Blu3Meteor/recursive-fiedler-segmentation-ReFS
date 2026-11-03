#!/usr/bin/env python3
"""Verify DINOv2 weights are loaded and measure actual feature extraction time."""

import sys
import time
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, "d:/Coding/Recursive Segmentation Master's Project/CutLER Implementation/TFSS-CutLER/maskcut")

import dinov2_feat

print("=" * 70)
print("DINOv2 Weight Loading & Speed Test")
print("=" * 70)

# 1. Verify model loads with pretrained weights
print("\n1. Loading DINOv2 model...")
try:
    feat_extractor = dinov2_feat.DINOv2Feat("dinov2_vits14", feat_type="k", device="cuda")
    print(f"   ✓ Model loaded: {feat_extractor.model_id}")
    print(f"   ✓ Weights loaded: {feat_extractor.weights_loaded}")
    print(f"   ✓ Feature dim: {feat_extractor.feat_dim}")
    print(f"   ✓ Patch size: {feat_extractor.patch_size}")

    if not feat_extractor.weights_loaded:
        print("\n   ⚠️  WARNING: Weights did NOT load (weights_loaded=False)")
        print("   This would explain suspicious speed!")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. Check model parameter count
print("\n2. Checking model parameters...")
total_params = sum(p.numel() for p in feat_extractor.model.parameters())
trainable_params = sum(p.numel() for p in feat_extractor.model.parameters() if p.requires_grad)
print(f"   Total params: {total_params:,}")
print(f"   Trainable params: {trainable_params:,}")
if trainable_params == 0:
    print("   ✓ All weights frozen (expected for inference)")
else:
    print("   ⚠️  WARNING: Found trainable parameters (unexpected for pretrained model)")

# 3. Measure feature extraction time on a real image
print("\n3. Testing feature extraction speed...")
test_image_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2010_002639.jpg")

if not test_image_path.exists():
    print(f"   ✗ Test image not found: {test_image_path}")
    sys.exit(1)

try:
    img = Image.open(test_image_path).convert("RGB")
    print(f"   Image size: {img.size}")

    # Warmup (first run is slower due to compilation)
    print("   Running warmup...")
    with torch.no_grad():
        _, _ = feat_extractor.extract_features(img)

    # Actual timing (average of 3 runs)
    print("   Measuring extraction time (3 runs)...")
    times = []
    for i in range(3):
        torch.cuda.synchronize()  # Ensure all GPU ops are done
        start = time.time()
        with torch.no_grad():
            features, grid = feat_extractor.extract_features(img)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"     Run {i+1}: {elapsed:.3f}s → features {features.shape}, grid {grid}")

    avg_time = sum(times) / len(times)
    print(f"\n   Average extraction time: {avg_time:.3f}s per image")
    print(f"   For 1505 PASCAL images: ~{avg_time * 1505 / 60:.1f} minutes")
    print(f"   For 1505 PASCAL images: ~{avg_time * 1505 / 3600:.2f} hours")

    # Sanity checks on features
    print(f"\n4. Feature sanity checks...")
    print(f"   Mean: {features.mean():.6f}")
    print(f"   Std:  {features.std():.6f}")
    print(f"   Min:  {features.min():.6f}")
    print(f"   Max:  {features.max():.6f}")

    if features.std() < 0.1:
        print("   ⚠️  WARNING: Very low std dev (features might be zeros or near-zeros)")
    else:
        print("   ✓ Feature statistics look reasonable")

except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ DINOv2 weights and speed test PASSED")
print("=" * 70)
