#!/usr/bin/env python3
"""Visualize extracted CNN vs ViT feature similarity maps."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATASET_ROOT = Path("Fiedler Evaluation/data/coco20i_backbone_comparison")
FEATURES_ROOT = Path("Fiedler Evaluation/output/01_backbone_comparison/backbone_comparison_features")
OUTPUT_ROOT = Path("Fiedler Evaluation/output/01_backbone_comparison/backbone_comparison_visualizations")

def load_image(img_path):
    """Load image as RGB array."""
    return cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

def normalize_heatmap(sim_map):
    """Normalize similarity map to [0, 1]."""
    sim_min = sim_map.min()
    sim_max = sim_map.max()
    if sim_max > sim_min:
        sim_norm = (sim_map - sim_min) / (sim_max - sim_min)
    else:
        sim_norm = np.zeros_like(sim_map)
    return sim_norm

def overlay_heatmap(image, heatmap, alpha=0.5, cmap='hot'):
    """Overlay heatmap on image."""
    # Normalize heatmap to [0, 1]
    heatmap_norm = normalize_heatmap(heatmap)

    # Apply colormap
    cmap_fn = plt.cm.get_cmap(cmap)
    heatmap_colored = cmap_fn(heatmap_norm)[:, :, :3]  # (H, W, 3)
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay

def visualize_image(img_idx, category):
    """Create side-by-side visualization for one image."""
    cat_dir = DATASET_ROOT / category
    feat_dir = FEATURES_ROOT / category

    images = sorted([f for f in cat_dir.glob("*.jpg")])
    if img_idx >= len(images):
        return

    img_path = images[img_idx]
    img_name = img_path.stem
    feat_path = feat_dir / img_name

    output_dir = OUTPUT_ROOT / category / img_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image = load_image(str(img_path))
    H, W = image.shape[:2]

    # Load feature similarity maps
    resnet_sim = np.load(str(feat_path / "resnet50_sim.npy"))
    dinov2_sim = np.load(str(feat_path / "dinov2_sim.npy"))

    # Resize to image size
    resnet_sim = cv2.resize(resnet_sim, (W, H), interpolation=cv2.INTER_LINEAR)
    dinov2_sim = cv2.resize(dinov2_sim, (W, H), interpolation=cv2.INTER_LINEAR)

    # Create overlays
    resnet_overlay = overlay_heatmap(image, resnet_sim, alpha=0.6, cmap='hot')
    dinov2_overlay = overlay_heatmap(image, dinov2_sim, alpha=0.6, cmap='hot')

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(resnet_overlay)
    axes[1].set_title("ResNet50 (Layer4)", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(dinov2_overlay)
    axes[2].set_title("DINOv2 (Keys)", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(str(output_dir / "comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Save individual heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(normalize_heatmap(resnet_sim), cmap='hot')
    axes[0].set_title("ResNet50 Similarity", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(normalize_heatmap(dinov2_sim), cmap='hot')
    axes[1].set_title("DINOv2 Similarity", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(str(output_dir / "heatmaps.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ {category}/{img_name}")

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BACKBONE FEATURE VISUALIZATION")
    print("=" * 60)

    for category in ["same_class_instances", "multi_class"]:
        cat_dir = DATASET_ROOT / category
        images = sorted([f for f in cat_dir.glob("*.jpg")])

        print(f"\nVisualizing {category} ({len(images)} images)...")
        for idx in range(len(images)):
            try:
                visualize_image(idx, category)
            except Exception as e:
                print(f"✗ Error on image {idx}: {e}")

    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {OUTPUT_ROOT}")
    print("=" * 60)

if __name__ == "__main__":
    main()
