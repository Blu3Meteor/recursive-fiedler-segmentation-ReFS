#!/usr/bin/env python3
"""Extract and visualize CNN vs ViT features for backbone comparison.

ResNet50 vs DINOv2 on same-class and multi-class images.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

# Config
DATASET_ROOT = Path("Fiedler Evaluation/data/coco20i_backbone_comparison")
OUTPUT_ROOT = Path("Fiedler Evaluation/output/01_backbone_comparison/backbone_comparison_features")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_resnet50():
    """Load pretrained ResNet50."""
    model = models.resnet50(pretrained=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_dinov2():
    """Load DINOv2-base."""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def extract_resnet50_features(model, image_tensor):
    """Extract ResNet50 layer4 features (dense)."""
    # Hook to capture layer4 output
    features = {}
    def hook(module, input, output):
        features['layer4'] = output

    handle = model.layer4.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(image_tensor)
    handle.remove()

    feat = features['layer4']  # (1, 2048, H, W)
    return feat

def extract_dinov2_features(model, image_tensor):
    """Extract DINOv2 patch features via hook."""
    features_dict = {}

    def hook_fn(module, input, output):
        features_dict['output'] = output

    # Register hook on last block
    handle = model.blocks[-1].register_forward_hook(hook_fn)

    with torch.no_grad():
        model(image_tensor)

    handle.remove()

    # Get features from hook
    features = features_dict.get('output', None)
    if features is None:
        raise RuntimeError("Failed to extract features via hook")

    # Features are (B, N, 768) where N includes cls token
    # Remove cls token
    features = features[:, 1:, :]  # (1, N, 768)

    # Infer patch grid size (224x224 image / 14 patch size = 16x16 patches)
    N = features.shape[1]
    h = w = int(np.sqrt(N))

    # Reshape to spatial: (1, h, w, 768) then to (1, 768, h, w)
    features = features.reshape(1, h, w, 768)  # (1, 36, 36, 768)
    features = features.permute(0, 3, 1, 2)  # (1, 768, 36, 36)

    return features

def compute_similarity_map(features, reference_idx=None):
    """Compute cosine similarity between all patches and a reference patch.

    Args:
        features: (1, C, H, W) feature map
        reference_idx: (h, w) index to use as reference. If None, use center.

    Returns:
        similarity: (H, W) similarity map
    """
    B, C, H, W = features.shape

    # Normalize features
    features_norm = F.normalize(features, p=2, dim=1)  # (1, C, H, W)

    # Flatten spatial dims
    feat_flat = features_norm.reshape(1, C, -1)  # (1, C, HW)

    # Select reference patch
    if reference_idx is None:
        reference_idx = (H // 2, W // 2)

    ref_idx_flat = reference_idx[0] * W + reference_idx[1]
    ref_feat = feat_flat[:, :, ref_idx_flat:ref_idx_flat+1]  # (1, C, 1)

    # Compute cosine similarity
    similarity = torch.mm(ref_feat.squeeze(0).T, feat_flat.squeeze(0))  # (1, HW)
    similarity = similarity.reshape(H, W).cpu().numpy()

    return similarity, ref_idx_flat

def load_image(image_path, size=224):
    """Load and preprocess image using torchvision transforms."""
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    from PIL import Image as PILImage

    # Define transforms
    transform = Compose([
        Resize((size, size), interpolation=Image.BILINEAR),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and transform
    img = PILImage.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    return img_tensor

def process_dataset():
    """Extract features for all images with memory management."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_path = DATASET_ROOT / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Collect all image paths
    all_images = {}
    for category in ["same_class_instances", "multi_class"]:
        cat_dir = DATASET_ROOT / category
        cat_output = OUTPUT_ROOT / category
        cat_output.mkdir(parents=True, exist_ok=True)
        all_images[category] = sorted([f for f in cat_dir.glob("*.jpg")])

    # Process ResNet50 first
    print("Loading ResNet50...")
    resnet = load_resnet50().to(DEVICE)

    for category in ["same_class_instances", "multi_class"]:
        print(f"\nProcessing {category} with ResNet50 ({len(all_images[category])} images)...")

        for img_path in tqdm(all_images[category], desc=category):
            img_name = img_path.stem
            img_output = OUTPUT_ROOT / category / img_name
            img_output.mkdir(parents=True, exist_ok=True)

            try:
                # Load image
                img_tensor = load_image(str(img_path), size=476).to(DEVICE)

                # Extract ResNet50 features
                with torch.no_grad():
                    resnet_feat = extract_resnet50_features(resnet, img_tensor)
                    resnet_feat_up = F.interpolate(resnet_feat, size=(224, 224), mode='bilinear', align_corners=False)

                # Compute similarity map
                resnet_sim, _ = compute_similarity_map(resnet_feat_up)

                # Save similarity map
                np.save(str(img_output / "resnet50_sim.npy"), resnet_sim)

                # Save stats
                stats = {
                    "image": img_name,
                    "resnet50_shape": str(resnet_feat.shape),
                    "resnet50_sim_range": [float(resnet_sim.min()), float(resnet_sim.max())],
                }
                with open(img_output / "stats.json", "w") as f:
                    json.dump(stats, f, indent=2)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM on {img_name}, clearing cache...")
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

    # Unload ResNet50
    del resnet
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Process DINOv2
    print("\nLoading DINOv2...")
    dinov2 = load_dinov2().to(DEVICE)

    for category in ["same_class_instances", "multi_class"]:
        print(f"\nProcessing {category} with DINOv2 ({len(all_images[category])} images)...")

        for img_path in tqdm(all_images[category], desc=category):
            img_name = img_path.stem
            img_output = OUTPUT_ROOT / category / img_name
            img_output.mkdir(parents=True, exist_ok=True)

            try:
                # Load image
                img_tensor = load_image(str(img_path), size=476).to(DEVICE)

                # Extract DINOv2 features
                with torch.no_grad():
                    dinov2_feat = extract_dinov2_features(dinov2, img_tensor)
                    dinov2_feat_up = F.interpolate(dinov2_feat, size=(224, 224), mode='bilinear', align_corners=False)

                # Compute similarity map
                dinov2_sim, _ = compute_similarity_map(dinov2_feat_up)

                # Save similarity map
                np.save(str(img_output / "dinov2_sim.npy"), dinov2_sim)

                # Update stats
                stats_path = img_output / "stats.json"
                with open(stats_path) as f:
                    stats = json.load(f)
                stats["dinov2_shape"] = str(dinov2_feat.shape)
                stats["dinov2_sim_range"] = [float(dinov2_sim.min()), float(dinov2_sim.max())]
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM on {img_name}, clearing cache...")
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

if __name__ == "__main__":
    process_dataset()
    print(f"\nFeatures saved to: {OUTPUT_ROOT}")
