#!/usr/bin/env python3
"""Debug script to visualize mask mismatch for image 15291916."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

sys.path.insert(0, "CutLER Implementation/TFSS-CutLER")
sys.path.insert(0, "CutLER Implementation/TFSS-CutLER/maskcut")

from fiedler_class_transfer import resize_with_aspect_ratio, build_backbone, overlay_mask
from recursive_fiedler import collect_recursive_candidates, upsample_mask


def mask_iou(mask_a, mask_b):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(mask_a, mask_b).sum()
    intersection = np.logical_and(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


class ArgNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


print("Loading annotations...")
pascal_coco = COCO("Fiedler Evaluation/datasets/pascal_voc2012_val_coco.json")

print("Finding image 15291916...")
coco_img = pascal_coco.loadImgs([15291916])[0]
print(f"Image: {coco_img['file_name']}")
print(f"Original size: {coco_img['width']} x {coco_img['height']}")

# Get ALL cow annotations (class-level, not instance-level!)
anns = pascal_coco.loadAnns(pascal_coco.getAnnIds(imgIds=[15291916]))
cow_anns = [a for a in anns if a["category_id"] == 10]
if not cow_anns:
    print("No cow annotations found!")
    sys.exit(1)

print(f"Found {len(cow_anns)} cow instances in image 15291916")

# Union all cow instances (class-level mask)
cow_masks = [pascal_coco.annToMask(a).astype(bool) for a in cow_anns]
gt_mask_orig = np.logical_or.reduce(cow_masks).astype(bool)

print(f"\nOriginal GT CLASS mask (union of {len(cow_anns)} instances):")
print(f"  Shape: {gt_mask_orig.shape}")
print(f"  Coverage: {gt_mask_orig.sum() / (gt_mask_orig.shape[0] * gt_mask_orig.shape[1]) * 100:.1f}%")
print(f"  Pixels: {gt_mask_orig.sum()}")

# Load and resize image
image_path = Path("Fiedler Evaluation/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages") / coco_img['file_name']
image_pil = Image.open(image_path).convert("RGB")
image_resized, resize_metadata = resize_with_aspect_ratio(image_pil, 476)

print(f"\nResize metadata:")
print(f"  Scale: {resize_metadata['scale']}")
print(f"  Resized dims: {resize_metadata['resized_w']} x {resize_metadata['resized_h']}")
print(f"  Padding: top={resize_metadata['pad_top']}, left={resize_metadata['pad_left']}")

# Resize GT mask
gt_mask_resized = np.zeros((476, 476), dtype=bool)
orig_h, orig_w = gt_mask_orig.shape
pad_top = resize_metadata.get("pad_top", 0)
pad_left = resize_metadata.get("pad_left", 0)
resized_h = resize_metadata.get("resized_h", orig_h)
resized_w = resize_metadata.get("resized_w", orig_w)
scale = resize_metadata.get("scale", 1.0)

if scale != 1.0:
    downscaled = Image.fromarray(gt_mask_orig.astype(np.uint8) * 255)
    downscaled = downscaled.resize((resized_w, resized_h), Image.NEAREST)
    gt_mask_resized_temp = np.asarray(downscaled) > 128
else:
    gt_mask_resized_temp = gt_mask_orig.copy()

gt_mask_resized[pad_top:pad_top+resized_h, pad_left:pad_left+resized_w] = gt_mask_resized_temp[:resized_h, :resized_w]

print(f"\nResized GT mask (476x476):")
print(f"  Coverage: {gt_mask_resized.sum() / (476*476) * 100:.1f}%")
print(f"  Pixels: {gt_mask_resized.sum()}")

# Get candidates
print("\nRunning Fiedler to get candidates...")
backbone = build_backbone("small", "k", 14, pretrain_path=None, dino_version="v2")

tree_args = ArgNamespace(
    image_path=str(image_path),
    resize_metadata=resize_metadata,
    vit_arch="small",
    vit_feat="k",
    patch_size=14,
    pretrain_path=None,
    dino_version="v2",
    debiased=False,
    fixed_size=476,
    tau=0.0,
    max_depth=float('inf'),
    max_nodes=float('inf'),
    min_region_patches=5,
    min_split_ratio=0.0,
    noise_threshold=float('inf'),
    unlimited_depth=True,
    unlimited_nodes=True,
    unlimited_noise=True,
    unlimited_imbalance=True,
    cpu=False,
)

tree = collect_recursive_candidates(tree_args, backbone=backbone)
candidate_masks = [upsample_mask(node["region_mask"], 476) for node in tree["nodes"]]

print(f"Generated {len(candidate_masks)} candidate masks")

# Find node with ID 6 (from CSV: best_candidate_id = 6)
target_node_id = 6
target_node = None
target_mask_idx = None
for idx, node in enumerate(tree["nodes"]):
    if node.get("id") == target_node_id:
        target_node = node
        target_mask_idx = idx
        break

# Find all IoU scores for all nodes
print("\nComputing IoU for all candidate nodes...")
all_ious = []
for idx, candidate_mask in enumerate(candidate_masks):
    iou = mask_iou(candidate_mask, gt_mask_resized)
    node = tree["nodes"][idx]
    all_ious.append({
        'idx': idx,
        'node_id': node.get('id'),
        'depth': node.get('depth'),
        'status': node.get('status'),
        'iou': iou,
        'coverage': candidate_mask.sum() / (476*476),
    })

# Sort by IoU descending
all_ious_sorted = sorted(all_ious, key=lambda x: x['iou'], reverse=True)

print("\nTop 15 nodes by IoU:")
for i, entry in enumerate(all_ious_sorted[:15], 1):
    print(f"  [{i:2d}] node_id={entry['node_id']:3d} depth={entry['depth']} status={entry['status']:6s} IoU={entry['iou']:.4f} coverage={entry['coverage']*100:.1f}%")

# Check if any node achieves close to 0.9337
target_iou = 0.933692572713942
close_nodes = [e for e in all_ious if abs(e['iou'] - target_iou) < 0.01]
if close_nodes:
    print(f"\nFound {len(close_nodes)} nodes with IoU close to {target_iou:.4f}:")
    for e in close_nodes:
        print(f"  node_id={e['node_id']} IoU={e['iou']:.4f}")
else:
    print(f"\nNo nodes found with IoU close to {target_iou:.4f}")
    print(f"Max IoU found: {all_ious_sorted[0]['iou']:.4f}")

# Use best match
best_iou = all_ious_sorted[0]['iou']
best_mask_idx = all_ious_sorted[0]['idx']
best_mask = candidate_masks[best_mask_idx]

print(f"\nBest candidate (index {best_mask_idx}):")
print(f"  IoU: {best_iou:.4f}")
print(f"  Coverage: {best_mask.sum() / (476*476) * 100:.1f}%")
print(f"  Pixels: {best_mask.sum()}")

# Analyze the overlap
intersection = np.logical_and(gt_mask_resized, best_mask).sum()
union = np.logical_or(gt_mask_resized, best_mask).sum()
gt_only = (gt_mask_resized & ~best_mask).sum()
candidate_only = (best_mask & ~gt_mask_resized).sum()

print(f"\nMask analysis:")
print(f"  GT pixels: {gt_mask_resized.sum()}")
print(f"  Candidate pixels: {best_mask.sum()}")
print(f"  Intersection: {intersection}")
print(f"  Union: {union}")
print(f"  GT-only: {gt_only}")
print(f"  Candidate-only: {candidate_only}")
print(f"  IoU = {intersection} / {union} = {intersection/union:.4f}")

# Save visualizations
print("\nSaving visualizations...")
img_array = np.asarray(image_resized)

# GT mask visualization
gt_viz = overlay_mask(img_array, gt_mask_resized, color=(230, 65, 65), alpha=0.5)
gt_img = Image.fromarray(gt_viz)
gt_img.save("Fiedler Evaluation/debug_gt_mask.png")
print("  GT mask: debug_gt_mask.png")

# Candidate mask visualization
cand_viz = overlay_mask(img_array, best_mask, color=(70, 200, 110), alpha=0.5)
cand_img = Image.fromarray(cand_viz)
cand_img.save("Fiedler Evaluation/debug_candidate_mask.png")
print("  Candidate mask: debug_candidate_mask.png")

# Overlap visualization (red=GT-only, green=candidate-only, yellow=both)
overlap = np.zeros_like(img_array)
overlap[gt_mask_resized & ~best_mask] = [255, 0, 0]  # Red: GT only
overlap[best_mask & ~gt_mask_resized] = [0, 255, 0]  # Green: candidate only
overlap[gt_mask_resized & best_mask] = [255, 255, 0]  # Yellow: both
overlap_img = Image.fromarray(overlap)
overlap_img.save("Fiedler Evaluation/debug_overlap.png")
print("  Overlap map: debug_overlap.png")

print("\nDone! Check debug_*.png for visualizations.")
