#!/usr/bin/env python3
"""Generate improved top 10 galleries using ranks 8-17 (better visual variety)."""

import json
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

sys.path.insert(0, "CutLER Implementation/TFSS-CutLER")
sys.path.insert(0, "CutLER Implementation/TFSS-CutLER/maskcut")

from fiedler_class_transfer import build_backbone, overlay_mask, resize_with_aspect_ratio
from recursive_fiedler import collect_recursive_candidates, upsample_mask


def mask_iou(mask_a, mask_b):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection) / float(union)


def overlay_with_label(image_array, mask, color, label_text):
    canvas = overlay_mask(image_array, mask, color=color, alpha=0.5)
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    draw.rectangle((0, 0, pil_canvas.width, 28), fill=(255, 255, 255))
    draw.text((6, 6), label_text, fill=(0, 0, 0))
    return pil_canvas


def build_sample_panel(image_path, gt_mask, candidate_mask, title):
    image_pil = Image.open(image_path).convert("RGB")
    image_resized, _ = resize_with_aspect_ratio(image_pil, 476)
    image_array = np.asarray(image_resized)
    display_h, display_w = image_array.shape[:2]

    gt_panel = overlay_with_label(image_array, gt_mask, (230, 65, 65), "ground truth")
    candidate_panel = overlay_with_label(image_array, candidate_mask, (70, 200, 110), "best candidate")

    panels = [image_pil, gt_panel, candidate_panel]
    canvas = Image.new("RGB", (display_w * len(panels), display_h + 44), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 44), fill=(248, 248, 248))
    draw.text((8, 6), title, fill=(0, 0, 0))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * display_w, 44))
    return canvas


def save_gallery(samples, output_path, title):
    if not samples:
        return None
    tiles = [build_sample_panel(s["image_path"], s["gt_mask"], s["best_candidate_mask"], s["title"]) for s in samples]
    if not tiles:
        return None
    cols = 2 if len(tiles) > 1 else 1
    rows = (len(tiles) + cols - 1) // cols
    tile_widths = [tile.size[0] for tile in tiles]
    tile_heights = [tile.size[1] for tile in tiles]
    max_width = max(tile_widths)
    max_height = max(tile_heights)

    canvas = Image.new("RGB", (max_width * cols, max_height * rows + 44), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 44), fill=(248, 248, 248))
    draw.text((8, 6), title, fill=(0, 0, 0))

    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        canvas.paste(tile, (col * max_width, row * max_height + 44))

    canvas.save(output_path, dpi=(160, 160))
    return output_path


class ArgNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Process COCO-20i
print("\n" + "="*80)
print("COCO-20i Instance-Level: Ranks 8-17 Gallery")
print("="*80)

coco_csv = "Fiedler Evaluation/output/02_oracle_proposals/COCO/coco_20i_dinov2_min_patches_only_instance/instance_results.csv"
coco_anno = "Fiedler Evaluation/data/coco20i/coco20i_test.json"
coco_image_root = Path("Fiedler Evaluation/datasets/val2017")
coco_output_dir = Path("Fiedler Evaluation/output/02_oracle_proposals/COCO/coco_20i_dinov2_min_patches_only_instance")

print(f"Loading COCO COCO annotations from {coco_anno}...")
coco = COCO(coco_anno)

print(f"Loading COCO results from {coco_csv}...")
coco_results = []
with open(coco_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        coco_results.append(row)

# Sort by best_iou descending
coco_sorted = sorted(coco_results, key=lambda x: float(x["best_iou"]), reverse=True)

print(f"Total instances: {len(coco_sorted)}")
print(f"Selecting ranks 8-17 (IoU range: {float(coco_sorted[7]['best_iou']):.4f} to {float(coco_sorted[16]['best_iou']):.4f})")

# Extract ranks 8-17
ranks_8_17_coco = coco_sorted[7:17]

print(f"\nBuilding DINOv2 backbone...")
backbone = build_backbone("small", "k", 14, pretrain_path=None, dino_version="v2")

print(f"Processing {len(ranks_8_17_coco)} COCO instances...")
samples_coco = []
for idx, result in enumerate(ranks_8_17_coco, 1):
    try:
        image_id = int(result["image_id"])
        ann_id = int(result["annotation_id"])
        iou = float(result["best_iou"])
        category_name = result["category_name"]

        image_info = coco.loadImgs([image_id])[0]
        image_path = coco_image_root / image_info["file_name"]
        if not image_path.exists():
            print(f"  [{idx}/10] Skipping image {image_id} (not found)")
            continue

        # Get annotation for this instance
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        # Find the annotation with matching category and instance in this image
        # For COCO-20i, we match by category_id and find the right instance
        cat_id = int(result["category_id"])
        matching_anns = [a for a in anns if a["category_id"] == cat_id]
        if not matching_anns:
            print(f"  [{idx}/10] No annotation found for image {image_id} category {cat_id}")
            continue

        # Use the first matching annotation (simplified - assumes one instance per category per image)
        ann = matching_anns[0]
        gt_mask = coco.annToMask(ann).astype(bool)

        # Load and prepare image
        image_pil = Image.open(image_path).convert("RGB")
        image_resized, resize_metadata = resize_with_aspect_ratio(image_pil, 476)

        # Get candidate tree
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

        # Find best matching candidate
        best_mask = None
        best_mask_iou = -1.0
        for candidate_mask in candidate_masks:
            gt_mask_resized = np.zeros((476, 476), dtype=bool)
            orig_h, orig_w = gt_mask.shape
            pad_top = resize_metadata.get("pad_top", 0)
            pad_left = resize_metadata.get("pad_left", 0)
            resized_h = resize_metadata.get("resized_h", orig_h)
            resized_w = resize_metadata.get("resized_w", orig_w)
            scale = resize_metadata.get("scale", 1.0)
            if scale != 1.0:
                downscaled = Image.fromarray(gt_mask.astype(np.uint8) * 255)
                downscaled = downscaled.resize((resized_w, resized_h), Image.NEAREST)
                gt_mask_resized_temp = np.asarray(downscaled) > 128
            else:
                gt_mask_resized_temp = gt_mask.copy()
            gt_mask_resized[pad_top:pad_top+resized_h, pad_left:pad_left+resized_w] = gt_mask_resized_temp[:resized_h, :resized_w]

            iou_val = mask_iou(candidate_mask, gt_mask_resized)
            if iou_val > best_mask_iou:
                best_mask_iou = iou_val
                best_mask = candidate_mask

        if best_mask is None:
            best_mask = candidate_masks[0] if candidate_masks else np.zeros((476, 476), dtype=bool)

        samples_coco.append({
            "image_path": str(image_path),
            "gt_mask": gt_mask_resized,
            "best_candidate_mask": best_mask,
            "title": f'{category_name} | img {image_id} | IoU {iou:.3f}',
        })
        print(f"  [{idx}/10] OK {category_name} (IoU {iou:.3f})")

    except Exception as e:
        print(f"  [{idx}/10] FAIL {e}")
        import traceback
        traceback.print_exc()
        continue

if samples_coco:
    coco_gallery_path = coco_output_dir / "top_10_better_ranks_8_17.png"
    save_gallery(samples_coco, str(coco_gallery_path), "COCO-20i DINOv2 Ranks 8-17 (Better Masks)")
    print(f"\nOK Saved COCO gallery to {coco_gallery_path}")
else:
    print("\nFAIL No COCO samples generated")


# Process PASCAL VOC 2012
print("\n" + "="*80)
print("PASCAL VOC 2012 Instance-Level: Ranks 8-17 Gallery")
print("="*80)

pascal_csv = "Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc_dinov2_min_patches_only_instance/instance_results.csv"
pascal_anno = "Fiedler Evaluation/datasets/pascal_voc2012_val_coco.json"
pascal_image_root = Path("Fiedler Evaluation/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages")
pascal_output_dir = Path("Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc_dinov2_min_patches_only_instance")

print(f"Loading PASCAL annotations from {pascal_anno}...")
pascal_coco = COCO(pascal_anno)

print(f"Loading PASCAL results from {pascal_csv}...")
pascal_results = []
with open(pascal_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pascal_results.append(row)

# Sort by best_iou descending
pascal_sorted = sorted(pascal_results, key=lambda x: float(x["best_iou"]), reverse=True)

print(f"Total instances: {len(pascal_sorted)}")
print(f"Selecting ranks 8-17 (IoU range: {float(pascal_sorted[7]['best_iou']):.4f} to {float(pascal_sorted[16]['best_iou']):.4f})")

ranks_8_17_pascal = pascal_sorted[7:17]

print(f"\nProcessing {len(ranks_8_17_pascal)} PASCAL instances...")
samples_pascal = []
for idx, result in enumerate(ranks_8_17_pascal, 1):
    try:
        image_id = int(result["image_id"])
        iou = float(result["best_iou"])
        category_name = result["category_name"]
        cat_id = int(result["category_id"])

        image_info = pascal_coco.loadImgs([image_id])[0]
        image_path = pascal_image_root / image_info["file_name"]
        if not image_path.exists():
            print(f"  [{idx}/10] Skipping image {image_id} (not found)")
            continue

        # Find annotation for this image+category
        anns = pascal_coco.loadAnns(pascal_coco.getAnnIds(imgIds=[image_id]))
        matching_anns = [a for a in anns if a["category_id"] == cat_id and not a.get("iscrowd", False)]
        if not matching_anns:
            print(f"  [{idx}/10] FAIL No annotation for image {image_id} category {cat_id}")
            continue

        ann = matching_anns[0]
        gt_mask = pascal_coco.annToMask(ann).astype(bool)

        # Load and prepare image
        image_pil = Image.open(image_path).convert("RGB")
        image_resized, resize_metadata = resize_with_aspect_ratio(image_pil, 476)

        # Get candidate tree
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

        # Find best matching candidate
        best_mask = None
        best_mask_iou = -1.0
        for candidate_mask in candidate_masks:
            gt_mask_resized = np.zeros((476, 476), dtype=bool)
            orig_h, orig_w = gt_mask.shape
            pad_top = resize_metadata.get("pad_top", 0)
            pad_left = resize_metadata.get("pad_left", 0)
            resized_h = resize_metadata.get("resized_h", orig_h)
            resized_w = resize_metadata.get("resized_w", orig_w)
            scale = resize_metadata.get("scale", 1.0)
            if scale != 1.0:
                downscaled = Image.fromarray(gt_mask.astype(np.uint8) * 255)
                downscaled = downscaled.resize((resized_w, resized_h), Image.NEAREST)
                gt_mask_resized_temp = np.asarray(downscaled) > 128
            else:
                gt_mask_resized_temp = gt_mask.copy()
            gt_mask_resized[pad_top:pad_top+resized_h, pad_left:pad_left+resized_w] = gt_mask_resized_temp[:resized_h, :resized_w]

            iou_val = mask_iou(candidate_mask, gt_mask_resized)
            if iou_val > best_mask_iou:
                best_mask_iou = iou_val
                best_mask = candidate_mask

        if best_mask is None:
            best_mask = candidate_masks[0] if candidate_masks else np.zeros((476, 476), dtype=bool)

        samples_pascal.append({
            "image_path": str(image_path),
            "gt_mask": gt_mask_resized,
            "best_candidate_mask": best_mask,
            "title": f'{category_name} | img {image_id} | IoU {iou:.3f}',
        })
        print(f"  [{idx}/10] OK {category_name} (IoU {iou:.3f})")

    except Exception as e:
        print(f"  [{idx}/10] FAIL {e}")
        import traceback
        traceback.print_exc()
        continue

if samples_pascal:
    pascal_gallery_path = pascal_output_dir / "top_10_better_ranks_8_17.png"
    save_gallery(samples_pascal, str(pascal_gallery_path), "PASCAL VOC DINOv2 Ranks 8-17 (Better Masks)")
    print(f"\nOK Saved PASCAL gallery to {pascal_gallery_path}")
else:
    print("\nFAIL No PASCAL samples generated")

print("\n" + "="*80)
print("Done!")
print("="*80)
