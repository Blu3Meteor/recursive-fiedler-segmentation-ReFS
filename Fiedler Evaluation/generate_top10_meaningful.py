#!/usr/bin/env python3
"""Generate top-10 gallery excluding full-image GT masks (100%)."""

import csv
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

sys.path.insert(0, "d:/Coding/Recursive Segmentation Master's Project/CutLER Implementation/TFSS-CutLER")
sys.path.insert(0, "d:/Coding/Recursive Segmentation Master's Project/CutLER Implementation/TFSS-CutLER/maskcut")

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

def build_sample_panel(image_path, gt_mask, candidate_mask, fixed_size, title):
    image_pil = Image.open(image_path).convert("RGB")
    if fixed_size > 0:
        image_pil, _ = resize_with_aspect_ratio(image_pil, fixed_size)
    image_array = np.asarray(image_pil)
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

def save_gallery(samples, output_path, fixed_size, title):
    if not samples:
        return None
    tiles = [build_sample_panel(s["image_path"], s["gt_mask_fixed"], s["best_candidate_mask"], fixed_size, s["title"])
             for s in samples]
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

# Paths
csv_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc2012_dinov2_patchlevel_final/instance_results.csv")
anno_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/datasets/pascal_voc2012_val_coco.json")
image_root = Path("D:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages")
output_dir = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc2012_dinov2_patchlevel_final")
output_path = output_dir / "top_10_gallery_meaningful.png"

print("Loading results...")
all_results = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_results.append(row)

print(f"Loaded {len(all_results)} instances")

# Load COCO
print("Loading COCO annotations...")
coco = COCO(str(anno_path))

# Filter out instances with 100% GT mask coverage
print("Filtering instances with GT mask coverage < 100%...")
filtered_results = []

for result in all_results:
    try:
        ann_id = int(result["annotation_id"])
        image_id = int(result["image_id"])
        ann = coco.loadAnns([ann_id])[0]
        gt_mask = coco.annToMask(ann).astype(bool)

        # Calculate GT mask coverage
        coverage = gt_mask.sum() / gt_mask.size

        # Keep if not 100% coverage
        if coverage < 0.99:  # Use 0.99 to handle floating point
            result["coverage"] = coverage
            result["iou"] = float(result["best_iou"])
            filtered_results.append(result)
    except Exception as e:
        continue

print(f"After filtering: {len(filtered_results)} instances with GT coverage < 100%")

# Sort by IoU and get top 10
filtered_results_sorted = sorted(filtered_results, key=lambda x: x["iou"], reverse=True)
top_10 = filtered_results_sorted[:10]

print(f"\nTop 10 instances (GT coverage < 100%):")
for i, r in enumerate(top_10, 1):
    print(f"  {i}. image_id={r['image_id']}, annotation_id={r['annotation_id']}, IoU={r['iou']:.4f}, GT_coverage={r['coverage']:.1%}")

# Generate gallery
print("\nGenerating gallery...")
print("Loading backbone...")
backbone = build_backbone("small", "k", 14, None, dino_version="v2", debiased=False)

class ArgNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

samples = []
for result in top_10:
    try:
        image_id = int(result["image_id"])
        ann_id = int(result["annotation_id"])
        iou = result["iou"]

        image_info = coco.loadImgs([image_id])[0]
        image_path = image_root / image_info["file_name"]
        if not image_path.exists():
            print(f"  Skipping {image_path} (not found)")
            continue

        ann = coco.loadAnns([ann_id])[0]
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
            tau=0.15,
            max_depth=9,
            max_nodes=255,
            min_region_patches=6,
            min_split_ratio=0.05,
            noise_threshold=3.0,
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

        category_name = result["category_name"]
        samples.append({
            "image_path": str(image_path),
            "gt_mask_fixed": gt_mask_resized,
            "best_candidate_mask": best_mask,
            "title": f'{category_name} | img {image_id} | IoU {iou:.3f} | GT {result["coverage"]:.0%}',
        })
        print(f"  ✓ Processed {category_name} (IoU {iou:.3f})")

    except Exception as e:
        print(f"  ✗ Failed to process: {e}")
        continue

# Save gallery
if samples:
    save_gallery(samples, str(output_path), 476, "PASCAL VOC 2012 DINOv2 Patch-Level Top 10 (GT < 100%)")
    print(f"\n✓ Saved gallery with {len(samples)} instances to {output_path}")
else:
    print("\n✗ No samples generated")
