#!/usr/bin/env python3
"""Generate PASCAL gallery filtering by GT mask area coverage (<80% of image)."""

import json
import csv
import sys
from pathlib import Path

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


print("\n" + "="*80)
print("PASCAL VOC 2012: Filtering by GT mask coverage (<80% of image)")
print("="*80)

pascal_csv = "Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc_dinov2_min_patches_only_instance/instance_results.csv"
pascal_anno = "Fiedler Evaluation/datasets/pascal_voc2012_val_coco.json"
pascal_image_root = Path("Fiedler Evaluation/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages")
pascal_output_dir = Path("Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc_dinov2_min_patches_only_instance")

print(f"\nLoading PASCAL annotations from {pascal_anno}...")
pascal_coco = COCO(pascal_anno)

print(f"Loading PASCAL results from {pascal_csv}...")
pascal_results = []
with open(pascal_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pascal_results.append(row)

# Filter by GT mask coverage
print("\nFiltering instances by GT mask coverage (<80%)...")
filtered_results = []
for result in pascal_results:
    image_id = int(result["image_id"])
    cat_id = int(result["category_id"])
    iou = float(result["best_iou"])

    try:
        image_info = pascal_coco.loadImgs([image_id])[0]

        # Find annotation for this image+category
        anns = pascal_coco.loadAnns(pascal_coco.getAnnIds(imgIds=[image_id]))
        matching_anns = [a for a in anns if a["category_id"] == cat_id and not a.get("iscrowd", False)]
        if not matching_anns:
            continue

        ann = matching_anns[0]
        gt_mask = pascal_coco.annToMask(ann).astype(bool)

        # Calculate coverage percentage
        total_pixels = gt_mask.shape[0] * gt_mask.shape[1]
        mask_pixels = gt_mask.sum()
        coverage = mask_pixels / total_pixels

        if coverage < 0.80:
            filtered_results.append({
                **result,
                "coverage": coverage,
                "mask_pixels": int(mask_pixels),
                "total_pixels": int(total_pixels),
            })
    except Exception as e:
        continue

print(f"Found {len(filtered_results)} instances with coverage <80%")

# Sort by best_iou descending
filtered_sorted = sorted(filtered_results, key=lambda x: float(x["best_iou"]), reverse=True)

# Show top candidates
print("\nTop 15 candidates by IoU (coverage <80%):")
for i, r in enumerate(filtered_sorted[:15], 1):
    print(f"  [{i:2d}] {r['category_name']:20s} | IoU {float(r['best_iou']):.4f} | coverage {r['coverage']*100:.1f}%")

# Select top 10
selected = filtered_sorted[:10]
print(f"\nSelecting top 10 from filtered results...")

print("\nBuilding DINOv2 backbone...")
backbone = build_backbone("small", "k", 14, pretrain_path=None, dino_version="v2")

print(f"Processing {len(selected)} PASCAL instances...")
samples_pascal = []
for idx, result in enumerate(selected, 1):
    try:
        image_id = int(result["image_id"])
        iou = float(result["best_iou"])
        category_name = result["category_name"]
        cat_id = int(result["category_id"])
        coverage = result["coverage"]

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
            "title": f'{category_name} | img {image_id} | IoU {iou:.3f} | cov {coverage*100:.0f}%',
        })
        print(f"  [{idx}/10] OK {category_name} (IoU {iou:.3f}, coverage {coverage*100:.1f}%)")

    except Exception as e:
        print(f"  [{idx}/10] FAIL {e}")
        import traceback
        traceback.print_exc()
        continue

if samples_pascal:
    pascal_gallery_path = pascal_output_dir / "top_10_filtered_by_coverage.png"
    save_gallery(samples_pascal, str(pascal_gallery_path), "PASCAL VOC DINOv2 Filtered by Coverage <80%")
    print(f"\nOK Saved PASCAL gallery to {pascal_gallery_path}")
else:
    print("\nFAIL No PASCAL samples generated")

print("\n" + "="*80)
print("Done!")
print("="*80)
