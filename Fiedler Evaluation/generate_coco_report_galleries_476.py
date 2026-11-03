#!/usr/bin/env python3
"""Generate COCO val2017 DINOv2 evaluation report and galleries (476 resolution)."""

import json
import csv
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

sys.path.insert(0, "d:/Coding/Recursive Segmentation Master's Project/CutLER Implementation/TFSS-CutLER")
sys.path.insert(0, "d:/Coding/Recursive Segmentation Master's Project/CutLER Implementation/TFSS-CutLER/maskcut")

from fiedler_class_transfer import build_backbone, overlay_mask, resize_with_aspect_ratio
from recursive_fiedler import collect_recursive_candidates, upsample_mask

# Paths
csv_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/COCO/coco_val2017_dinov2_patchlevel_476/instance_results.csv")
anno_path = Path("d:/Coding/Recursive Segmentation Master's Project/CutLER Implementation/TFSS-CutLER/datasets/coco/annotations/instances_val2017.json")
image_root = Path("d:/Coding/Recursive Segmentation Master's Project/CutLER Implementation/TFSS-CutLER/datasets/coco/val2017")
output_dir = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/COCO/coco_val2017_dinov2_patchlevel_476")
summary_json = output_dir / "summary.json"
report_md = output_dir / "report.md"
histogram_path = output_dir / "overall_iou_histogram.png"
class_histogram_path = output_dir / "class_mean_iou.png"

print("Loading results from CSV...")
all_results = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_results.append(row)

print(f"Loaded {len(all_results)} results")

# Compute statistics
class_scores = defaultdict(list)
per_class_stats = defaultdict(lambda: {"count": 0, "scores": []})
image_id_to_results = defaultdict(list)
seen_images = set()

for result in all_results:
    image_id = int(result["image_id"])
    category_name = result["category_name"]
    best_iou = float(result["best_iou"])

    class_scores[category_name].append(best_iou)
    per_class_stats[category_name]["count"] += 1
    per_class_stats[category_name]["scores"].append(best_iou)
    image_id_to_results[image_id].append(best_iou)
    seen_images.add(image_id)

# Image-level summary
image_summaries = []
for image_id, ious in image_id_to_results.items():
    image_summaries.append({
        "image_id": image_id,
        "num_instances": len(ious),
        "best_instance_iou": max(ious),
        "mean_instance_iou": float(np.mean(ious)),
    })

# Aggregate statistics
all_ious = [float(r["best_iou"]) for r in all_results]
per_image_best_ious = [s["best_instance_iou"] for s in image_summaries]

mean_best_iou = float(np.mean(all_ious))
median_best_iou = float(np.median(all_ious))
min_best_iou = float(np.min(all_ious))
max_best_iou = float(np.max(all_ious))
mean_best_iou_per_image = float(np.mean(per_image_best_ious))
median_best_iou_per_image = float(np.median(per_image_best_ious))

# Per-class summary
per_class_summary = {}
for class_name, stats in per_class_stats.items():
    scores = stats["scores"]
    per_class_summary[class_name] = {
        "count": int(stats["count"]),
        "mean_iou": float(np.mean(scores)),
        "median_iou": float(np.median(scores)),
        "min_iou": float(np.min(scores)),
        "max_iou": float(np.max(scores)),
    }

# Write summary.json
summary = {
    "dataset_name": "coco_val2017_dinov2_patchlevel_476",
    "fixed_size": 476,
    "patch_size": 14,
    "max_depth": 9,
    "max_nodes": 255,
    "min_region_patches": 6,
    "num_images": len(seen_images),
    "num_instances": len(all_results),
    "mean_best_iou": mean_best_iou,
    "median_best_iou": median_best_iou,
    "min_best_iou": min_best_iou,
    "max_best_iou": max_best_iou,
    "mean_best_iou_per_image": mean_best_iou_per_image,
    "median_best_iou_per_image": median_best_iou_per_image,
    "per_class": per_class_summary,
}

with open(summary_json, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print(f"Saved summary to {summary_json}")

# Write report.md
with open(report_md, 'w', encoding='utf-8') as f:
    f.write("# COCO val2017 DINOv2 Patch-Level Recursive Fiedler Evaluation\n\n")
    f.write("## Configuration\n")
    f.write("- Dataset: COCO val2017\n")
    f.write("- Model: DINOv2-small\n")
    f.write("- Fixed size: 476\n")
    f.write("- Patch size: 14\n")
    f.write("- Max depth: 9\n")
    f.write("- Max nodes: 255\n")
    f.write("- Min region patches: 6\n\n")
    f.write("## Results\n")
    f.write(f"- Images: {len(seen_images)}\n")
    f.write(f"- Instances: {len(all_results)}\n")
    f.write(f"- Mean best IoU: {mean_best_iou:.4f}\n")
    f.write(f"- Median best IoU: {median_best_iou:.4f}\n")
    f.write(f"- Min best IoU: {min_best_iou:.4f}\n")
    f.write(f"- Max best IoU: {max_best_iou:.4f}\n")
    f.write(f"- Mean best IoU per image: {mean_best_iou_per_image:.4f}\n")
    f.write(f"- Median best IoU per image: {median_best_iou_per_image:.4f}\n\n")
    f.write("## Outputs\n")
    f.write("- Results CSV: instance_results.csv\n")
    f.write("- Summary JSON: summary.json\n")
    f.write("- Overall histogram: overall_iou_histogram.png\n")
    f.write("- Class histogram: class_mean_iou.png\n")
    f.write("- Top 10 gallery: top_10_gallery.png\n")
    f.write("- Bottom 10 gallery: bottom_10_gallery.png\n")

print(f"Saved report to {report_md}")

# Generate plots
print("Generating visualizations...")

# Overall histogram
plt.figure(figsize=(8, 5))
plt.hist(all_ious, bins=np.linspace(0.0, 1.0, 21), color="#72b7b2", edgecolor="white")
plt.xlim(0.0, 1.0)
plt.xlabel("Best IoU")
plt.ylabel("Count")
plt.title("COCO val2017 DINOv2 Patch-Level Best IoU Distribution (476 resolution)")
plt.tight_layout()
plt.savefig(histogram_path, dpi=160)
plt.close()
print(f"Saved histogram to {histogram_path}")

# Class histogram
ordered = sorted(per_class_summary.items(), key=lambda item: (-item[1]["count"], item[0]))
labels = [name for name, _ in ordered]
means = [stats["mean_iou"] for _, stats in ordered]

plt.figure(figsize=(max(12, len(labels) * 0.25), 6))
plt.bar(range(len(labels)), means, color="#72b7b2")
plt.ylim(0.0, 1.0)
plt.ylabel("Mean best IoU")
plt.title("COCO val2017 DINOv2 Patch-Level Mean Best IoU by Class (476 resolution)")
plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
plt.tight_layout()
plt.savefig(class_histogram_path, dpi=160)
plt.close()
print(f"Saved class histogram to {class_histogram_path}")

# Load COCO for gallery generation
print("\nGenerating galleries...")
print("Loading COCO annotations...")
coco = COCO(str(anno_path))

# Sort by IoU and get top/bottom 10
results_sorted = sorted(all_results, key=lambda x: float(x["best_iou"]), reverse=True)
top_10 = results_sorted[:10]
bottom_10 = results_sorted[-10:]

print(f"\nTop 10 instances:")
for i, r in enumerate(top_10, 1):
    print(f"  {i}. image_id={r['image_id']}, IoU={float(r['best_iou']):.4f}")

print(f"\nBottom 10 instances:")
for i, r in enumerate(bottom_10, 1):
    print(f"  {i}. image_id={r['image_id']}, IoU={float(r['best_iou']):.4f}")

# Load backbone for gallery generation
print("Loading backbone...")
backbone = build_backbone("small", "k", 14, None, dino_version="v2", debiased=False)

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

# Generate top 10 gallery
samples_top = []
for result in top_10:
    try:
        image_id = int(result["image_id"])
        ann_id = int(result["annotation_id"])
        iou = float(result["best_iou"])

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
        samples_top.append({
            "image_path": str(image_path),
            "gt_mask": gt_mask_resized,
            "best_candidate_mask": best_mask,
            "title": f'{category_name} | img {image_id} | IoU {iou:.3f}',
        })
        print(f"  ✓ Processed top {category_name} (IoU {iou:.3f})")

    except Exception as e:
        print(f"  ✗ Failed to process top: {e}")
        continue

if samples_top:
    top_gallery_path = output_dir / "top_10_gallery.png"
    save_gallery(samples_top, str(top_gallery_path), "COCO val2017 DINOv2 Patch-Level Top 10 (476)")
    print(f"\n✓ Saved top 10 gallery with {len(samples_top)} instances to {top_gallery_path}")
else:
    print("\n✗ No top 10 samples generated")

# Generate bottom 10 gallery
samples_bottom = []
for result in bottom_10:
    try:
        image_id = int(result["image_id"])
        ann_id = int(result["annotation_id"])
        iou = float(result["best_iou"])

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
        samples_bottom.append({
            "image_path": str(image_path),
            "gt_mask": gt_mask_resized,
            "best_candidate_mask": best_mask,
            "title": f'{category_name} | img {image_id} | IoU {iou:.3f}',
        })
        print(f"  ✓ Processed bottom {category_name} (IoU {iou:.3f})")

    except Exception as e:
        print(f"  ✗ Failed to process bottom: {e}")
        continue

if samples_bottom:
    bottom_gallery_path = output_dir / "bottom_10_gallery.png"
    save_gallery(samples_bottom, str(bottom_gallery_path), "COCO val2017 DINOv2 Patch-Level Bottom 10 (476)")
    print(f"\n✓ Saved bottom 10 gallery with {len(samples_bottom)} instances to {bottom_gallery_path}")
else:
    print("\n✗ No bottom 10 samples generated")

print("\n" + "="*70)
print("COCO EVALUATION COMPLETE (476 resolution)")
print("="*70)
print(f"Images: {len(seen_images)}")
print(f"Instances: {len(all_results)}")
print(f"Mean best IoU: {mean_best_iou:.4f}")
print(f"Median best IoU: {median_best_iou:.4f}")
print(f"Output directory: {output_dir}")
print("="*70)
