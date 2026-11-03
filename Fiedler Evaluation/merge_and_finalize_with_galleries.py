#!/usr/bin/env python3
"""Merge results, generate summary, and create top/bottom galleries."""

import json
import csv
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
CUTLER_ROOT = WORKSPACE_ROOT / "CutLER Implementation" / "TFSS-CutLER"
MASKCUT_ROOT = CUTLER_ROOT / "maskcut"
if str(CUTLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTLER_ROOT))
if str(MASKCUT_ROOT) not in sys.path:
    sys.path.insert(0, str(MASKCUT_ROOT))

from fiedler_class_transfer import build_backbone, overlay_mask, resize_with_aspect_ratio
from recursive_fiedler import collect_recursive_candidates, upsample_mask
from pycocotools.coco import COCO

def mask_iou(mask_a, mask_b):
    """Compute IoU between two binary masks."""
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection) / float(union)

def overlay_with_label(image_array, mask, color, label_text):
    """Overlay mask on image with label."""
    canvas = overlay_mask(image_array, mask, color=color, alpha=0.5)
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    draw.rectangle((0, 0, pil_canvas.width, 28), fill=(255, 255, 255))
    draw.text((6, 6), label_text, fill=(0, 0, 0))
    return pil_canvas

def build_sample_panel(image_path, gt_mask, candidate_mask, fixed_size, title):
    """Build visualization panel with ground truth and best candidate."""
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
    """Save gallery of samples."""
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
original_csv = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc2012_dinov2_patchlevel/instance_results.csv")
retry_csv = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc2012_dinov2_patchlevel_failed_retry/instance_results.csv")
final_output_dir = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc2012_dinov2_patchlevel_final")
anno_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/datasets/pascal_voc2012_val_coco.json")
image_root = Path("D:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages")

final_csv = final_output_dir / "instance_results.csv"
summary_json = final_output_dir / "summary.json"
report_md = final_output_dir / "report.md"
histogram_path = final_output_dir / "overall_iou_histogram.png"
class_overview_path = final_output_dir / "class_mean_iou.png"
top_gallery_path = final_output_dir / "top_10_gallery.png"
bottom_gallery_path = final_output_dir / "bottom_10_gallery.png"

os.makedirs(final_output_dir, exist_ok=True)

print("="*70)
print("MERGING RESULTS AND GENERATING FINAL EVALUATION")
print("="*70)

# Merge CSVs
print("\n[1/5] Merging results from original and retry runs...")
all_results = []
seen_rows = set()

for csv_file in [original_csv, retry_csv]:
    if not csv_file.exists():
        print(f"Warning: {csv_file} not found, skipping")
        continue
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["image_id"], row["annotation_id"])
            if key not in seen_rows:
                all_results.append(row)
                seen_rows.add(key)

print(f"Total merged instances: {len(all_results)}")

# Write merged CSV
print("Writing merged results to CSV...")
if all_results:
    fieldnames = list(all_results[0].keys())
    with open(final_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

# Compute statistics
print("[2/5] Computing statistics...")
class_scores = defaultdict(list)
per_class_stats = defaultdict(lambda: {"count": 0, "scores": []})
image_summaries = []
seen_images = set()

for result in all_results:
    image_id = int(result["image_id"])
    category_name = result["category_name"]
    best_iou = float(result["best_iou"])
    class_scores[category_name].append(best_iou)
    per_class_stats[category_name]["count"] += 1
    per_class_stats[category_name]["scores"].append(best_iou)
    seen_images.add(image_id)

image_id_to_results = defaultdict(list)
for result in all_results:
    image_id = int(result["image_id"])
    image_id_to_results[image_id].append(float(result["best_iou"]))

for image_id, ious in image_id_to_results.items():
    image_summaries.append({
        "image_id": image_id,
        "num_instances": len(ious),
        "best_instance_iou": max(ious),
        "mean_instance_iou": float(np.mean(ious)),
    })

# Aggregate stats
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
print("Writing summary JSON...")
summary = {
    "dataset_name": "pascal_voc2012_dinov2_patchlevel_complete",
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

# Write report.md
with open(report_md, 'w', encoding='utf-8') as f:
    f.write("# PASCAL VOC 2012 DINOv2 Patch-Level Recursive Fiedler Evaluation (Complete)\n\n")
    f.write("## Configuration\n- Model: DINOv2-small\n- Fixed size: 476\n- Patch size: 14\n")
    f.write("- Max depth: 9\n- Max nodes: 255\n- Min region patches: 6\n\n## Results\n")
    f.write(f"- Images: {len(seen_images)}\n- Instances: {len(all_results)}\n")
    f.write(f"- Mean best IoU: {mean_best_iou:.4f}\n- Median best IoU: {median_best_iou:.4f}\n")
    f.write(f"- Mean best IoU per image: {mean_best_iou_per_image:.4f}\n\n## Outputs\n")
    f.write("- Results CSV: instance_results.csv\n- Summary JSON: summary.json\n")
    f.write("- Overall histogram: overall_iou_histogram.png\n- Class overview: class_mean_iou.png\n")
    f.write("- Top 10 gallery: top_10_gallery.png\n- Bottom 10 gallery: bottom_10_gallery.png\n")

# Generate plots
print("[3/5] Generating plots...")
plt.figure(figsize=(8, 5))
plt.hist(all_ious, bins=np.linspace(0.0, 1.0, 21), color="#72b7b2", edgecolor="white")
plt.xlim(0.0, 1.0)
plt.xlabel("Best IoU")
plt.ylabel("Count")
plt.title("PASCAL VOC 2012 DINOv2 Patch-Level Best IoU Distribution")
plt.tight_layout()
plt.savefig(histogram_path, dpi=160)
plt.close()

ordered = sorted(per_class_summary.items(), key=lambda item: (-item[1]["count"], item[0]))
labels = [name for name, _ in ordered]
means = [stats["mean_iou"] for _, stats in ordered]
plt.figure(figsize=(max(10, len(labels) * 0.22), 6))
plt.bar(range(len(labels)), means, color="#72b7b2")
plt.ylim(0.0, 1.0)
plt.ylabel("Mean best IoU")
plt.title("PASCAL VOC 2012 DINOv2 Patch-Level Mean Best IoU by Class")
plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
plt.tight_layout()
plt.savefig(class_overview_path, dpi=160)
plt.close()

# Generate galleries
print("[4/5] Generating top 10 and bottom 10 galleries...")
print("Loading backbone...")
backbone = build_backbone("small", "k", 14, None, dino_version="v2", debiased=False)

print("Loading COCO annotations...")
coco = COCO(str(anno_path))

# Get top and bottom 10
results_with_iou = [(float(r["best_iou"]), int(r["image_id"]), int(r["annotation_id"]), r)
                     for r in all_results]
results_sorted = sorted(results_with_iou, key=lambda x: x[0], reverse=True)
top_10_rows = results_sorted[:10]
bottom_10_rows = results_sorted[-10:]

def generate_gallery_samples(result_rows, title_prefix):
    """Generate gallery samples from result rows."""
    samples = []
    for iou, image_id, annotation_id, result_row in result_rows:
        try:
            image_info = coco.loadImgs([image_id])[0]
            image_path = image_root / image_info["file_name"]
            if not image_path.exists():
                print(f"  Warning: Image not found {image_path}")
                continue

            ann = coco.loadAnns([annotation_id])[0]
            gt_mask = coco.annToMask(ann).astype(bool)

            # Load and prepare image
            image_pil = Image.open(image_path).convert("RGB")
            image_resized, resize_metadata = resize_with_aspect_ratio(image_pil, 476)

            # Get candidate tree
            class ArgNamespace:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

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
            best_iou = -1.0
            best_mask = None
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
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_mask = candidate_mask

            if best_mask is None:
                best_mask = candidate_masks[0] if candidate_masks else np.zeros((476, 476), dtype=bool)

            category_name = result_row["category_name"]
            samples.append({
                "image_path": str(image_path),
                "gt_mask_fixed": gt_mask_resized,
                "best_candidate_mask": best_mask,
                "title": f'{category_name} | img {image_id} | IoU {iou:.3f}',
            })
        except Exception as e:
            print(f"  Warning: Failed to process instance {annotation_id}: {e}")
            continue

    return samples

print(f"  Processing top 10...")
top_samples = generate_gallery_samples(top_10_rows, "top")
print(f"  Processing bottom 10...")
bottom_samples = generate_gallery_samples(bottom_10_rows, "bottom")

# Save galleries
print("[5/5] Saving galleries...")
if top_samples:
    save_gallery(top_samples, str(top_gallery_path), 476, "PASCAL VOC 2012 DINOv2 Patch-Level Top 10 Instances")
    print(f"Saved top gallery with {len(top_samples)} instances")

if bottom_samples:
    save_gallery(bottom_samples, str(bottom_gallery_path), 476, "PASCAL VOC 2012 DINOv2 Patch-Level Bottom 10 Instances")
    print(f"Saved bottom gallery with {len(bottom_samples)} instances")

print("\n" + "="*70)
print("COMPLETE EVALUATION FINISHED")
print("="*70)
print(f"Images: {len(seen_images)}")
print(f"Instances: {len(all_results)}")
print(f"Mean best IoU: {mean_best_iou:.4f}")
print(f"Median best IoU: {median_best_iou:.4f}")
print(f"Output directory: {final_output_dir}")
print("="*70)
