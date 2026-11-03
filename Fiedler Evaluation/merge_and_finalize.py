#!/usr/bin/env python3
"""Merge results from initial run and failed retry, then generate final summary."""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
original_csv = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc2012_dinov2_patchlevel/instance_results.csv")
retry_csv = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc2012_dinov2_patchlevel_failed_retry/instance_results.csv")
final_output_dir = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc2012_dinov2_patchlevel_final")
final_csv = final_output_dir / "instance_results.csv"
summary_json = final_output_dir / "summary.json"
report_md = final_output_dir / "report.md"
histogram_path = final_output_dir / "overall_iou_histogram.png"
class_overview_path = final_output_dir / "class_mean_iou.png"

os.makedirs(final_output_dir, exist_ok=True)

print("Merging results from original and retry runs...")

# Read original and retry CSVs
all_results = []
seen_rows = set()

for csv_file in [original_csv, retry_csv]:
    if not csv_file.exists():
        print(f"Warning: {csv_file} not found, skipping")
        continue

    print(f"Loading {csv_file}...")
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create unique key to avoid duplicates
            key = (row["image_id"], row["annotation_id"])
            if key not in seen_rows:
                all_results.append(row)
                seen_rows.add(key)

print(f"Total merged instances: {len(all_results)}")

# Write merged CSV
print(f"Writing merged results to {final_csv}...")
if all_results:
    fieldnames = list(all_results[0].keys())
    with open(final_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

# Compute statistics
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

# Image-level summary
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

print(f"Saved summary to {summary_json}")

# Write report.md
with open(report_md, 'w', encoding='utf-8') as f:
    f.write("# PASCAL VOC 2012 DINOv2 Patch-Level Recursive Fiedler Evaluation (Complete)\n\n")
    f.write("## Configuration\n")
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
    f.write("- Class overview: class_mean_iou.png\n")

print(f"Saved report to {report_md}")

# Generate plots
print("Generating visualizations...")

# Overall histogram
plt.figure(figsize=(8, 5))
plt.hist(all_ious, bins=np.linspace(0.0, 1.0, 21), color="#72b7b2", edgecolor="white")
plt.xlim(0.0, 1.0)
plt.xlabel("Best IoU")
plt.ylabel("Count")
plt.title("PASCAL VOC 2012 DINOv2 Patch-Level Best IoU Distribution")
plt.tight_layout()
plt.savefig(histogram_path, dpi=160)
plt.close()
print(f"Saved histogram to {histogram_path}")

# Class overview
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
print(f"Saved class overview to {class_overview_path}")

print("\n" + "="*70)
print("MERGE AND FINALIZATION COMPLETE")
print("="*70)
print(f"Images: {len(seen_images)}")
print(f"Instances: {len(all_results)}")
print(f"Mean best IoU: {mean_best_iou:.4f}")
print(f"Median best IoU: {median_best_iou:.4f}")
print(f"Output directory: {final_output_dir}")
print("="*70)
