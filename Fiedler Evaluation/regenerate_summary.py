#!/usr/bin/env python3
"""Regenerate summary.json and visualizations from instance_results.csv."""

import json
import csv
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

# Add plot functions
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_overall_histogram(scores, output_path, title):
    """Plot IoU distribution histogram."""
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=np.linspace(0.0, 1.0, 21), color="#72b7b2", edgecolor="white")
    plt.xlim(0.0, 1.0)
    plt.xlabel("Best IoU")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Saved histogram to {output_path}")

def plot_class_overview(class_stats, output_path, title):
    """Plot mean IoU by class."""
    ordered = sorted(class_stats.items(), key=lambda item: (-item[1]["count"], item[0]))
    labels = [name for name, _ in ordered]
    means = [stats["mean_iou"] for _, stats in ordered]

    plt.figure(figsize=(max(10, len(labels) * 0.22), 6))
    plt.bar(range(len(labels)), means, color="#72b7b2")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean best IoU")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Saved class overview to {output_path}")

# Paths
output_dir = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc2012_dinov2_patchlevel")
csv_path = output_dir / "instance_results.csv"
summary_path = output_dir / "summary.json"
report_path = output_dir / "report.md"
histogram_path = output_dir / "overall_iou_histogram.png"
class_overview_path = output_dir / "class_mean_iou.png"

print(f"Reading CSV from {csv_path}...")

# Load results
all_results = []
class_scores = defaultdict(list)
per_class_stats = defaultdict(lambda: {"count": 0, "sum": 0.0, "scores": []})
image_summaries = []
seen_images = set()

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            result = {
                "image_id": int(row["image_id"]),
                "annotation_id": int(row["annotation_id"]),
                "category_id": int(row["category_id"]),
                "category_name": row["category_name"],
                "best_iou": float(row["best_iou"]),
                "best_candidate_id": int(row["best_candidate_id"]),
                "best_candidate_depth": int(row["best_candidate_depth"]),
                "best_candidate_status": row["best_candidate_status"],
                "best_candidate_area_patches": int(row["best_candidate_area_patches"]),
                "candidate_count": int(row["candidate_count"]),
                "image_path": row["image_path"],
            }
            all_results.append(result)
            class_scores[result["category_name"]].append(result["best_iou"])
            per_class_stats[result["category_name"]]["count"] += 1
            per_class_stats[result["category_name"]]["sum"] += result["best_iou"]
            per_class_stats[result["category_name"]]["scores"].append(result["best_iou"])
            seen_images.add(result["image_id"])
        except Exception as e:
            print(f"Error parsing row: {e}")
            continue

print(f"Loaded {len(all_results)} instances from {len(seen_images)} images")

# Compute stats
overall_scores = [r["best_iou"] for r in all_results]
per_image_scores = []
image_id_to_results = defaultdict(list)
for result in all_results:
    image_id_to_results[result["image_id"]].append(result)

for image_id, results in image_id_to_results.items():
    best_iou = max(r["best_iou"] for r in results)
    mean_iou = float(np.mean([r["best_iou"] for r in results]))
    per_image_scores.append(best_iou)
    image_summaries.append({
        "image_id": image_id,
        "num_instances": len(results),
        "best_instance_iou": best_iou,
        "mean_instance_iou": mean_iou,
    })

# Aggregate stats
mean_best_iou = float(np.mean(overall_scores))
median_best_iou = float(np.median(overall_scores))
min_best_iou = float(np.min(overall_scores))
max_best_iou = float(np.max(overall_scores))
mean_best_iou_per_image = float(np.mean(per_image_scores))
median_best_iou_per_image = float(np.median(per_image_scores))

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

# Top/bottom samples
top_samples = sorted(all_results, key=lambda x: x["best_iou"], reverse=True)[:10]
bottom_samples = sorted(all_results, key=lambda x: x["best_iou"])[:10]

# Write summary.json
summary = {
    "dataset_name": "pascal_voc2012_dinov2_patchlevel",
    "fixed_size": 476,
    "patch_size": 14,
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

with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary.json")

# Write report.md
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# pascal_voc2012_dinov2_patchlevel recursive Fiedler evaluation\n\n")
    f.write(f"- images: {len(seen_images)}\n")
    f.write(f"- instances: {len(all_results)}\n")
    f.write(f"- mean best IoU: {mean_best_iou:.4f}\n")
    f.write(f"- median best IoU: {median_best_iou:.4f}\n")
    f.write(f"- mean best IoU per image: {mean_best_iou_per_image:.4f}\n\n")
    f.write(f"- results CSV: instance_results.csv\n")
    f.write(f"- summary JSON: summary.json\n")
    f.write(f"- overall histogram: overall_iou_histogram.png\n")
    f.write(f"- class mean chart: class_mean_iou.png\n")

print(f"Saved report.md")

# Generate plots
print("Generating plots...")
plot_overall_histogram(overall_scores, str(histogram_path), "PASCAL VOC 2012 DINOv2 Patch-Level best IoU distribution")
plot_class_overview(per_class_summary, str(class_overview_path), "PASCAL VOC 2012 DINOv2 Patch-Level mean best IoU by class")

print("\n" + "="*70)
print("Summary regenerated successfully!")
print("="*70)
print(f"Mean best IoU: {mean_best_iou:.4f}")
print(f"Median best IoU: {median_best_iou:.4f}")
print(f"Images processed: {len(seen_images)}")
print(f"Instances: {len(all_results)}")
print("="*70)
