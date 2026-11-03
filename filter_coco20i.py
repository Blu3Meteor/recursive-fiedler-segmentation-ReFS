#!/usr/bin/env python3
"""Filter instance_results.csv to only COCO-20i's 20 classes and recompute IoU stats."""

import csv
import json
import sys
from collections import defaultdict

import numpy as np

COCO20I_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "boat", "dog", "cat", "bird", "horse", "sheep", "cow", "bottle",
    "cup", "chair", "dining table", "backpack", "handbag",
}


def filter_and_report(csv_path, label):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category_name"] in COCO20I_CLASSES:
                rows.append(row)

    if not rows:
        print(f"{label}: no matching rows found")
        return

    ious = [float(r["best_iou"]) for r in rows]
    image_ids = set(r["image_id"] for r in rows)

    per_class = defaultdict(list)
    for r in rows:
        per_class[r["category_name"]].append(float(r["best_iou"]))

    print(f"\n{'='*60}")
    print(f"{label} — COCO-20i classes only")
    print(f"{'='*60}")
    print(f"Images (with >=1 COCO-20i instance): {len(image_ids)}")
    print(f"Instances: {len(rows)}")
    print(f"Mean IoU: {np.mean(ious):.4f}")
    print(f"Median IoU: {np.median(ious):.4f}")
    print(f"Min/Max IoU: {np.min(ious):.4f} / {np.max(ious):.4f}")
    print(f"\nPer-class breakdown:")
    for cls in sorted(per_class, key=lambda c: -len(per_class[c])):
        scores = per_class[cls]
        print(f"  {cls:15s} count={len(scores):4d}  mean_iou={np.mean(scores):.4f}")

    return {
        "label": label,
        "num_images": len(image_ids),
        "num_instances": len(rows),
        "mean_best_iou": float(np.mean(ious)),
        "median_best_iou": float(np.median(ious)),
        "per_class": {cls: {"count": len(scores), "mean_iou": float(np.mean(scores))} for cls, scores in per_class.items()},
    }


if __name__ == "__main__":
    fiedler_csv = "Fiedler Evaluation/output/02_oracle_proposals/COCO/coco_500_1022x1022_patch14/instance_results.csv"
    insid3_csv = "Fiedler Evaluation/output/baselines_insid3/coco_500_insid3_zeroshot/instance_results.csv"

    fiedler_result = filter_and_report(fiedler_csv, "Fiedler (500-img subset)")
    insid3_result = filter_and_report(insid3_csv, "INSID3 (500-img subset)")

    with open("Fiedler Evaluation/output/baselines_insid3/coco20i_filtered_comparison.json", "w") as f:
        json.dump({"fiedler": fiedler_result, "insid3": insid3_result}, f, indent=2)

    print(f"\n\nSaved comparison to: Fiedler Evaluation/output/baselines_insid3/coco20i_filtered_comparison.json")
