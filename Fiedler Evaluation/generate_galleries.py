#!/usr/bin/env python3
"""Generate top 10 and bottom 10 galleries from merged results."""

import json
import csv
from pathlib import Path
from pycocotools.coco import COCO

# Load merged results CSV
csv_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc2012_dinov2_patchlevel_final/instance_results.csv")
print(f"Loading results from {csv_path}...")

results = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append({
            "image_id": int(row["image_id"]),
            "annotation_id": int(row["annotation_id"]),
            "best_iou": float(row["best_iou"]),
        })

print(f"Loaded {len(results)} results")

# Sort by IoU
results_sorted = sorted(results, key=lambda x: x["best_iou"], reverse=True)
top_10 = results_sorted[:10]
bottom_10 = results_sorted[-10:]

print(f"\nTop 10 instances:")
for i, r in enumerate(top_10, 1):
    print(f"  {i}. image_id={r['image_id']}, annotation_id={r['annotation_id']}, IoU={r['best_iou']:.4f}")

print(f"\nBottom 10 instances:")
for i, r in enumerate(bottom_10, 1):
    print(f"  {i}. image_id={r['image_id']}, annotation_id={r['annotation_id']}, IoU={r['best_iou']:.4f}")

# Load full PASCAL COCO data
anno_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/datasets/pascal_voc2012_val_coco.json")
print(f"\nLoading PASCAL annotations from {anno_path}...")
coco = COCO(str(anno_path))

# Extract annotations for top/bottom instances
top_bottom_ids = {r["annotation_id"] for r in top_10 + bottom_10}
top_bottom_image_ids = {r["image_id"] for r in top_10 + bottom_10}

# Create filtered COCO data
filtered_data = {
    "images": [img for img in coco.dataset["images"] if img["id"] in top_bottom_image_ids],
    "annotations": [ann for ann in coco.dataset["annotations"] if ann["id"] in top_bottom_ids],
    "categories": coco.dataset["categories"],
}

filtered_anno_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/pascal_voc2012_top_bottom.json")
with open(filtered_anno_path, 'w') as f:
    json.dump(filtered_data, f)

print(f"Created filtered annotation file: {filtered_anno_path}")
print(f"Contains {len(filtered_data['images'])} images and {len(filtered_data['annotations'])} annotations\n")

# Print command to generate galleries
cmd = f"""python "d:\\Coding\\Recursive Segmentation Master's Project\\Fiedler Evaluation\\fiedler_candidate_eval.py" `
  --annotations "{filtered_anno_path}" `
  --image-root "D:\\Coding\\Recursive Segmentation Master's Project\\Fiedler Evaluation\\datasets\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages" `
  --output-dir "d:\\Coding\\Recursive Segmentation Master's Project\\Fiedler Evaluation\\output\\02_oracle_proposals\\PASCAL\\pascal_voc2012_dinov2_patchlevel_galleries" `
  --dino-version v2 `
  --vit-arch small `
  --patch-size 14 `
  --fixed-size 476 `
  --max-depth 9 `
  --max-nodes 255 `
  --min-region-patches 6 `
  --top-k 10
"""

print("Run this command to generate the gallery images:")
print(cmd)
print("\nNote: This will generate top_10_gallery.png and bottom_10_gallery.png")
