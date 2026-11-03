#!/usr/bin/env python3
"""Identify which COCO images failed and create filtered annotation file for retry."""

import json
import csv
from pathlib import Path
from pycocotools.coco import COCO

# Load all COCO val2017 image IDs
anno_path = Path("d:/Coding/Recursive Segmentation Master's Project/CutLER Implementation/TFSS-CutLER/datasets/coco/annotations/instances_val2017.json")
coco = COCO(str(anno_path))
all_image_ids = set(coco.getImgIds())

print(f"Total COCO val2017 images: {len(all_image_ids)}")

# Load successfully processed image IDs from CSV
csv_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/COCO/coco_val2017_dinov2_patchlevel/instance_results.csv")
processed_ids = set()

if csv_path.exists():
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_ids.add(int(row["image_id"]))
    print(f"Successfully processed images: {len(processed_ids)}")
else:
    print(f"Warning: {csv_path} not found")

# Find failed images
failed_ids = sorted(all_image_ids - processed_ids)
print(f"Failed images: {len(failed_ids)}")

if failed_ids:
    print(f"\nFirst 10 failed image IDs: {failed_ids[:10]}")
    print(f"Last 10 failed image IDs: {failed_ids[-10:]}")

# Create filtered annotation JSON with only failed images
if failed_ids:
    failed_ids_set = set(failed_ids)
    coco_data = {
        "images": [img for img in coco.dataset["images"] if img["id"] in failed_ids_set],
        "annotations": [ann for ann in coco.dataset["annotations"] if ann["image_id"] in failed_ids_set],
        "categories": coco.dataset["categories"],
    }

    filtered_anno_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/coco_val2017_failed_only.json")
    with open(filtered_anno_path, 'w') as f:
        json.dump(coco_data, f)

    print(f"\nCreated filtered annotation file: {filtered_anno_path}")
    print(f"Contains {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations\n")

    # Print retry command
    cmd = f"""python "d:\\Coding\\Recursive Segmentation Master's Project\\Fiedler Evaluation\\fiedler_candidate_eval.py" `
  --annotations "{filtered_anno_path}" `
  --image-root "d:\\Coding\\Recursive Segmentation Master's Project\\CutLER Implementation\\TFSS-CutLER\\datasets\\coco\\val2017" `
  --output-dir "d:\\Coding\\Recursive Segmentation Master's Project\\Fiedler Evaluation\\output\\02_oracle_proposals\\COCO\\coco_val2017_dinov2_patchlevel_failed_retry" `
  --dino-version v2 `
  --vit-arch small `
  --patch-size 14 `
  --fixed-size 476 `
  --max-depth 9 `
  --max-nodes 255 `
  --min-region-patches 6"""

    print("Run this command to retry failed images:")
    print(cmd)
