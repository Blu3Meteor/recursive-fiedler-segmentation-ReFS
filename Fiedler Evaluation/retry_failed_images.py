#!/usr/bin/env python3
"""Retry processing only the failed images from a previous run."""

import json
import sys
from pathlib import Path

# Extract failed image IDs
failures_log = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/output/02_oracle_proposals/PASCAL/pascal_voc2012_dinov2_patchlevel/failures.jsonl")
failed_ids = []

if failures_log.exists():
    with open(failures_log, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    failed_ids.append(data["image_id"])
                except:
                    pass

print(f"Found {len(failed_ids)} failed images:")
for iid in failed_ids:
    print(f"  - {iid}")

if not failed_ids:
    print("No failed images to retry.")
    sys.exit(0)

# Create a modified annotations JSON with only failed images
print("\nLoading PASCAL VOC 2012 annotations...")
anno_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/datasets/pascal_voc2012_val_coco.json")
with open(anno_path, 'r') as f:
    coco_data = json.load(f)

# Filter to only failed images
original_images = len(coco_data["images"])
coco_data["images"] = [img for img in coco_data["images"] if img["id"] in failed_ids]
coco_data["annotations"] = [ann for ann in coco_data["annotations"] if ann["image_id"] in failed_ids]

filtered_anno_path = Path("d:/Coding/Recursive Segmentation Master's Project/Fiedler Evaluation/pascal_voc2012_failed_only.json")
with open(filtered_anno_path, 'w') as f:
    json.dump(coco_data, f)

print(f"Created filtered annotation file with {len(coco_data['images'])} failed images")
print(f"Path: {filtered_anno_path}\n")

# Print command
cmd = f"""
python "d:\\Coding\\Recursive Segmentation Master's Project\\Fiedler Evaluation\\fiedler_candidate_eval.py" `
  --annotations "{filtered_anno_path}" `
  --image-root "D:\\Coding\\Recursive Segmentation Master's Project\\Fiedler Evaluation\\datasets\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages" `
  --output-dir "d:\\Coding\\Recursive Segmentation Master's Project\\Fiedler Evaluation\\output\\02_oracle_proposals\\PASCAL\\pascal_voc2012_dinov2_patchlevel_failed_retry" `
  --dino-version v2 `
  --vit-arch small `
  --patch-size 14 `
  --fixed-size 476 `
  --max-depth 9 `
  --max-nodes 255 `
  --min-region-patches 6
"""

print("Run this command to retry failed images:")
print(cmd)
