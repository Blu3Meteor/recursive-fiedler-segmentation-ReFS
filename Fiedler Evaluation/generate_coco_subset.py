#!/usr/bin/env python3
"""Generate random COCO val2017 subset with annotations."""

import json
import random
from pathlib import Path

# Config
COCO_JSON = Path("CutLER Implementation/TFSS-CutLER/datasets/coco/annotations/instances_val2017.json")
OUTPUT_DIR = Path("Fiedler Evaluation/data")
OUTPUT_JSON = OUTPUT_DIR / "coco_500_random.json"

random.seed(42)

# Load full COCO annotations
with open(COCO_JSON) as f:
    coco_full = json.load(f)

print(f"Full COCO val2017: {len(coco_full['images'])} images, {len(coco_full['annotations'])} annotations")

# Select 500 random images
image_ids = [img['id'] for img in coco_full['images']]
selected_ids = set(random.sample(image_ids, min(500, len(image_ids))))

# Filter images and annotations
selected_images = [img for img in coco_full['images'] if img['id'] in selected_ids]
selected_annotations = [ann for ann in coco_full['annotations'] if ann['image_id'] in selected_ids]

print(f"Selected 500 random images: {len(selected_images)} images, {len(selected_annotations)} annotations")

# Create new COCO dict
coco_subset = {
    'info': coco_full['info'],
    'licenses': coco_full['licenses'],
    'images': selected_images,
    'annotations': selected_annotations,
    'categories': coco_full['categories'],
}

# Save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(coco_subset, f)

print(f"Saved to: {OUTPUT_JSON}")
