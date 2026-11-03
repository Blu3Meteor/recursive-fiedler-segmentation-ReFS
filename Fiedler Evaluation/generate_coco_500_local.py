#!/usr/bin/env python3
"""Generate 500-image subset from local COCO dataset."""

import json
import random
from pathlib import Path

COCO_JSON = Path("Fiedler Evaluation/data/coco20i/coco20i_test_100_random.json")
OUTPUT_JSON = Path("Fiedler Evaluation/data/coco_500_random_local.json")

random.seed(42)

# Load COCO-20i annotations
with open(COCO_JSON) as f:
    coco_full = json.load(f)

print(f"Source COCO-20i: {len(coco_full['images'])} images")

# Select 500 random images
image_ids = [img['id'] for img in coco_full['images']]
selected_ids = set(random.sample(image_ids, min(500, len(image_ids))))

# Filter images and annotations
selected_images = [img for img in coco_full['images'] if img['id'] in selected_ids]
selected_annotations = [ann for ann in coco_full['annotations'] if ann['image_id'] in selected_ids]

print(f"Selected 500 images: {len(selected_images)} images, {len(selected_annotations)} annotations")

# Create new COCO dict
coco_subset = {
    'info': coco_full['info'],
    'licenses': coco_full['licenses'],
    'images': selected_images,
    'annotations': selected_annotations,
    'categories': coco_full['categories'],
}

# Save
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(coco_subset, f)

print(f"Saved to: {OUTPUT_JSON}")
