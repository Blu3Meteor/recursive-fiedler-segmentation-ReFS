#!/usr/bin/env python3
"""Find larger/more prominent candidate reference instances for the thin-reference
classes (bird, sheep, cow, boat) from the 1000-image COCO-20i pool, excluding any
image already used in the 500-image test subset (avoid leakage)."""

import json

TARGET_CLASSES = {"bird", "sheep", "cow", "boat"}

with open("Fiedler Evaluation/data/coco20i/coco20i_test_1000_random.json") as f:
    pool = json.load(f)

with open("Fiedler Evaluation/data/coco_500_from_1000.json") as f:
    test_subset = json.load(f)

test_image_ids = set(img["id"] for img in test_subset["images"])
cat_names = {c["id"]: c["name"] for c in pool["categories"]}
images_by_id = {img["id"]: img for img in pool["images"]}

candidates = {cls: [] for cls in TARGET_CLASSES}
for ann in pool["annotations"]:
    cls = cat_names.get(ann["category_id"])
    if cls not in TARGET_CLASSES:
        continue
    if ann["image_id"] in test_image_ids:
        continue  # avoid reference/test leakage
    if ann.get("iscrowd", 0):
        continue
    img = images_by_id.get(ann["image_id"])
    if img is None:
        continue
    area_ratio = ann["area"] / (img["width"] * img["height"])
    candidates[cls].append((area_ratio, ann["image_id"], img["file_name"], ann["id"], ann["area"]))

for cls in TARGET_CLASSES:
    candidates[cls].sort(reverse=True)
    print(f"\n=== {cls} (top 5 by area ratio) ===")
    for area_ratio, image_id, file_name, ann_id, area in candidates[cls][:5]:
        print(f"  image_id={image_id:>8} ann_id={ann_id:>8} area_ratio={area_ratio:.3f} area={area:.0f} file={file_name}")
