#!/usr/bin/env python3
"""Create class-level annotations by merging instance masks of the same class."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO


def main():
    if len(sys.argv) < 3:
        print("Usage: python create_class_level_annotations.py <coco_json> <output_json>")
        sys.exit(1)

    coco_json = sys.argv[1]
    output_json = sys.argv[2]

    print(f"Loading COCO annotations from {coco_json}...")
    coco = COCO(coco_json)

    # Group annotations by image and category
    print("Grouping instances by image and category...")
    class_annotations = defaultdict(lambda: defaultdict(list))

    for image_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        annotations = coco.loadAnns(ann_ids)

        for ann in annotations:
            if ann.get("iscrowd"):
                continue

            cat_id = ann["category_id"]
            class_annotations[image_id][cat_id].append(ann)

    # Create class-level masks
    print("Creating class-level masks...")
    class_level_annotations = []
    annotation_id = 1

    for image_id in sorted(class_annotations.keys()):
        for cat_id, instance_anns in class_annotations[image_id].items():
            if not instance_anns:
                continue

            # Load masks for all instances of this class
            masks = [coco.annToMask(ann) for ann in instance_anns]

            # Create union mask
            class_mask = np.logical_or.reduce(masks).astype(np.uint8)

            # Encode as RLE (COCO format)
            from pycocotools import mask as coco_mask

            fortran_mask = np.asfortranarray(class_mask)
            rle = coco_mask.encode(fortran_mask)
            rle["counts"] = rle["counts"].decode("utf-8")

            # Create annotation
            class_ann = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cat_id,
                "area": float(class_mask.sum()),
                "bbox": [0, 0, class_mask.shape[1], class_mask.shape[0]],  # Full image bbox
                "segmentation": rle,
                "iscrowd": 0,
            }

            class_level_annotations.append(class_ann)
            annotation_id += 1

    # Create output COCO JSON
    print(f"Writing {len(class_level_annotations)} class-level annotations...")
    output_coco = {
        "images": coco.loadImgs(coco.getImgIds()),
        "annotations": class_level_annotations,
        "categories": coco.loadCats(coco.getCatIds()),
    }

    with open(output_json, "w") as f:
        json.dump(output_coco, f, indent=2)

    print(f"✓ Saved class-level annotations: {output_json}")
    print(f"  - {len(output_coco['images'])} images")
    print(f"  - {len(class_level_annotations)} class-level instances")
    print(f"  - {len(output_coco['categories'])} categories")


if __name__ == "__main__":
    main()
