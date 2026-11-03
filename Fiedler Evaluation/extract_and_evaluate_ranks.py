#!/usr/bin/env python3
"""Extract ranks 9-19 from results and create subset for re-evaluation."""

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

from pycocotools.coco import COCO


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python extract_and_evaluate_ranks.py <csv_path> <annotations_json> <output_json>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    annotations_json = sys.argv[2]
    output_json = sys.argv[3]

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    # Load original COCO
    print("Loading COCO annotations...")
    coco = COCO(annotations_json)

    # Read CSV and find ranks 9-19
    print(f"Reading results from {csv_path}...")
    all_results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                result = {
                    "image_id": int(row["image_id"]),
                    "annotation_id": int(row["annotation_id"]),
                    "category_id": int(row["category_id"]),
                    "category_name": row["category_name"],
                    "best_iou": float(row["best_iou"]),
                }
                all_results.append(result)
            except Exception as e:
                print(f"  Warning: Error parsing row: {e}")
                continue

    print(f"Loaded {len(all_results)} results")

    # Get ranks 9-19
    sorted_results = sorted(all_results, key=lambda x: x["best_iou"], reverse=True)
    ranks_9_19 = sorted_results[9:20]

    print(f"\nRanks 9-19 ({len(ranks_9_19)} class-level samples):")
    for idx, r in enumerate(ranks_9_19, start=10):
        level = "class-level" if r["annotation_id"] == -1 else "instance"
        print(f"  {idx}. {level} | image_id={r['image_id']}, IoU={r['best_iou']:.4f}, class={r['category_name']}")

    # Build subset COCO annotation with just these instances
    print(f"\nBuilding subset COCO annotations...")

    subset_images = []
    subset_annotations = []
    added_image_ids = set()
    added_ann_ids = set()

    # For each rank 9-19 result
    for r in ranks_9_19:
        image_id = r["image_id"]
        annotation_id = r["annotation_id"]
        category_id = r["category_id"]

        # Add image if not already added
        if image_id not in added_image_ids:
            img_info = coco.loadImgs([image_id])[0]
            subset_images.append(img_info)
            added_image_ids.add(image_id)

        # If annotation_id is -1, it's a class-level result
        # Add all annotations of that category in that image
        if annotation_id == -1:
            ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=[category_id])
            for ann_id in ann_ids:
                if ann_id not in added_ann_ids:
                    ann_info = coco.loadAnns([ann_id])[0]
                    subset_annotations.append(ann_info)
                    added_ann_ids.add(ann_id)
        else:
            # Regular instance-level result
            if annotation_id not in added_ann_ids:
                ann_info = coco.loadAnns([annotation_id])[0]
                subset_annotations.append(ann_info)
                added_ann_ids.add(annotation_id)

    # Create subset COCO JSON
    subset_coco = {
        "images": subset_images,
        "annotations": subset_annotations,
        "categories": coco.loadCats(sorted(coco.getCatIds())),
    }

    # Save subset
    with open(output_json, "w") as f:
        json.dump(subset_coco, f, indent=2)

    print(f"Saved subset annotations: {output_json}")
    print(f"  - {len(subset_images)} images")
    print(f"  - {len(subset_annotations)} annotations")
    print(f"\nNext, run evaluation on this subset:")
    print(f"  python fiedler_candidate_eval.py --annotations {output_json} ...")
