#!/usr/bin/env python3
"""Construct a small dataset for the class-group-hierarchy Fiedler experiment:
images containing 2+ distinct classes, where at least one of those classes has
3+ prominent (non-crowd) instances -- e.g. a street scene with several cars and
a person. Used to test whether the Fiedler tree's first split tends to isolate
the multi-instance class as a group before any node separates its individual
instances, or whether that doesn't hold.

Reuses the COCO-API-derived category naming already fixed in
construct_class_vs_instance_dataset.py (no hand-rolled class_names dict).
"""

import json
import random
import shutil
from collections import Counter
from pathlib import Path

from pycocotools.coco import COCO

COCO_JSON = Path("CutLER Implementation/TFSS-CutLER/datasets/coco/annotations/instances_val2017.json")
IMAGE_ROOT = Path("Fiedler Evaluation/datasets/val2017")
OUTPUT_ROOT = Path("Fiedler Evaluation/data/class_group_hierarchy")

MIN_MASK_AREA_RATIO = 0.01  # prominence threshold per instance (relative to image area)
MIN_GROUP_INSTANCES = 3     # the multi-instance class needs at least this many prominent instances
MAX_TOTAL_INSTANCES = 12    # keep images from being too cluttered to visualize
N_IMAGES = 10
PRIORITY_CLASSES = {"car", "person", "sheep", "cow", "chair", "bottle", "bus", "elephant"}
SEED = 7

random.seed(SEED)


def filter_images(coco, categories):
    candidates = []
    stats = Counter()

    for image_id in coco.getImgIds():
        image_info = coco.loadImgs([image_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        annotations = [a for a in coco.loadAnns(ann_ids) if not a.get("iscrowd", 0)]
        if not annotations:
            stats["no_annotations"] += 1
            continue

        image_area = image_info["width"] * image_info["height"]
        min_area = image_area * MIN_MASK_AREA_RATIO
        valid_anns = [a for a in annotations if a.get("area", 0) > min_area]
        if len(valid_anns) < 2:
            stats["too_few_prominent"] += 1
            continue
        if len(valid_anns) > MAX_TOTAL_INSTANCES:
            stats["too_cluttered"] += 1
            continue

        class_counts = Counter(a["category_id"] for a in valid_anns)
        if len(class_counts) < 2:
            stats["single_class"] += 1
            continue

        group_candidates = [cid for cid, count in class_counts.items() if count >= MIN_GROUP_INSTANCES]
        if not group_candidates:
            stats["no_multi_instance_class"] += 1
            continue

        # Pick exactly ONE category as "the group" -- the one with the most prominent
        # instances -- so the group is a genuine single multi-instance class (e.g. just
        # "car"), never an artificial union of two unrelated classes that both happened
        # to cross the instance threshold (e.g. "person"+"tie" counted together).
        group_cid = max(group_candidates, key=lambda cid: class_counts[cid])

        class_distribution = {categories.get(cid, f"class_{cid}"): count for cid, count in class_counts.items()}
        record = {
            "image_id": image_id,
            "file_name": image_info["file_name"],
            "width": image_info["width"],
            "height": image_info["height"],
            "num_instances": len(valid_anns),
            "num_classes": len(class_counts),
            "class_distribution": class_distribution,
            "group_category_ids": [group_cid],
            "group_category_names": [categories.get(group_cid, f"class_{group_cid}")],
        }
        candidates.append(record)
        stats["candidates"] += 1

    return candidates, stats


def select(candidates, n):
    """Guarantee a few "car"-group examples (the illustrative case asked about),
    then fill the rest with other priority classes, then anything else."""
    car_candidates = [r for r in candidates if "car" in r["group_category_names"]]
    random.shuffle(car_candidates)
    selected = car_candidates[:3]
    selected_ids = {r["image_id"] for r in selected}

    remaining = [r for r in candidates if r["image_id"] not in selected_ids]

    def sort_key(record):
        has_priority = any(name in PRIORITY_CLASSES for name in record["group_category_names"])
        return (0 if has_priority else 1, random.random())

    ordered = sorted(remaining, key=sort_key)
    selected += ordered[: max(0, n - len(selected))]
    return selected[:n]


def create_dataset():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(COCO_JSON))
    categories = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    candidates, stats = filter_images(coco, categories)

    print("=" * 60)
    print("DATASET CONSTRUCTION STATISTICS")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    selected = select(candidates, min(N_IMAGES, len(candidates)))

    print(f"\nSelected {len(selected)} images:")
    all_metadata = []
    for i, record in enumerate(selected, 1):
        src = IMAGE_ROOT / record["file_name"]
        local_file_name = f"img_{i:03d}.jpg"
        dst = OUTPUT_ROOT / local_file_name
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  [warning] missing source image: {src}")
        record["local_file_name"] = local_file_name
        all_metadata.append(record)
        print(f"{i:2d}. {record['file_name']:50s} | group={record['group_category_names']} | {record['class_distribution']}")

    metadata_path = OUTPUT_ROOT / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Dataset created at: {OUTPUT_ROOT}")
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    create_dataset()
