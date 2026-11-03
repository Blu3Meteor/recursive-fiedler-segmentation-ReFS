#!/usr/bin/env python3
"""Construct a ~20-image dataset for the class-vs-instance Fiedler visualization
experiment: images with 2+ distinct classes present (e.g. a dog and a cat) versus
images with 2+ instances of the same class (e.g. two dogs).

Fixes a real bug found in the earlier construct_backbone_comparison_dataset.py:
that script's hand-rolled class_names dict was off-by-one relative to real COCO
category ids (e.g. it called id 16 "dog" when COCO id 16 is actually "bird").
Here, category names are always derived directly from the COCO API so they can
never drift from the real ids.
"""

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from pycocotools.coco import COCO

COCO_JSON = Path("CutLER Implementation/TFSS-CutLER/datasets/coco/annotations/instances_val2017.json")
IMAGE_ROOT = Path("Fiedler Evaluation/datasets/val2017")
OUTPUT_ROOT = Path("Fiedler Evaluation/data/class_vs_instance_20")

CLASS_IDS = {1, 16, 17, 18, 19, 20, 21, 22, 23}  # person, bird, cat, dog, horse, sheep, cow, elephant, bear
PRIORITY_PAIR = {17, 18}  # cat, dog -- prioritized within the multi_class bucket
MIN_MASK_AREA_RATIO = 0.05
MIN_INSTANCES = 1
MAX_INSTANCES = 6
N_PER_BUCKET = 10
SEED = 42

random.seed(SEED)


def filter_images(coco, categories):
    """Filter images by criteria and categorize into multi_class / same_class_instances."""
    same_class_candidates = []
    multi_class_candidates = []

    stats = {
        "total_images": len(coco.imgs),
        "no_target_class": 0,
        "too_few_instances": 0,
        "too_many_instances": 0,
        "no_prominent_instances": 0,
        "single_class_single_instance_dropped": 0,
        "same_class_candidates": 0,
        "multi_class_candidates": 0,
        "dog_and_cat_present": 0,
    }

    for image_id in coco.getImgIds():
        image_info = coco.loadImgs([image_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        annotations = coco.loadAnns(ann_ids)

        annotations = [a for a in annotations if a.get("category_id") in CLASS_IDS and not a.get("iscrowd", 0)]
        if not annotations:
            stats["no_target_class"] += 1
            continue

        if len(annotations) < MIN_INSTANCES or len(annotations) > MAX_INSTANCES:
            if len(annotations) < MIN_INSTANCES:
                stats["too_few_instances"] += 1
            else:
                stats["too_many_instances"] += 1
            continue

        image_area = image_info["width"] * image_info["height"]
        min_mask_area = image_area * MIN_MASK_AREA_RATIO
        valid_anns = [a for a in annotations if a.get("area", 0) > min_mask_area]
        if len(valid_anns) < MIN_INSTANCES:
            stats["no_prominent_instances"] += 1
            continue

        class_counts = Counter(a["category_id"] for a in valid_anns)
        class_ids_present = set(class_counts.keys())
        class_distribution = {categories.get(cid, f"class_{cid}"): count for cid, count in class_counts.items()}

        record = {
            "image_id": image_id,
            "file_name": image_info["file_name"],
            "width": image_info["width"],
            "height": image_info["height"],
            "num_instances": len(valid_anns),
            "class_distribution": class_distribution,
            "num_classes": len(class_counts),
            "category_ids_present": sorted(class_ids_present),
        }

        if PRIORITY_PAIR.issubset(class_ids_present):
            stats["dog_and_cat_present"] += 1

        if len(class_counts) >= 2:
            multi_class_candidates.append(record)
            stats["multi_class_candidates"] += 1
        elif any(count >= 2 for count in class_counts.values()):
            same_class_candidates.append(record)
            stats["same_class_candidates"] += 1
        else:
            stats["single_class_single_instance_dropped"] += 1

    return same_class_candidates, multi_class_candidates, stats


def select_multi_class(candidates, n):
    """Prioritize images containing both cat and dog, fill the rest randomly."""
    def sort_key(record):
        has_pair = PRIORITY_PAIR.issubset(set(record["category_ids_present"]))
        return (0 if has_pair else 1, random.random())

    ordered = sorted(candidates, key=sort_key)
    return ordered[:n]


def create_dataset():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(COCO_JSON))
    categories = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    same_class_cands, multi_class_cands, stats = filter_images(coco, categories)

    print("=" * 60)
    print("DATASET CONSTRUCTION STATISTICS")
    print("=" * 60)
    print(f"Total images in val2017: {stats['total_images']}")
    print(f"Filtered (no target class present): {stats['no_target_class']}")
    print(f"Filtered (too few instances): {stats['too_few_instances']}")
    print(f"Filtered (too many instances): {stats['too_many_instances']}")
    print(f"Filtered (no prominent instances): {stats['no_prominent_instances']}")
    print(f"Dropped (single class, single instance): {stats['single_class_single_instance_dropped']}")
    print(f"\nCandidates for same_class_instances: {stats['same_class_candidates']}")
    print(f"Candidates for multi_class: {stats['multi_class_candidates']}")
    print(f"Images with BOTH cat and dog prominent: {stats['dog_and_cat_present']}")

    n_same_class = min(N_PER_BUCKET, len(same_class_cands))
    n_multi_class = min(N_PER_BUCKET, len(multi_class_cands))

    same_class_selected = random.sample(same_class_cands, n_same_class)
    multi_class_selected = select_multi_class(multi_class_cands, n_multi_class)

    print(f"\nSelected for same_class_instances: {n_same_class}")
    print(f"Selected for multi_class: {n_multi_class}")
    n_pairs_selected = sum(1 for r in multi_class_selected if PRIORITY_PAIR.issubset(set(r["category_ids_present"])))
    print(f"  ...of which contain both cat and dog: {n_pairs_selected}")

    same_class_dir = OUTPUT_ROOT / "same_class_instances"
    multi_class_dir = OUTPUT_ROOT / "multi_class"
    same_class_dir.mkdir(parents=True, exist_ok=True)
    multi_class_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = []

    print("\n" + "=" * 60)
    print("SAME CLASS INSTANCES")
    print("=" * 60)
    for i, record in enumerate(same_class_selected, 1):
        src = IMAGE_ROOT / record["file_name"]
        local_file_name = f"img_{i:03d}.jpg"
        dst = same_class_dir / local_file_name
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  [warning] missing source image: {src}")
        record["category"] = "same_class_instances"
        record["local_file_name"] = local_file_name
        all_metadata.append(record)
        print(f"{i:2d}. {record['file_name']:50s} | {record['class_distribution']}")

    print("\n" + "=" * 60)
    print("MULTI CLASS")
    print("=" * 60)
    for i, record in enumerate(multi_class_selected, 1):
        src = IMAGE_ROOT / record["file_name"]
        local_file_name = f"img_{i:03d}.jpg"
        dst = multi_class_dir / local_file_name
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  [warning] missing source image: {src}")
        record["category"] = "multi_class"
        record["local_file_name"] = local_file_name
        all_metadata.append(record)
        print(f"{i:2d}. {record['file_name']:50s} | {record['class_distribution']}")

    metadata_path = OUTPUT_ROOT / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Dataset created at: {OUTPUT_ROOT}")
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    create_dataset()
