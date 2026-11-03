#!/usr/bin/env python3
"""Construct curated dataset for CNN vs ViT backbone comparison.

Filter COCO-20i for simple images with prominent cat/dog instances.
Categorize into: same_class_instances, multi_class.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from PIL import Image

# Config
COCO_JSON = Path("CutLER Implementation/TFSS-CutLER/datasets/coco/annotations/instances_val2017.json")
IMAGE_ROOT = Path("CutLER Implementation/TFSS-CutLER/datasets/coco/val2017")
OUTPUT_ROOT = Path("Fiedler Evaluation/data/coco20i_backbone_comparison")

# Class filter: animals + person (broader set for multi-class)
CLASS_IDS = {1, 16, 17, 18, 19, 20, 21, 22, 23}  # person, dog, horse, cat/sheep, cow, elephant, bear, zebra, giraffe
MIN_MASK_AREA_RATIO = 0.05  # 5% of image area (per instance)
MIN_INSTANCES = 1
MAX_INSTANCES = 6

random.seed(42)

def mask_area(annotation, coco):
    """Compute mask area in pixels."""
    rle = annotation.get("area", 0)
    return rle

def filter_images():
    """Filter images by criteria and categorize."""
    coco = COCO(str(COCO_JSON))

    same_class_candidates = []
    multi_class_candidates = []

    stats = {
        "total_images": len(coco.imgs),
        "filtered_no_cat_dog": 0,
        "filtered_too_few_instances": 0,
        "filtered_too_many_instances": 0,
        "filtered_small_instances": 0,
        "same_class_candidates": 0,
        "multi_class_candidates": 0,
        "found_both_classes": 0,  # debug
    }

    for image_id in coco.getImgIds():
        image_info = coco.loadImgs([image_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        annotations = coco.loadAnns(ann_ids)

        # Filter by class
        annotations = [a for a in annotations if a.get("category_id") in CLASS_IDS]
        if not annotations:
            stats["filtered_no_cat_dog"] += 1
            continue

        # Filter by instance count
        if len(annotations) < MIN_INSTANCES or len(annotations) > MAX_INSTANCES:
            if len(annotations) < MIN_INSTANCES:
                stats["filtered_too_few_instances"] += 1
            else:
                stats["filtered_too_many_instances"] += 1
            continue

        # Filter by mask area (exclude tiny instances)
        image_area = image_info["width"] * image_info["height"]
        min_mask_area = int(image_area * MIN_MASK_AREA_RATIO)
        valid_anns = [a for a in annotations if mask_area(a, coco) > min_mask_area]
        if len(valid_anns) < MIN_INSTANCES:
            stats["filtered_small_instances"] += 1
            continue

        annotations = valid_anns

        # Categorize
        class_counts = defaultdict(int)
        for ann in annotations:
            class_counts[ann["category_id"]] += 1

        class_names = {1: "person", 16: "dog", 17: "horse", 18: "cat", 19: "sheep", 20: "cow", 21: "elephant", 22: "bear", 23: "zebra"}
        class_dist = {class_names.get(cid, f"class_{cid}"): count for cid, count in class_counts.items()}

        record = {
            "image_id": image_id,
            "file_name": image_info["file_name"],
            "width": image_info["width"],
            "height": image_info["height"],
            "num_instances": len(annotations),
            "class_distribution": class_dist,
            "num_classes": len(class_counts),
        }

        # Check for both classes
        if len(class_counts) >= 2:
            stats["found_both_classes"] += 1

        # Multi-class: both cat and dog present (prioritize)
        if len(class_counts) >= 2:
            multi_class_candidates.append(record)
            stats["multi_class_candidates"] += 1
        # Same class: only one class OR 2+ of the same class
        elif any(count >= 2 for count in class_counts.values()):
            same_class_candidates.append(record)
            stats["same_class_candidates"] += 1

    return same_class_candidates, multi_class_candidates, stats

def create_dataset():
    """Create folder structure and copy/symlink images."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    same_class_cands, multi_class_cands, stats = filter_images()

    # Print statistics
    print("=" * 60)
    print("DATASET CONSTRUCTION STATISTICS")
    print("=" * 60)
    print(f"Total images in COCO-20i: {stats['total_images']}")
    print(f"Filtered (no cat/dog): {stats['filtered_no_cat_dog']}")
    print(f"Filtered (too few instances): {stats['filtered_too_few_instances']}")
    print(f"Filtered (too many instances): {stats['filtered_too_many_instances']}")
    print(f"Filtered (small masks): {stats['filtered_small_instances']}")
    print(f"\nFound images with both cat AND dog: {stats['found_both_classes']}")
    print(f"Candidates for same_class_instances: {stats['same_class_candidates']}")
    print(f"Candidates for multi_class: {stats['multi_class_candidates']}")

    # Select randomly
    n_same_class = min(10, len(same_class_cands))
    n_multi_class = min(10, len(multi_class_cands))

    same_class_selected = random.sample(same_class_cands, n_same_class)
    multi_class_selected = random.sample(multi_class_cands, n_multi_class)

    print(f"\nSelected for same_class_instances: {n_same_class}")
    print(f"Selected for multi_class: {n_multi_class}")

    # Create folders
    same_class_dir = OUTPUT_ROOT / "same_class_instances"
    multi_class_dir = OUTPUT_ROOT / "multi_class"
    same_class_dir.mkdir(parents=True, exist_ok=True)
    multi_class_dir.mkdir(parents=True, exist_ok=True)

    # Copy/symlink images and save metadata
    all_metadata = []

    print("\n" + "=" * 60)
    print("SAME CLASS INSTANCES")
    print("=" * 60)
    for i, record in enumerate(same_class_selected, 1):
        src = IMAGE_ROOT / record["file_name"]
        dst = same_class_dir / f"img_{i:03d}.jpg"
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
        record["category"] = "same_class_instances"
        all_metadata.append(record)
        print(f"{i:2d}. {record['file_name']:50s} | {record['class_distribution']}")

    print("\n" + "=" * 60)
    print("MULTI CLASS")
    print("=" * 60)
    for i, record in enumerate(multi_class_selected, 1):
        src = IMAGE_ROOT / record["file_name"]
        dst = multi_class_dir / f"img_{i:03d}.jpg"
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
        record["category"] = "multi_class"
        all_metadata.append(record)
        print(f"{i:2d}. {record['file_name']:50s} | {record['class_distribution']}")

    # Save metadata
    metadata_path = OUTPUT_ROOT / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Dataset created at: {OUTPUT_ROOT}")
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 60)

if __name__ == "__main__":
    create_dataset()
