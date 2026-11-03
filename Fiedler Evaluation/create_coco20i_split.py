#!/usr/bin/env python3
"""Create COCO-20i dataset: 20-class subset for one-shot segmentation evaluation."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from pycocotools.coco import COCO


# Standard COCO-20i classes (widely used in one-shot literature)
# Selected to be diverse and representative
COCO_20i_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "boat",
    "dog",
    "cat",
    "bird",
    "horse",
    "sheep",
    "cow",
    "bottle",
    "cup",
    "chair",
    "dining table",
    "backpack",
    "handbag",
]


def create_coco20i_split(coco_json_path, output_dir, reference_count_per_class=1, random_seed=42):
    """
    Create COCO-20i split with reference and test sets.

    Args:
        coco_json_path: Path to COCO annotations JSON (instances_val2017.json)
        output_dir: Directory to save reference and test annotations
        reference_count_per_class: Number of reference examples per class (default: 1 for one-shot)
        random_seed: Random seed for reproducibility
    """
    import random

    random.seed(random_seed)

    # Load COCO
    print(f"Loading COCO annotations from {coco_json_path}...")
    coco = COCO(coco_json_path)

    # Get class ID to name mapping
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco.loadCats(sorted(coco.getCatIds()))}
    category_name_to_id = {name: cid for cid, name in category_id_to_name.items()}

    # Get class IDs for COCO-20i
    class_ids = []
    missing_classes = []
    for class_name in COCO_20i_CLASSES:
        if class_name in category_name_to_id:
            class_ids.append(category_name_to_id[class_name])
        else:
            missing_classes.append(class_name)

    if missing_classes:
        print(f"Warning: Could not find classes {missing_classes} in COCO")

    print(f"Using {len(class_ids)} classes: {[category_id_to_name[cid] for cid in class_ids]}")

    # Group images and annotations by class
    images_by_class = defaultdict(list)
    annotations_by_class = defaultdict(list)

    for image_id in coco.getImgIds():
        image_info = coco.loadImgs([image_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        annotations = coco.loadAnns(ann_ids)

        for ann in annotations:
            if ann.get("iscrowd"):
                continue
            cat_id = ann["category_id"]
            if cat_id not in class_ids:
                continue

            # Store annotation with image info
            ann_with_image = {**ann, "image_id": image_id, "image_info": image_info}
            annotations_by_class[cat_id].append(ann_with_image)

            # Track unique images per class
            if image_id not in images_by_class[cat_id]:
                images_by_class[cat_id].append((image_id, image_info))

    # Split into reference and test
    reference_annotations = []
    reference_images = {}
    test_annotations = []
    test_images = {}

    for class_id in sorted(class_ids):
        class_name = category_id_to_name[class_id]
        anns = annotations_by_class[class_id]
        imgs = images_by_class[class_id]

        if not anns:
            print(f"  Warning: No annotations for class {class_name}")
            continue

        # Shuffle for random selection
        shuffled_anns = list(anns)
        random.shuffle(shuffled_anns)

        # Select reference annotations (first N per class)
        for i, ann in enumerate(shuffled_anns[:reference_count_per_class]):
            reference_annotations.append(ann)
            image_id = ann["image_id"]
            if image_id not in reference_images:
                reference_images[image_id] = ann["image_info"]

        # Remaining annotations go to test set
        for ann in shuffled_anns[reference_count_per_class:]:
            test_annotations.append(ann)
            image_id = ann["image_id"]
            if image_id not in test_images:
                test_images[image_id] = ann["image_info"]

        print(
            f"  {class_name:20} | reference: {reference_count_per_class:3} | test: {len(shuffled_anns) - reference_count_per_class:4} | images: {len(imgs):4}"
        )

    # Build reference and test COCO-format annotations
    print(f"\nBuilding reference and test sets...")

    reference_coco = {
        "images": list(reference_images.values()),
        "annotations": reference_annotations,
        "categories": [{"id": cid, "name": category_id_to_name[cid]} for cid in sorted(class_ids)],
    }

    test_coco = {
        "images": list(test_images.values()),
        "annotations": test_annotations,
        "categories": [{"id": cid, "name": category_id_to_name[cid]} for cid in sorted(class_ids)],
    }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save reference set
    reference_json_path = os.path.join(output_dir, "coco20i_reference.json")
    with open(reference_json_path, "w") as f:
        json.dump(reference_coco, f, indent=2)
    print(f"Saved reference set: {reference_json_path}")
    print(f"  - {len(reference_images)} images")
    print(f"  - {len(reference_annotations)} annotations")

    # Save test set
    test_json_path = os.path.join(output_dir, "coco20i_test.json")
    with open(test_json_path, "w") as f:
        json.dump(test_coco, f, indent=2)
    print(f"Saved test set: {test_json_path}")
    print(f"  - {len(test_images)} images")
    print(f"  - {len(test_annotations)} annotations")

    # Save split info
    split_info = {
        "dataset": "COCO-20i",
        "num_classes": len(class_ids),
        "classes": COCO_20i_CLASSES,
        "reference_count_per_class": reference_count_per_class,
        "num_reference_images": len(reference_images),
        "num_reference_annotations": len(reference_annotations),
        "num_test_images": len(test_images),
        "num_test_annotations": len(test_annotations),
        "random_seed": random_seed,
    }

    split_info_path = os.path.join(output_dir, "coco20i_split_info.json")
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved split info: {split_info_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"COCO-20i split created successfully")
    print(f"{'='*70}")
    print(f"Classes: {len(class_ids)}")
    print(f"Reference set: {len(reference_images)} images, {len(reference_annotations)} instances")
    print(f"Test set: {len(test_images)} images, {len(test_annotations)} instances")
    print(f"Total: {len(reference_images) + len(test_images)} images, {len(reference_annotations) + len(test_annotations)} instances")

    return {
        "reference_json": reference_json_path,
        "test_json": test_json_path,
        "split_info": split_info_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Create COCO-20i dataset split")
    parser.add_argument(
        "--coco-json",
        type=str,
        default="CutLER Implementation/TFSS-CutLER/datasets/coco/annotations/instances_val2017.json",
        help="Path to COCO annotations JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Fiedler Evaluation/data/coco20i",
        help="Output directory for COCO-20i split",
    )
    parser.add_argument(
        "--reference-per-class",
        type=int,
        default=1,
        help="Number of reference annotations per class (default: 1 for one-shot)",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Convert paths to absolute if needed
    coco_json = os.path.abspath(args.coco_json)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(coco_json):
        raise FileNotFoundError(f"COCO annotations not found: {coco_json}")

    create_coco20i_split(coco_json, output_dir, args.reference_per_class, args.random_seed)


if __name__ == "__main__":
    main()
