#!/usr/bin/env python3
"""Convert PASCAL VOC 2012 XML annotations to COCO JSON format."""

import json
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


PASCAL_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)


def load_voc_annotations(voc_dir, split="val", subset=None):
    """Load PASCAL VOC annotations from XML files."""
    image_set_file = os.path.join(voc_dir, "ImageSets", "Main", f"{split}.txt")
    annotations_dir = os.path.join(voc_dir, "Annotations")
    images_dir = os.path.join(voc_dir, "JPEGImages")

    with open(image_set_file, "r") as f:
        fileids = [line.strip() for line in f.readlines()]

    if subset and subset > 0:
        fileids = fileids[:subset]

    images = []
    annotations = []
    annotation_id = 1

    for fileid in fileids:
        xml_file = os.path.join(annotations_dir, f"{fileid}.xml")
        jpg_file = os.path.join(images_dir, f"{fileid}.jpg")

        if not os.path.exists(xml_file):
            continue

        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract image info
        height = int(root.find("size/height").text)
        width = int(root.find("size/width").text)

        image_id = int(fileid) if fileid.isdigit() else hash(fileid) % (10 ** 8)

        images.append({
            "id": image_id,
            "file_name": f"{fileid}.jpg",
            "height": height,
            "width": width
        })

        # Extract objects (instances)
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in PASCAL_CLASSES:
                continue

            category_id = PASCAL_CLASSES.index(class_name) + 1  # COCO uses 1-indexed categories

            # Get bounding box (XYXY format in PASCAL, 1-indexed)
            bbox_elem = obj.find("bndbox")
            xmin = float(bbox_elem.find("xmin").text) - 1.0  # Convert to 0-indexed
            ymin = float(bbox_elem.find("ymin").text) - 1.0
            xmax = float(bbox_elem.find("xmax").text)
            ymax = float(bbox_elem.find("ymax").text)

            # Convert to COCO format (XYWH)
            w = xmax - xmin
            h = ymax - ymin
            area = w * h

            # Generate segmentation polygon from bbox (4-point rectangle)
            segmentation = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, w, h],
                "segmentation": segmentation,
                "area": area,
                "iscrowd": 0
            })
            annotation_id += 1

    return images, annotations


def create_coco_json(images, annotations, output_path):
    """Create COCO JSON structure and save to file."""
    categories = [
        {"id": i + 1, "name": name}
        for i, name in enumerate(PASCAL_CLASSES)
    ]

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_path, "w") as f:
        json.dump(coco_data, f)

    print(f"Saved {len(images)} images, {len(annotations)} annotations to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert PASCAL VOC 2012 to COCO JSON format")
    parser.add_argument("--voc-dir", type=str, required=True, help="Path to VOC2012 directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "trainval", "test"],
                        help="Dataset split to convert")
    parser.add_argument("--output", type=str, required=True, help="Output COCO JSON path")
    parser.add_argument("--subset", type=int, default=None, help="Limit to first N images (None for full dataset)")

    args = parser.parse_args()

    print(f"Loading PASCAL VOC from {args.voc_dir} (split: {args.split})...")
    images, annotations = load_voc_annotations(args.voc_dir, args.split, subset=args.subset)

    print(f"Creating COCO JSON with {len(images)} images, {len(annotations)} annotations...")
    create_coco_json(images, annotations, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
