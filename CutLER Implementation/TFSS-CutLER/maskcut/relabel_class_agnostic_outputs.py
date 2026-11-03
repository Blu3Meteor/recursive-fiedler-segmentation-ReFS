#!/usr/bin/env python3

import argparse
import copy
import json
import os
from collections import Counter, defaultdict


def image_id_from_path(image_path):
    stem = os.path.splitext(os.path.basename(str(image_path).replace("\\", "/")))[0]
    return int(stem)


def safe_name(value):
    keep = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "unknown"


def load_coco_labels(coco_json):
    with open(coco_json, "r", encoding="utf-8") as file_handle:
        dataset = json.load(file_handle)

    categories = {category["id"]: category["name"] for category in dataset.get("categories", [])}
    labels_by_image = defaultdict(Counter)
    for annotation in dataset.get("annotations", []):
        category_name = categories.get(annotation["category_id"], str(annotation["category_id"]))
        labels_by_image[annotation["image_id"]][category_name] += 1
    return labels_by_image


def label_record(record, labels_by_image):
    image_id = image_id_from_path(record["image_path"])
    label_counts = labels_by_image.get(image_id, Counter())
    labels = sorted(label_counts)
    corrected = copy.deepcopy(record)
    corrected["original_class_name"] = record.get("class_name")
    corrected["coco_image_id"] = image_id
    corrected["coco_labels"] = labels
    corrected["coco_label_counts"] = dict(sorted(label_counts.items()))
    if len(labels) == 1:
        corrected["class_name"] = labels[0]
        corrected["label_status"] = "single_class"
    elif labels:
        corrected["class_name"] = "mixed"
        corrected["label_status"] = "multi_class_image"
    else:
        corrected["class_name"] = "unknown"
        corrected["label_status"] = "missing_coco_annotations"
    return corrected


def relabel_fiedler_summary(summary, labels_by_image):
    corrected = copy.deepcopy(summary)
    corrected["label_source"] = {
        "type": "coco_instances",
        "note": "Labels corrected from class-specific COCO annotations. Multi-class images are marked mixed.",
    }
    grouped = defaultdict(list)
    for class_summary in summary.get("classes", []):
        for record in class_summary.get("images", []):
            if not record.get("image_path"):
                continue
            corrected_record = label_record(record, labels_by_image)
            grouped[corrected_record["class_name"]].append(corrected_record)

    corrected["classes"] = []
    for class_name in sorted(grouped):
        records = grouped[class_name]
        corrected["classes"].append(
            {
                "class_name": class_name,
                "num_samples": len(records),
                "num_valid": sum(1 for record in records if record.get("status") == "ok"),
                "num_errors": sum(1 for record in records if record.get("status") != "ok"),
                "images": records,
            }
        )
    return corrected


def relabel_recursive_summary(summary, labels_by_image):
    corrected = copy.deepcopy(summary)
    corrected["label_source"] = {
        "type": "coco_instances",
        "note": "Labels corrected from class-specific COCO annotations. Multi-class images are marked mixed.",
    }
    corrected_images = []
    for record in summary.get("images", []):
        corrected_record = label_record(record, labels_by_image)
        corrected_images.append(corrected_record)
    corrected["images"] = corrected_images
    corrected["label_counts"] = dict(sorted(Counter(record["class_name"] for record in corrected_images).items()))
    return corrected


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def write_audit(path, corrected_images):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        file_handle.write("sample_id,image_file,corrected_class,label_status,coco_labels\n")
        for record in corrected_images:
            labels = ";".join(record.get("coco_labels", []))
            file_handle.write(
                f"{record.get('sample_id','')},"
                f"{os.path.basename(record.get('image_path',''))},"
                f"{record.get('class_name','')},"
                f"{record.get('label_status','')},"
                f"{labels}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Relabel old class-agnostic Fiedler outputs using COCO instance labels.")
    parser.add_argument("--instances-json", required=True, help="Class-specific COCO instances JSON")
    parser.add_argument("--fiedler-summary", default=None, help="Old fiedler_class_transfer_summary.json")
    parser.add_argument("--recursive-summary", default=None, help="Old recursive_fiedler_batch_summary.json")
    parser.add_argument("--out-dir", required=True, help="Directory for corrected summaries and audit CSV")
    args = parser.parse_args()

    labels_by_image = load_coco_labels(args.instances_json)
    all_images = []

    if args.fiedler_summary:
        with open(args.fiedler_summary, "r", encoding="utf-8") as file_handle:
            summary = json.load(file_handle)
        corrected = relabel_fiedler_summary(summary, labels_by_image)
        write_json(os.path.join(args.out_dir, "fiedler_class_transfer_summary_relabelled.json"), corrected)
        for class_summary in corrected["classes"]:
            all_images.extend(class_summary["images"])

    if args.recursive_summary:
        with open(args.recursive_summary, "r", encoding="utf-8") as file_handle:
            summary = json.load(file_handle)
        corrected = relabel_recursive_summary(summary, labels_by_image)
        write_json(os.path.join(args.out_dir, "recursive_fiedler_batch_summary_relabelled.json"), corrected)
        all_images.extend(corrected["images"])

    if all_images:
        write_audit(os.path.join(args.out_dir, "relabel_audit.csv"), all_images)
        counts = Counter(record["class_name"] for record in all_images)
        print("Corrected labels:")
        for class_name, count in sorted(counts.items()):
            print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
