#!/usr/bin/env python3
"""Zero-shot INSID3 candidate generation evaluation on COCO-20i."""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

# Add INSID3 to path
INSID3_ROOT = Path(__file__).parent.parent / "INSID3"
sys.path.insert(0, str(INSID3_ROOT))

from models import build_insid3_from_args
import opts
from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import build_transform, load_image


def mask_iou(mask_a, mask_b):
    """Compute IoU between two binary masks."""
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection) / float(union)


def resize_binary_mask(mask, fixed_size):
    """Resize binary mask to fixed size."""
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (fixed_size, fixed_size), Image.NEAREST)) > 0


def load_instance_records(annotation_json, image_root, max_images=None):
    """Load COCO instances for evaluation."""
    coco = COCO(annotation_json)
    records_by_image = defaultdict(list)
    image_ids = sorted(coco.getImgIds())

    for image_id in image_ids[:max_images] if max_images else image_ids:
        image_info = coco.loadImgs([image_id])[0]
        image_path = os.path.join(image_root, image_info["file_name"])
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        annotations = coco.loadAnns(ann_ids)

        image_records = []
        for annotation in annotations:
            if annotation.get("iscrowd", 0):
                continue
            segmentation = annotation.get("segmentation")
            if not segmentation:
                continue
            gt_mask = coco.annToMask(annotation).astype(bool)
            if not gt_mask.any():
                continue

            image_records.append({
                "annotation_id": annotation["id"],
                "image_id": image_id,
                "image_path": image_path,
                "image_size": (image_info["width"], image_info["height"]),
                "category_id": annotation["category_id"],
                "gt_mask": gt_mask,
            })

        if image_records:
            records_by_image[image_id] = image_records

    return records_by_image


@torch.no_grad()
def extract_and_cluster(model, image_path, device, tau=0.6):
    """Extract DINOv3 features and run zero-shot clustering."""
    img_tensor, orig_size = load_image(image_path, model._transform, device)

    # Extract features
    fmaps = model._extract_features(img_tensor.unsqueeze(0))
    fmaps_norm = F.normalize(fmaps, p=2, dim=2)
    _, _, C, h, w = fmaps_norm.shape

    # Fine-grained clustering (zero-shot, no reference)
    feat_flat = fmaps_norm[0, 0].reshape(C, -1).permute(1, 0)
    cluster_labels = agglomerative_clustering(feat_flat, tau).reshape(h, w)
    K = int(cluster_labels.max().item()) + 1

    # Generate cluster masks
    cluster_masks = []
    for k in range(K):
        cluster_mask = (cluster_labels == k).cpu().numpy()
        # Resize to original image size
        resized_mask = np.array(Image.fromarray(cluster_mask.astype(np.uint8) * 255).resize(
            orig_size, Image.NEAREST)) > 128
        cluster_masks.append(resized_mask)

    return cluster_masks


def evaluate_image(model, record, device, fixed_size=480, tau=0.6):
    """Evaluate all GT instances in an image using zero-shot INSID3 clustering."""
    image_path = record[0]["image_path"]

    try:
        # Generate cluster candidates
        cluster_masks = extract_and_cluster(model, image_path, device, tau)

        # Evaluate each GT instance
        results = []
        for instance_record in record:
            gt_mask = instance_record["gt_mask"]
            gt_mask_resized = resize_binary_mask(gt_mask, fixed_size)

            # Find best IoU match among clusters
            best_iou = 0.0
            for cluster_mask in cluster_masks:
                cluster_mask_resized = resize_binary_mask(cluster_mask, fixed_size)
                iou = mask_iou(gt_mask_resized, cluster_mask_resized)
                best_iou = max(best_iou, iou)

            results.append({
                "image_id": instance_record["image_id"],
                "annotation_id": instance_record["annotation_id"],
                "category_id": instance_record["category_id"],
                "best_iou": best_iou,
            })

        return results, None

    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser("Zero-shot INSID3 evaluation on COCO-20i")
    parser.add_argument("--annotations", required=True, help="COCO annotations JSON")
    parser.add_argument("--image-root", required=True, help="Image directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")
    parser.add_argument("--tau", type=float, default=0.6, help="Clustering threshold")
    parser.add_argument("--fixed-size", type=int, default=480, help="Fixed size for evaluation")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-size", default="large", choices=["small", "base", "large"], help="DINOv3 size")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build INSID3 model (zero-shot, no reference)
    insid3_args = argparse.Namespace(
        model_size=args.model_size,
        image_size=1024,
        svd_comps=500,
        tau=args.tau,
        merge_thresh=0.2,
        crf_size=640,
        crf_mask_refinement=False,
        device=args.device,
        seed=args.seed,
    )

    print("Loading INSID3 model...")
    model = build_insid3_from_args(insid3_args)
    model.eval()

    # Load dataset
    print("Loading dataset...")
    records_by_image = load_instance_records(
        args.annotations, args.image_root, args.max_images
    )

    # Evaluate
    csv_path = os.path.join(args.output_dir, "instance_results.csv")
    failures_path = os.path.join(args.output_dir, "failures.jsonl")

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "image_id", "annotation_id", "category_id", "best_iou"
    ])
    csv_writer.writeheader()

    failures_file = open(failures_path, "w")

    all_ious = []
    num_instances = 0
    elapsed_times = []

    pbar = tqdm(records_by_image.items(), desc="Evaluating")
    for image_id, records in pbar:
        start_time = time.time()

        results, error = evaluate_image(model, records, args.device, args.fixed_size, args.tau)
        elapsed = time.time() - start_time
        elapsed_times.append(elapsed)

        if error:
            failures_file.write(json.dumps({
                "image_id": image_id,
                "error": error,
                "elapsed": elapsed,
            }) + "\n")
            failures_file.flush()
            pbar.set_description(f"Error on image {image_id}")
        else:
            for result in results:
                csv_writer.writerow(result)
                all_ious.append(result["best_iou"])
                num_instances += 1

            mean_elapsed = np.mean(elapsed_times)
            remaining = len(records_by_image) - len(elapsed_times)
            eta = mean_elapsed * remaining

            pbar.set_description(
                f"mIoU: {np.mean(all_ious):.4f} | "
                f"elapsed: {elapsed:.2f}s | ETA: {eta/60:.1f}m"
            )

        csv_file.flush()

    csv_file.close()
    failures_file.close()

    # Summary
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    median_iou = np.median(all_ious) if all_ious else 0.0

    summary = {
        "num_images": len(records_by_image),
        "num_instances": num_instances,
        "mean_best_iou": float(mean_iou),
        "median_best_iou": float(median_iou),
        "tau": args.tau,
        "fixed_size": args.fixed_size,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nZero-shot INSID3 Results:")
    print(f"  Images: {summary['num_images']}")
    print(f"  Instances: {summary['num_instances']}")
    print(f"  Mean IoU: {summary['mean_best_iou']:.4f}")
    print(f"  Median IoU: {summary['median_best_iou']:.4f}")


if __name__ == "__main__":
    main()
