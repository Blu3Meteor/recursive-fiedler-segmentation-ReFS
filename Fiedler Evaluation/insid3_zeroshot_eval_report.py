#!/usr/bin/env python3
"""Zero-shot INSID3 candidate generation evaluation on COCO-20i, with full reporting
(matches the report structure produced by fiedler_candidate_eval.py: summary.json,
instance_results.csv, failures.jsonl, overall_iou_histogram.png, class_mean_iou.png,
class_histograms/*.png + index.md, top/bottom-K galleries, report.md)."""

import argparse
import csv
import heapq
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from tqdm import tqdm

# Add INSID3 to path
INSID3_ROOT = Path(__file__).parent.parent / "INSID3"
sys.path.insert(0, str(INSID3_ROOT))

from models import build_insid3_from_args
import opts  # noqa: F401
from utils.clustering import agglomerative_clustering
from utils.data import build_transform, load_image  # noqa: F401


def mask_iou(mask_a, mask_b):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection) / float(union)


def resize_binary_mask(mask, fixed_size):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (fixed_size, fixed_size), Image.NEAREST)) > 0


def safe_name(name):
    return str(name).strip().replace(" ", "_").replace("/", "_")


def load_instance_records(annotation_json, image_root, max_images=None):
    coco = COCO(annotation_json)
    categories = {cat["id"]: cat["name"] for cat in coco.loadCats(sorted(coco.getCatIds()))}
    records_by_image = defaultdict(list)
    image_ids = sorted(coco.getImgIds())

    for image_id in (image_ids[:max_images] if max_images else image_ids):
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
                "category_name": categories.get(annotation["category_id"], f"category_{annotation['category_id']}"),
                "gt_mask": gt_mask,
            })

        if image_records:
            records_by_image[image_id] = image_records

    return records_by_image, categories


@torch.no_grad()
def extract_and_cluster(model, image_path, device, tau=0.6):
    """Extract DINOv3 features and run zero-shot clustering."""
    img_tensor, orig_size = load_image(image_path, model._transform, device)

    fmaps = model._extract_features(img_tensor.unsqueeze(0))
    fmaps_norm = F.normalize(fmaps, p=2, dim=2)
    _, _, C, h, w = fmaps_norm.shape

    feat_flat = fmaps_norm[0, 0].reshape(C, -1).permute(1, 0)
    cluster_labels = agglomerative_clustering(feat_flat, tau).reshape(h, w)
    K = int(cluster_labels.max().item()) + 1

    cluster_masks = []
    for k in range(K):
        cluster_mask = (cluster_labels == k).cpu().numpy()
        resized_mask = np.array(Image.fromarray(cluster_mask.astype(np.uint8) * 255).resize(
            orig_size, Image.NEAREST)) > 128
        cluster_masks.append(resized_mask)

    return cluster_masks


def evaluate_image(model, record, device, fixed_size=480, tau=0.6):
    """Evaluate all GT instances in an image; return lightweight scores plus a
    per-instance best-candidate mask ONLY (not all candidates) so callers can
    build galleries without retaining every cluster mask in memory."""
    image_path = record[0]["image_path"]

    try:
        cluster_masks = extract_and_cluster(model, image_path, device, tau)
        cluster_masks_resized = [resize_binary_mask(m, fixed_size) for m in cluster_masks]

        results = []
        for instance_record in record:
            gt_mask_resized = resize_binary_mask(instance_record["gt_mask"], fixed_size)

            best_iou = -1.0
            best_idx = -1
            for idx, cluster_mask_resized in enumerate(cluster_masks_resized):
                iou = mask_iou(gt_mask_resized, cluster_mask_resized)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            category_name = instance_record["category_name"]
            results.append({
                "image_id": instance_record["image_id"],
                "annotation_id": instance_record["annotation_id"],
                "category_id": instance_record["category_id"],
                "category_name": category_name,
                "title": f'{category_name} (instance {instance_record["annotation_id"]}) | img {instance_record["image_id"]} | IoU {best_iou:.3f}',
                "best_iou": float(best_iou),
                "best_candidate_id": int(best_idx),
                "candidate_count": len(cluster_masks_resized),
                "image_path": image_path,
                "gt_mask_fixed": gt_mask_resized,
                "best_candidate_mask": cluster_masks_resized[best_idx] if best_idx >= 0 else None,
            })

        return results, None

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Reporting helpers (mirrors fiedler_candidate_eval.py's report structure)
# ---------------------------------------------------------------------------

def overlay_mask(image_array, mask, color=(255, 0, 0), alpha=0.5):
    overlay = image_array.copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image_array.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (image_array.shape[1], image_array.shape[0]), Image.NEAREST)) > 128
    color_arr = np.array(color, dtype=np.float32)
    region = overlay[mask].astype(np.float32)
    overlay[mask] = (region * (1 - alpha) + color_arr * alpha).astype(np.uint8)
    return overlay


def overlay_with_label(image_array, mask, color, label_text):
    canvas = overlay_mask(image_array, mask, color=color, alpha=0.5)
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    draw.rectangle((0, 0, pil_canvas.width, 28), fill=(255, 255, 255))
    draw.text((6, 6), label_text, fill=(0, 0, 0))
    return pil_canvas


def build_sample_panel(image_path, gt_mask, candidate_mask, fixed_size, title):
    image_pil = Image.open(image_path).convert("RGB")
    if fixed_size > 0:
        image_pil = image_pil.resize((fixed_size, fixed_size), Image.LANCZOS)
    else:
        image_pil.thumbnail((480, 480), Image.LANCZOS)

    image_array = np.asarray(image_pil)
    display_h, display_w = image_array.shape[:2]

    gt_panel = overlay_with_label(image_array, gt_mask, (230, 65, 65), "ground truth")
    if candidate_mask is not None:
        candidate_panel = overlay_with_label(image_array, candidate_mask, (70, 200, 110), "best candidate")
    else:
        candidate_panel = Image.fromarray(image_array)

    panels = [image_pil, gt_panel, candidate_panel]
    canvas = Image.new("RGB", (display_w * len(panels), display_h + 44), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 44), fill=(248, 248, 248))
    draw.text((8, 6), title, fill=(0, 0, 0))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * display_w, 44))
    return canvas


def save_gallery(samples, output_path, fixed_size, title):
    if not samples:
        return None

    tiles = [
        build_sample_panel(
            sample["image_path"],
            sample["gt_mask_fixed"],
            sample["best_candidate_mask"],
            fixed_size,
            sample["title"],
        )
        for sample in samples
    ]

    cols = 2 if len(tiles) > 1 else 1
    rows = (len(tiles) + cols - 1) // cols

    tile_widths = [tile.size[0] for tile in tiles]
    tile_heights = [tile.size[1] for tile in tiles]

    col_widths = [0] * cols
    row_heights = [0] * rows
    for idx, (w, h) in enumerate(zip(tile_widths, tile_heights)):
        col = idx % cols
        row = idx // cols
        col_widths[col] = max(col_widths[col], w)
        row_heights[row] = max(row_heights[row], h)

    total_width = sum(col_widths) + 10
    total_height = sum(row_heights) + 44 + 10

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, total_width, 44), fill=(242, 242, 242))
    draw.text((8, 6), title, fill=(0, 0, 0))

    for idx, tile in enumerate(tiles):
        col = idx % cols
        row = idx // cols
        x = sum(col_widths[:col]) + 5
        y = 44 + sum(row_heights[:row]) + 5
        canvas.paste(tile, (x, y))

    canvas.save(output_path)
    return output_path


def plot_overall_histogram(values, output_path, title):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=np.linspace(0.0, 1.0, 21), color="#4c78a8", edgecolor="white")
    plt.xlim(0.0, 1.0)
    plt.xlabel("Best IoU")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_class_overview(class_stats, output_path, title):
    ordered = sorted(class_stats.items(), key=lambda item: (-item[1]["count"], item[0]))
    labels = [name for name, _ in ordered]
    means = [stats["mean_iou"] for _, stats in ordered]

    plt.figure(figsize=(max(10, len(labels) * 0.22), 6))
    plt.bar(range(len(labels)), means, color="#72b7b2")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean best IoU")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_class_histograms(class_scores, output_dir, title_prefix):
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "index.md")
    lines = [f"# {title_prefix} class histograms", ""]

    for class_name in sorted(class_scores):
        scores = class_scores[class_name]
        if not scores:
            continue
        output_path = os.path.join(output_dir, f"{safe_name(class_name)}.png")
        plt.figure(figsize=(6, 4))
        plt.hist(scores, bins=np.linspace(0.0, 1.0, 21), color="#f58518", edgecolor="white")
        plt.xlim(0.0, 1.0)
        plt.xlabel("Best IoU")
        plt.ylabel("Count")
        plt.title(class_name)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        lines.append(f"- {class_name}: {os.path.basename(output_path)}")

    with open(index_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines) + "\n")
    return index_path


class TopKHeap:
    """Keeps the K largest (or smallest, via negation) items by IoU without storing all results."""

    def __init__(self, k, largest=True):
        self.k = k
        self.largest = largest
        self.heap = []  # min-heap of (sort_key, counter, result)
        self.counter = 0

    def push(self, result):
        key = result["best_iou"] if self.largest else -result["best_iou"]
        self.counter += 1
        entry = (key, self.counter, result)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, entry)
        elif key > self.heap[0][0]:
            heapq.heapreplace(self.heap, entry)

    def sorted_results(self):
        ordered = sorted(self.heap, key=lambda item: item[0], reverse=True)
        return [item[2] for item in ordered]


def main():
    parser = argparse.ArgumentParser("Zero-shot INSID3 evaluation on COCO-20i with full reporting")
    parser.add_argument("--dataset-name", type=str, default="insid3_zeroshot", help="Label used in reports and plots")
    parser.add_argument("--annotations", required=True, help="COCO annotations JSON")
    parser.add_argument("--image-root", required=True, help="Image directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")
    parser.add_argument("--tau", type=float, default=0.6, help="Clustering threshold")
    parser.add_argument("--fixed-size", type=int, default=480, help="Fixed size for evaluation")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-size", default="large", choices=["small", "base", "large"], help="DINOv3 size")
    parser.add_argument("--top-k", type=int, default=10, help="Number of best and worst samples to export")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

    print("Loading dataset...")
    records_by_image, categories = load_instance_records(args.annotations, args.image_root, args.max_images)
    total_images = len(records_by_image)

    csv_path = os.path.join(args.output_dir, "instance_results.csv")
    failures_path = os.path.join(args.output_dir, "failures.jsonl")
    csv_fieldnames = [
        "image_id", "annotation_id", "category_id", "category_name",
        "best_iou", "best_candidate_id", "candidate_count", "image_path",
    ]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    failures_file = open(failures_path, "w", encoding="utf-8")

    all_ious = []
    per_image_scores = []
    image_summaries = []
    class_scores = defaultdict(list)
    per_class_stats = defaultdict(lambda: {"count": 0, "scores": []})
    top_heap = TopKHeap(args.top_k, largest=True)
    bottom_heap = TopKHeap(args.top_k, largest=False)

    elapsed_times = []

    pbar = tqdm(records_by_image.items(), desc="Evaluating", total=total_images)
    for image_id, records in pbar:
        start_time = time.time()

        results, error = evaluate_image(model, records, args.device, args.fixed_size, args.tau)
        elapsed = time.time() - start_time
        elapsed_times.append(elapsed)

        if error:
            failures_file.write(json.dumps({
                "image_id": image_id,
                "image_path": records[0]["image_path"],
                "error": error,
                "elapsed": elapsed,
            }) + "\n")
            failures_file.flush()
            pbar.set_description(f"Error on image {image_id}")
        else:
            image_ious = []
            for result in results:
                csv_writer.writerow({key: result.get(key) for key in csv_fieldnames})
                all_ious.append(result["best_iou"])
                image_ious.append(result["best_iou"])
                class_scores[result["category_name"]].append(result["best_iou"])
                per_class_stats[result["category_name"]]["count"] += 1
                per_class_stats[result["category_name"]]["scores"].append(result["best_iou"])
                top_heap.push(result)
                bottom_heap.push(result)

            if image_ious:
                per_image_scores.append(max(image_ious))
                image_summaries.append({
                    "image_id": image_id,
                    "image_path": records[0]["image_path"],
                    "num_instances": len(image_ious),
                    "best_instance_iou": max(image_ious),
                    "mean_instance_iou": float(np.mean(image_ious)),
                })

            mean_elapsed = np.mean(elapsed_times)
            remaining = total_images - len(elapsed_times)
            eta = mean_elapsed * remaining
            pbar.set_description(f"mIoU: {np.mean(all_ious):.4f} | elapsed: {elapsed:.2f}s | ETA: {eta/60:.1f}m")

        csv_file.flush()

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    csv_file.close()
    failures_file.close()

    per_class_summary = {}
    for class_name, stats in per_class_stats.items():
        scores = stats["scores"]
        if scores:
            per_class_summary[class_name] = {
                "count": int(stats["count"]),
                "mean_iou": float(np.mean(scores)),
                "median_iou": float(np.median(scores)),
                "min_iou": float(np.min(scores)),
                "max_iou": float(np.max(scores)),
                "scores": scores,
            }

    mean_best_iou = float(np.mean(all_ious)) if all_ious else 0.0
    median_best_iou = float(np.median(all_ious)) if all_ious else 0.0
    min_best_iou = float(np.min(all_ious)) if all_ious else 0.0
    max_best_iou = float(np.max(all_ious)) if all_ious else 0.0
    mean_best_iou_per_image = float(np.mean(per_image_scores)) if per_image_scores else 0.0
    median_best_iou_per_image = float(np.median(per_image_scores)) if per_image_scores else 0.0

    top_samples = top_heap.sorted_results()
    bottom_samples = bottom_heap.sorted_results()

    summary = {
        "dataset_name": args.dataset_name,
        "annotations": os.path.abspath(args.annotations),
        "image_root": os.path.abspath(args.image_root),
        "fixed_size": args.fixed_size,
        "tau": args.tau,
        "model_size": args.model_size,
        "num_images": total_images,
        "num_instances": len(all_ious),
        "mean_best_iou": mean_best_iou,
        "median_best_iou": median_best_iou,
        "min_best_iou": min_best_iou,
        "max_best_iou": max_best_iou,
        "mean_best_iou_per_image": mean_best_iou_per_image,
        "median_best_iou_per_image": median_best_iou_per_image,
        "per_class": per_class_summary,
        "image_summaries": image_summaries,
        "top_samples": [
            {key: value for key, value in sample.items() if key not in {"gt_mask_fixed", "best_candidate_mask"}}
            for sample in top_samples
        ],
        "bottom_samples": [
            {key: value for key, value in sample.items() if key not in {"gt_mask_fixed", "best_candidate_mask"}}
            for sample in bottom_samples
        ],
    }

    summary_json = os.path.join(args.output_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if all_ious:
        plot_overall_histogram(all_ious, os.path.join(args.output_dir, "overall_iou_histogram.png"), f"{args.dataset_name} best IoU distribution")
    if per_class_summary:
        plot_class_overview(per_class_summary, os.path.join(args.output_dir, "class_mean_iou.png"), f"{args.dataset_name} mean best IoU by class")
    class_histogram_dir = os.path.join(args.output_dir, "class_histograms")
    class_index = plot_class_histograms(class_scores, class_histogram_dir, args.dataset_name) if class_scores else None

    top_gallery = save_gallery(top_samples, os.path.join(args.output_dir, "top_10_gallery.png"), args.fixed_size, f"{args.dataset_name} top {args.top_k} instances by best IoU")
    bottom_gallery = save_gallery(bottom_samples, os.path.join(args.output_dir, "bottom_10_gallery.png"), args.fixed_size, f"{args.dataset_name} bottom {args.top_k} instances by best IoU")

    report_md = os.path.join(args.output_dir, "report.md")
    with open(report_md, "w", encoding="utf-8") as file_handle:
        file_handle.write(f"# {args.dataset_name} zero-shot INSID3 evaluation\n\n")
        file_handle.write(f"- annotations: {os.path.abspath(args.annotations)}\n")
        file_handle.write(f"- image root: {os.path.abspath(args.image_root)}\n")
        file_handle.write(f"- model size: {args.model_size}\n")
        file_handle.write(f"- tau: {args.tau}\n")
        file_handle.write(f"- fixed size: {args.fixed_size}\n")
        file_handle.write(f"- images: {total_images}\n")
        file_handle.write(f"- instances: {len(all_ious)}\n")
        file_handle.write(f"- mean best IoU: {summary['mean_best_iou']:.4f}\n")
        file_handle.write(f"- median best IoU: {summary['median_best_iou']:.4f}\n")
        file_handle.write(f"- mean best IoU per image: {summary['mean_best_iou_per_image']:.4f}\n\n")
        file_handle.write(f"- results CSV: {os.path.basename(csv_path)}\n")
        file_handle.write(f"- summary JSON: {os.path.basename(summary_json)}\n")
        if all_ious:
            file_handle.write(f"- overall histogram: overall_iou_histogram.png\n")
        if per_class_summary:
            file_handle.write(f"- class mean chart: class_mean_iou.png\n")
        if class_index is not None:
            file_handle.write(f"- class histograms: {os.path.basename(class_histogram_dir)}/index.md\n")
        if top_gallery is not None:
            file_handle.write(f"- top gallery: {os.path.basename(top_gallery)}\n")
        if bottom_gallery is not None:
            file_handle.write(f"- bottom gallery: {os.path.basename(bottom_gallery)}\n")

    print(f"\nZero-shot INSID3 Results:")
    print(f"  Images: {summary['num_images']}")
    print(f"  Instances: {summary['num_instances']}")
    print(f"  Mean IoU: {summary['mean_best_iou']:.4f}")
    print(f"  Median IoU: {summary['median_best_iou']:.4f}")
    print(f"Saved report to {report_md}")


if __name__ == "__main__":
    main()
