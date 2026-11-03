#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CUTLER_ROOT = os.path.join(WORKSPACE_ROOT, "CutLER Implementation", "TFSS-CutLER")
MASKCUT_ROOT = os.path.join(CUTLER_ROOT, "maskcut")
if CUTLER_ROOT not in sys.path:
    sys.path.insert(0, CUTLER_ROOT)
if MASKCUT_ROOT not in sys.path:
    sys.path.insert(0, MASKCUT_ROOT)

from fiedler_class_transfer import build_backbone, overlay_mask, resize_stretch, resize_with_aspect_ratio, upsample_mask_with_metadata  # noqa: E402
from recursive_fiedler import collect_recursive_candidates, safe_name, upsample_mask  # noqa: E402


def mask_iou(mask_a, mask_b):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection) / float(union)


def resize_binary_mask(mask, fixed_size, metadata=None):
    """Resize binary mask, optionally using aspect-ratio preservation metadata."""
    if metadata is None:
        return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((fixed_size, fixed_size), Image.NEAREST)) > 0

    if fixed_size <= 0 or not metadata:
        return mask

    scale = metadata.get("scale", 1.0)
    pad_left = metadata.get("pad_left", 0)
    pad_top = metadata.get("pad_top", 0)
    resized_w = metadata.get("resized_w", int(metadata["orig_w"] * scale))
    resized_h = metadata.get("resized_h", int(metadata["orig_h"] * scale))

    # Downscale (and, for stretch mode, independently re-aspect-ratio) mask to
    # match the resized image. Gated on actual target size vs. the mask's own
    # size rather than a single uniform "scale" value, since stretch mode has
    # no single scale (x and y scale independently) -- this is also correct
    # for the letterbox path, which always has resized_w/h != the mask's size
    # for any real (non-1.0-scale) image.
    if (resized_w, resized_h) != (mask.shape[1], mask.shape[0]):
        downscaled = Image.fromarray(mask.astype(np.uint8) * 255)
        downscaled = downscaled.resize((resized_w, resized_h), Image.NEAREST)
        mask = np.asarray(downscaled) > 128
    else:
        mask = mask.copy()

    # Add padding
    padded = np.zeros((fixed_size, fixed_size), dtype=bool)
    padded[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = mask
    return padded


def load_instance_records(annotation_json, image_root, max_images=None, max_instances=None):
    coco = COCO(annotation_json)
    categories = {cat["id"]: cat["name"] for cat in coco.loadCats(sorted(coco.getCatIds()))}

    records_by_image = defaultdict(list)
    image_ids = sorted(coco.getImgIds())
    image_count = 0
    instance_count = 0

    for image_id in image_ids:
        if max_images is not None and image_count >= max_images:
            break

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

            image_records.append(
                {
                    "image_id": image_id,
                    "annotation_id": annotation["id"],
                    "category_id": annotation["category_id"],
                    "category_name": categories.get(annotation["category_id"], f"category_{annotation['category_id']}"),
                    "image_path": image_path,
                    "gt_mask": gt_mask,
                    "width": image_info["width"],
                    "height": image_info["height"],
                }
            )
            instance_count += 1
            if max_instances is not None and instance_count >= max_instances:
                break

        if image_records:
            records_by_image[image_id] = image_records
            image_count += 1
        if max_instances is not None and instance_count >= max_instances:
            break

    return records_by_image


def prepare_args(namespace, image_path, resize_metadata=None):
    image_args = argparse.Namespace(**vars(namespace))
    image_args.image_path = image_path
    image_args.resize_metadata = resize_metadata
    return image_args


def candidate_masks_from_tree(tree, fixed_size, metadata=None, mask_interp="nearest"):
    """Extract candidate masks from tree, upsampling to fixed_size."""
    # Always return masks in fixed_size coordinates (with padding)
    # This matches how gt_mask_fixed is computed
    return [upsample_mask(node["region_mask"], fixed_size, mode=mask_interp) for node in tree["nodes"]]


def overlay_with_label(image_array, mask, color, label_text):
    canvas = overlay_mask(image_array, mask, color=color, alpha=0.5)
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    draw.rectangle((0, 0, pil_canvas.width, 28), fill=(255, 255, 255))
    draw.text((6, 6), label_text, fill=(0, 0, 0))
    return pil_canvas


def build_sample_panel(image_path, gt_mask, candidate_mask, fixed_size, title, metadata=None):
    """Build visualization panel with ground truth and best candidate."""
    image_pil = Image.open(image_path).convert("RGB")

    if fixed_size > 0:
        # Stretch-mode metadata carries scale_x/scale_y; match the masks' geometry.
        if metadata and "scale_x" in metadata:
            # Display at the true aspect ratio: un-stretch the fixed_size x fixed_size
            # masks back to the original proportions instead of showing a squashed image.
            display_size = (fixed_size, max(1, round(fixed_size * metadata["orig_h"] / metadata["orig_w"])))
            image_pil = image_pil.resize(display_size, Image.LANCZOS)
            gt_mask = np.asarray(Image.fromarray(np.asarray(gt_mask).astype(np.uint8) * 255).resize(display_size, Image.NEAREST)) > 127
            candidate_mask = np.asarray(Image.fromarray(np.asarray(candidate_mask).astype(np.uint8) * 255).resize(display_size, Image.NEAREST)) > 127
        else:
            image_pil, _ = resize_with_aspect_ratio(image_pil, fixed_size)
    else:
        # Scale to reasonable size for visualization if not using fixed size
        max_dim = 480
        image_pil.thumbnail((max_dim, max_dim), Image.LANCZOS)

    image_array = np.asarray(image_pil)
    display_h, display_w = image_array.shape[:2]

    gt_panel = overlay_with_label(image_array, gt_mask, (230, 65, 65), "ground truth")
    candidate_panel = overlay_with_label(image_array, candidate_mask, (70, 200, 110), "best candidate")

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
            sample.get("resize_metadata"),
        )
        for sample in samples
    ]

    if not tiles:
        return None

    # Layout tiles with variable sizes
    cols = 2 if len(tiles) > 1 else 1
    rows = (len(tiles) + cols - 1) // cols

    # Compute canvas size based on actual tile dimensions
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


def write_results_csv(results, output_path):
    fieldnames = [
        "image_id",
        "annotation_id",
        "category_id",
        "category_name",
        "best_iou",
        "best_candidate_id",
        "best_candidate_depth",
        "best_candidate_status",
        "best_candidate_area_patches",
        "candidate_count",
        "image_path",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_summary_json(summary, output_path):
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2)


def evaluate_dataset(args):
    os.makedirs(args.output_dir, exist_ok=True)
    records_by_image = load_instance_records(args.annotations, args.image_root, max_images=args.max_images, max_instances=args.max_instances)
    if not records_by_image:
        raise ValueError(f"No valid instances found in {args.annotations}")

    backbone = build_backbone(
        args.vit_arch,
        args.vit_feat,
        args.patch_size,
        args.pretrain_path,
        dino_version=args.dino_version,
        debiased=getattr(args, "debiased", False),
    )
    if not args.cpu and args.dino_version == "v1":
        backbone.half()
        backbone.cuda()

    all_results = []
    class_scores = defaultdict(list)
    per_class_stats = defaultdict(lambda: {"count": 0, "sum": 0.0, "scores": []})
    per_image_best = []
    image_summaries = []
    total_images = len(records_by_image)

    # Set up incremental CSV writing
    results_csv = os.path.join(args.output_dir, "instance_results.csv")
    csv_fieldnames = [
        "image_id",
        "annotation_id",
        "category_id",
        "category_name",
        "best_iou",
        "best_candidate_id",
        "best_candidate_depth",
        "best_candidate_status",
        "best_candidate_area_patches",
        "candidate_count",
        "image_path",
    ]
    csv_file = open(results_csv, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    # Failures log for fault isolation
    failures_log = os.path.join(args.output_dir, "failures.jsonl")
    failures_file = open(failures_log, "w", encoding="utf-8")

    start_time = time.time()
    elapsed_times = []

    for index, (image_id, records) in enumerate(records_by_image.items(), start=1):
        image_path = records[0]["image_path"]
        image_start = time.time()

        try:
            # Stage 1: Load and prepare image
            t1 = time.time()
            image_pil = Image.open(image_path).convert("RGB")
            if getattr(args, "resize_mode", "letterbox") == "stretch":
                image_resized, resize_metadata = resize_stretch(image_pil, args.fixed_size)
            else:
                image_resized, resize_metadata = resize_with_aspect_ratio(image_pil, args.fixed_size)
            time_load_image = time.time() - t1

            # Stage 2: Recursive Fiedler splitting (includes feature extraction + graph + eigensolver)
            t2 = time.time()
            tree_args = prepare_args(args, image_path, resize_metadata)
            tree = collect_recursive_candidates(tree_args, backbone=backbone)
            time_fiedler = time.time() - t2

            # Stage 3: Extract candidate masks from tree
            t3 = time.time()
            candidate_masks = candidate_masks_from_tree(tree, args.fixed_size, resize_metadata)
            time_mask_extraction = time.time() - t3

            if not candidate_masks:
                continue

            # Stage 4: Evaluate instances
            t4 = time.time()
            image_results = []

            if args.eval_mode == "class":
                # Class-level evaluation: group by category and union all instances per class
                records_by_category = defaultdict(list)
                for record in records:
                    records_by_category[record["category_id"]].append(record)

                for category_id, class_records in records_by_category.items():
                    class_masks_fixed = [
                        resize_binary_mask(record["gt_mask"], args.fixed_size, resize_metadata)
                        for record in class_records
                    ]
                    class_mask_union = np.logical_or.reduce(class_masks_fixed) if class_masks_fixed else np.zeros((args.fixed_size, args.fixed_size), dtype=bool)

                    # Find best candidate for class mask
                    best_iou = -1.0
                    best_candidate_index = -1
                    for candidate_index, candidate_mask in enumerate(candidate_masks):
                        iou = mask_iou(candidate_mask, class_mask_union)
                        if iou > best_iou:
                            best_iou = iou
                            best_candidate_index = candidate_index

                    class_name = class_records[0]["category_name"]
                    best_node = tree["nodes"][best_candidate_index]
                    best_candidate_mask = candidate_masks[best_candidate_index]

                    result = {
                        "image_id": image_id,
                        "annotation_id": -1,
                        "category_id": category_id,
                        "category_name": class_name,
                        "title": f'{class_name} (class mask) | img {image_id} | IoU {best_iou:.3f}',
                        "best_iou": float(best_iou),
                        "best_candidate_id": best_node["id"],
                        "best_candidate_depth": best_node["depth"],
                        "best_candidate_status": best_node["status"],
                        "best_candidate_area_patches": best_node["region_area_patches"],
                        "candidate_count": len(candidate_masks),
                        "image_path": records[0]["image_path"],
                        "best_candidate_label": best_node["status"],
                        "gt_mask_fixed": class_mask_union,
                        "best_candidate_mask": best_candidate_mask,
                    }
                    all_results.append(result)
                    image_results.append(result)
                    class_scores[class_name].append(float(best_iou))
                    per_class_stats[class_name]["count"] += 1
                    per_class_stats[class_name]["sum"] += float(best_iou)
                    per_class_stats[class_name]["scores"].append(float(best_iou))

                    csv_row = {key: result.get(key) for key in csv_fieldnames}
                    csv_writer.writerow(csv_row)

                del records_by_category, class_masks_fixed, class_mask_union
                gc.collect()

            else:  # instance-level evaluation
                # Evaluate each instance individually with its own annotation ID
                for record in records:
                    gt_mask_fixed = resize_binary_mask(record["gt_mask"], args.fixed_size, resize_metadata)

                    # Find best candidate for this instance
                    best_iou = -1.0
                    best_candidate_index = -1
                    for candidate_index, candidate_mask in enumerate(candidate_masks):
                        iou = mask_iou(candidate_mask, gt_mask_fixed)
                        if iou > best_iou:
                            best_iou = iou
                            best_candidate_index = candidate_index

                    category_name = record["category_name"]
                    best_node = tree["nodes"][best_candidate_index]
                    best_candidate_mask = candidate_masks[best_candidate_index]

                    result = {
                        "image_id": image_id,
                        "annotation_id": record["annotation_id"],
                        "category_id": record["category_id"],
                        "category_name": category_name,
                        "title": f'{category_name} (instance {record["annotation_id"]}) | img {image_id} | IoU {best_iou:.3f}',
                        "best_iou": float(best_iou),
                        "best_candidate_id": best_node["id"],
                        "best_candidate_depth": best_node["depth"],
                        "best_candidate_status": best_node["status"],
                        "best_candidate_area_patches": best_node["region_area_patches"],
                        "candidate_count": len(candidate_masks),
                        "image_path": record["image_path"],
                        "best_candidate_label": best_node["status"],
                        "gt_mask_fixed": gt_mask_fixed,
                        "best_candidate_mask": best_candidate_mask,
                        "resize_metadata": resize_metadata,
                    }
                    all_results.append(result)
                    image_results.append(result)
                    class_scores[category_name].append(float(best_iou))
                    per_class_stats[category_name]["count"] += 1
                    per_class_stats[category_name]["sum"] += float(best_iou)
                    per_class_stats[category_name]["scores"].append(float(best_iou))

                    csv_row = {key: result.get(key) for key in csv_fieldnames}
                    csv_writer.writerow(csv_row)

                gc.collect()

            time_iou_matching = time.time() - t4

            if image_results:
                per_image_best.append(max(image_results, key=lambda item: item["best_iou"]))
                image_summaries.append(
                    {
                        "image_id": image_id,
                        "image_path": image_path,
                        "num_instances": len(image_results),
                        "best_instance_iou": max(item["best_iou"] for item in image_results),
                        "mean_instance_iou": float(np.mean([item["best_iou"] for item in image_results])),
                    }
                )

                # Generate candidate tree visualization if enabled
                if args.visualize_candidates:
                    try:
                        from visualize_candidates import save_candidate_visualization
                        viz_dir = os.path.join(args.output_dir, "candidate_visualizations")
                        os.makedirs(viz_dir, exist_ok=True)
                        highlights = {r["best_candidate_id"]: f'{r["category_name"]} {r["best_iou"]:.2f}' for r in image_results}
                        save_candidate_visualization(image_id, image_path, tree, viz_dir, resize_metadata=resize_metadata, fixed_size=args.fixed_size, highlights=highlights)
                    except Exception as viz_err:
                        print(f"  [Visualization warning] Failed to visualize candidates for {image_id}: {viz_err}", flush=True)

            # Timing and progress logging
            elapsed = time.time() - image_start
            elapsed_times.append(elapsed)
            avg_elapsed = np.mean(elapsed_times)
            remaining = total_images - index
            eta_seconds = avg_elapsed * remaining
            eta_minutes = eta_seconds / 60.0

            # Print detailed timing breakdown
            num_candidates = len(candidate_masks)
            print(f"[{index}/{total_images}] {os.path.basename(image_path)} ({len(records)} instances, {num_candidates} candidates)", flush=True)
            print(f"  Timing: load={time_load_image:.3f}s | fiedler={time_fiedler:.3f}s | masks={time_mask_extraction:.3f}s | iou={time_iou_matching:.3f}s | total={elapsed:.2f}s | ETA={eta_minutes:.1f}m", flush=True)

            # Clean up GPU and CPU memory after each image to prevent accumulation
            # Explicitly delete large intermediate variables
            del image_pil, image_resized, tree, candidate_masks, image_results, records
            if 'image_args' in locals():
                del image_args

            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

            # Only aggressive gc for DINOv3 (has memory issues); DINOv2 doesn't need it
            if args.dino_version == "v3":
                # Force garbage collection aggressively - multiple passes
                for _ in range(3):
                    gc.collect()

                # On every 5th image, do an additional deep clean
                if index % 5 == 0:
                    for _ in range(5):
                        gc.collect()
            else:
                # Single gc.collect() for other versions
                gc.collect()

        except Exception as e:
            elapsed = time.time() - image_start
            error_msg = f"Image {image_id} ({image_path}) failed after {elapsed:.2f}s: {str(e)}"
            print(f"ERROR: {error_msg}", flush=True)
            traceback.print_exc()
            failures_file.write(json.dumps({"image_id": image_id, "image_path": image_path, "error": str(e), "elapsed": elapsed}) + "\n")
            failures_file.flush()

            # GPU and CPU recovery: try to free memory
            # Aggressively delete any variables that might exist
            for var_name in ['image_pil', 'image_resized', 'tree', 'candidate_masks', 'image_results', 'records', 'tree_args', 'image_args', 'gt_mask_fixed', 'best_candidate_mask']:
                try:
                    del locals()[var_name]
                except (KeyError, TypeError):
                    pass

            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

            # Only aggressive gc for DINOv3 (has memory issues)
            if args.dino_version == "v3":
                # Force aggressive garbage collection after error
                for _ in range(5):
                    gc.collect()
            else:
                gc.collect()
            continue

    csv_file.close()
    failures_file.close()

    if not all_results:
        print("WARNING: No valid evaluation results were produced. Partial results may be in the CSV.")

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

    overall_scores = [result["best_iou"] for result in all_results]
    per_image_scores = [summary["best_instance_iou"] for summary in image_summaries]
    top_samples = sorted(all_results, key=lambda item: item["best_iou"], reverse=True)[: args.top_k]
    bottom_samples = sorted(all_results, key=lambda item: item["best_iou"])[: args.top_k]

    # Compute aggregate stats safely
    if overall_scores:
        mean_best_iou = float(np.mean(overall_scores))
        median_best_iou = float(np.median(overall_scores))
        min_best_iou = float(np.min(overall_scores))
        max_best_iou = float(np.max(overall_scores))
    else:
        mean_best_iou = median_best_iou = min_best_iou = max_best_iou = 0.0

    if per_image_scores:
        mean_best_iou_per_image = float(np.mean(per_image_scores))
        median_best_iou_per_image = float(np.median(per_image_scores))
    else:
        mean_best_iou_per_image = median_best_iou_per_image = 0.0

    report = {
        "dataset_name": args.dataset_name,
        "eval_mode": args.eval_mode,
        "annotations": os.path.abspath(args.annotations),
        "image_root": os.path.abspath(args.image_root),
        "fixed_size": args.fixed_size,
        "patch_size": args.patch_size,
        "tau": args.tau,
        "num_images": len(records_by_image),
        "num_instances": len(all_results),
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

    # Note: results_csv was already written incrementally, no need to call write_results_csv
    summary_json = os.path.join(args.output_dir, "summary.json")
    write_summary_json(report, summary_json)

    # Plot only if there are results
    if overall_scores:
        plot_overall_histogram(overall_scores, os.path.join(args.output_dir, "overall_iou_histogram.png"), f"{args.dataset_name} best IoU distribution")
    if per_class_summary:
        plot_class_overview(per_class_summary, os.path.join(args.output_dir, "class_mean_iou.png"), f"{args.dataset_name} mean best IoU by class")
    class_histogram_dir = os.path.join(args.output_dir, "class_histograms")
    if class_scores:
        class_index = plot_class_histograms(class_scores, class_histogram_dir, args.dataset_name)
    else:
        class_index = None

    top_gallery = save_gallery(top_samples, os.path.join(args.output_dir, "top_10_gallery.png"), args.fixed_size, f"{args.dataset_name} top {args.top_k} instances by best IoU")
    bottom_gallery = save_gallery(bottom_samples, os.path.join(args.output_dir, "bottom_10_gallery.png"), args.fixed_size, f"{args.dataset_name} bottom {args.top_k} instances by best IoU")

    report_md = os.path.join(args.output_dir, "report.md")
    with open(report_md, "w", encoding="utf-8") as file_handle:
        file_handle.write(f"# {args.dataset_name} recursive Fiedler evaluation\n\n")
        file_handle.write(f"- eval mode: {args.eval_mode.upper()}\n")
        file_handle.write(f"- annotations: {os.path.abspath(args.annotations)}\n")
        file_handle.write(f"- image root: {os.path.abspath(args.image_root)}\n")
        file_handle.write(f"- images: {len(records_by_image)}\n")
        file_handle.write(f"- instances: {len(all_results)}\n")
        file_handle.write(f"- mean best IoU: {report['mean_best_iou']:.4f}\n")
        file_handle.write(f"- median best IoU: {report['median_best_iou']:.4f}\n")
        file_handle.write(f"- mean best IoU per image: {report['mean_best_iou_per_image']:.4f}\n\n")
        file_handle.write(f"- results CSV: {os.path.basename(results_csv)}\n")
        file_handle.write(f"- summary JSON: {os.path.basename(summary_json)}\n")
        if overall_scores:
            file_handle.write(f"- overall histogram: {os.path.basename(args.output_dir)}/overall_iou_histogram.png\n")
        if per_class_summary:
            file_handle.write(f"- class mean chart: {os.path.basename(args.output_dir)}/class_mean_iou.png\n")
        if class_index is not None:
            file_handle.write(f"- class histograms: {os.path.basename(class_index)}\n")
        if top_gallery is not None:
            file_handle.write(f"- top gallery: {os.path.basename(top_gallery)}\n")
        if bottom_gallery is not None:
            file_handle.write(f"- bottom gallery: {os.path.basename(bottom_gallery)}\n")

    total_elapsed = time.time() - start_time
    print(f"\nCompleted in {total_elapsed:.1f}s", flush=True)
    print(f"Saved results CSV to {results_csv}", flush=True)
    print(f"Saved summary JSON to {summary_json}", flush=True)
    print(f"Saved report to {report_md}", flush=True)
    if overall_scores:
        print(f"Saved overall histogram to {os.path.join(args.output_dir, 'overall_iou_histogram.png')}", flush=True)
    if per_class_summary:
        print(f"Saved class overview to {os.path.join(args.output_dir, 'class_mean_iou.png')}", flush=True)
    if class_index is not None:
        print(f"Saved class histograms to {class_histogram_dir}", flush=True)
    if top_gallery is not None:
        print(f"Saved top gallery to {top_gallery}", flush=True)
    if bottom_gallery is not None:
        print(f"Saved bottom gallery to {bottom_gallery}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate recursive Fiedler candidate masks against COCO-format instance annotations.")
    parser.add_argument("--dataset-name", type=str, default="dataset", help="Label used in reports and plots")
    parser.add_argument("--image-root", type=str, required=True, help="Root directory containing the dataset images")
    parser.add_argument("--annotations", type=str, required=True, help="COCO-format annotation JSON with instance masks")
    parser.add_argument("--output-dir", type=str, default=os.path.join("output", "fiedler_candidate_eval"), help="Directory for reports and plots")
    parser.add_argument("--vit-arch", type=str, default="small", choices=["base", "small"], help="Backbone architecture")
    parser.add_argument("--vit-feat", type=str, default="k", choices=["k", "q", "v", "kqv"], help="Backbone feature type")
    parser.add_argument("--patch-size", type=int, default=8, choices=[16, 8, 14], help="Patch size")
    parser.add_argument("--dino-version", type=str, default="v1", choices=["v1", "v2", "v3"], help="DINO backbone version")
    parser.add_argument("--debiased", action="store_true", help="Use positional debiasing (DINOv3 only)")
    parser.add_argument("--tau", type=float, default=0.15, help="Graph threshold used by the recursive splitter")
    parser.add_argument("--eig-index", type=int, default=1, help="Which Laplacian generalized eigenvector to split on (ascending eigenvalue order; 0=trivial, 1=classic Fiedler vector [default], 2/3/4=next eigenvectors -- for the eigenvector-choice ablation)")
    parser.add_argument("--split-method", type=str, default="laplacian", choices=["laplacian", "pca"], help="Splitting criterion: 'laplacian' (default) thresholds a graph-Laplacian eigenvector (see --eig-index); 'pca' thresholds a principal component of the region's raw patch features instead (see --pca-index), no affinity graph involved")
    parser.add_argument("--pca-index", type=int, default=0, help="Which principal component to split on when --split-method=pca (0=top/most-variance component, 1=2nd, ...)")
    parser.add_argument("--resize-mode", type=str, default="letterbox", choices=["letterbox", "stretch"], help="'letterbox' (default) preserves aspect ratio and pads with black -- wastes some patch budget on padding for non-square images. 'stretch' resizes directly to fixed_size x fixed_size with no padding (every patch is real content) at the cost of distorting proportions.")
    parser.add_argument("--fixed-size", type=int, default=480, help="Square resize used before feature extraction (0 to preserve aspect ratio)")
    parser.add_argument("--pretrain-path", type=str, default=None, help="Optional local DINO checkpoint path")
    parser.add_argument("--cpu", action="store_true", help="Run the backbone on CPU")
    parser.add_argument("--visualize-candidates", action="store_true", help="Generate per-image candidate visualizations with tree structure")
    parser.add_argument("--max-images", type=int, default=None, help="Limit the number of images")
    parser.add_argument("--max-instances", type=int, default=None, help="Limit the number of instances")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum recursive split depth")
    parser.add_argument("--unlimited-depth", action="store_true", help="Disable max-depth limit (rely only on min-region-patches)")
    parser.add_argument("--unlimited-nodes", action="store_true", help="Disable max-nodes limit")
    parser.add_argument("--unlimited-noise", action="store_true", help="Disable noise-threshold filtering")
    parser.add_argument("--unlimited-imbalance", action="store_true", help="Disable min-split-ratio filtering")
    parser.add_argument("--max-nodes", type=int, default=63, help="Maximum number of solved nodes")
    parser.add_argument("--min-region-patches", type=int, default=36, help="Minimum active patch count to split")
    parser.add_argument("--min-split-ratio", type=float, default=0.05, help="Reject very imbalanced splits")
    parser.add_argument("--noise-threshold", type=float, default=3.0, help="Stop when the Fiedler field becomes too noisy")
    parser.add_argument("--top-k", type=int, default=10, help="Number of best and worst samples to export")
    parser.add_argument("--eval-mode", type=str, default="class", choices=["class", "instance"], help="Evaluate against class-level masks (union of all instances) or instance-level masks (each annotation individually)")
    return parser


def main():
    args = build_parser().parse_args()
    evaluate_dataset(args)


if __name__ == "__main__":
    main()
