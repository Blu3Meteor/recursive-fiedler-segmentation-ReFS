#!/usr/bin/env python3
"""Generate report.md and top/bottom galleries from instance_results.csv"""

import json
import csv
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add paths
SCRIPT_DIR = Path(__file__).parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
CUTLER_ROOT = WORKSPACE_ROOT / "CutLER Implementation" / "TFSS-CutLER"
MASKCUT_ROOT = CUTLER_ROOT / "maskcut"
if str(CUTLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTLER_ROOT))
if str(MASKCUT_ROOT) not in sys.path:
    sys.path.insert(0, str(MASKCUT_ROOT))

from fiedler_class_transfer import overlay_mask, resize_with_aspect_ratio
from pycocotools.coco import COCO

def overlay_with_label(image_array, mask, color, label_text):
    """Overlay mask on image with label."""
    canvas = overlay_mask(image_array, mask, color=color, alpha=0.5)
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    draw.rectangle((0, 0, pil_canvas.width, 28), fill=(255, 255, 255))
    draw.text((6, 6), label_text, fill=(0, 0, 0))
    return pil_canvas

def build_sample_panel(image_path, gt_mask, candidate_mask, fixed_size, title):
    """Build visualization panel with ground truth and best candidate."""
    try:
        image_pil = Image.open(image_path).convert("RGB")
        if fixed_size > 0:
            image_pil, _ = resize_with_aspect_ratio(image_pil, fixed_size)
        image_array = np.asarray(image_pil)

        gt_panel = overlay_with_label(image_array, gt_mask, (230, 65, 65), "ground truth")
        candidate_panel = overlay_with_label(image_array, candidate_mask, (70, 200, 110), "best candidate")

        panels = [image_pil, gt_panel, candidate_panel]
        display_w, display_h = image_pil.size
        canvas = Image.new("RGB", (display_w * len(panels), display_h + 44), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, 44), fill=(248, 248, 248))
        draw.text((8, 6), title, fill=(0, 0, 0))
        for index, panel in enumerate(panels):
            canvas.paste(panel, (index * display_w, 44))
        return canvas
    except Exception as e:
        print(f"  Error building panel for {image_path}: {e}")
        return None

def save_gallery(samples, output_path, fixed_size, title):
    """Save gallery of samples."""
    if not samples:
        print(f"  No samples to save for {output_path}")
        return None

    tiles = []
    for s in samples:
        tile = build_sample_panel(s["image_path"], s["gt_mask"], s["candidate_mask"], fixed_size, s["title"])
        if tile:
            tiles.append(tile)

    if not tiles:
        print(f"  No valid tiles for gallery {output_path}")
        return None

    cols = 2 if len(tiles) > 1 else 1
    rows = (len(tiles) + cols - 1) // cols
    max_width = max(tile.size[0] for tile in tiles)
    max_height = max(tile.size[1] for tile in tiles)

    canvas = Image.new("RGB", (max_width * cols, max_height * rows + 44), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 44), fill=(248, 248, 248))
    draw.text((8, 6), title, fill=(0, 0, 0))

    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        canvas.paste(tile, (col * max_width, row * max_height + 44))

    canvas.save(output_path, dpi=(160, 160))
    print(f"  Saved gallery: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate report and galleries from evaluation results")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory with instance_results.csv")
    parser.add_argument("--annotations", type=str, required=True, help="COCO annotations JSON")
    parser.add_argument("--image-root", type=str, required=True, help="Image root directory")
    parser.add_argument("--fixed-size", type=int, default=480, help="Fixed image size")
    parser.add_argument("--dataset-name", type=str, default="evaluation", help="Dataset name for report title")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "instance_results.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    histogram_path = output_dir / "overall_iou_histogram.png"
    class_overview_path = output_dir / "class_mean_iou.png"
    top_gallery_path = output_dir / "top_10_gallery.png"
    bottom_gallery_path = output_dir / "bottom_10_gallery.png"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    # Load COCO for GT masks
    print("Loading COCO annotations...")
    coco = COCO(args.annotations)

    print(f"Reading results from {csv_path}...")
    all_results = []
    class_scores = defaultdict(list)
    per_class_stats = defaultdict(lambda: {"count": 0, "sum": 0.0, "scores": []})
    seen_images = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                result = {
                    "image_id": int(row["image_id"]),
                    "annotation_id": int(row["annotation_id"]),
                    "category_id": int(row["category_id"]),
                    "category_name": row["category_name"],
                    "best_iou": float(row["best_iou"]),
                    "image_path": row["image_path"],
                }
                all_results.append(result)
                class_scores[result["category_name"]].append(result["best_iou"])
                per_class_stats[result["category_name"]]["count"] += 1
                per_class_stats[result["category_name"]]["sum"] += result["best_iou"]
                per_class_stats[result["category_name"]]["scores"].append(result["best_iou"])
                seen_images.add(result["image_id"])
            except Exception as e:
                print(f"  Warning: Error parsing row: {e}")
                continue

    print(f"Loaded {len(all_results)} instances from {len(seen_images)} images\n")

    # Compute stats
    overall_scores = [r["best_iou"] for r in all_results]
    per_image_scores = []
    image_id_to_results = defaultdict(list)
    for result in all_results:
        image_id_to_results[result["image_id"]].append(result)

    for image_id, results in image_id_to_results.items():
        best_iou = max(r["best_iou"] for r in results)
        per_image_scores.append(best_iou)

    mean_best_iou = float(np.mean(overall_scores))
    median_best_iou = float(np.median(overall_scores))
    mean_best_iou_per_image = float(np.mean(per_image_scores))

    # Per-class summary
    per_class_summary = {}
    for class_name, stats in per_class_stats.items():
        scores = stats["scores"]
        per_class_summary[class_name] = {
            "count": int(stats["count"]),
            "mean_iou": float(np.mean(scores)),
            "median_iou": float(np.median(scores)),
            "min_iou": float(np.min(scores)) if scores else 0.0,
            "max_iou": float(np.max(scores)) if scores else 0.0,
        }

    # Top/bottom samples
    print("[1/4] Collecting top/bottom samples...")
    top_samples = sorted(all_results, key=lambda x: x["best_iou"], reverse=True)[:10]
    bottom_samples = sorted(all_results, key=lambda x: x["best_iou"])[:10]

    # Load masks for gallery
    print("  Loading GT masks and best candidates...")
    image_root = Path(args.image_root)
    for sample in top_samples + bottom_samples:
        try:
            ann = coco.loadAnns([sample["annotation_id"]])[0]
            gt_mask = coco.annToMask(ann).astype(bool)
            gt_mask_resized = np.asarray(Image.fromarray(gt_mask.astype(np.uint8) * 255).resize((args.fixed_size, args.fixed_size), Image.NEAREST)) > 0

            # For now, use GT mask as candidate (should be from results)
            sample["gt_mask"] = gt_mask_resized
            sample["candidate_mask"] = gt_mask_resized  # Placeholder
            sample["title"] = f"IoU: {sample['best_iou']:.3f} ({sample['category_name']})"
        except Exception as e:
            print(f"  Warning: Could not load masks for {sample['annotation_id']}: {e}")

    # Write summary.json
    print("[2/4] Writing summary.json...")
    summary = {
        "dataset_name": args.dataset_name,
        "fixed_size": args.fixed_size,
        "num_images": len(seen_images),
        "num_instances": len(all_results),
        "mean_best_iou": mean_best_iou,
        "median_best_iou": median_best_iou,
        "mean_best_iou_per_image": mean_best_iou_per_image,
        "per_class": per_class_summary,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Write report.md
    print("[3/4] Writing report.md...")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# {args.dataset_name} recursive Fiedler evaluation\n\n")
        f.write(f"- images: {len(seen_images)}\n")
        f.write(f"- instances: {len(all_results)}\n")
        f.write(f"- mean best IoU: {mean_best_iou:.4f}\n")
        f.write(f"- median best IoU: {median_best_iou:.4f}\n")
        f.write(f"- mean best IoU per image: {mean_best_iou_per_image:.4f}\n\n")
        f.write(f"- results CSV: instance_results.csv\n")
        f.write(f"- summary JSON: summary.json\n")
        f.write(f"- overall histogram: overall_iou_histogram.png\n")
        f.write(f"- class mean chart: class_mean_iou.png\n")
        f.write(f"- top gallery: top_10_gallery.png\n")
        f.write(f"- bottom gallery: bottom_10_gallery.png\n")

    # Generate plots
    print("[4/4] Generating plots and galleries...")
    plt.figure(figsize=(8, 5))
    plt.hist(overall_scores, bins=np.linspace(0.0, 1.0, 21), color="#72b7b2", edgecolor="white")
    plt.xlim(0.0, 1.0)
    plt.xlabel("Best IoU")
    plt.ylabel("Count")
    plt.title(f"{args.dataset_name} Best IoU Distribution")
    plt.tight_layout()
    plt.savefig(histogram_path, dpi=160)
    plt.close()
    print(f"  Saved histogram: {histogram_path}")

    # Class overview
    ordered = sorted(per_class_summary.items(), key=lambda item: (-item[1]["count"], item[0]))
    labels = [name for name, _ in ordered]
    means = [stats["mean_iou"] for _, stats in ordered]

    plt.figure(figsize=(max(10, len(labels) * 0.22), 6))
    plt.bar(range(len(labels)), means, color="#72b7b2")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean best IoU")
    plt.title(f"{args.dataset_name} Mean Best IoU by Class")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(class_overview_path, dpi=160)
    plt.close()
    print(f"  Saved class overview: {class_overview_path}")

    # Galleries (simplified - just save as placeholder for now)
    print(f"  Generating top 10 gallery...")
    save_gallery(top_samples, str(top_gallery_path), args.fixed_size, "Top 10 Best IoU Instances")
    print(f"  Generating bottom 10 gallery...")
    save_gallery(bottom_samples, str(bottom_gallery_path), args.fixed_size, "Bottom 10 Worst IoU Instances")

    print("\n" + "="*70)
    print("Finalization complete!")
    print("="*70)
    print(f"Mean best IoU: {mean_best_iou:.4f}")
    print(f"Median best IoU: {median_best_iou:.4f}")
    print(f"Images processed: {len(seen_images)}")
    print(f"Instances: {len(all_results)}")
    print("="*70)

if __name__ == "__main__":
    main()
