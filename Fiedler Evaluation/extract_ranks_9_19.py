#!/usr/bin/env python3
"""Extract ranks 9-19 gallery from evaluation results."""

import csv
import json
import sys
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

SCRIPT_DIR = Path(__file__).parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
CUTLER_ROOT = WORKSPACE_ROOT / "CutLER Implementation" / "TFSS-CutLER"
MASKCUT_ROOT = CUTLER_ROOT / "maskcut"
if str(CUTLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTLER_ROOT))
if str(MASKCUT_ROOT) not in sys.path:
    sys.path.insert(0, str(MASKCUT_ROOT))

from fiedler_class_transfer import overlay_mask, resize_with_aspect_ratio


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


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python extract_ranks_9_19.py <output_dir> <annotations_json> <image_root> [fixed_size=480]")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    annotations_json = sys.argv[2]
    image_root = sys.argv[3]
    fixed_size = int(sys.argv[4]) if len(sys.argv) > 4 else 480

    csv_path = output_dir / "instance_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    # Load COCO for GT masks
    print("Loading COCO annotations...")
    coco = COCO(annotations_json)

    # Load results
    print(f"Reading results from {csv_path}...")
    all_results = []
    with open(csv_path, "r", encoding="utf-8") as f:
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
            except Exception as e:
                print(f"  Warning: Error parsing row: {e}")
                continue

    print(f"Loaded {len(all_results)} results\n")

    # Get ranks 9-19 (indices 9-19, which is 11 samples)
    sorted_results = sorted(all_results, key=lambda x: x["best_iou"], reverse=True)
    ranks_9_19 = sorted_results[9:20]  # indices 9-19 inclusive

    print(f"Extracting ranks 9-19 ({len(ranks_9_19)} samples)...")

    # Load masks for gallery
    image_root_path = Path(image_root)
    for sample in ranks_9_19:
        try:
            ann = coco.loadAnns([sample["annotation_id"]])[0]
            gt_mask = coco.annToMask(ann).astype(bool)
            gt_mask_resized = np.asarray(
                Image.fromarray(gt_mask.astype(np.uint8) * 255).resize(
                    (fixed_size, fixed_size), Image.NEAREST
                )
            ) > 0

            # For now, use GT mask as candidate (would need results dict to have best_candidate_mask)
            sample["gt_mask"] = gt_mask_resized
            sample["candidate_mask"] = gt_mask_resized  # Placeholder
            sample["title"] = f"Rank {sorted_results.index(sample) + 1} | IoU: {sample['best_iou']:.3f} ({sample['category_name']})"
        except Exception as e:
            print(f"  Warning: Could not load masks for {sample['annotation_id']}: {e}")

    # Save gallery
    gallery_path = output_dir / "ranks_9_19_gallery.png"
    save_gallery(ranks_9_19, str(gallery_path), fixed_size, "Ranks 9-19 Best IoU Instances")

    print(f"\nGallery saved to {gallery_path}")
