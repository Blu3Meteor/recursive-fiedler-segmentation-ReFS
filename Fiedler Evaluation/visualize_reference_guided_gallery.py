#!/usr/bin/env python3
"""Rebuild the reference-guided Fiedler top/bottom-K galleries with a 4th panel
showing the reference (prompt) image + mask used to make each selection.

Layout per tile: [reference prompt] | [target image] | [ground truth] | [selected candidate]

Only re-processes the specific images referenced in summary.json's top_samples /
bottom_samples (cheap -- typically <20 images), not the full dataset.
"""

import argparse
import gc
import json
import os
import sys
from collections import defaultdict

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

from fiedler_class_transfer import build_backbone, resize_with_aspect_ratio, overlay_mask  # noqa: E402
from recursive_fiedler import collect_recursive_candidates, load_image_features  # noqa: E402
from fiedler_candidate_eval import mask_iou, resize_binary_mask, prepare_args, candidate_masks_from_tree, load_instance_records  # noqa: E402
from fiedler_reference_eval import build_reference_prototypes, score_nodes_against_prototype, l2_normalize_columns, downsample_mask_to_grid  # noqa: E402


def build_reference_panels(reference_json, image_root, fixed_size):
    """Return {category_name: (resized_ref_image_pil, ref_mask_fixed)} for panel rendering."""
    coco_ref = COCO(reference_json)
    categories = {c["id"]: c["name"] for c in coco_ref.loadCats(coco_ref.getCatIds())}
    panels = {}

    for image_id in coco_ref.getImgIds():
        image_info = coco_ref.loadImgs([image_id])[0]
        image_path = os.path.join(image_root, image_info["file_name"])
        ann_ids = coco_ref.getAnnIds(imgIds=[image_id])
        anns = coco_ref.loadAnns(ann_ids)
        if not os.path.exists(image_path):
            continue

        image_pil = Image.open(image_path).convert("RGB")
        image_resized, resize_metadata = resize_with_aspect_ratio(image_pil, fixed_size)

        for ann in anns:
            category_name = categories.get(ann["category_id"], f"category_{ann['category_id']}")
            gt_mask = coco_ref.annToMask(ann).astype(bool)
            gt_mask_fixed = resize_binary_mask(gt_mask, fixed_size, resize_metadata)
            panels[category_name] = (image_resized, gt_mask_fixed)

    return panels


def build_4panel(ref_image, ref_mask, tgt_image_path, gt_mask, candidate_mask, fixed_size, title):
    tgt_image_pil = Image.open(tgt_image_path).convert("RGB")
    tgt_image_pil, _ = resize_with_aspect_ratio(tgt_image_pil, fixed_size)
    tgt_array = np.asarray(tgt_image_pil)
    display_h, display_w = tgt_array.shape[:2]

    # Reference panel may be a different aspect-ratio size -- resize to match display dims.
    ref_array = np.asarray(ref_image.resize((display_w, display_h), Image.LANCZOS))
    ref_mask_resized = np.asarray(
        Image.fromarray(ref_mask.astype(np.uint8) * 255).resize((display_w, display_h), Image.NEAREST)
    ) > 128

    def overlay_with_label(base_array, mask, color, label_text):
        canvas = overlay_mask(base_array, mask, color=color, alpha=0.5)
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)
        draw.rectangle((0, 0, pil_canvas.width, 28), fill=(255, 255, 255))
        draw.text((6, 6), label_text, fill=(0, 0, 0))
        return pil_canvas

    ref_panel = overlay_with_label(ref_array, ref_mask_resized, (240, 165, 0), "reference (prompt)")
    tgt_panel = Image.fromarray(tgt_array)
    gt_panel = overlay_with_label(tgt_array, gt_mask, (230, 65, 65), "ground truth")
    pred_panel = overlay_with_label(tgt_array, candidate_mask, (70, 200, 110), "selected (ref-guided)")

    panels = [ref_panel, tgt_panel, gt_panel, pred_panel]
    canvas = Image.new("RGB", (display_w * len(panels), display_h + 44), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 44), fill=(248, 248, 248))
    draw.text((8, 6), title, fill=(0, 0, 0))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * display_w, 44))
    return canvas


def save_gallery(tiles, output_path, title):
    if not tiles:
        return None
    cols = 1
    rows = len(tiles)
    tw = [t.size[0] for t in tiles]
    th = [t.size[1] for t in tiles]
    total_w = max(tw) + 10
    total_h = sum(th) + 44 + 10 * rows

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, total_w, 44), fill=(242, 242, 242))
    draw.text((8, 6), title, fill=(0, 0, 0))
    y = 44
    for tile in tiles:
        canvas.paste(tile, (5, y))
        y += tile.size[1] + 10
    canvas.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True, help="Path to the completed summary.json from fiedler_reference_eval.py")
    parser.add_argument("--annotations", required=True, help="Test set COCO annotations (same as used for the eval run)")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--reference-annotations", required=True)
    parser.add_argument("--reference-image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vit-arch", default="small")
    parser.add_argument("--vit-feat", default="k")
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--dino-version", default="v2")
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--fixed-size", type=int, default=1022)
    parser.add_argument("--pretrain-path", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--unlimited-depth", action="store_true")
    parser.add_argument("--unlimited-nodes", action="store_true")
    parser.add_argument("--unlimited-noise", action="store_true")
    parser.add_argument("--unlimited-imbalance", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=63)
    parser.add_argument("--min-region-patches", type=int, default=5)
    parser.add_argument("--min-split-ratio", type=float, default=0.05)
    parser.add_argument("--noise-threshold", type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.summary_json) as f:
        summary = json.load(f)

    top_samples = summary["top_samples"]
    bottom_samples = summary["bottom_samples"]

    # Build image_id -> full record map so we can locate the exact GT mask again.
    records_by_image = load_instance_records(args.annotations, args.image_root)

    print("Loading backbone...")
    backbone = build_backbone(args.vit_arch, args.vit_feat, args.patch_size, args.pretrain_path, dino_version=args.dino_version, debiased=False)

    print("Building reference prototypes...")
    prototypes = build_reference_prototypes(args.reference_annotations, args.reference_image_root, backbone, args.fixed_size, args.cpu)

    print("Building reference panel images...")
    ref_panels = build_reference_panels(args.reference_annotations, args.reference_image_root, args.fixed_size)

    def process_samples(samples, label):
        tiles = []
        tree_cache = {}
        remaining_uses = defaultdict(int)
        for sample in samples:
            remaining_uses[sample["image_id"]] += 1

        for sample in samples:
            image_id = sample["image_id"]
            annotation_id = sample["annotation_id"]
            category_name = sample["category_name"]
            image_path = sample["image_path"]

            if image_id not in tree_cache:
                image_pil = Image.open(image_path).convert("RGB")
                _, resize_metadata = resize_with_aspect_ratio(image_pil, args.fixed_size)
                tree_args = prepare_args(args, image_path, resize_metadata)
                tree = collect_recursive_candidates(tree_args, backbone=backbone)
                _, features, grid_h, grid_w = load_image_features(image_path, backbone, args.fixed_size, args.cpu, resize_metadata)
                features_np = features.numpy() if hasattr(features, "numpy") else np.asarray(features)
                candidate_masks = candidate_masks_from_tree(tree, args.fixed_size, resize_metadata)
                tree_cache[image_id] = (tree, features_np, candidate_masks, resize_metadata)
                print(f"  [{label}] built tree for image {image_id}")

                del image_pil
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()

            tree, features_np, candidate_masks, resize_metadata = tree_cache[image_id]

            prototype = prototypes.get(category_name)
            if prototype is None:
                print(f"  [warn] no prototype for {category_name}, skipping sample")
                continue
            scores = score_nodes_against_prototype(tree, features_np, prototype)
            best_idx = int(np.argmax(scores))
            selected_mask = candidate_masks[best_idx]

            # Find the exact GT mask for this annotation_id.
            gt_mask = None
            for record in records_by_image.get(image_id, []):
                if record["annotation_id"] == annotation_id:
                    gt_mask = resize_binary_mask(record["gt_mask"], args.fixed_size, resize_metadata)
                    break
            if gt_mask is None:
                print(f"  [warn] could not locate GT for annotation {annotation_id}, skipping")
                continue

            ref_image, ref_mask = ref_panels.get(category_name, (None, None))
            if ref_image is None:
                print(f"  [warn] no reference panel for {category_name}, skipping")
                continue

            tile = build_4panel(ref_image, ref_mask, image_path, gt_mask, selected_mask, args.fixed_size, sample["title"])
            tiles.append(tile)

            remaining_uses[image_id] -= 1
            if remaining_uses[image_id] <= 0 and image_id in tree_cache:
                del tree_cache[image_id]
                gc.collect()

        return tiles

    print("Processing top samples...")
    top_tiles = process_samples(top_samples, "top")
    print("Processing bottom samples...")
    bottom_tiles = process_samples(bottom_samples, "bottom")

    top_path = save_gallery(top_tiles, os.path.join(args.output_dir, "top_10_gallery_with_reference.png"), f"{summary['dataset_name']} top instances (reference + target + GT + prediction)")
    bottom_path = save_gallery(bottom_tiles, os.path.join(args.output_dir, "bottom_10_gallery_with_reference.png"), f"{summary['dataset_name']} bottom instances (reference + target + GT + prediction)")

    print(f"Saved: {top_path}")
    print(f"Saved: {bottom_path}")


if __name__ == "__main__":
    main()
