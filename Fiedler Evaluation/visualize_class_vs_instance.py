#!/usr/bin/env python3
"""Visualize recursive Fiedler behavior on class-vs-instance images.

For each image in the class_vs_instance_20 dataset, produces one composite PNG
with 6 labeled panels: [original] [DINO feature map (PCA-RGB)]
[Fiedler mask: root] [Fiedler mask: child A] [Fiedler mask: child B] [ground truth].

Exploratory/visualization only -- no IoU or other metrics are computed.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CUTLER_ROOT = os.path.join(WORKSPACE_ROOT, "CutLER Implementation", "TFSS-CutLER")
MASKCUT_ROOT = os.path.join(CUTLER_ROOT, "maskcut")
if CUTLER_ROOT not in sys.path:
    sys.path.insert(0, CUTLER_ROOT)
if MASKCUT_ROOT not in sys.path:
    sys.path.insert(0, MASKCUT_ROOT)

from fiedler_class_transfer import build_backbone, overlay_mask, resize_with_aspect_ratio  # noqa: E402
from recursive_fiedler import collect_recursive_candidates, load_image_features, upsample_mask  # noqa: E402

CLASS_IDS = {1, 16, 17, 18, 19, 20, 21, 22, 23}

NODE_COLORS = [(255, 180, 0), (0, 220, 80), (70, 146, 230)]  # root, child A, child B
GT_PALETTE = [
    (230, 65, 65), (65, 140, 230), (240, 200, 40), (170, 80, 220),
    (60, 200, 170), (240, 130, 40), (100, 100, 220), (200, 60, 130),
]


def mask_iou(mask_a, mask_b):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection) / float(union)


def downsample_mask_to_grid(mask_fixed, grid_h, grid_w):
    return np.asarray(
        Image.fromarray(mask_fixed.astype(np.uint8) * 255).resize((grid_w, grid_h), Image.NEAREST)
    ) > 128


def find_best_matching_node(tree_nodes, gt_mask_grid):
    """Search the full tree for the node whose region_mask best matches gt_mask_grid
    (both at grid/patch resolution) -- oracle-style search, used here purely to
    visualize where in the tree (if anywhere) an instance actually gets separated."""
    best_node = None
    best_iou = -1.0
    for node in tree_nodes:
        iou = mask_iou(node["region_mask"], gt_mask_grid)
        if iou > best_iou:
            best_iou = iou
            best_node = node
    return best_node, best_iou


def resize_binary_mask(mask, fixed_size, metadata=None):
    """Resize a binary mask into the same fixed_size x fixed_size, aspect-preserving,
    letterboxed coordinate system used for the displayed image (mirrors
    fiedler_candidate_eval.py's resize_binary_mask)."""
    if metadata is None:
        return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((fixed_size, fixed_size), Image.NEAREST)) > 0

    if fixed_size <= 0 or not metadata:
        return mask

    scale = metadata.get("scale", 1.0)
    pad_left = metadata.get("pad_left", 0)
    pad_top = metadata.get("pad_top", 0)
    resized_w = metadata.get("resized_w", int(metadata["orig_w"] * scale))
    resized_h = metadata.get("resized_h", int(metadata["orig_h"] * scale))

    if scale != 1.0:
        downscaled = Image.fromarray(mask.astype(np.uint8) * 255)
        downscaled = downscaled.resize((resized_w, resized_h), Image.NEAREST)
        mask = np.asarray(downscaled) > 128
    else:
        mask = mask.copy()

    padded = np.zeros((fixed_size, fixed_size), dtype=bool)
    padded[pad_top: pad_top + resized_h, pad_left: pad_left + resized_w] = mask
    return padded


def pca_feature_map(features, grid_h, grid_w, display_size):
    """features: (feat_dim, grid_h*grid_w) array -> (display_size, display_size, 3) uint8 RGB."""
    feats = np.asarray(features)
    if hasattr(feats, "numpy"):
        feats = feats.numpy()
    feats = feats.T  # (N, feat_dim)

    pca = PCA(n_components=3)
    projected = pca.fit_transform(feats)  # (N, 3)

    mins = projected.min(axis=0, keepdims=True)
    maxs = projected.max(axis=0, keepdims=True)
    ranges = np.clip(maxs - mins, 1e-6, None)
    normalized = ((projected - mins) / ranges * 255.0).astype(np.uint8)

    rgb_grid = normalized.reshape(grid_h, grid_w, 3)
    rgb_image = Image.fromarray(rgb_grid, mode="RGB").resize((display_size, display_size), Image.NEAREST)
    return np.asarray(rgb_image)


def overlay_with_label(image_array, mask, color, label_text):
    canvas = overlay_mask(image_array, mask, color=color, alpha=0.5)
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    draw.rectangle((0, 0, pil_canvas.width, 28), fill=(255, 255, 255))
    draw.text((6, 6), label_text, fill=(0, 0, 0))
    return pil_canvas


def plain_panel(image_array, label_text):
    pil_canvas = Image.fromarray(image_array)
    draw = ImageDraw.Draw(pil_canvas)
    draw.rectangle((0, 0, pil_canvas.width, 28), fill=(255, 255, 255))
    draw.text((6, 6), label_text, fill=(0, 0, 0))
    return pil_canvas


def overlay_multi_with_label(image_array, masks_colors, label_text):
    canvas = image_array.copy()
    for mask, color in masks_colors:
        canvas = overlay_mask(canvas, mask, color=color, alpha=0.5)
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    draw.rectangle((0, 0, pil_canvas.width, 28), fill=(255, 255, 255))
    draw.text((6, 6), label_text, fill=(0, 0, 0))
    return pil_canvas


def build_panel_row(panels, title):
    display_w = panels[0].size[0]
    display_h = panels[0].size[1]
    canvas = Image.new("RGB", (display_w * len(panels), display_h + 32), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 32), fill=(240, 240, 240))
    draw.text((8, 8), title, fill=(0, 0, 0))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * display_w, 32))
    return canvas


def process_image(record, backbone, coco, categories, args):
    image_path = os.path.join(args.dataset_root, record["category"], record["local_file_name"])
    image_id = record["image_id"]

    image_pil = Image.open(image_path).convert("RGB")
    image_resized, resize_metadata = resize_with_aspect_ratio(image_pil, args.fixed_size)
    image_np = np.asarray(image_resized)

    # DINO feature map (separate direct call, decoupled from the tree builder).
    _, features, grid_h, grid_w = load_image_features(image_path, backbone, args.fixed_size, args.cpu, resize_metadata)
    feature_map = pca_feature_map(features, grid_h, grid_w, args.fixed_size)

    # Recursive Fiedler tree, capped to just the first split (root + 2 children).
    tree_args = argparse.Namespace(
        image_path=image_path,
        vit_arch=args.vit_arch,
        vit_feat=args.vit_feat,
        patch_size=args.patch_size,
        pretrain_path=None,
        cpu=args.cpu,
        dino_version=args.dino_version,
        debiased=False,
        fixed_size=args.fixed_size,
        tau=args.tau,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        min_region_patches=args.min_region_patches,
        min_split_ratio=args.min_split_ratio,
        noise_threshold=args.noise_threshold,
        unlimited_depth=args.unlimited_depth,
        unlimited_nodes=args.unlimited_nodes,
        unlimited_imbalance=args.unlimited_imbalance,
        unlimited_noise=args.unlimited_noise,
        resize_metadata=resize_metadata,
    )
    tree = collect_recursive_candidates(tree_args, backbone=backbone)
    first_three = tree["nodes"][:3]
    if args.debug:
        print(f"  [debug] {len(tree['nodes'])} total nodes")
        for n in first_three:
            print(f"  [debug] id={n['id']} depth={n['depth']} status={n['status']} area_ratio={n.get('region_area_ratio')} stop_reason={n.get('stop_reason')}")

    # GT masks.
    ann_ids = coco.getAnnIds(imgIds=[image_id])
    anns = [a for a in coco.loadAnns(ann_ids) if a.get("category_id") in CLASS_IDS and not a.get("iscrowd", 0)]
    gt_masks_colors = []
    gt_labels = []
    best_match_panels = []
    for i, ann in enumerate(anns):
        gt_mask = coco.annToMask(ann).astype(bool)
        gt_mask_fixed = resize_binary_mask(gt_mask, args.fixed_size, resize_metadata)
        color = GT_PALETTE[i % len(GT_PALETTE)]
        gt_masks_colors.append((gt_mask_fixed, color))
        category_name = categories.get(ann["category_id"], ann["category_id"])
        gt_labels.append(f"{category_name} #{i+1}")

        gt_mask_grid = downsample_mask_to_grid(gt_mask_fixed, grid_h, grid_w)
        best_node, best_iou = find_best_matching_node(tree["nodes"], gt_mask_grid)
        if best_node is not None:
            best_mask = upsample_mask(best_node["region_mask"], args.fixed_size)
            label = f"best match: {category_name} #{i+1} @ depth={best_node['depth']} IoU={best_iou:.3f}"
            if args.debug:
                print(f"  [debug] best match for {category_name} #{i+1}: node id={best_node['id']} depth={best_node['depth']} iou={best_iou:.3f}")
            best_match_panels.append(overlay_with_label(image_np, best_mask, color, label))

    # Build panels.
    original_panel = plain_panel(image_np, "original")
    feature_panel = plain_panel(feature_map, "DINO feature map (PCA-RGB)")

    node_panels = []
    for i, node in enumerate(first_three):
        mask = upsample_mask(node["region_mask"], args.fixed_size)
        color = NODE_COLORS[i % len(NODE_COLORS)]
        label = f"depth={node['depth']} id={node['id']} status={node['status']}"
        node_panels.append(overlay_with_label(image_np, mask, color, label))

    if len(first_three) < 3:
        stop_reason = first_three[0].get("stop_reason", "unknown") if first_three else "no nodes"
        for _ in range(3 - len(first_three)):
            node_panels.append(plain_panel(image_np, f"(root did not split: {stop_reason})"))

    gt_panel = overlay_multi_with_label(image_np, gt_masks_colors, "GT: " + ", ".join(gt_labels) if gt_labels else "GT: (none)")

    panels = [original_panel, feature_panel] + node_panels + [gt_panel] + best_match_panels
    title = f"{record['category']} | {record['file_name']} | image_id={image_id} | classes={record['class_distribution']}"
    row = build_panel_row(panels, title)

    del image_pil, image_resized, features, tree
    return row


def save_gallery(rows, output_path, title):
    if not rows:
        return None
    width = rows[0].size[0]
    total_height = sum(r.size[1] for r in rows) + 36
    canvas = Image.new("RGB", (width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 36), fill=(220, 220, 220))
    draw.text((10, 10), title, fill=(0, 0, 0))
    y = 36
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.size[1]
    canvas.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="Fiedler Evaluation/data/class_vs_instance_20")
    parser.add_argument("--metadata", default=None, help="Defaults to <dataset-root>/metadata.json")
    parser.add_argument("--coco-json", default="CutLER Implementation/TFSS-CutLER/datasets/coco/annotations/instances_val2017.json")
    parser.add_argument("--output-dir", default="Fiedler Evaluation/output/03_class_vs_instance/class_vs_instance_viz")
    parser.add_argument("--vit-arch", default="small")
    parser.add_argument("--vit-feat", default="k")
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--dino-version", default="v2")
    parser.add_argument("--fixed-size", type=int, default=1022)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=5, help="Ignored when --unlimited-depth is set (default)")
    parser.add_argument("--max-nodes", type=int, default=63, help="Ignored when --unlimited-nodes is set (default)")
    parser.add_argument("--min-region-patches", type=int, default=5, help="The only real stopping criterion when all --unlimited-* flags are set")
    parser.add_argument("--min-split-ratio", type=float, default=0.05, help="Ignored when --unlimited-imbalance is set (default)")
    parser.add_argument("--noise-threshold", type=float, default=3.0, help="Ignored when --unlimited-noise is set (default)")
    parser.add_argument("--unlimited-depth", dest="unlimited_depth", action="store_true", default=True)
    parser.add_argument("--no-unlimited-depth", dest="unlimited_depth", action="store_false")
    parser.add_argument("--unlimited-nodes", dest="unlimited_nodes", action="store_true", default=True)
    parser.add_argument("--no-unlimited-nodes", dest="unlimited_nodes", action="store_false")
    parser.add_argument("--unlimited-imbalance", dest="unlimited_imbalance", action="store_true", default=True)
    parser.add_argument("--no-unlimited-imbalance", dest="unlimited_imbalance", action="store_false")
    parser.add_argument("--unlimited-noise", dest="unlimited_noise", action="store_true", default=True)
    parser.add_argument("--no-unlimited-noise", dest="unlimited_noise", action="store_false")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--make-gallery", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N records (smoke testing)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    metadata_path = args.metadata or os.path.join(args.dataset_root, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    if args.limit is not None:
        metadata = metadata[: args.limit]

    for record in metadata:
        os.makedirs(os.path.join(args.output_dir, record["category"]), exist_ok=True)

    print("Loading backbone...")
    backbone = build_backbone(args.vit_arch, args.vit_feat, args.patch_size, None, dino_version=args.dino_version, debiased=False)

    print("Loading COCO annotations...")
    coco = COCO(args.coco_json)
    categories = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    rows_by_category = {"multi_class": [], "same_class_instances": []}
    failures = []

    for index, record in enumerate(metadata, start=1):
        start_time = time.time()
        try:
            row = process_image(record, backbone, coco, categories, args)
            out_path = os.path.join(args.output_dir, record["category"], record["local_file_name"].replace(".jpg", ".png"))
            row.save(out_path)
            rows_by_category[record["category"]].append(row)
            elapsed = time.time() - start_time
            print(f"[{index}/{len(metadata)}] {record['category']}/{record['local_file_name']} -> {out_path} ({elapsed:.2f}s)", flush=True)
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"ERROR: {record['category']}/{record['local_file_name']} failed after {elapsed:.2f}s: {e}", flush=True)
            failures.append({"record": record, "error": str(e)})

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    if args.make_gallery:
        for category, rows in rows_by_category.items():
            if not rows:
                continue
            gallery_path = os.path.join(args.output_dir, f"gallery_{category}.png")
            save_gallery(rows, gallery_path, f"{category} -- class-vs-instance Fiedler visualization")
            print(f"Saved gallery: {gallery_path}")

    if failures:
        failures_path = os.path.join(args.output_dir, "failures.json")
        with open(failures_path, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"\n{len(failures)} failures logged to {failures_path}")

    print(f"\nDone. {sum(len(v) for v in rows_by_category.values())}/{len(metadata)} images processed successfully.")


if __name__ == "__main__":
    main()
