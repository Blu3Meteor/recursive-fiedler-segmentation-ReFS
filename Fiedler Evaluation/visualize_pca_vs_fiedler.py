#!/usr/bin/env python3
"""Visualize the biggest per-(image,category) IoU deltas between the Fiedler-vector
split criterion and the top-PCA-component split criterion, found via
eig_ablation/eig1 vs pca_ablation/pca0 (both oracle best-node search, 100-image
sample). For each selected (image, category), rebuilds both trees from scratch
and overlays each method's best-matching node + GT on the original image.
"""

import argparse
import csv
import os
import sys

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

from fiedler_class_transfer import build_backbone, overlay_mask, resize_with_aspect_ratio  # noqa: E402
from recursive_fiedler import collect_recursive_candidates, upsample_mask  # noqa: E402

FIEDLER_COLOR = (70, 146, 230)
PCA_COLOR = (230, 120, 40)
GT_COLOR = (230, 65, 65)

FIXED_SIZE = 1022
VIT_ARCH, VIT_FEAT, PATCH_SIZE, DINO_VERSION = "small", "k", 14, "v2"


def mask_iou(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / float(union) if union else 0.0


def downsample_mask_to_grid(mask_fixed, grid_h, grid_w):
    return np.asarray(Image.fromarray(mask_fixed.astype(np.uint8) * 255).resize((grid_w, grid_h), Image.NEAREST)) > 128


def resize_binary_mask(mask, fixed_size, metadata):
    scale = metadata.get("scale", 1.0)
    pad_left, pad_top = metadata.get("pad_left", 0), metadata.get("pad_top", 0)
    resized_w = metadata.get("resized_w", int(metadata["orig_w"] * scale))
    resized_h = metadata.get("resized_h", int(metadata["orig_h"] * scale))
    downscaled = Image.fromarray(mask.astype(np.uint8) * 255).resize((resized_w, resized_h), Image.NEAREST)
    mask = np.asarray(downscaled) > 128
    padded = np.zeros((fixed_size, fixed_size), dtype=bool)
    padded[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = mask
    return padded


def find_best_matching_node(nodes, gt_mask_grid):
    best_node, best_iou = None, -1.0
    for node in nodes:
        iou = mask_iou(node["region_mask"], gt_mask_grid)
        if iou > best_iou:
            best_iou, best_node = iou, node
    return best_node, best_iou


def build_tree(image_path, resize_metadata, backbone, split_method, pca_index=0):
    tree_args = argparse.Namespace(
        image_path=image_path, vit_arch=VIT_ARCH, vit_feat=VIT_FEAT, patch_size=PATCH_SIZE,
        pretrain_path=None, cpu=False, dino_version=DINO_VERSION, debiased=False,
        fixed_size=FIXED_SIZE, tau=0.0, max_depth=5, max_nodes=63, min_region_patches=5,
        min_split_ratio=0.05, noise_threshold=3.0,
        unlimited_depth=True, unlimited_nodes=True, unlimited_imbalance=True, unlimited_noise=True,
        resize_metadata=resize_metadata, eig_index=1, split_method=split_method, pca_index=pca_index,
        resize_mode="letterbox",
    )
    return collect_recursive_candidates(tree_args, backbone=backbone)


def label_panel(image_array, mask, color, label_lines):
    canvas = overlay_mask(image_array, mask, color=color, alpha=0.5) if mask is not None else image_array.copy()
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    h = 16 * len(label_lines) + 8
    draw.rectangle((0, 0, pil_canvas.width, h), fill=(255, 255, 255))
    for i, line in enumerate(label_lines):
        draw.text((6, 4 + 16 * i), line, fill=(0, 0, 0))
    return pil_canvas


def build_panel_row(panels, title):
    total_w = sum(p.size[0] for p in panels)
    row_h = max(p.size[1] for p in panels)
    canvas = Image.new("RGB", (total_w, row_h + 32), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 32), fill=(240, 240, 240))
    draw.text((8, 8), title, fill=(0, 0, 0))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 32))
        x += p.size[0]
    return canvas


def save_gallery(rows, output_path, title):
    if not rows:
        return
    width = max(r.size[0] for r in rows)
    total_h = sum(r.size[1] for r in rows) + 36
    canvas = Image.new("RGB", (width, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 36), fill=(220, 220, 220))
    draw.text((10, 10), title, fill=(0, 0, 0))
    y = 36
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.size[1]
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def crop_to_bbox(image_array, masks, pad_frac=0.6, min_size=160):
    """Crop to the union bbox of the given masks, padded, upscaled to a minimum
    display size so small/thin objects are actually visible."""
    union = np.zeros(image_array.shape[:2], dtype=bool)
    for m in masks:
        if m is not None:
            union |= m
    if not union.any():
        return image_array
    ys, xs = np.where(union)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    pad_y, pad_x = int(h * pad_frac) + 10, int(w * pad_frac) + 10
    H, W = image_array.shape[:2]
    y0, y1 = max(0, y0 - pad_y), min(H, y1 + pad_y + 1)
    x0, x1 = max(0, x0 - pad_x), min(W, x1 + pad_x + 1)
    return image_array[y0:y1, x0:x1]


def process_case(image_id, category_name, coco, image_root, backbone):
    image_info = coco.loadImgs([image_id])[0]
    image_path = os.path.join(image_root, image_info["file_name"])
    image_pil = Image.open(image_path).convert("RGB")
    image_resized, resize_metadata = resize_with_aspect_ratio(image_pil, FIXED_SIZE)
    image_np = np.asarray(image_resized)

    cat_id = next(c["id"] for c in coco.loadCats(coco.getCatIds()) if c["name"] == category_name)
    ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)
    gt_mask_fixed = np.zeros((FIXED_SIZE, FIXED_SIZE), dtype=bool)
    for a in anns:
        gt_mask_fixed |= resize_binary_mask(coco.annToMask(a).astype(bool), FIXED_SIZE, resize_metadata)

    fiedler_tree = build_tree(image_path, resize_metadata, backbone, "laplacian")
    pca_tree = build_tree(image_path, resize_metadata, backbone, "pca", pca_index=0)

    grid_h, grid_w = fiedler_tree["grid_h"], fiedler_tree["grid_w"]
    gt_grid = downsample_mask_to_grid(gt_mask_fixed, grid_h, grid_w)

    fiedler_node, fiedler_iou = find_best_matching_node(fiedler_tree["nodes"], gt_grid)
    pca_node, pca_iou = find_best_matching_node(pca_tree["nodes"], gt_grid)

    fiedler_mask = upsample_mask(fiedler_node["region_mask"], FIXED_SIZE) if fiedler_node else None
    pca_mask = upsample_mask(pca_node["region_mask"], FIXED_SIZE) if pca_node else None

    # Full-frame context thumbnail (small) + a cropped, upscaled zoom on the
    # object region, since the biggest-delta cases skew toward small/thin
    # objects that are invisible at full 1022x1022 scale.
    context_panel = Image.fromarray(image_np).resize((256, 256), Image.LANCZOS)
    draw = ImageDraw.Draw(context_panel)
    draw.rectangle((0, 0, context_panel.width, 20), fill=(255, 255, 255))
    draw.text((4, 2), "context (full frame)", fill=(0, 0, 0))

    crop_box = crop_to_bbox(image_np, [gt_mask_fixed, fiedler_mask, pca_mask])
    zoom_size = max(300, min(700, crop_box.shape[0] * 2))

    def zoomed_panel(mask, color, label):
        cropped_img = crop_to_bbox(image_np, [gt_mask_fixed, fiedler_mask, pca_mask])
        cropped_mask = crop_to_bbox(mask, [gt_mask_fixed, fiedler_mask, pca_mask]) if mask is not None else None
        zoom_img = np.asarray(Image.fromarray(cropped_img).resize((zoom_size, zoom_size), Image.NEAREST))
        zoom_mask = np.asarray(Image.fromarray(cropped_mask.astype(np.uint8) * 255).resize((zoom_size, zoom_size), Image.NEAREST)) > 0 if cropped_mask is not None else None
        return label_panel(zoom_img, zoom_mask, color, [label])

    gt_panel = zoomed_panel(gt_mask_fixed, GT_COLOR, f"GT: {category_name}")
    fiedler_panel = zoomed_panel(fiedler_mask, FIEDLER_COLOR, f"Fiedler  IoU={fiedler_iou:.3f}")
    pca_panel = zoomed_panel(pca_mask, PCA_COLOR, f"PCA (PC1)  IoU={pca_iou:.3f}")

    title = f"img {image_id} | {category_name} | Fiedler={fiedler_iou:.3f} PCA={pca_iou:.3f} delta={pca_iou - fiedler_iou:+.3f}"
    return build_panel_row([context_panel, gt_panel, fiedler_panel, pca_panel], title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="Fiedler Evaluation/data/coco_500_from_1000.json")
    parser.add_argument("--image-root", default="Fiedler Evaluation/datasets/val2017")
    parser.add_argument("--eig-csv", default="Fiedler Evaluation/output/06_alternate_dims/eig_ablation/eig1/instance_results.csv")
    parser.add_argument("--pca-csv", default="Fiedler Evaluation/output/06_alternate_dims/pca_ablation/pca0/instance_results.csv")
    parser.add_argument("--output-dir", default="Fiedler Evaluation/output/06_alternate_dims/pca_vs_fiedler_viz")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    def load_rows(path):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    eig = {(int(r["image_id"]), r["category_name"]): float(r["best_iou"]) for r in load_rows(args.eig_csv)}
    pca = {(int(r["image_id"]), r["category_name"]): float(r["best_iou"]) for r in load_rows(args.pca_csv)}
    common = sorted(set(eig) & set(pca))
    deltas = [(k, pca[k] - eig[k]) for k in common]

    def top_distinct(sorted_list, n):
        seen, out = set(), []
        for k, d in sorted_list:
            if k[0] in seen:
                continue
            seen.add(k[0])
            out.append(k)
            if len(out) == n:
                break
        return out

    pca_wins = top_distinct(sorted(deltas, key=lambda x: -x[1]), args.top_n)
    fiedler_wins = top_distinct(sorted(deltas, key=lambda x: x[1]), args.top_n)

    print("PCA >> Fiedler cases:", pca_wins)
    print("Fiedler >> PCA cases:", fiedler_wins)

    print("Loading backbone...")
    backbone = build_backbone(VIT_ARCH, VIT_FEAT, PATCH_SIZE, None, dino_version=DINO_VERSION, debiased=False)
    coco = COCO(args.annotations)

    for label, cases in [("pca_wins", pca_wins), ("fiedler_wins", fiedler_wins)]:
        rows = []
        for image_id, category_name in cases:
            print(f"[{label}] image {image_id}, category {category_name}")
            row = process_case(image_id, category_name, coco, args.image_root, backbone)
            rows.append(row)
        save_gallery(rows, os.path.join(args.output_dir, f"{label}_gallery.png"), f"{label}: Fiedler vs PCA(PC1)")


if __name__ == "__main__":
    main()
