#!/usr/bin/env python3
"""Tune MaskCut on ADE20K samples and report class-agnostic mask metrics."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "maskcut"))
sys.path.insert(0, str(ROOT / "third_party"))
sys.path.insert(0, str(ROOT))

import dino  # noqa: E402
from crf import densecrf  # noqa: E402
from maskcut import maskcut  # noqa: E402
from TokenCut.unsupervised_saliency_detection import metric  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ade-root", default=str(ROOT / "datasets" / "ade_20k"))
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "ade20k_maskcut_tuning"))
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.1, 0.15, 0.2])
    parser.add_argument("--n-masks", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--fixed-size", type=int, default=480)
    parser.add_argument("--vit-arch", choices=["small", "base"], default="small")
    parser.add_argument("--patch-size", choices=[8, 16], type=int, default=8)
    parser.add_argument("--vit-feat", choices=["k", "q", "v", "kqv"], default="k")
    parser.add_argument("--pretrain-path", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-crf", action="store_true")
    parser.add_argument("--no-visualizations", action="store_true")
    return parser.parse_args()


def samples(ade_root, split, limit):
    split_dir = Path(ade_root) / split
    img_dir = split_dir / "img"
    ann_dir = split_dir / "ann"
    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        ann_path = ann_dir / f"{img_path.name}.json"
        if ann_path.exists():
            pairs.append((img_path, ann_path))
    return pairs[:limit]


def ade_instances(ann_path):
    data = json.loads(Path(ann_path).read_text())
    width = int(data["size"]["width"])
    height = int(data["size"]["height"])
    masks = []
    for obj in data.get("objects", []):
        points = obj.get("points", {})
        exterior = points.get("exterior")
        if obj.get("geometryType") != "polygon" or not exterior or len(exterior) < 3:
            continue
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon([tuple(p) for p in exterior], fill=1)
        for interior in points.get("interior", []):
            if len(interior) >= 3:
                draw.polygon([tuple(p) for p in interior], fill=0)
        mask = np.asarray(mask_img, dtype=bool)
        if mask.any():
            masks.append(mask)
    return masks, (width, height)


def load_backbone(args):
    if args.pretrain_path:
        url = args.pretrain_path
    elif args.vit_arch == "base" and args.patch_size == 8:
        url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    elif args.vit_arch == "small" and args.patch_size == 8:
        url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    else:
        raise ValueError("This repository only wires DINO patch-size 8 weights.")
    feat_dim = 768 if args.vit_arch == "base" else 384
    backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)
    backbone.eval()
    if not args.cpu:
        backbone.cuda()
    return backbone


def predict_masks(img_path, backbone, args, tau, n_masks):
    bipartitions, _, resized = maskcut(
        str(img_path), backbone, args.patch_size, tau, N=n_masks,
        fixed_size=args.fixed_size, cpu=args.cpu
    )
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    predictions = []
    for bipartition in bipartitions:
        pseudo = bipartition
        if not args.no_crf:
            pseudo = densecrf(np.array(resized), bipartition)
            pseudo = ndimage.binary_fill_holes(pseudo >= 0.5)
            mask1 = torch.from_numpy(bipartition)
            mask2 = torch.from_numpy(pseudo)
            if not args.cpu:
                mask1 = mask1.cuda()
                mask2 = mask2.cuda()
            if metric.IoU(mask1, mask2) < 0.5:
                pseudo = pseudo * -1
        pseudo = (pseudo > 0).astype(np.uint8) * 255
        pseudo = Image.fromarray(pseudo).resize((width, height), Image.NEAREST)
        mask = np.asarray(pseudo) > 0
        if mask.any():
            predictions.append(mask)
    return predictions


def iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union else 0.0


def evaluate_pairs(pairs, backbone, args, tau, n_masks, out_dir=None, split_name="validation"):
    rows = []
    all_matches = []
    total_gt = 0
    viz_dir = None
    if out_dir is not None and not args.no_visualizations:
        viz_dir = Path(out_dir) / "visualizations" / split_name
        viz_dir.mkdir(parents=True, exist_ok=True)
    for img_path, ann_path in pairs:
        gt_masks, _ = ade_instances(ann_path)
        pred_masks = predict_masks(img_path, backbone, args, tau, n_masks)
        total_gt += len(gt_masks)
        best_pred_ious = [max([iou(pred, gt) for gt in gt_masks] or [0.0]) for pred in pred_masks]
        best_gt_ious = [max([iou(gt, pred) for pred in pred_masks] or [0.0]) for gt in gt_masks]
        matches = greedy_matches(pred_masks, gt_masks, threshold=0.5)
        per_image_ap50 = average_precision(matches, len(gt_masks))
        all_matches.extend(matches)
        visualization = None
        if viz_dir is not None:
            visualization = viz_dir / f"{img_path.stem}_gt_vs_maskcut.png"
            save_visualization(img_path, gt_masks, pred_masks, visualization)
        rows.append({
            "image": img_path.name,
            "gt_instances": len(gt_masks),
            "pred_instances": len(pred_masks),
            "mean_best_pred_iou": float(np.mean(best_pred_ious)) if best_pred_ious else 0.0,
            "mean_best_gt_iou": float(np.mean(best_gt_ious)) if best_gt_ious else 0.0,
            "ap50": per_image_ap50,
            "ap50_tp": sum(matches),
            "visualization": str(visualization.relative_to(Path(out_dir))) if visualization else None,
        })
    mean_iou = float(np.mean([r["mean_best_gt_iou"] for r in rows])) if rows else 0.0
    ap50 = average_precision(all_matches, total_gt)
    return {"mean_iou": mean_iou, "ap50": ap50, "images": rows, "total_gt": total_gt}


def mask_union(masks, size):
    width, height = size
    union = np.zeros((height, width), dtype=bool)
    for mask in masks:
        union |= mask
    return union


def overlay_masks(image, masks, colors):
    arr = np.asarray(image).astype(np.float32)
    for idx, mask in enumerate(masks):
        color = np.asarray(colors[idx % len(colors)], dtype=np.float32)
        arr[mask] = arr[mask] * 0.45 + color * 0.55
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def draw_label(panel, label):
    panel = panel.copy()
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, 24), fill=(0, 0, 0))
    draw.text((8, 5), label, fill=(255, 255, 255))
    return panel


def save_visualization(img_path, gt_masks, pred_masks, out_path):
    image = Image.open(img_path).convert("RGB")
    colors = [
        (230, 57, 70), (42, 157, 143), (69, 123, 157),
        (244, 162, 97), (131, 56, 236), (255, 183, 3),
    ]
    gt_overlay = overlay_masks(image, gt_masks, colors)
    pred_overlay = overlay_masks(image, pred_masks, list(reversed(colors)))
    gt_union = mask_union(gt_masks, image.size)
    pred_union = mask_union(pred_masks, image.size)
    matched = Image.fromarray(np.asarray(image).copy())
    matched_arr = np.asarray(matched).astype(np.float32)
    tp = gt_union & pred_union
    fp = ~gt_union & pred_union
    fn = gt_union & ~pred_union
    matched_arr[tp] = matched_arr[tp] * 0.35 + np.array((35, 180, 95)) * 0.65
    matched_arr[fp] = matched_arr[fp] * 0.35 + np.array((230, 57, 70)) * 0.65
    matched_arr[fn] = matched_arr[fn] * 0.35 + np.array((255, 183, 3)) * 0.65
    matched = Image.fromarray(np.clip(matched_arr, 0, 255).astype(np.uint8))

    panels = [
        draw_label(image, "Image"),
        draw_label(gt_overlay, "Ground truth masks"),
        draw_label(pred_overlay, "Tuned MaskCut masks"),
        draw_label(matched, "Union compare: green TP, red FP, yellow FN"),
    ]
    canvas = Image.new("RGB", (image.width * len(panels), image.height), (255, 255, 255))
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (idx * image.width, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def greedy_matches(pred_masks, gt_masks, threshold):
    used = set()
    hits = []
    for pred in pred_masks:
        scored = sorted(((iou(pred, gt), idx) for idx, gt in enumerate(gt_masks)), reverse=True)
        if scored and scored[0][0] >= threshold and scored[0][1] not in used:
            used.add(scored[0][1])
            hits.append(True)
        else:
            hits.append(False)
    return hits


def average_precision(matches, total_gt):
    if total_gt == 0 or not matches:
        return 0.0
    tp = np.cumsum(np.asarray(matches, dtype=float))
    fp = np.cumsum(~np.asarray(matches, dtype=bool))
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    for idx in range(len(precision) - 2, -1, -1):
        precision[idx] = max(precision[idx], precision[idx + 1])
    changed = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1]))


def write_report(out_dir, args, train_pairs, val_pairs, grid, best, val_metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "settings": vars(args),
        "best": best,
        "grid": grid,
        "validation": val_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# ADE20K MaskCut Sample Tuning Report",
        "",
        f"- Training samples: {len(train_pairs)}",
        f"- Validation samples: {len(val_pairs)}",
        f"- Best tau: {best['tau']}",
        f"- Best N: {best['n_masks']}",
        f"- Validation mean IoU: {val_metrics['mean_iou']:.4f}",
        f"- Validation AP50: {val_metrics['ap50']:.4f}",
        "",
        "## Training Grid",
        "",
        "| tau | N | mean IoU | AP50 |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in grid:
        lines.append(f"| {row['tau']} | {row['n_masks']} | {row['mean_iou']:.4f} | {row['ap50']:.4f} |")
    lines.extend([
        "",
        "## Validation Images",
        "",
        "| image | gt | pred | IoU | AP50 | TP@50 | visualization |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in val_metrics["images"]:
        viz = row.get("visualization") or ""
        viz_link = f"[view]({viz.replace(chr(92), '/')})" if viz else ""
        lines.append(
            f"| {row['image']} | {row['gt_instances']} | {row['pred_instances']} | "
            f"{row['mean_best_gt_iou']:.4f} | {row['ap50']:.4f} | {row['ap50_tp']} | {viz_link} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    train_pairs = samples(args.ade_root, "training", args.train_samples)
    val_pairs = samples(args.ade_root, "validation", args.val_samples)
    if not train_pairs or not val_pairs:
        raise SystemExit("ADE20K training/validation image+annotation pairs were not found.")
    backbone = load_backbone(args)
    grid = []
    for tau, n_masks in itertools.product(args.taus, args.n_masks):
        metrics = evaluate_pairs(train_pairs, backbone, args, tau, n_masks)
        grid.append({"tau": tau, "n_masks": n_masks, "mean_iou": metrics["mean_iou"], "ap50": metrics["ap50"]})
    best = max(grid, key=lambda row: (row["mean_iou"], row["ap50"]))
    out_dir = Path(args.output_dir)
    val_metrics = evaluate_pairs(val_pairs, backbone, args, best["tau"], best["n_masks"], out_dir, "validation")
    write_report(out_dir, args, train_pairs, val_pairs, grid, best, val_metrics)
    print(f"Wrote {out_dir / 'report.md'}")
    if not args.no_visualizations:
        print(f"Wrote visualizations to {out_dir / 'visualizations' / 'validation'}")
    print(f"Validation mean IoU: {val_metrics['mean_iou']:.4f}")
    print(f"Validation AP50: {val_metrics['ap50']:.4f}")


if __name__ == "__main__":
    main()
