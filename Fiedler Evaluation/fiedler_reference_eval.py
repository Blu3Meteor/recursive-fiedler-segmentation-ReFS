#!/usr/bin/env python3
"""Reference-guided Recursive Fiedler evaluation.

Instead of oracle-selecting the best candidate node by comparing every node
against the GT mask (as fiedler_candidate_eval.py does), this script:

  1. Builds one "reference prototype" per class from coco20i_reference.json
     (one labeled example image + mask per class), by averaging L2-normalized
     patch features under the reference mask -- mirroring how INSID3 builds
     its ref_prototype (models/insid3.py:149-158), but without DINOv3's
     positional-debiasing step (Fiedler uses a different backbone/feature type
     and debiasing isn't part of its existing pipeline).
  2. For each test image, builds the full recursive Fiedler candidate tree
     exactly as before (no GT involved).
  3. Scores every node by cosine similarity between the node's mean patch
     feature and the category's reference prototype.
  4. Picks the single highest-scoring node as the prediction -- GT is never
     consulted during selection.
  5. Only afterward, for evaluation purposes only, compares the selected
     node's mask against GT to report IoU.

This produces a real (non-oracle) few-shot number, comparable in spirit to
INSID3's own reported few-shot results (though with a materially different,
simpler selection mechanism -- no forward/backward localization or
cluster-merging step).
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
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

from fiedler_class_transfer import build_backbone, resize_with_aspect_ratio  # noqa: E402
from recursive_fiedler import collect_recursive_candidates, load_image_features, upsample_mask  # noqa: E402

# Reuse the exact reporting helpers from the oracle eval script so both report
# structures stay bit-for-bit comparable.
from fiedler_candidate_eval import (  # noqa: E402
    mask_iou,
    resize_binary_mask,
    prepare_args,
    candidate_masks_from_tree,
    save_gallery,
    plot_overall_histogram,
    plot_class_overview,
    plot_class_histograms,
    load_instance_records,
)


def l2_normalize_columns(features):
    """features: (C, N) -> L2-normalize each column (patch) independently."""
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return features / norms


def downsample_mask_to_grid(mask_fixed, grid_h, grid_w, mode="nearest"):
    """Downsample a fixed_size x fixed_size bool mask to (grid_h, grid_w).

    mode="nearest" (default, unchanged): a single NEAREST-neighbor point-sample
    per patch (PIL resize) -- a sliver smaller than the one sampled pixel
    vanishes with zero contribution; a sliver that happens to cover the sampled
    pixel inflates the whole patch to foreground. No partial credit either way.

    mode="bilinear": mirrors INSID3's utils/data.py downsample_mask's three-tier
    fallback -- bilinear interpolation + threshold 0.5 first (a soft local-area
    vote, not a single point-sample); if that's all-empty, retry with nearest;
    if still empty, force-set the single patch containing the mask's true
    pixel-space centroid. Guarantees at least one foreground patch for any
    non-empty input mask, which nearest-only cannot.
    """
    if mode == "bilinear":
        import torch
        import torch.nn.functional as F

        mask_t = torch.from_numpy(mask_fixed.astype(np.float32))[None, None]
        down = F.interpolate(mask_t, size=(grid_h, grid_w), mode="bilinear", align_corners=False)[0, 0] > 0.5
        if down.sum() == 0:
            down = F.interpolate(mask_t, size=(grid_h, grid_w), mode="nearest")[0, 0] > 0.5
            if down.sum() == 0:
                ys, xs = np.nonzero(mask_fixed)
                if ys.size > 0:
                    cy = int(ys.mean()) * grid_h // mask_fixed.shape[0]
                    cx = int(xs.mean()) * grid_w // mask_fixed.shape[1]
                    down[min(cy, grid_h - 1), min(cx, grid_w - 1)] = True
        return down.numpy()

    return np.asarray(
        Image.fromarray(mask_fixed.astype(np.uint8) * 255).resize((grid_w, grid_h), Image.NEAREST)
    ) > 128


def build_reference_prototypes(reference_json, image_root, backbone, fixed_size, cpu):
    """Build one L2-normalized feature prototype per class from the reference set."""
    coco_ref = COCO(reference_json)
    categories = {c["id"]: c["name"] for c in coco_ref.loadCats(coco_ref.getCatIds())}

    prototypes = {}
    for image_id in coco_ref.getImgIds():
        image_info = coco_ref.loadImgs([image_id])[0]
        image_path = os.path.join(image_root, image_info["file_name"])
        ann_ids = coco_ref.getAnnIds(imgIds=[image_id])
        anns = coco_ref.loadAnns(ann_ids)

        if not os.path.exists(image_path):
            print(f"  [ref warning] missing reference image: {image_path}")
            continue

        image_pil = Image.open(image_path).convert("RGB")
        image_resized, resize_metadata = resize_with_aspect_ratio(image_pil, fixed_size)
        _, features, grid_h, grid_w = load_image_features(image_path, backbone, fixed_size, cpu, resize_metadata)
        features_np = features.numpy() if hasattr(features, "numpy") else np.asarray(features)  # (C, N)

        for ann in anns:
            category_name = categories.get(ann["category_id"], f"category_{ann['category_id']}")
            gt_mask = coco_ref.annToMask(ann).astype(bool)
            gt_mask_fixed = resize_binary_mask(gt_mask, fixed_size, resize_metadata)
            mask_small = downsample_mask_to_grid(gt_mask_fixed, grid_h, grid_w)
            fg_indices = mask_small.reshape(-1)

            if fg_indices.sum() == 0:
                print(f"  [ref warning] empty downsampled mask for {category_name}, skipping")
                continue

            fg_feats = features_np[:, fg_indices]  # (C, n_fg)
            fg_feats_norm = l2_normalize_columns(fg_feats)
            prototype = fg_feats_norm.mean(axis=1)
            prototype = prototype / max(np.linalg.norm(prototype), 1e-8)
            prototypes[category_name] = prototype
            print(f"  Built reference prototype: {category_name} (from image {image_id}, {int(fg_indices.sum())} fg patches)")

    return prototypes


def score_nodes_against_prototype(tree, features_np, prototype):
    """Return a list of (node_index, score) for every node in tree['nodes']."""
    features_norm = l2_normalize_columns(features_np)  # (C, N)
    scores = []
    for idx, node in enumerate(tree["nodes"]):
        region_flat = node["region_mask"].reshape(-1)
        if region_flat.sum() == 0:
            scores.append(-1.0)
            continue
        node_feats = features_norm[:, region_flat]
        node_proto = node_feats.mean(axis=1)
        node_proto = node_proto / max(np.linalg.norm(node_proto), 1e-8)
        score = float(np.dot(node_proto, prototype))
        scores.append(score)
    return scores


def build_sample_panel(image_path, gt_mask, candidate_mask, fixed_size, title, metadata=None):
    from fiedler_class_transfer import overlay_mask

    image_pil = Image.open(image_path).convert("RGB")
    if fixed_size > 0:
        image_pil, _ = resize_with_aspect_ratio(image_pil, fixed_size)
    else:
        image_pil.thumbnail((480, 480), Image.LANCZOS)

    image_array = np.asarray(image_pil)
    display_h, display_w = image_array.shape[:2]

    def overlay_with_label(mask, color, label_text):
        canvas = overlay_mask(image_array, mask, color=color, alpha=0.5)
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)
        draw.rectangle((0, 0, pil_canvas.width, 28), fill=(255, 255, 255))
        draw.text((6, 6), label_text, fill=(0, 0, 0))
        return pil_canvas

    gt_panel = overlay_with_label(gt_mask, (230, 65, 65), "ground truth")
    candidate_panel = overlay_with_label(candidate_mask, (70, 200, 110), "selected (ref-guided)")

    panels = [image_pil, gt_panel, candidate_panel]
    canvas = Image.new("RGB", (display_w * len(panels), display_h + 44), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 44), fill=(248, 248, 248))
    draw.text((8, 6), title, fill=(0, 0, 0))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * display_w, 44))
    return canvas


def evaluate_dataset(args):
    os.makedirs(args.output_dir, exist_ok=True)

    coco20i_classes = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "boat", "dog", "cat", "bird", "horse", "sheep", "cow", "bottle",
        "cup", "chair", "dining table", "backpack", "handbag",
    }

    records_by_image = load_instance_records(args.annotations, args.image_root, max_images=args.max_images)
    # Filter to only COCO-20i classes
    filtered = {}
    for image_id, records in records_by_image.items():
        kept = [r for r in records if r["category_name"] in coco20i_classes]
        if kept:
            filtered[image_id] = kept
    records_by_image = filtered
    total_images = len(records_by_image)
    print(f"Evaluating on {total_images} images (COCO-20i classes only)")

    backbone = build_backbone(
        args.vit_arch, args.vit_feat, args.patch_size, args.pretrain_path,
        dino_version=args.dino_version, debiased=False,
    )
    if not args.cpu and args.dino_version == "v1":
        backbone.half()
        backbone.cuda()

    print("Building reference prototypes...")
    prototypes = build_reference_prototypes(args.reference_annotations, args.reference_image_root, backbone, args.fixed_size, args.cpu)
    missing_protos = coco20i_classes - set(prototypes.keys())
    if missing_protos:
        print(f"WARNING: no reference prototype for: {sorted(missing_protos)} -- instances of these classes will be skipped")

    csv_path = os.path.join(args.output_dir, "instance_results.csv")
    csv_fieldnames = [
        "image_id", "annotation_id", "category_id", "category_name",
        "best_iou", "best_candidate_id", "best_candidate_depth", "best_candidate_status",
        "best_candidate_area_patches", "candidate_count", "selection_score", "image_path",
    ]
    csv_file = open(csv_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    failures_path = os.path.join(args.output_dir, "failures.jsonl")
    failures_file = open(failures_path, "w", encoding="utf-8")

    all_results = []
    class_scores = defaultdict(list)
    per_class_stats = defaultdict(lambda: {"count": 0, "scores": []})
    image_summaries = []
    elapsed_times = []

    for index, (image_id, records) in enumerate(records_by_image.items(), start=1):
        image_path = records[0]["image_path"]
        image_start = time.time()

        try:
            image_pil = Image.open(image_path).convert("RGB")
            image_resized, resize_metadata = resize_with_aspect_ratio(image_pil, args.fixed_size)

            tree_args = prepare_args(args, image_path, resize_metadata)
            tree = collect_recursive_candidates(tree_args, backbone=backbone)

            # Re-extract features (cheap: ~0.02-0.03s) to score nodes against prototypes.
            _, features, grid_h, grid_w = load_image_features(image_path, backbone, args.fixed_size, args.cpu, resize_metadata)
            features_np = features.numpy() if hasattr(features, "numpy") else np.asarray(features)

            candidate_masks = candidate_masks_from_tree(tree, args.fixed_size, resize_metadata)
            if not candidate_masks:
                continue

            image_results = []
            # Group records by category so we score nodes once per category present in the image.
            records_by_category = defaultdict(list)
            for record in records:
                records_by_category[record["category_name"]].append(record)

            for category_name, cat_records in records_by_category.items():
                prototype = prototypes.get(category_name)
                if prototype is None:
                    continue

                scores = score_nodes_against_prototype(tree, features_np, prototype)
                best_node_idx = int(np.argmax(scores))
                best_score = float(scores[best_node_idx])
                best_node = tree["nodes"][best_node_idx]
                selected_mask = candidate_masks[best_node_idx]

                for record in cat_records:
                    gt_mask_fixed = resize_binary_mask(record["gt_mask"], args.fixed_size, resize_metadata)
                    iou = mask_iou(selected_mask, gt_mask_fixed)

                    result = {
                        "image_id": image_id,
                        "annotation_id": record["annotation_id"],
                        "category_id": record["category_id"],
                        "category_name": category_name,
                        "title": f'{category_name} (instance {record["annotation_id"]}) | img {image_id} | IoU {iou:.3f}',
                        "best_iou": float(iou),
                        "best_candidate_id": best_node["id"],
                        "best_candidate_depth": best_node["depth"],
                        "best_candidate_status": best_node["status"],
                        "best_candidate_area_patches": best_node["region_area_patches"],
                        "candidate_count": len(candidate_masks),
                        "selection_score": best_score,
                        "image_path": record["image_path"],
                        "gt_mask_fixed": gt_mask_fixed,
                        "best_candidate_mask": selected_mask,
                    }
                    all_results.append(result)
                    image_results.append(result)
                    class_scores[category_name].append(float(iou))
                    per_class_stats[category_name]["count"] += 1
                    per_class_stats[category_name]["scores"].append(float(iou))

                    csv_writer.writerow({k: result.get(k) for k in csv_fieldnames})

            csv_file.flush()

            if image_results:
                image_summaries.append({
                    "image_id": image_id,
                    "image_path": image_path,
                    "num_instances": len(image_results),
                    "best_instance_iou": max(r["best_iou"] for r in image_results),
                    "mean_instance_iou": float(np.mean([r["best_iou"] for r in image_results])),
                })

            elapsed = time.time() - image_start
            elapsed_times.append(elapsed)
            avg_elapsed = np.mean(elapsed_times)
            eta_minutes = avg_elapsed * (total_images - index) / 60.0
            print(f"[{index}/{total_images}] {os.path.basename(image_path)} ({len(records)} instances) | elapsed={elapsed:.2f}s | ETA={eta_minutes:.1f}m", flush=True)

            del image_pil, image_resized, tree, features, candidate_masks
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

        except Exception as e:
            elapsed = time.time() - image_start
            error_msg = f"Image {image_id} ({image_path}) failed after {elapsed:.2f}s: {str(e)}"
            print(f"ERROR: {error_msg}", flush=True)
            failures_file.write(json.dumps({"image_id": image_id, "image_path": image_path, "error": str(e), "elapsed": elapsed}) + "\n")
            failures_file.flush()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            continue

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

    overall_scores = [r["best_iou"] for r in all_results]
    per_image_scores = [s["best_instance_iou"] for s in image_summaries]
    top_samples = sorted(all_results, key=lambda item: item["best_iou"], reverse=True)[: args.top_k]
    bottom_samples = sorted(all_results, key=lambda item: item["best_iou"])[: args.top_k]

    summary = {
        "dataset_name": args.dataset_name,
        "selection_method": "reference-guided (prototype similarity, no GT)",
        "annotations": os.path.abspath(args.annotations),
        "reference_annotations": os.path.abspath(args.reference_annotations),
        "image_root": os.path.abspath(args.image_root),
        "fixed_size": args.fixed_size,
        "patch_size": args.patch_size,
        "tau": args.tau,
        "num_images": len(records_by_image),
        "num_instances": len(all_results),
        "mean_best_iou": float(np.mean(overall_scores)) if overall_scores else 0.0,
        "median_best_iou": float(np.median(overall_scores)) if overall_scores else 0.0,
        "min_best_iou": float(np.min(overall_scores)) if overall_scores else 0.0,
        "max_best_iou": float(np.max(overall_scores)) if overall_scores else 0.0,
        "mean_best_iou_per_image": float(np.mean(per_image_scores)) if per_image_scores else 0.0,
        "median_best_iou_per_image": float(np.median(per_image_scores)) if per_image_scores else 0.0,
        "per_class": per_class_summary,
        "image_summaries": image_summaries,
        "top_samples": [{k: v for k, v in s.items() if k not in {"gt_mask_fixed", "best_candidate_mask"}} for s in top_samples],
        "bottom_samples": [{k: v for k, v in s.items() if k not in {"gt_mask_fixed", "best_candidate_mask"}} for s in bottom_samples],
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if overall_scores:
        plot_overall_histogram(overall_scores, os.path.join(args.output_dir, "overall_iou_histogram.png"), f"{args.dataset_name} best IoU distribution")
    if per_class_summary:
        plot_class_overview(per_class_summary, os.path.join(args.output_dir, "class_mean_iou.png"), f"{args.dataset_name} mean IoU by class")
    class_histogram_dir = os.path.join(args.output_dir, "class_histograms")
    class_index = plot_class_histograms(class_scores, class_histogram_dir, args.dataset_name) if class_scores else None

    def save_ref_gallery(samples, output_path, title):
        if not samples:
            return None
        tiles = [
            build_sample_panel(s["image_path"], s["gt_mask_fixed"], s["best_candidate_mask"], args.fixed_size, s["title"])
            for s in samples
        ]
        cols = 2 if len(tiles) > 1 else 1
        rows = (len(tiles) + cols - 1) // cols
        tw = [t.size[0] for t in tiles]
        th = [t.size[1] for t in tiles]
        col_w = [0] * cols
        row_h = [0] * rows
        for i, (w, h) in enumerate(zip(tw, th)):
            col_w[i % cols] = max(col_w[i % cols], w)
            row_h[i // cols] = max(row_h[i // cols], h)
        total_w = sum(col_w) + 10
        total_h = sum(row_h) + 44 + 10
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, total_w, 44), fill=(242, 242, 242))
        draw.text((8, 6), title, fill=(0, 0, 0))
        for i, tile in enumerate(tiles):
            x = sum(col_w[: i % cols]) + 5
            y = 44 + sum(row_h[: i // cols]) + 5
            canvas.paste(tile, (x, y))
        canvas.save(output_path)
        return output_path

    top_gallery = save_ref_gallery(top_samples, os.path.join(args.output_dir, "top_10_gallery.png"), f"{args.dataset_name} top {args.top_k} (reference-guided selection)")
    bottom_gallery = save_ref_gallery(bottom_samples, os.path.join(args.output_dir, "bottom_10_gallery.png"), f"{args.dataset_name} bottom {args.top_k} (reference-guided selection)")

    report_md = os.path.join(args.output_dir, "report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(f"# {args.dataset_name} reference-guided recursive Fiedler evaluation\n\n")
        f.write("Selection method: prototype similarity (no GT used for selection, GT used only for scoring)\n\n")
        f.write(f"- annotations: {os.path.abspath(args.annotations)}\n")
        f.write(f"- reference annotations: {os.path.abspath(args.reference_annotations)}\n")
        f.write(f"- image root: {os.path.abspath(args.image_root)}\n")
        f.write(f"- images: {len(records_by_image)}\n")
        f.write(f"- instances: {len(all_results)}\n")
        f.write(f"- mean best IoU: {summary['mean_best_iou']:.4f}\n")
        f.write(f"- median best IoU: {summary['median_best_iou']:.4f}\n")
        f.write(f"- mean best IoU per image: {summary['mean_best_iou_per_image']:.4f}\n\n")
        f.write(f"- results CSV: instance_results.csv\n")
        f.write(f"- summary JSON: summary.json\n")
        if overall_scores:
            f.write("- overall histogram: overall_iou_histogram.png\n")
        if per_class_summary:
            f.write("- class mean chart: class_mean_iou.png\n")
        if class_index is not None:
            f.write(f"- class histograms: class_histograms/index.md\n")
        if top_gallery is not None:
            f.write(f"- top gallery: top_10_gallery.png\n")
        if bottom_gallery is not None:
            f.write(f"- bottom gallery: bottom_10_gallery.png\n")

    print(f"\nReference-guided Fiedler Results:")
    print(f"  Images: {summary['num_images']}")
    print(f"  Instances: {summary['num_instances']}")
    print(f"  Mean IoU: {summary['mean_best_iou']:.4f}")
    print(f"  Median IoU: {summary['median_best_iou']:.4f}")
    print(f"Saved report to {report_md}")


def build_parser():
    parser = argparse.ArgumentParser(description="Reference-guided (non-oracle) Recursive Fiedler evaluation on COCO-20i.")
    parser.add_argument("--dataset-name", type=str, default="fiedler_reference_guided")
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--annotations", type=str, required=True, help="Test set COCO-format annotations")
    parser.add_argument("--reference-image-root", type=str, required=True)
    parser.add_argument("--reference-annotations", type=str, required=True, help="coco20i_reference.json (one labeled example per class)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--vit-arch", type=str, default="small", choices=["base", "small"])
    parser.add_argument("--vit-feat", type=str, default="k", choices=["k", "q", "v", "kqv"])
    parser.add_argument("--patch-size", type=int, default=14, choices=[16, 8, 14])
    parser.add_argument("--dino-version", type=str, default="v2", choices=["v1", "v2", "v3"])
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--fixed-size", type=int, default=1022)
    parser.add_argument("--pretrain-path", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--unlimited-depth", action="store_true")
    parser.add_argument("--unlimited-nodes", action="store_true")
    parser.add_argument("--unlimited-noise", action="store_true")
    parser.add_argument("--unlimited-imbalance", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=63)
    parser.add_argument("--min-region-patches", type=int, default=5)
    parser.add_argument("--min-split-ratio", type=float, default=0.05)
    parser.add_argument("--noise-threshold", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main():
    args = build_parser().parse_args()
    evaluate_dataset(args)


if __name__ == "__main__":
    main()
