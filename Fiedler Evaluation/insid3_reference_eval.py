#!/usr/bin/env python3
"""Reference-guided INSID3 evaluation on COCO-20i (real pipeline, no GT bypass).

Unlike the earlier insid3_zeroshot_eval*.py scripts (which bypassed INSID3's
reference matching entirely and only tested raw agglomerative clustering),
this script calls INSID3's actual predict_mask() -- forward/backward
localization, clustering, seed selection, and cluster aggregation -- exactly
as the model is designed to be used, with a real one-shot reference image+mask
per class (same coco20i_reference.json used for fiedler_reference_eval_v2.py).

This is the fair, apples-to-apples counterpart to fiedler_reference_eval_v2.py:
both pipelines use a reference example (never the test image's own GT) to pick
a final mask; only the reported IoU compares that pick against real GT.

For a small set of specific (image_id, category) pairs -- by default, whatever
appears in a given Fiedler-v2 run's top/bottom-K sample lists -- this script
also renders a visualization panel exposing INSID3's intermediate clustering
result, by re-orchestrating predict_mask's own steps (calling the model's
existing _extract_features/_debias_features/_locate_candidates/
_seed_and_aggregate/_finalize_mask methods in sequence, the same steps
predict_mask itself calls) rather than duplicating any of their internals.
"""

import argparse
import csv
import gc
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

INSID3_ROOT = Path(__file__).parent.parent / "INSID3"
if str(INSID3_ROOT) not in sys.path:
    sys.path.insert(0, str(INSID3_ROOT))

from models import build_insid3_from_args  # noqa: E402
from utils.data import load_image, load_mask, downsample_mask  # noqa: E402
from utils.clustering import agglomerative_clustering, compute_cluster_prototypes  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from fiedler_candidate_eval import (  # noqa: E402
    plot_overall_histogram,
    plot_class_overview,
    plot_class_histograms,
)

COCO20I_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "boat", "dog", "cat", "bird", "horse", "sheep", "cow", "bottle",
    "cup", "chair", "dining table", "backpack", "handbag",
}


def mask_iou(mask_a, mask_b):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection) / float(union)


def resize_mask_squish(mask, image_size):
    """Resize a binary mask to (image_size, image_size) via nearest-neighbor,
    matching INSID3's own squish-resize convention (no aspect-ratio padding --
    see utils/data.py build_transform, which uses a plain Resize((s, s)))."""
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((image_size, image_size), Image.NEAREST)
    ) > 128


def load_instance_records(annotation_json, image_root, max_images=None):
    coco = COCO(annotation_json)
    categories = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
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
            category_name = categories.get(annotation["category_id"], f"category_{annotation['category_id']}")
            if category_name not in COCO20I_CLASSES:
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
                "category_id": annotation["category_id"],
                "category_name": category_name,
                "gt_mask": gt_mask,
            })

        if image_records:
            records_by_image[image_id] = image_records

    return records_by_image, categories, coco


def reference_mask_for(ann, coco_ref, full_coco):
    """Reference mask for one reference annotation. With full_coco given, returns
    the union of EVERY instance of that category in the reference image (a
    class-level mask, as in INSID3's benchmark); otherwise just the single
    annotated instance."""
    if full_coco is None:
        return coco_ref.annToMask(ann).astype(bool), 1
    ids = full_coco.getAnnIds(imgIds=[ann["image_id"]], catIds=[ann["category_id"]])
    mask = None
    for a in full_coco.loadAnns(ids):
        m = full_coco.annToMask(a).astype(bool)
        mask = m if mask is None else (mask | m)
    return mask, len(ids)


def build_reference_tensors(reference_json, image_root, model, device, full_coco=None):
    """One (ref_images, ref_masks) tensor pair per class, built once and kept on
    CPU -- with a 4GB GPU and a large DINOv3 model, holding all 20 reference
    tensors resident on GPU simultaneously leaves no headroom for the per-image
    forward pass. Each pair is moved to GPU only for the instant it's needed."""
    coco_ref = COCO(reference_json)
    categories = {c["id"]: c["name"] for c in coco_ref.loadCats(coco_ref.getCatIds())}

    reference_tensors = {}
    reference_panels = {}  # category_name -> (resized ref image np array, ref mask at image_size)
    for image_id in coco_ref.getImgIds():
        image_info = coco_ref.loadImgs([image_id])[0]
        image_path = os.path.join(image_root, image_info["file_name"])
        ann_ids = coco_ref.getAnnIds(imgIds=[image_id])
        anns = coco_ref.loadAnns(ann_ids)
        if not os.path.exists(image_path):
            print(f"  [ref warning] missing reference image: {image_path}")
            continue

        for ann in anns:
            category_name = categories.get(ann["category_id"], f"category_{ann['category_id']}")
            gt_mask, n_ref_instances = reference_mask_for(ann, coco_ref, full_coco)

            ref_image_tensor, _ = load_image(image_path, model._transform, "cpu")  # (1,C,H,W)
            ref_mask_tensor = load_mask(torch.from_numpy(gt_mask), model.image_size, "cpu")  # (1,H,W)

            reference_tensors[category_name] = (ref_image_tensor, ref_mask_tensor)

            ref_image_pil = Image.open(image_path).convert("RGB").resize((model.image_size, model.image_size), Image.BILINEAR)
            ref_mask_display = resize_mask_squish(gt_mask, model.image_size)
            reference_panels[category_name] = (np.asarray(ref_image_pil), ref_mask_display)

            print(f"  Built reference: {category_name} (from image {image_id}, {n_ref_instances} instance(s) in mask)")

    return reference_tensors, reference_panels


@torch.no_grad()
def predict_mask_with_intermediates(model, ref_images, ref_masks, tgt_image):
    """Re-orchestrates INSID3.predict_mask()'s own steps (calling the model's
    existing methods in the same sequence predict_mask itself uses) so we can
    additionally capture the clustering result and localized candidate region
    for visualization. Does not duplicate any of those methods' internals.

    Note: unlike the vendored predict_mask (which concatenates reference and
    target into one batch before feature extraction), this runs them as two
    separate single-image forward passes and concatenates the resulting
    feature maps afterward -- same total compute, roughly half the peak GPU
    memory during the forward pass itself, which matters on a 4GB card."""
    S = ref_images.shape[0]
    fmaps_ref = model._extract_features(ref_images.unsqueeze(0)).clone()  # (1,S,C,h,w)
    torch.cuda.empty_cache()
    fmaps_tgt_only = model._extract_features(tgt_image.unsqueeze(0)).clone()  # (1,1,C,h,w)
    torch.cuda.empty_cache()
    fmaps = torch.cat([fmaps_ref, fmaps_tgt_only], dim=1)  # (1,S+1,C,h,w)
    fmaps_norm = F.normalize(fmaps, p=2, dim=2)
    _, _, C, h, w = fmaps_norm.shape

    ref_masks_ = ref_masks.unsqueeze(1)
    feat_tgt = fmaps_norm[:, S]

    fmaps_debiased = model._debias_features(fmaps_norm)
    feat_refs_deb = fmaps_debiased[:, :S]
    feat_tgt_deb = fmaps_debiased[:, S]

    ref_prototypes = []
    for s in range(S):
        mask_s = downsample_mask(ref_masks_[s:s + 1], h, w)
        fg = feat_refs_deb[0, s, :, mask_s]
        if fg.shape[1] > 0:
            ref_prototypes.append(fg.mean(dim=1))
    ref_prototype = F.normalize(torch.stack(ref_prototypes).mean(dim=0), p=2, dim=0).unsqueeze(1)

    sim_maps = []
    for m in range(S):
        feat_ref_m = feat_refs_deb[:, m]
        sim_m = torch.einsum('bchw,bcxy->bhwxy', feat_ref_m, feat_tgt_deb)
        sim_maps.append(sim_m)
    candidate_mask = model._locate_candidates(sim_maps, ref_masks_, feat_tgt_deb, ref_prototype, h, w)

    if candidate_mask.sum() == 0:
        pred_mask = model._finalize_mask(candidate_mask, tgt_image)
        return pred_mask, candidate_mask.cpu().numpy(), None

    feat_tgt_flat = feat_tgt[0].reshape(C, -1).permute(1, 0)
    cluster_labels = agglomerative_clustering(feat_tgt_flat, model.tau).reshape(h, w)
    K = int(cluster_labels.max().item()) + 1

    feat_tgt_deb_flat = feat_tgt_deb[0].reshape(C, -1).permute(1, 0)
    cluster_protos = compute_cluster_prototypes(feat_tgt_deb_flat, cluster_labels.view(-1), K)

    pred_mask_grid = model._seed_and_aggregate(
        candidate_mask, cluster_labels, cluster_protos, K, ref_prototype, feat_tgt, feat_tgt_deb, h, w
    )
    pred_mask = model._finalize_mask(pred_mask_grid, tgt_image)

    return pred_mask, candidate_mask.cpu().numpy(), cluster_labels.cpu().numpy()


def cluster_map_to_rgb(cluster_labels, display_size):
    """Colorize an integer cluster-label grid for visualization."""
    if cluster_labels is None:
        return np.zeros((display_size, display_size, 3), dtype=np.uint8)
    K = int(cluster_labels.max()) + 1
    cmap = plt.get_cmap("nipy_spectral")
    colors = (np.array([cmap(i / max(K - 1, 1))[:3] for i in range(K)]) * 255).astype(np.uint8)
    rgb = colors[cluster_labels.reshape(-1)].reshape(cluster_labels.shape[0], cluster_labels.shape[1], 3)
    rgb_image = Image.fromarray(rgb, mode="RGB").resize((display_size, display_size), Image.NEAREST)
    return np.asarray(rgb_image)


def overlay_mask_np(image_array, mask, color=(255, 0, 0), alpha=0.5):
    overlay = image_array.copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image_array.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (image_array.shape[1], image_array.shape[0]), Image.NEAREST)) > 128
    color_arr = np.array(color, dtype=np.float32)
    region = overlay[mask].astype(np.float32)
    overlay[mask] = (region * (1 - alpha) + color_arr * alpha).astype(np.uint8)
    return overlay


def panel_with_label(image_array, label_text):
    pil_canvas = Image.fromarray(image_array)
    draw = ImageDraw.Draw(pil_canvas)
    draw.rectangle((0, 0, pil_canvas.width, 28), fill=(255, 255, 255))
    draw.text((6, 6), label_text, fill=(0, 0, 0))
    return pil_canvas


def build_visualization_row(ref_image, ref_mask, tgt_image_np, gt_mask, cluster_labels, pred_mask, image_size, title):
    ref_panel = panel_with_label(overlay_mask_np(ref_image, ref_mask, (240, 165, 0)), "reference (prompt)")
    tgt_panel = panel_with_label(tgt_image_np, "target")
    gt_panel = panel_with_label(overlay_mask_np(tgt_image_np, gt_mask, (230, 65, 65)), "ground truth")
    cluster_panel = panel_with_label(cluster_map_to_rgb(cluster_labels, image_size), "INSID3 clustering")
    pred_panel = panel_with_label(overlay_mask_np(tgt_image_np, pred_mask, (70, 200, 110)), "INSID3 final prediction")

    panels = [ref_panel, tgt_panel, gt_panel, cluster_panel, pred_panel]
    display_w, display_h = panels[0].size
    canvas = Image.new("RGB", (display_w * len(panels), display_h + 32), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 32), fill=(240, 240, 240))
    draw.text((8, 8), title, fill=(0, 0, 0))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * display_w, 32))
    return canvas


def visualize_extremes(model, reference_tensors, reference_panels, records_by_image, all_results, args, viz_dir):
    """Post-run: re-run inference for this run's own top-K and bottom-K
    (image, category) pairs by class-level IoU and save a 5-panel figure for
    each. Done after the run (rather than caching every prediction) because
    only ~2K pairs are needed and re-inference for them is cheap."""
    ranked = sorted(all_results, key=lambda r: r["iou"])
    k = min(args.viz_top_k, len(ranked) // 2) if len(ranked) >= 2 else len(ranked)
    bottom = ranked[:k]
    top = ranked[-k:][::-1]  # best first

    selections = [("top", i + 1, r) for i, r in enumerate(top)] + [("bottom", i + 1, r) for i, r in enumerate(bottom)]
    print(f"\nVisualizing {len(top)} top and {len(bottom)} bottom (image, category) pairs...", flush=True)

    for group, rank, result in selections:
        image_id = result["image_id"]
        category_name = result["category_name"]
        try:
            records = [r for r in records_by_image[image_id] if r["category_name"] == category_name]
            ref_images = reference_tensors[category_name][0].to(args.device)
            ref_masks = reference_tensors[category_name][1].to(args.device)
            tgt_image, _ = load_image(result["image_path"], model._transform, args.device)
            tgt_image_np = np.asarray(Image.open(result["image_path"]).convert("RGB").resize((args.image_size, args.image_size), Image.BILINEAR))

            pred_mask, _, cluster_labels = predict_mask_with_intermediates(model, ref_images, ref_masks, tgt_image)
            pred_mask_np = pred_mask.cpu().numpy() if hasattr(pred_mask, "cpu") else np.asarray(pred_mask)

            gt_union = np.zeros_like(pred_mask_np, dtype=bool)
            for record in records:
                gt_union |= resize_mask_squish(record["gt_mask"], args.image_size)

            ref_image_np, ref_mask_np = reference_panels[category_name]
            title = f"{group} #{rank} | {category_name} | img {image_id} | class-level IoU {result['iou']:.3f} | {len(records)} instance(s)"
            row = build_visualization_row(ref_image_np, ref_mask_np, tgt_image_np, gt_union, cluster_labels, pred_mask_np, args.image_size, title)
            out_path = os.path.join(viz_dir, f"{group}_{rank:02d}_iou{result['iou']:.3f}_{image_id}_{category_name.replace(' ', '_')}.png")
            row.save(out_path)
            print(f"  [viz] saved {out_path}", flush=True)

            del ref_images, ref_masks, tgt_image
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
        except Exception as e:
            print(f"  [viz] FAILED for {group} #{rank} ({image_id}, {category_name}): {e}", flush=True)


def evaluate_dataset(args):
    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    records_by_image, categories, coco = load_instance_records(args.annotations, args.image_root, max_images=None)
    ordered_ids = sorted(records_by_image.keys())
    if args.max_images is not None:
        ordered_ids = ordered_ids[: args.max_images]
    records_by_image = {image_id: records_by_image[image_id] for image_id in ordered_ids}
    total_images = len(records_by_image)
    print(f"Evaluating on {total_images} images (COCO-20i classes only)")

    insid3_args = argparse.Namespace(
        model_size=args.model_size,
        image_size=args.image_size,
        svd_comps=args.svd_comps,
        tau=args.tau,
        merge_thresh=args.merge_thresh,
        crf_size=args.crf_size,
        crf_mask_refinement=False,
        device=args.device,
        seed=args.seed,
    )
    print("Loading INSID3 model...")
    model = build_insid3_from_args(insid3_args)
    model.eval()

    print("Building reference tensors...")
    full_coco = COCO(args.full_annotations) if args.class_level_reference else None
    reference_tensors, reference_panels = build_reference_tensors(args.reference_annotations, args.reference_image_root, model, args.device, full_coco=full_coco)
    missing = COCO20I_CLASSES - set(reference_tensors.keys())
    if missing:
        print(f"WARNING: no reference for: {sorted(missing)} -- instances of these classes will be skipped")

    csv_path = os.path.join(args.output_dir, "class_level_results.csv")
    csv_fieldnames = ["image_id", "category_id", "category_name", "num_instances", "iou", "image_path"]
    csv_file = open(csv_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    failures_path = os.path.join(args.output_dir, "failures.jsonl")
    failures_file = open(failures_path, "w", encoding="utf-8")

    all_results = []
    class_scores = defaultdict(list)
    per_class_stats = defaultdict(lambda: {"count": 0, "scores": []})
    # Cumulative intersection/union per class, matching INSID3's own AverageMeter
    # (utils/metrics.py): iou_per_class = sum(intersections) / sum(unions) across
    # the whole run, then averaged over classes -- not a flat per-episode average.
    cumulative_inter = defaultdict(float)
    cumulative_union = defaultdict(float)
    image_summaries = []
    elapsed_times = []

    for index, (image_id, records) in enumerate(records_by_image.items(), start=1):
        image_path = records[0]["image_path"]
        image_start = time.time()

        try:
            tgt_image, _ = load_image(image_path, model._transform, args.device)  # (1,C,H,W)

            records_by_category = defaultdict(list)
            for record in records:
                records_by_category[record["category_name"]].append(record)

            image_results = []
            for category_name, cat_records in records_by_category.items():
                ref_entry = reference_tensors.get(category_name)
                if ref_entry is None:
                    continue
                ref_images = ref_entry[0].to(args.device)
                ref_masks = ref_entry[1].to(args.device)

                pred_mask, _, _ = predict_mask_with_intermediates(model, ref_images, ref_masks, tgt_image)

                pred_mask_np = pred_mask.cpu().numpy() if hasattr(pred_mask, "cpu") else np.asarray(pred_mask)
                del ref_images, ref_masks
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                # Class-level GT: union of every instance of this category present in
                # the image -- matches the real benchmark's semantic tgt_mask (see
                # datasets/coco.py), not any single instance's mask.
                gt_union = np.zeros_like(pred_mask_np, dtype=bool)
                for record in cat_records:
                    gt_union |= resize_mask_squish(record["gt_mask"], args.image_size)

                intersection = np.logical_and(pred_mask_np, gt_union).sum()
                union = np.logical_or(pred_mask_np, gt_union).sum()
                iou = mask_iou(pred_mask_np, gt_union)

                cumulative_inter[category_name] += float(intersection)
                cumulative_union[category_name] += float(union)

                result = {
                    "image_id": image_id,
                    "category_id": cat_records[0]["category_id"],
                    "category_name": category_name,
                    "num_instances": len(cat_records),
                    "iou": float(iou),
                    "image_path": image_path,
                }
                all_results.append(result)
                image_results.append(result)
                class_scores[category_name].append(float(iou))
                per_class_stats[category_name]["count"] += 1
                per_class_stats[category_name]["scores"].append(float(iou))
                csv_writer.writerow(result)

            csv_file.flush()

            if image_results:
                image_summaries.append({
                    "image_id": image_id,
                    "image_path": image_path,
                    "num_categories": len(image_results),
                    "best_category_iou": max(r["iou"] for r in image_results),
                    "mean_category_iou": float(np.mean([r["iou"] for r in image_results])),
                })

            elapsed = time.time() - image_start
            elapsed_times.append(elapsed)
            avg_elapsed = np.mean(elapsed_times)
            eta_minutes = avg_elapsed * (total_images - index) / 60.0
            print(f"[{index}/{total_images}] {os.path.basename(image_path)} ({len(records)} instances) | elapsed={elapsed:.2f}s | ETA={eta_minutes:.1f}m", flush=True)

            del tgt_image
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

        except Exception as e:
            elapsed = time.time() - image_start
            print(f"ERROR: Image {image_id} ({image_path}) failed after {elapsed:.2f}s: {e}", flush=True)
            failures_file.write(json.dumps({"image_id": image_id, "image_path": image_path, "error": str(e), "elapsed": elapsed}) + "\n")
            failures_file.flush()
            try:
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

    overall_scores = [r["iou"] for r in all_results]
    per_image_scores = [s["best_category_iou"] for s in image_summaries]

    # Official-style cumulative-area mIoU, matching INSID3's own utils/metrics.py
    # AverageMeter: per-class IoU = sum(intersections)/sum(unions) across the
    # whole run, then averaged over classes. This is the number comparable to
    # the paper's reported figure -- not the flat per-episode average above.
    per_class_cumulative_iou = {}
    for class_name in cumulative_union:
        u = cumulative_union[class_name]
        per_class_cumulative_iou[class_name] = (cumulative_inter[class_name] / u) if u > 0 else 0.0
    cumulative_miou = float(np.mean(list(per_class_cumulative_iou.values()))) if per_class_cumulative_iou else 0.0

    summary = {
        "dataset_name": args.dataset_name,
        "selection_method": "INSID3 reference-guided (full pipeline: forward/backward localization + clustering + seed/aggregate), class-level (union-of-instances) GT",
        "annotations": os.path.abspath(args.annotations),
        "reference_annotations": os.path.abspath(args.reference_annotations),
        "image_root": os.path.abspath(args.image_root),
        "model_size": args.model_size,
        "image_size": args.image_size,
        "tau": args.tau,
        "merge_thresh": args.merge_thresh,
        "svd_comps": args.svd_comps,
        "num_images": len(records_by_image),
        "num_image_category_pairs": len(all_results),
        "cumulative_miou": cumulative_miou,
        "cumulative_iou_per_class": per_class_cumulative_iou,
        "mean_iou_flat": float(np.mean(overall_scores)) if overall_scores else 0.0,
        "median_iou_flat": float(np.median(overall_scores)) if overall_scores else 0.0,
        "min_iou_flat": float(np.min(overall_scores)) if overall_scores else 0.0,
        "max_iou_flat": float(np.max(overall_scores)) if overall_scores else 0.0,
        "mean_iou_per_image": float(np.mean(per_image_scores)) if per_image_scores else 0.0,
        "median_iou_per_image": float(np.median(per_image_scores)) if per_image_scores else 0.0,
        "per_class": per_class_summary,
        "image_summaries": image_summaries,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if overall_scores:
        plot_overall_histogram(overall_scores, os.path.join(args.output_dir, "overall_iou_histogram.png"), f"{args.dataset_name} best IoU distribution")
    if per_class_summary:
        plot_class_overview(per_class_summary, os.path.join(args.output_dir, "class_mean_iou.png"), f"{args.dataset_name} mean IoU by class")
    if class_scores:
        plot_class_histograms(class_scores, os.path.join(args.output_dir, "class_histograms"), args.dataset_name)

    print("\nINSID3 Reference-Guided Results (class-level):")
    print(f"  Images: {summary['num_images']}")
    print(f"  (image, category) pairs: {summary['num_image_category_pairs']}")
    print(f"  Cumulative mIoU (paper-style, per-class area sums): {summary['cumulative_miou']:.4f}")
    print(f"  Flat mean IoU (per-episode average): {summary['mean_iou_flat']:.4f}")
    print(f"  Flat median IoU: {summary['median_iou_flat']:.4f}")

    if all_results and not args.no_viz:
        visualize_extremes(model, reference_tensors, reference_panels, records_by_image, all_results, args, viz_dir)
        print(f"\nVisualizations saved to {viz_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Reference-guided (real pipeline) INSID3 evaluation on COCO-20i.")
    parser.add_argument("--dataset-name", default="insid3_reference_guided")
    parser.add_argument("--annotations", default="Fiedler Evaluation/data/coco_500_from_1000.json")
    parser.add_argument("--image-root", default="Fiedler Evaluation/datasets/val2017")
    parser.add_argument("--reference-annotations", default="Fiedler Evaluation/data/coco20i/coco20i_reference.json")
    parser.add_argument("--reference-image-root", default="Fiedler Evaluation/datasets/val2017")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--viz-top-k", type=int, default=10, help="Number of top and bottom (image, category) pairs to visualize after the run")
    parser.add_argument("--class-level-reference", action="store_true", help="Build each reference mask as the union of ALL instances of that class in the reference image (needs --full-annotations), instead of the single annotated instance")
    parser.add_argument("--full-annotations", default="CutLER Implementation/TFSS-CutLER/datasets/coco/annotations/instances_val2017.json")
    parser.add_argument("--no-viz", action="store_true", help="Skip the post-run visualization step")
    parser.add_argument("--model-size", default="large", choices=["small", "base", "large"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--svd-comps", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--merge-thresh", type=float, default=0.2)
    parser.add_argument("--crf-size", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    evaluate_dataset(build_parser())
