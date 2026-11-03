#!/usr/bin/env python3
"""Reference-guided Recursive Fiedler evaluation, v2: adds positional debiasing +
forward/backward localization (INSID3-style) plus a size-aware node-selection rule,
on top of the plain-prototype-similarity baseline in fiedler_reference_eval.py.

Baseline (fiedler_reference_eval.py) selection:
    score(node) = cosine( mean_feature(node), reference_prototype )
    pick node = argmax score

v2 selection here:
    1. Build a positional basis (SVD over a blank/noise image's features) and project
       it out of every feature before comparing anything -- removes the "where in the
       frame" bias that plain ViT features carry (mirrors models/insid3.py:246-271).
    2. Forward localization: per-patch cosine similarity to the reference prototype;
       patches with positive similarity are forward candidates (falls back to top
       decile if none are positive) -- mirrors models/insid3.py:_locate_candidates
       forward branch.
    3. Backward localization: for every target patch, find its single nearest-neighbor
       patch in the *reference image* (full patch-to-patch similarity, not the
       prototype) and check whether that neighbor lies inside the reference mask --
       mirrors the backward branch of the same function.
    4. candidate_patch_mask = forward_mask & backward_mask (never touches GT).
    5. Node selection: instead of picking whichever single node has the highest mean
       feature similarity (which can favor tiny, overfit-to-noise fragments), pick the
       node whose region best matches candidate_patch_mask by IoU. This folds in a
       natural size preference for free -- a node that's much smaller or much larger
       than the localized region scores worse on IoU than one that matches it well.
    6. GT is used only afterward, to score the selected node -- never during selection.
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
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CUTLER_ROOT = os.path.join(WORKSPACE_ROOT, "CutLER Implementation", "TFSS-CutLER")
MASKCUT_ROOT = os.path.join(CUTLER_ROOT, "maskcut")
if CUTLER_ROOT not in sys.path:
    sys.path.insert(0, CUTLER_ROOT)
if MASKCUT_ROOT not in sys.path:
    sys.path.insert(0, MASKCUT_ROOT)

from fiedler_class_transfer import build_backbone, resize_stretch, resize_with_aspect_ratio  # noqa: E402
from recursive_fiedler import collect_recursive_candidates, load_image_features, upsample_mask  # noqa: E402

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
from fiedler_reference_eval import downsample_mask_to_grid  # noqa: E402


COCO20I_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "boat", "dog", "cat", "bird", "horse", "sheep", "cow", "bottle",
    "cup", "chair", "dining table", "backpack", "handbag",
}


def resize_for_mode(image_pil, fixed_size, resize_mode):
    if resize_mode == "stretch":
        return resize_stretch(image_pil, fixed_size)
    return resize_with_aspect_ratio(image_pil, fixed_size)


def l2_normalize_columns(features):
    """features: (C, N) -> L2-normalize each column (patch) independently."""
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return features / norms


def build_positional_basis(backbone, fixed_size, cpu, n_components):
    """Estimate a positional feature subspace from a blank (content-free) image,
    following the same logic as models/insid3.py:_build_positional_basis."""
    blank = Image.fromarray(np.zeros((fixed_size, fixed_size, 3), dtype=np.uint8), mode="RGB")
    blank_resized, _ = resize_with_aspect_ratio(blank, fixed_size)
    tmp_path = os.path.join(WORKSPACE_ROOT, "Fiedler Evaluation", "output", "_positional_basis_tmp.png")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    blank_resized.save(tmp_path)

    _, features, grid_h, grid_w = load_image_features(tmp_path, backbone, fixed_size, cpu, None)
    features_np = features.numpy() if hasattr(features, "numpy") else np.asarray(features)  # (C, N)

    try:
        os.remove(tmp_path)
    except OSError:
        pass

    E = l2_normalize_columns(features_np)  # (C, N)
    E = E - E.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(E, full_matrices=False)
    n_components = min(n_components, U.shape[1] - 1)
    return U[:, :n_components].astype(np.float32)  # (C, n_components)


def debias_and_normalize(features_np, basis):
    """Project out the positional subspace, then re-L2-normalize each patch."""
    C = features_np.shape[0]
    P_perp = np.eye(C, dtype=np.float32) - basis @ basis.T
    deb = P_perp @ features_np
    return l2_normalize_columns(deb)


class DebiasedBackbone:
    """Wraps a (features, grid)-returning backbone so the Fiedler tree is built on
    debiased features (--debias-tree). Without it, only localization is debiased."""

    def __init__(self, inner, basis):
        self.inner = inner
        self.basis = basis

    def __call__(self, tensor):
        import torch

        features, grid = self.inner(tensor)  # (N, C)
        deb = debias_and_normalize(features.float().cpu().numpy().T, self.basis)  # (C, N)
        return torch.from_numpy(np.ascontiguousarray(deb.T)), grid


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


def build_reference_data(reference_json, image_root, backbone, fixed_size, cpu, basis, full_coco=None, mask_interp="nearest", resize_mode="letterbox"):
    """Return {category_name: {"prototype": (C,), "feats": (C, N_ref) L2-norm debiased,
    "mask_flat": (N_ref,) bool}} for every class in the reference set."""
    coco_ref = COCO(reference_json)
    categories = {c["id"]: c["name"] for c in coco_ref.loadCats(coco_ref.getCatIds())}

    data = {}
    for image_id in coco_ref.getImgIds():
        image_info = coco_ref.loadImgs([image_id])[0]
        image_path = os.path.join(image_root, image_info["file_name"])
        ann_ids = coco_ref.getAnnIds(imgIds=[image_id])
        anns = coco_ref.loadAnns(ann_ids)
        if not os.path.exists(image_path):
            print(f"  [ref warning] missing reference image: {image_path}")
            continue

        image_pil = Image.open(image_path).convert("RGB")
        _, resize_metadata = resize_for_mode(image_pil, fixed_size, resize_mode)
        _, features, grid_h, grid_w = load_image_features(image_path, backbone, fixed_size, cpu, resize_metadata, resize_mode=resize_mode)
        features_np = features.numpy() if hasattr(features, "numpy") else np.asarray(features)
        features_deb = debias_and_normalize(features_np, basis)  # (C, N_ref)

        for ann in anns:
            category_name = categories.get(ann["category_id"], f"category_{ann['category_id']}")
            gt_mask, n_ref_instances = reference_mask_for(ann, coco_ref, full_coco)
            gt_mask_fixed = resize_binary_mask(gt_mask, fixed_size, resize_metadata)
            mask_small = downsample_mask_to_grid(gt_mask_fixed, grid_h, grid_w, mode=mask_interp)
            mask_flat = mask_small.reshape(-1)

            if mask_flat.sum() == 0:
                print(f"  [ref warning] empty downsampled mask for {category_name}, skipping")
                continue

            fg_feats = features_deb[:, mask_flat]  # (C, n_fg)
            prototype = fg_feats.mean(axis=1)
            prototype = prototype / max(np.linalg.norm(prototype), 1e-8)

            data[category_name] = {
                "prototype": prototype.astype(np.float32),
                "feats": features_deb.astype(np.float32),
                "mask_flat": mask_flat,
            }
            print(f"  Built reference data: {category_name} (from image {image_id}, {n_ref_instances} instance(s) in mask, {int(mask_flat.sum())} fg patches, {features_deb.shape[1]} total patches)")

    return data


def localize_candidate_patches(tgt_feats_deb, ref_entry):
    """Return (candidate_mask (N_tgt,) bool, forward_mask, backward_mask) -- GT never touched."""
    prototype = ref_entry["prototype"]  # (C,)
    ref_feats = ref_entry["feats"]      # (C, N_ref)
    ref_mask_flat = ref_entry["mask_flat"]  # (N_ref,)

    # Forward: prototype similarity per target patch.
    fwd_sim = prototype @ tgt_feats_deb  # (N_tgt,)
    forward_mask = fwd_sim > 0
    if not forward_mask.any():
        thresh = np.quantile(fwd_sim, 0.9)
        forward_mask = fwd_sim >= thresh

    # Backward: nearest-neighbor patch in the reference image, check if inside ref mask.
    # Chunked to avoid materializing the full (N_tgt, N_ref) matrix at once -- on this
    # machine's tight memory budget, repeated large transient allocations fragment the
    # process heap over hundreds of images and eventually get the process killed by the
    # OS outright (no Python exception to catch), even though any single allocation is
    # well within budget.
    n_tgt = tgt_feats_deb.shape[1]
    chunk_size = 512
    best_ref_idx = np.empty(n_tgt, dtype=np.int64)
    for start in range(0, n_tgt, chunk_size):
        end = min(start + chunk_size, n_tgt)
        sim_chunk = tgt_feats_deb[:, start:end].T @ ref_feats  # (chunk, N_ref)
        best_ref_idx[start:end] = np.argmax(sim_chunk, axis=1)
        del sim_chunk
    backward_mask = ref_mask_flat[best_ref_idx]

    candidate_mask = forward_mask & backward_mask
    if not candidate_mask.any():
        candidate_mask = forward_mask  # fall back to forward-only if backward is too strict

    return candidate_mask, forward_mask, backward_mask


def select_node_by_localization(tree, candidate_patch_mask):
    """Pick the node whose region best matches (by IoU) the localized candidate
    patch mask -- this naturally folds in a size preference (a node much smaller
    or much larger than the localized region scores worse)."""
    best_idx = -1
    best_iou = -1.0
    for idx, node in enumerate(tree["nodes"]):
        region_flat = node["region_mask"].reshape(-1)
        union = np.logical_or(region_flat, candidate_patch_mask).sum()
        if union == 0:
            continue
        intersection = np.logical_and(region_flat, candidate_patch_mask).sum()
        iou = intersection / union
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    return best_idx, best_iou


def node_purities(tree, candidate_patch_mask):
    """Per node: fraction of the node's patches that lie inside the localized
    candidate region (|node & candidate| / |node|)."""
    purities = np.zeros(len(tree["nodes"]), dtype=np.float32)
    for idx, node in enumerate(tree["nodes"]):
        region_flat = node["region_mask"].reshape(-1)
        area = int(region_flat.sum())
        if area > 0:
            purities[idx] = np.logical_and(region_flat, candidate_patch_mask).sum() / area
    return purities


def union_of_pure_nodes(tree, purities, tau, fallback_idx):
    """Union (at patch-grid resolution) of every node whose purity >= tau -- the
    tree analogue of INSID3's threshold-based cluster aggregation. Falls back to
    the single best-IoU node if nothing clears the threshold."""
    idxs = np.where(purities >= tau)[0]
    if len(idxs) == 0:
        idxs = np.array([fallback_idx])
    grid_shape = tree["nodes"][0]["region_mask"].shape
    union_flat = np.zeros(grid_shape[0] * grid_shape[1], dtype=bool)
    for i in idxs:
        union_flat |= tree["nodes"][int(i)]["region_mask"].reshape(-1)
    return union_flat.reshape(grid_shape), len(idxs)


def evaluate_dataset(args):
    os.makedirs(args.output_dir, exist_ok=True)

    union_taus = [float(t) for t in args.union_taus.split(",")] if args.union_taus else []
    if union_taus and not args.class_level:
        raise ValueError("--union-taus requires --class-level")

    records_by_image = load_instance_records(args.annotations, args.image_root, max_images=None)
    filtered = {}
    for image_id, records in records_by_image.items():
        kept = [r for r in records if r["category_name"] in COCO20I_CLASSES]
        if kept:
            filtered[image_id] = kept

    # Slice by image_id order (stable across batched subprocess invocations).
    ordered_ids = sorted(filtered.keys())
    batch_ids = ordered_ids[args.skip_images:]
    if args.max_images is not None:
        batch_ids = batch_ids[: args.max_images]
    records_by_image = {image_id: filtered[image_id] for image_id in batch_ids}
    total_images = len(records_by_image)
    print(f"Evaluating on {total_images} images (COCO-20i classes only, skip={args.skip_images}, of {len(filtered)} total filtered)")

    backbone = build_backbone(
        args.vit_arch, args.vit_feat, args.patch_size, args.pretrain_path,
        dino_version=args.dino_version, debiased=False,
    )
    if not args.cpu and args.dino_version == "v1":
        backbone.half()
        backbone.cuda()

    print("Building positional basis...")
    basis = build_positional_basis(backbone, args.fixed_size, args.cpu, args.svd_components)
    print(f"  basis shape: {basis.shape}")
    tree_backbone = DebiasedBackbone(backbone, basis) if args.debias_tree else backbone
    print(f"  tree built on {'debiased' if args.debias_tree else 'raw'} features")

    print("Building reference data (prototypes + full patch features + masks)...")
    full_coco = COCO(args.full_annotations) if args.class_level_reference else None
    reference_data = build_reference_data(args.reference_annotations, args.reference_image_root, backbone, args.fixed_size, args.cpu, basis, full_coco=full_coco, mask_interp=args.mask_interp, resize_mode=args.resize_mode)
    missing = COCO20I_CLASSES - set(reference_data.keys())
    if missing:
        print(f"WARNING: no reference data for: {sorted(missing)} -- instances of these classes will be skipped")

    csv_path = os.path.join(args.output_dir, "instance_results.csv")
    csv_fieldnames = [
        "image_id", "annotation_id", "category_id", "category_name",
        "best_iou", "best_candidate_id", "best_candidate_depth", "best_candidate_status",
        "best_candidate_area_patches", "candidate_count", "localization_iou",
        "candidate_patch_fraction", "num_instances", "intersection", "union", "image_path",
        "candidate_iou", "candidate_intersection", "candidate_union",
        "combo_and_intersection", "combo_and_union", "combo_or_intersection", "combo_or_union",
    ]
    for t in union_taus:
        csv_fieldnames += [f"inter_tau{t:g}", f"union_tau{t:g}", f"nsel_tau{t:g}"]
    write_header = not (args.append and os.path.exists(csv_path))
    csv_file = open(csv_path, "a" if args.append else "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    if write_header:
        csv_writer.writeheader()

    failures_path = os.path.join(args.output_dir, "failures.jsonl")
    failures_file = open(failures_path, "a" if args.append else "w", encoding="utf-8")

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
            _, resize_metadata = resize_for_mode(image_pil, args.fixed_size, args.resize_mode)

            tree_args = prepare_args(args, image_path, resize_metadata)
            tree = collect_recursive_candidates(tree_args, backbone=tree_backbone)

            _, features, grid_h, grid_w = load_image_features(image_path, backbone, args.fixed_size, args.cpu, resize_metadata, resize_mode=args.resize_mode)
            features_np = features.numpy() if hasattr(features, "numpy") else np.asarray(features)
            features_deb = debias_and_normalize(features_np, basis)

            candidate_masks = candidate_masks_from_tree(tree, args.fixed_size, resize_metadata, mask_interp=args.mask_interp)
            if not candidate_masks:
                continue

            image_results = []
            records_by_category = defaultdict(list)
            for record in records:
                records_by_category[record["category_name"]].append(record)

            for category_name, cat_records in records_by_category.items():
                ref_entry = reference_data.get(category_name)
                if ref_entry is None:
                    continue

                candidate_patch_mask, _, _ = localize_candidate_patches(features_deb, ref_entry)
                best_node_idx, loc_iou = select_node_by_localization(tree, candidate_patch_mask)
                if best_node_idx < 0:
                    continue

                best_node = tree["nodes"][best_node_idx]
                selected_mask = candidate_masks[best_node_idx]
                patch_fraction = float(candidate_patch_mask.mean())

                # Raw localization quality: how well does candidate_patch_mask itself
                # (forward & backward matching, before any tree-node selection) line
                # up with GT -- isolates the localization step from the Fiedler tree's
                # contribution.
                candidate_grid = candidate_patch_mask.reshape(grid_h, grid_w)
                candidate_mask_px = upsample_mask(candidate_grid, args.fixed_size, mode=args.mask_interp)

                # Scoring units: per-instance (default) or one class-level unit per
                # category scored against the union of all its instances, matching
                # INSID3's benchmark protocol (semantic tgt_mask).
                if args.class_level:
                    gt_union = np.zeros((args.fixed_size, args.fixed_size), dtype=bool)
                    for record in cat_records:
                        gt_union |= resize_binary_mask(record["gt_mask"], args.fixed_size, resize_metadata)
                    units = [(-1, gt_union, len(cat_records))]
                else:
                    units = [
                        (record["annotation_id"], resize_binary_mask(record["gt_mask"], args.fixed_size, resize_metadata), 1)
                        for record in cat_records
                    ]

                purities = node_purities(tree, candidate_patch_mask) if union_taus else None

                for annotation_id, gt_mask_fixed, num_instances in units:
                    union_fields = {}
                    for t in union_taus:
                        union_grid, n_sel = union_of_pure_nodes(tree, purities, t, best_node_idx)
                        union_mask = upsample_mask(union_grid, args.fixed_size, mode=args.mask_interp)
                        union_fields[f"inter_tau{t:g}"] = int(np.logical_and(union_mask, gt_mask_fixed).sum())
                        union_fields[f"union_tau{t:g}"] = int(np.logical_or(union_mask, gt_mask_fixed).sum())
                        union_fields[f"nsel_tau{t:g}"] = n_sel

                    iou = mask_iou(selected_mask, gt_mask_fixed)
                    intersection = int(np.logical_and(selected_mask, gt_mask_fixed).sum())
                    union_area = int(np.logical_or(selected_mask, gt_mask_fixed).sum())

                    candidate_iou = mask_iou(candidate_mask_px, gt_mask_fixed)
                    candidate_intersection = int(np.logical_and(candidate_mask_px, gt_mask_fixed).sum())
                    candidate_union = int(np.logical_or(candidate_mask_px, gt_mask_fixed).sum())

                    # "Mean of node and candidate" check: with two binary masks, mean
                    # thresholding only has two non-trivial outcomes -- AND (both agree)
                    # or OR (either agrees). Both derive from the same noisy reference
                    # signal as selected_mask itself (selected_mask was chosen to match
                    # candidate_mask_px), so neither is expected to add information --
                    # computed here only to confirm that expectation.
                    combo_and = np.logical_and(selected_mask, candidate_mask_px)
                    combo_or = np.logical_or(selected_mask, candidate_mask_px)
                    combo_and_intersection = int(np.logical_and(combo_and, gt_mask_fixed).sum())
                    combo_and_union = int(np.logical_or(combo_and, gt_mask_fixed).sum())
                    combo_or_intersection = int(np.logical_and(combo_or, gt_mask_fixed).sum())
                    combo_or_union = int(np.logical_or(combo_or, gt_mask_fixed).sum())

                    result = {
                        "image_id": image_id,
                        "annotation_id": annotation_id,
                        "category_id": cat_records[0]["category_id"],
                        "category_name": category_name,
                        "num_instances": num_instances,
                        "intersection": intersection,
                        "union": union_area,
                        "title": f'{category_name} ({"class-level" if args.class_level else "instance " + str(annotation_id)}) | img {image_id} | IoU {iou:.3f}',
                        "best_iou": float(iou),
                        "best_candidate_id": best_node["id"],
                        "best_candidate_depth": best_node["depth"],
                        "best_candidate_status": best_node["status"],
                        "best_candidate_area_patches": best_node["region_area_patches"],
                        "candidate_count": len(candidate_masks),
                        "localization_iou": float(loc_iou),
                        "candidate_patch_fraction": patch_fraction,
                        "candidate_iou": float(candidate_iou),
                        "candidate_intersection": candidate_intersection,
                        "candidate_union": candidate_union,
                        "combo_and_intersection": combo_and_intersection,
                        "combo_and_union": combo_and_union,
                        "combo_or_intersection": combo_or_intersection,
                        "combo_or_union": combo_or_union,
                        "image_path": cat_records[0]["image_path"],
                        "gt_mask_fixed": gt_mask_fixed,
                        "best_candidate_mask": selected_mask,
                        "resize_metadata": resize_metadata,
                        **union_fields,
                    }
                    all_results.append(result)
                    image_results.append(result)
                    class_scores[category_name].append(float(iou))
                    per_class_stats[category_name]["count"] += 1
                    per_class_stats[category_name]["scores"].append(float(iou))

                    csv_writer.writerow({k: result.get(k) for k in csv_fieldnames})

                del candidate_patch_mask
                gc.collect()

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

            del image_pil, tree, features, features_np, features_deb, candidate_masks
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
        "selection_method": "reference-guided v2 (positional debiasing + forward/backward localization + IoU-based node selection, no GT)",
        "annotations": os.path.abspath(args.annotations),
        "reference_annotations": os.path.abspath(args.reference_annotations),
        "image_root": os.path.abspath(args.image_root),
        "fixed_size": args.fixed_size,
        "patch_size": args.patch_size,
        "tau": args.tau,
        "svd_components": args.svd_components,
        "debias_tree": args.debias_tree,
        "resize_mode": args.resize_mode,
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

    top_gallery = save_gallery(
        top_samples,
        os.path.join(args.output_dir, "top_10_gallery.png"), args.fixed_size,
        f"{args.dataset_name} top {args.top_k} (v2: localization-based selection)",
    )
    bottom_gallery = save_gallery(
        bottom_samples,
        os.path.join(args.output_dir, "bottom_10_gallery.png"), args.fixed_size,
        f"{args.dataset_name} bottom {args.top_k} (v2: localization-based selection)",
    )

    report_md = os.path.join(args.output_dir, "report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(f"# {args.dataset_name} reference-guided recursive Fiedler evaluation (v2)\n\n")
        f.write("Selection method: positional debiasing + forward/backward localization + IoU-based node selection (no GT used for selection, GT used only for scoring)\n\n")
        f.write(f"- annotations: {os.path.abspath(args.annotations)}\n")
        f.write(f"- reference annotations: {os.path.abspath(args.reference_annotations)}\n")
        f.write(f"- image root: {os.path.abspath(args.image_root)}\n")
        f.write(f"- svd components (positional basis): {args.svd_components}\n")
        f.write(f"- tree features: {'debiased' if args.debias_tree else 'raw'}\n")
        f.write(f"- resize mode: {args.resize_mode}\n")
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
            f.write("- class histograms: class_histograms/index.md\n")
        if top_gallery is not None:
            f.write("- top gallery: top_10_gallery.png\n")
        if bottom_gallery is not None:
            f.write("- bottom gallery: bottom_10_gallery.png\n")

    print(f"\nReference-guided Fiedler v2 Results:")
    print(f"  Images: {summary['num_images']}")
    print(f"  Instances: {summary['num_instances']}")
    print(f"  Mean IoU: {summary['mean_best_iou']:.4f}")
    print(f"  Median IoU: {summary['median_best_iou']:.4f}")
    print(f"Saved report to {report_md}")


def build_parser():
    parser = argparse.ArgumentParser(description="Reference-guided Recursive Fiedler v2: debiasing + localization + IoU-based selection.")
    parser.add_argument("--dataset-name", type=str, default="fiedler_reference_guided_v2")
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--reference-image-root", type=str, required=True)
    parser.add_argument("--reference-annotations", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--vit-arch", type=str, default="small", choices=["base", "small"])
    parser.add_argument("--vit-feat", type=str, default="k", choices=["k", "q", "v", "kqv"])
    parser.add_argument("--patch-size", type=int, default=14, choices=[16, 8, 14])
    parser.add_argument("--dino-version", type=str, default="v2", choices=["v1", "v2", "v3"])
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--fixed-size", type=int, default=1022)
    parser.add_argument("--resize-mode", type=str, default="letterbox", choices=["letterbox", "stretch"], help="Same as fiedler_candidate_eval.py --resize-mode; applied to both reference and target images")
    parser.add_argument("--svd-components", type=int, default=100, help="Number of components removed as the positional subspace")
    parser.add_argument("--debias-tree", action="store_true", help="Also build the Fiedler tree on debiased features (default: tree on raw features, debiasing only for localization)")
    parser.add_argument("--pretrain-path", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-images", type=int, default=0, help="Skip this many (filtered) images before starting -- for batched runs")
    parser.add_argument("--union-taus", type=str, default="", help="Comma-separated purity thresholds (e.g. 0.3,0.5,0.7). For each, also score the union of all tree nodes whose purity (fraction inside the localized candidate region) >= tau, alongside the single best node. Requires --class-level.")
    parser.add_argument("--class-level-reference", action="store_true", help="Build each reference mask as the union of ALL instances of that class in the reference image (needs --full-annotations), instead of the single annotated instance")
    parser.add_argument("--full-annotations", type=str, default="CutLER Implementation/TFSS-CutLER/datasets/coco/annotations/instances_val2017.json")
    parser.add_argument("--class-level", action="store_true", help="Score each (image, category) against the union of all its instances (INSID3-style semantic GT) instead of per instance")
    parser.add_argument("--mask-interp", type=str, default="nearest", choices=["nearest", "bilinear"], help="Patch-grid <-> pixel mask resampling. 'nearest' (default) matches all prior runs in this project: a single point-sample when downsampling GT, hard patch-block upsampling for predictions. 'bilinear' mirrors INSID3's own mask handling (bilinear+threshold with a nearest/centroid fallback on the downsample side) -- see the patch-discretization ablation.")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV/failures files instead of overwriting (used for batched runs)")
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
