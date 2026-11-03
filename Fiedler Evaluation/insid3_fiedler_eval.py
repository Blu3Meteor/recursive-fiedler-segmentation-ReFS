#!/usr/bin/env python3
"""INSID3-style evaluation with recursive Fiedler candidate generation."""

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CUTLER_ROOT = os.path.join(WORKSPACE_ROOT, "CutLER Implementation", "TFSS-CutLER")
MASKCUT_ROOT = os.path.join(CUTLER_ROOT, "maskcut")

if CUTLER_ROOT not in sys.path:
    sys.path.insert(0, CUTLER_ROOT)
if MASKCUT_ROOT not in sys.path:
    sys.path.insert(0, MASKCUT_ROOT)

from fiedler_class_transfer import overlay_mask, resize_with_aspect_ratio  # noqa: E402
from recursive_fiedler import collect_recursive_candidates, safe_name, upsample_mask  # noqa: E402


def mask_iou(mask_a, mask_b):
    """Compute Intersection over Union between two binary masks."""
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection) / float(union)


def build_backbone(vit_arch, vit_feat, patch_size, pretrain_path=None, device="cuda", debiased=False):
    """Build DINOv3 backbone with optional positional debiasing."""
    from dinov3_feat import DINOv3Feat

    # Determine model ID based on architecture
    if vit_arch == "small":
        if patch_size == 14:
            model_id = "dinov3_vits14_pretrain_lvd1689m"
        elif patch_size == 16:
            model_id = "dinov3_vits16plus_pretrain_lvd1689m"
        else:
            model_id = f"dinov3_vits{patch_size}_pretrain"
    else:
        if patch_size == 14:
            model_id = "dinov3_vitb14_pretrain_lvd1689m"
        else:
            model_id = f"dinov3_vitb{patch_size}_pretrain"

    # If pretrain_path provided, use it directly
    if pretrain_path:
        model_id = pretrain_path

    feat_extractor = DINOv3Feat(model_id=model_id, feat_type=vit_feat, device=device)
    feat_extractor.debiased = debiased

    print(f"Loaded DINOv3 backbone: {vit_arch} patch_size={patch_size}")
    if debiased:
        print("  Positional debiasing: ENABLED")
    else:
        print("  Positional debiasing: disabled")

    return feat_extractor


def prepare_args(namespace, image_path, resize_metadata=None):
    """Prepare image-specific arguments."""
    image_args = argparse.Namespace(**vars(namespace))
    image_args.image_path = image_path
    image_args.resize_metadata = resize_metadata
    # Ensure cpu flag exists (convert from device setting)
    if not hasattr(image_args, 'cpu'):
        image_args.cpu = (getattr(image_args, 'device', 'cuda') == 'cpu')
    return image_args


def candidate_masks_from_tree(tree, fixed_size, metadata=None):
    """Extract candidate masks from tree, upsampling to fixed_size."""
    return [upsample_mask(node["region_mask"], fixed_size) for node in tree["nodes"]]


def mask_to_patch_grid(mask_2d, grid_h, grid_w):
    """
    Resample a 2D spatial mask to patch grid coordinates.

    Args:
        mask_2d: 2D boolean array (H, W) in image space
        grid_h: Number of patches in height dimension
        grid_w: Number of patches in width dimension

    Returns:
        mask_1d: 1D boolean array (grid_h * grid_w,) where each element corresponds to a patch
    """
    if mask_2d.ndim == 1:
        # Already in patch-grid format, return as-is
        return mask_2d.astype(bool)

    H, W = mask_2d.shape
    patch_h = H // grid_h
    patch_w = W // grid_w

    mask_patched = np.zeros((grid_h, grid_w), dtype=bool)
    for i in range(grid_h):
        for j in range(grid_w):
            # Check if any pixel in this patch region is True
            patch_mask = mask_2d[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            mask_patched[i, j] = patch_mask.any()

    return mask_patched.flatten()


def mask_from_patch_grid(mask_1d, grid_h, grid_w, output_h, output_w):
    """
    Convert a mask from patch grid coordinates back to 2D spatial coordinates.

    Args:
        mask_1d: 1D boolean array (grid_h * grid_w,)
        grid_h: Number of patches in height dimension
        grid_w: Number of patches in width dimension
        output_h: Desired output height
        output_w: Desired output width

    Returns:
        mask_2d: 2D boolean array (output_h, output_w) in image space
    """
    if mask_1d.ndim == 2:
        # Already in spatial format
        return mask_1d.astype(bool)

    mask_grid = mask_1d.reshape(grid_h, grid_w)
    patch_h = output_h // grid_h
    patch_w = output_w // grid_w

    mask_2d = np.zeros((output_h, output_w), dtype=bool)
    for i in range(grid_h):
        for j in range(grid_w):
            if mask_grid[i, j]:
                mask_2d[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = True

    return mask_2d


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

    # Downscale and pad mask to match resized image
    if scale != 1.0:
        downscaled = Image.fromarray(mask.astype(np.uint8) * 255)
        downscaled = downscaled.resize((resized_w, resized_h), Image.NEAREST)
        mask = np.asarray(downscaled) > 128
    else:
        mask = mask.copy()

    # Add padding
    padded = np.zeros((fixed_size, fixed_size), dtype=bool)
    padded[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = mask
    return padded


def backward_correspondences_filter(target_features, target_candidates, reference_features, reference_mask, tau=0.15):
    """
    Filter candidates using backward correspondences (INSID3 Section 3.3).

    A region C_k is kept if for most patches in C_k, their NN in target has
    its NN in reference falling within the reference mask.

    Args:
        target_features: (num_patches_target, feat_dim)
        target_candidates: List of candidate masks (bool arrays of patch grid)
        reference_features: (num_patches_ref, feat_dim)
        reference_mask: (num_patches_ref,) bool array or flattened array
        tau: Threshold for ratio of valid patches in candidate

    Returns:
        filtered_candidates: List of boolean arrays, same format as input
        backward_corr_scores: List of scores (0-1) for each candidate
    """
    if len(target_candidates) == 0:
        return [], []

    # Handle reference mask shape
    if isinstance(reference_mask, np.ndarray) and reference_mask.ndim > 1:
        reference_mask = reference_mask.flatten()

    num_target_patches = len(target_features)
    num_ref_patches = len(reference_features)

    # If shapes don't match, skip backward correspondences filtering
    if num_ref_patches != len(reference_mask):
        print(f"Warning: Reference feature/mask shape mismatch ({num_ref_patches} vs {len(reference_mask)})")
        # Return all candidates without filtering
        return target_candidates, [1.0] * len(target_candidates)

    # Normalize features for cosine similarity
    target_features = F.normalize(torch.tensor(target_features, dtype=torch.float32), dim=1)
    reference_features = F.normalize(torch.tensor(reference_features, dtype=torch.float32), dim=1)

    # Compute similarity: (num_target, num_ref)
    similarity = torch.mm(target_features, reference_features.t())

    # For each target patch, find its NN in reference
    target_to_ref_nn = torch.argmax(similarity, dim=1)  # (num_target,)

    # Check which reference patches are in mask
    ref_in_mask = torch.tensor(reference_mask, dtype=torch.bool)

    # For each candidate, count valid patches
    filtered_candidates = []
    backward_corr_scores = []

    for candidate_mask in target_candidates:
        # Flatten candidate mask to get patch indices
        candidate_patches = np.where(candidate_mask.flatten())[0]

        if len(candidate_patches) == 0:
            # Empty candidate, skip
            continue

        # Check if NN of each target patch in candidate falls within reference mask
        valid_count = 0
        for patch_idx in candidate_patches:
            if patch_idx < len(target_to_ref_nn):
                ref_nn = int(target_to_ref_nn[patch_idx].item())
                if ref_nn < len(ref_in_mask) and ref_in_mask[ref_nn]:
                    valid_count += 1

        # Compute score as ratio of valid patches
        score = valid_count / len(candidate_patches) if len(candidate_patches) > 0 else 0.0

        # Keep candidate if score exceeds threshold
        if score >= tau:
            filtered_candidates.append(candidate_mask)
            backward_corr_scores.append(score)

    return filtered_candidates, backward_corr_scores


def cross_image_similarity(target_features, target_mask, reference_features, reference_mask):
    """
    Compute cross-image similarity between target and reference clusters.

    Args:
        target_features: (num_patches_target, feat_dim)
        target_mask: (num_patches_target,) bool array for candidate region
        reference_features: (num_patches_ref, feat_dim)
        reference_mask: (num_patches_ref,) bool array for reference region

    Returns:
        cross_sim: Float in [0, 1] representing cross-image similarity
    """
    # Get features in foreground regions
    target_fg = target_features[target_mask]
    reference_fg = reference_features[reference_mask]

    if len(target_fg) == 0 or len(reference_fg) == 0:
        return 0.0

    # Compute prototypes as mean features
    target_proto = torch.tensor(target_fg, dtype=torch.float32).mean(dim=0)
    ref_proto = torch.tensor(reference_fg, dtype=torch.float32).mean(dim=0)

    # Normalize for cosine similarity
    target_proto = F.normalize(target_proto, dim=0)
    ref_proto = F.normalize(ref_proto, dim=0)

    # Cosine similarity (shift from [-1, 1] to [0, 1])
    sim = torch.dot(target_proto, ref_proto).item()
    return max(0.0, (sim + 1.0) / 2.0)


def self_image_similarity(target_features, target_mask, tau=0.15):
    """
    Compute self-image (intra-image) coherence of a candidate region.

    Measures how well-connected patches are within the candidate.

    Args:
        target_features: (num_patches_target, feat_dim)
        target_mask: (num_patches_target,) bool array for candidate region
        tau: Similarity threshold for connectivity

    Returns:
        self_sim: Float in [0, 1] representing self-image coherence
    """
    target_fg = target_features[target_mask]

    if len(target_fg) <= 1:
        return 1.0

    # Compute pairwise similarity within foreground
    target_fg_tensor = torch.tensor(target_fg, dtype=torch.float32)
    target_fg_tensor = F.normalize(target_fg_tensor, dim=1)

    similarity = torch.mm(target_fg_tensor, target_fg_tensor.t())

    # Count pairs with similarity >= tau
    num_high_sim_pairs = (similarity >= tau).sum().item() - len(target_fg)  # Exclude diagonal
    total_pairs = len(target_fg) * (len(target_fg) - 1) / 2.0

    if total_pairs == 0:
        return 1.0

    return min(1.0, num_high_sim_pairs / total_pairs)


def hybrid_scoring(target_features, target_candidates, reference_features, reference_mask, alpha=0.5):
    """
    Compute hybrid scores combining cross-image and self-image similarity (INSID3 Eq. 13).

    Args:
        target_features: (num_patches_target, feat_dim)
        target_candidates: List of candidate masks
        reference_features: (num_patches_ref, feat_dim)
        reference_mask: (num_patches_ref,) bool array for reference region
        alpha: Weight for cross-image similarity (1-alpha for self-image)

    Returns:
        hybrid_scores: List of scores for each candidate
    """
    hybrid_scores = []

    for candidate_mask in target_candidates:
        if not candidate_mask.any():
            hybrid_scores.append(0.0)
            continue

        # Flatten candidate mask to get patch indices
        candidate_patches = candidate_mask.flatten()

        s_cross = cross_image_similarity(target_features, candidate_patches, reference_features, reference_mask)
        s_intra = self_image_similarity(target_features, candidate_patches)

        # Eq. 13: S_k = s_cross * s_intra
        hybrid_score = s_cross * s_intra

        hybrid_scores.append(hybrid_score)

    return hybrid_scores


def load_reference_data(reference_json, image_root, device="cuda", feat_extractor=None, debiased=False):
    """
    Load reference images and extract features for each class.

    Returns:
        reference_data: Dict mapping class_name -> {features, mask, image_path}
                       where mask is in patch-grid coordinates matching features
    """
    coco_ref = COCO(reference_json)
    categories = {cat["id"]: cat["name"] for cat in coco_ref.loadCats(sorted(coco_ref.getCatIds()))}

    reference_data = {}

    for ann in coco_ref.dataset["annotations"]:
        cat_id = ann["category_id"]
        cat_name = categories.get(cat_id, f"class_{cat_id}")

        image_id = ann["image_id"]
        image_info = coco_ref.loadImgs([image_id])[0]
        image_path = os.path.join(image_root, image_info["file_name"])

        if not os.path.exists(image_path):
            print(f"Warning: Reference image not found: {image_path}")
            continue

        # Load image and annotation
        image_pil = Image.open(image_path).convert("RGB")
        image_resized, resize_metadata = resize_with_aspect_ratio(image_pil, fixed_size=480)
        gt_mask = coco_ref.annToMask(ann).astype(bool)
        gt_mask_resized = resize_binary_mask(gt_mask, 480, resize_metadata)

        # Extract features using feat_extractor
        features = None
        grid_h = grid_w = None
        if feat_extractor is not None:
            image_tensor = torch.tensor(np.array(image_resized), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            if device == "cuda":
                image_tensor = image_tensor.cuda()

            try:
                result = feat_extractor.extract_features(image_resized, debiased=debiased)
                # Handle both single tensor and tuple (features, grid) returns
                if isinstance(result, tuple):
                    features = result[0]
                    grid_h, grid_w = result[1]
                else:
                    features = result
                    # Infer grid from features shape (assuming square grid)
                    num_patches = len(features)
                    grid_size = int(np.sqrt(num_patches))
                    if grid_size * grid_size == num_patches:
                        grid_h = grid_w = grid_size
                    else:
                        # Can't infer grid, will handle below
                        grid_h = grid_w = None

                if features is not None:
                    features = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
                else:
                    print(f"Warning: Failed to extract features for reference {cat_name}")
                    continue
            except Exception as e:
                print(f"Warning: Feature extraction failed for reference {cat_name}: {e}")
                continue
        else:
            features = None

        # Convert reference mask to patch-grid coordinates
        if features is not None and grid_h is not None and grid_w is not None:
            # Resample mask from image space (480, 480) to patch grid (grid_h, grid_w)
            mask_patch_grid = mask_to_patch_grid(gt_mask_resized, grid_h, grid_w)
            print(f"  Reference mask resampled to patch grid: {gt_mask_resized.shape} -> ({grid_h}, {grid_w}) = {len(mask_patch_grid)} patches")
        else:
            # Fall back to flattened image-space mask (shouldn't happen with DINOv3)
            mask_patch_grid = gt_mask_resized.flatten()

        reference_data[cat_name] = {
            "features": features,
            "mask": mask_patch_grid,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "image_path": image_path,
            "image_id": image_id,
            "annotation_id": ann["id"],
        }

        print(f"Loaded reference for {cat_name}: {image_path} ({len(mask_patch_grid)} patches)")

    return reference_data


def evaluate_with_insid3(args):
    """Run full INSID3 evaluation with recursive Fiedler candidates."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Load reference data
    print("Loading reference data...")
    feat_extractor = build_backbone(args.vit_arch, args.vit_feat, args.patch_size, args.pretrain_path, device=args.device, debiased=args.debiased)

    reference_data = load_reference_data(args.reference_json, args.image_root, device=args.device, feat_extractor=feat_extractor, debiased=args.debiased)

    if not reference_data:
        raise ValueError("No reference data loaded")

    print(f"Loaded reference data for {len(reference_data)} classes")

    # Load test set
    print("Loading test set...")
    coco_test = COCO(args.test_json)
    categories = {cat["id"]: cat["name"] for cat in coco_test.loadCats(sorted(coco_test.getCatIds()))}

    records_by_image = defaultdict(list)
    image_ids = sorted(coco_test.getImgIds())

    for image_id in image_ids[: args.max_images] if args.max_images else image_ids:
        image_info = coco_test.loadImgs([image_id])[0]
        image_path = os.path.join(args.image_root, image_info["file_name"])

        ann_ids = coco_test.getAnnIds(imgIds=[image_id])
        annotations = coco_test.loadAnns(ann_ids)

        for annotation in annotations:
            if annotation.get("iscrowd", 0):
                continue

            segmentation = annotation.get("segmentation")
            if not segmentation:
                continue

            gt_mask = coco_test.annToMask(annotation).astype(bool)
            if not gt_mask.any():
                continue

            cat_id = annotation["category_id"]
            cat_name = categories.get(cat_id, f"category_{cat_id}")

            records_by_image[image_id].append(
                {
                    "image_id": image_id,
                    "annotation_id": annotation["id"],
                    "category_id": cat_id,
                    "category_name": cat_name,
                    "image_path": image_path,
                    "gt_mask": gt_mask,
                    "width": image_info["width"],
                    "height": image_info["height"],
                }
            )

    if not records_by_image:
        raise ValueError("No valid test instances found")

    print(f"Loaded {len(records_by_image)} test images with {sum(len(v) for v in records_by_image.values())} instances")

    # Evaluation loop
    all_results = []
    class_scores = defaultdict(list)
    per_class_stats = defaultdict(lambda: {"count": 0, "sum": 0.0, "scores": []})
    failures_log = os.path.join(args.output_dir, "failures.jsonl")
    failures_file = open(failures_log, "w", encoding="utf-8")

    results_csv = os.path.join(args.output_dir, "instance_results_insid3.csv")
    csv_fieldnames = [
        "image_id",
        "annotation_id",
        "category_id",
        "category_name",
        "best_iou",
        "best_candidate_id",
        "candidate_count_before_bc",
        "candidate_count_after_bc",
        "mean_hybrid_score",
        "image_path",
    ]
    csv_file = open(results_csv, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    total_images = len(records_by_image)
    elapsed_times = []
    start_time = time.time()

    for index, (image_id, records) in enumerate(records_by_image.items(), start=1):
        image_path = records[0]["image_path"]
        image_start = time.time()

        try:
            # Load and prepare image
            image_pil = Image.open(image_path).convert("RGB")
            image_resized, resize_metadata = resize_with_aspect_ratio(image_pil, args.fixed_size)

            # Generate candidates using recursive Fiedler
            tree_args = prepare_args(args, image_path, resize_metadata)
            tree = collect_recursive_candidates(tree_args, backbone=feat_extractor)

            candidate_masks = candidate_masks_from_tree(tree, args.fixed_size, resize_metadata)

            if not candidate_masks:
                continue

            # Extract features for this image
            image_tensor = torch.tensor(np.array(image_resized), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            if args.device == "cuda":
                image_tensor = image_tensor.cuda()

            result = feat_extractor.extract_features(image_resized, debiased=args.debiased)
            # Handle both single tensor and tuple (features, grid) returns
            if isinstance(result, tuple):
                target_features = result[0]
                target_grid_h, target_grid_w = result[1]
            else:
                target_features = result
                # Infer grid from features shape
                num_patches = len(target_features)
                grid_size = int(np.sqrt(num_patches))
                if grid_size * grid_size == num_patches:
                    target_grid_h = target_grid_w = grid_size
                else:
                    # Can't infer, assume square grid
                    target_grid_h = target_grid_w = int(np.sqrt(num_patches))

            if target_features is None:
                continue
            target_features = target_features.cpu().numpy() if isinstance(target_features, torch.Tensor) else target_features

            # Convert candidate masks to patch-grid coordinates
            candidate_masks_patch_grid = []
            for cand_mask_spatial in candidate_masks:
                cand_mask_patch = mask_to_patch_grid(cand_mask_spatial, target_grid_h, target_grid_w)
                candidate_masks_patch_grid.append(cand_mask_patch)

            # Process each instance in this image
            image_results = []
            for record in records:
                cat_name = record["category_name"]

                # Skip if no reference for this class
                if cat_name not in reference_data:
                    continue

                ref_data = reference_data[cat_name]
                ref_features = ref_data["features"]
                ref_mask = ref_data["mask"]

                if ref_features is None:
                    continue

                # Apply backward correspondences filtering
                filtered_candidates, bc_scores = backward_correspondences_filter(
                    target_features, candidate_masks_patch_grid, ref_features, ref_mask, tau=args.bc_threshold
                )

                if not filtered_candidates:
                    # No candidates passed backward correspondences
                    continue

                # Apply hybrid scoring
                hybrid_scores = hybrid_scoring(target_features, filtered_candidates, ref_features, ref_mask, alpha=args.alpha)

                # Find best candidate by IoU
                gt_mask_resized = resize_binary_mask(record["gt_mask"], args.fixed_size, resize_metadata)
                best_iou = -1.0
                best_candidate_idx = -1
                mean_hybrid_score = np.mean(hybrid_scores) if hybrid_scores else 0.0

                for cand_idx, cand_mask_patch in enumerate(filtered_candidates):
                    # Convert candidate from patch-grid back to image space for IoU computation
                    cand_mask_spatial = mask_from_patch_grid(cand_mask_patch, target_grid_h, target_grid_w, args.fixed_size, args.fixed_size)
                    iou = mask_iou(cand_mask_spatial, gt_mask_resized)
                    if iou > best_iou:
                        best_iou = iou
                        best_candidate_idx = cand_idx

                if best_iou >= 0:
                    result = {
                        "image_id": record["image_id"],
                        "annotation_id": record["annotation_id"],
                        "category_id": record["category_id"],
                        "category_name": cat_name,
                        "best_iou": float(best_iou),
                        "best_candidate_id": best_candidate_idx,
                        "candidate_count_before_bc": len(candidate_masks),
                        "candidate_count_after_bc": len(filtered_candidates),
                        "mean_hybrid_score": mean_hybrid_score,
                        "image_path": record["image_path"],
                    }
                    all_results.append(result)
                    image_results.append(result)
                    class_scores[cat_name].append(float(best_iou))
                    per_class_stats[cat_name]["count"] += 1
                    per_class_stats[cat_name]["sum"] += float(best_iou)
                    per_class_stats[cat_name]["scores"].append(float(best_iou))

                    csv_row = {key: result.get(key) for key in csv_fieldnames}
                    csv_writer.writerow(csv_row)

            # Progress logging
            elapsed = time.time() - image_start
            elapsed_times.append(elapsed)
            avg_elapsed = np.mean(elapsed_times)
            remaining = total_images - index
            eta_seconds = avg_elapsed * remaining
            eta_minutes = eta_seconds / 60.0

            print(
                f"[{index}/{total_images}] {os.path.basename(image_path)} ({len(records)} instances) | {elapsed:.2f}s | ETA {eta_minutes:.1f}m",
                flush=True,
            )

            # Memory cleanup
            del image_pil, image_resized, tree, candidate_masks, image_results, records
            try:
                torch.cuda.empty_cache()
            except:
                pass
            gc.collect()

        except Exception as e:
            elapsed = time.time() - image_start
            error_msg = f"Image {image_id} failed after {elapsed:.2f}s: {str(e)}"
            print(f"ERROR: {error_msg}", flush=True)
            traceback.print_exc()
            failures_file.write(json.dumps({"image_id": image_id, "image_path": image_path, "error": str(e), "elapsed": elapsed}) + "\n")
            failures_file.flush()

            try:
                torch.cuda.empty_cache()
            except:
                pass
            gc.collect()
            continue

    csv_file.close()
    failures_file.close()

    # Compute summary statistics
    if all_results:
        overall_scores = [r["best_iou"] for r in all_results]
        mean_best_iou = float(np.mean(overall_scores))
        median_best_iou = float(np.median(overall_scores))
    else:
        mean_best_iou = median_best_iou = 0.0
        overall_scores = []

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
            }

    total_elapsed = time.time() - start_time

    report = {
        "dataset": "COCO-20i",
        "num_test_images": len(records_by_image),
        "num_test_instances": len(all_results),
        "mean_best_iou": mean_best_iou,
        "median_best_iou": median_best_iou,
        "per_class": per_class_summary,
        "total_elapsed_seconds": total_elapsed,
        "num_reference_classes": len(reference_data),
        "method": "INSID3+Fiedler",
        "backbone": f"DINOv3-{args.vit_arch}",
        "debiased": args.debiased,
        "alpha": args.alpha,
        "bc_threshold": args.bc_threshold,
    }

    summary_json = os.path.join(args.output_dir, "insid3_summary.json")
    with open(summary_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"INSID3+Fiedler Evaluation Complete")
    print(f"{'='*70}")
    print(f"Test instances evaluated: {len(all_results)}")
    print(f"Mean best IoU: {mean_best_iou:.4f}")
    print(f"Median best IoU: {median_best_iou:.4f}")
    print(f"INSID3 baseline (57.6%): {'✓ EXCEEDED' if mean_best_iou > 0.576 else '✗ Below'}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - {os.path.basename(results_csv)}")
    print(f"  - {os.path.basename(summary_json)}")


def build_parser():
    parser = argparse.ArgumentParser(description="INSID3 evaluation with recursive Fiedler candidates")
    parser.add_argument("--reference-json", type=str, default="Fiedler Evaluation/data/coco20i/coco20i_reference.json", help="Reference set annotations")
    parser.add_argument("--test-json", type=str, default="Fiedler Evaluation/data/coco20i/coco20i_test.json", help="Test set annotations")
    parser.add_argument("--image-root", type=str, default="CutLER Implementation/TFSS-CutLER/datasets/coco/val2017", help="Image root directory")
    parser.add_argument("--output-dir", type=str, default="Fiedler Evaluation/output/02_oracle_proposals/COCO/insid3_fiedler_dinov3", help="Output directory")
    parser.add_argument("--vit-arch", type=str, default="small", choices=["small", "base"], help="ViT architecture")
    parser.add_argument("--vit-feat", type=str, default="k", choices=["k", "q", "v", "kqv"], help="Feature type")
    parser.add_argument("--patch-size", type=int, default=14, help="Patch size")
    parser.add_argument("--fixed-size", type=int, default=480, help="Fixed image size")
    parser.add_argument("--pretrain-path", type=str, default=None, help="Path to DINOv3 weights")
    parser.add_argument("--debiased", action="store_true", help="Use positional debiasing")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for cross-image similarity in hybrid scoring")
    parser.add_argument("--bc-threshold", type=float, default=0.5, help="Backward correspondences threshold")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to evaluate")
    # Fiedler hyperparameters
    parser.add_argument("--tau", type=float, default=0.15, help="Graph threshold")
    parser.add_argument("--max-depth", type=int, default=5, help="Max recursion depth")
    parser.add_argument("--max-nodes", type=int, default=63, help="Max nodes in tree")
    parser.add_argument("--min-region-patches", type=int, default=36, help="Min patches to split")
    parser.add_argument("--min-split-ratio", type=float, default=0.05, help="Min split balance")
    parser.add_argument("--noise-threshold", type=float, default=3.0, help="Noise threshold")

    return parser


def main():
    args = build_parser().parse_args()
    evaluate_with_insid3(args)


if __name__ == "__main__":
    main()
