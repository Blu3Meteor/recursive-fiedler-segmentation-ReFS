#!/usr/bin/env python3

import argparse
import copy
import json
import os
import sys
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
THIRD_PARTY_ROOT = os.path.join(REPO_ROOT, "third_party")
if THIRD_PARTY_ROOT not in sys.path:
    sys.path.insert(0, THIRD_PARTY_ROOT)

from fiedler_class_transfer import (  # noqa: E402
    build_backbone,
    colorize_heatmap,
    get_affinity_matrix,
    largest_connected_component,
    make_transform,
    nth_eigenvector,
    overlay_mask,
    pca_component_vector,
    resize_square,
    second_smallest_eigenvector,
)


def infer_feature_grid(features):
    if features.dim() == 4:
        _, channels, height, width = features.shape
        return features.reshape(channels, height * width), height, width
    if features.dim() == 3:
        channels, height, width = features.shape
        return features.reshape(channels, height * width), height, width
    if features.dim() == 2:
        channels, num_patches = features.shape
        grid = int(round(num_patches ** 0.5))
        if grid * grid != num_patches:
            raise RuntimeError(f"Cannot infer square patch grid from feature shape: {tuple(features.shape)}")
        return features.reshape(channels, num_patches), grid, grid
    raise RuntimeError(f"Unexpected feature shape: {tuple(features.shape)}")


def load_image_features(image_path, backbone, fixed_size, cpu, resize_metadata=None, resize_mode="letterbox"):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    # Use aspect-ratio preservation (or stretch, per resize_mode) if metadata
    # provided, otherwise plain square resize.
    if resize_metadata is not None:
        try:
            if resize_mode == "stretch":
                from fiedler_class_transfer import resize_stretch
                image, _ = resize_stretch(image, fixed_size)
            else:
                from fiedler_class_transfer import resize_with_aspect_ratio
                image, _ = resize_with_aspect_ratio(image, fixed_size)
        except Exception:
            image = resize_square(image, fixed_size)
    else:
        image = resize_square(image, fixed_size)

    image_np = np.asarray(image)
    tensor = make_transform()(image).unsqueeze(0)
    if not cpu:
        tensor = tensor.cuda()
        # Only use half precision for DINO v1, not DINOv3
        backbone_class_name = backbone.__class__.__name__
        if backbone_class_name == "ViTFeat":  # DINO v1
            tensor = tensor.half()

    with torch.no_grad():
        output = backbone(tensor)

    # Handle both DINO v1 (returns features) and DINOv3 (returns (features, grid_info))
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], tuple):
        features, (grid_h, grid_w) = output
        # DINOv3 returns (num_patches, feat_dim), need to transpose to (feat_dim, num_patches)
        if features.shape[0] == grid_h * grid_w:
            features = features.T
    else:
        features = output[0] if isinstance(output, (list, tuple)) else output
        features, grid_h, grid_w = infer_feature_grid(features)

    if not cpu:
        features = features.float().cpu()

    return image_np, features, grid_h, grid_w


def orient_region_vector(vector):
    import numpy as np

    vector = np.asarray(vector, dtype=np.float32)
    if vector.size == 0:
        return vector
    seed = int(np.argmax(np.abs(vector)))
    if vector[seed] < 0:
        vector = -vector
    return vector


def component_for_split(split_values, region_mask, region_indices):
    import numpy as np

    threshold = float(np.mean(split_values))
    raw_mask = np.zeros(region_mask.size, dtype=bool)
    raw_mask[region_indices] = split_values > threshold
    component = largest_connected_component(raw_mask.reshape(region_mask.shape)).astype(bool)
    component = np.logical_and(component, region_mask)
    complement = np.logical_and(region_mask, np.logical_not(component))
    return component, complement


def region_noise_score(region_values, region_mask):
    import numpy as np

    values = np.zeros(region_mask.shape, dtype=np.float32)
    values[region_mask] = region_values.astype(np.float32)

    up = np.zeros_like(values)
    down = np.zeros_like(values)
    left = np.zeros_like(values)
    right = np.zeros_like(values)
    up[1:, :] = values[:-1, :]
    down[:-1, :] = values[1:, :]
    left[:, 1:] = values[:, :-1]
    right[:, :-1] = values[:, 1:]

    gradient_y = down - up
    gradient_x = right - left
    laplacian = up + down + left + right - 4.0 * values

    gradient_energy = float(np.mean((gradient_x[region_mask] ** 2) + (gradient_y[region_mask] ** 2)))
    laplacian_energy = float(np.mean(laplacian[region_mask] ** 2))
    if gradient_energy <= 1e-12:
        return 0.0
    return laplacian_energy / gradient_energy


def solve_region(features, grid_h, grid_w, region_mask, tau, eig_index=1, split_method="laplacian", pca_index=0):
    import numpy as np

    flat_region = region_mask.reshape(-1).astype(bool)
    region_indices = np.flatnonzero(flat_region)
    if region_indices.size < 3:
        return None

    region_features = features[:, region_indices]

    if split_method == "pca":
        # PCA-vs-Laplacian ablation: split on a principal component of the raw
        # patch features instead of a graph-Laplacian eigenvector. No affinity
        # graph is built at all in this branch.
        second_smallest_vec = pca_component_vector(region_features, pca_index=pca_index)
    else:
        affinity, degree_matrix = get_affinity_matrix(region_features, tau)
        if eig_index == 1:
            _, second_smallest_vec = second_smallest_eigenvector(affinity, degree_matrix)
        else:
            _, second_smallest_vec = nth_eigenvector(affinity, degree_matrix, eig_index=eig_index)

    # Handle ARPACK convergence failure (or a region too small to have eig_index+1
    # eigenvectors / pca_index+1 components): return None to make this region a leaf
    if second_smallest_vec is None:
        return None

    oriented_vec = orient_region_vector(second_smallest_vec)

    region_values = np.zeros(grid_h * grid_w, dtype=np.float32)
    region_values[region_indices] = oriented_vec
    component, complement = component_for_split(oriented_vec, region_mask, region_indices)
    noise_score = region_noise_score(oriented_vec, region_mask)

    return {
        "region_indices": region_indices,
        "region_values": region_values.reshape(grid_h, grid_w),
        "component": component,
        "complement": complement,
        "noise_score": noise_score,
        "split_area_ratio": float(component.sum() / max(1, region_mask.sum())),
    }


def upsample_mask(mask, fixed_size, mode="nearest"):
    """Upsample a patch-grid boolean mask to fixed_size x fixed_size.

    mode="nearest" (default, unchanged): each output pixel is one of the source
    patches' values verbatim -- boundaries are hard-snapped to the patch grid,
    and a mask can never represent anything finer than one whole patch block.

    mode="bilinear": mirrors INSID3's utils/refinement.py upsample_mask --
    bilinearly interpolate the patch grid as a continuous field, then threshold
    at 0.5. Boundaries are no longer forced onto patch-grid lines (the 0.5
    contour can fall anywhere between patch centers), which matters for
    sub-patch/thin objects -- see the patch-discretization discussion in this
    project's experiment log.
    """
    import numpy as np

    if mode == "bilinear":
        import torch
        import torch.nn.functional as F

        mask_t = torch.from_numpy(mask.astype(np.float32))
        up = F.interpolate(mask_t[None, None], size=(fixed_size, fixed_size), mode="bilinear", align_corners=False)
        return (up[0, 0] > 0.5).numpy()

    from PIL import Image

    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((fixed_size, fixed_size), Image.NEAREST)) > 0


def save_node_visual(image, node, output_path, fixed_size):
    import numpy as np
    from PIL import Image, ImageDraw

    region_mask = upsample_mask(node["region_mask"], fixed_size)
    split_mask = upsample_mask(node["split_mask"], fixed_size) if node.get("split_mask") is not None else region_mask
    heatmap = colorize_heatmap(node["fiedler_values"])
    heatmap = np.asarray(Image.fromarray(heatmap).resize((fixed_size, fixed_size), Image.NEAREST))

    panels = [
        (image, "image"),
        (overlay_mask(image, region_mask, color=(255, 180, 0), alpha=0.45), "active region"),
        (heatmap, "fiedler field"),
        (overlay_mask(image, split_mask, color=(0, 220, 80), alpha=0.50), "accepted split"),
    ]
    pil_panels = [Image.fromarray(panel) for panel, _ in panels]
    label_height = 30
    canvas = Image.new("RGB", (sum(panel.width for panel in pil_panels), fixed_size + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for (panel, label), panel_image in zip(panels, pil_panels):
        canvas.paste(panel_image, (x, label_height))
        draw.text((x + 5, 7), label, fill=(0, 0, 0))
        x += panel_image.width
    canvas.save(output_path)


def safe_name(value):
    keep = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "image"


def summary_uses_class_agnostic_source(summary):
    source = summary.get("source", {})
    coco_json = str(source.get("coco_json", "")).lower()
    return "cls_agnostic" in coco_json or "class_agnostic" in coco_json


def collect_recursive_candidates(args, backbone=None):
    import numpy as np

    if backbone is None:
        dino_version = getattr(args, "dino_version", "v1")
        debiased = getattr(args, "debiased", False)
        backbone = build_backbone(
            args.vit_arch,
            args.vit_feat,
            args.patch_size,
            args.pretrain_path,
            dino_version=dino_version,
            debiased=debiased,
        )
        if not args.cpu and dino_version == "v1":
            backbone.half()
            backbone.cuda()

    resize_metadata = getattr(args, "resize_metadata", None)
    image, features, grid_h, grid_w = load_image_features(
        args.image_path, backbone, args.fixed_size, args.cpu, resize_metadata,
        resize_mode=getattr(args, "resize_mode", "letterbox"),
    )
    root_mask = np.ones((grid_h, grid_w), dtype=bool)
    queue = deque([(0, root_mask, 0, "root")])
    nodes = []
    leaves = []
    next_id = 1

    max_nodes_limit = float('inf') if getattr(args, "unlimited_nodes", False) else args.max_nodes
    while queue and len(nodes) < max_nodes_limit:
        node_id, region_mask, depth, parent = queue.popleft()
        region_area = int(region_mask.sum())
        node = {
            "id": node_id,
            "parent": parent,
            "depth": depth,
            "region_area_patches": region_area,
            "region_area_ratio": float(region_area / region_mask.size),
            "region_mask": region_mask,
            "split_mask": None,
            "fiedler_values": np.zeros((grid_h, grid_w), dtype=np.float32),
        }

        if not getattr(args, "unlimited_depth", False) and depth >= args.max_depth:
            node["status"] = "leaf"
            node["stop_reason"] = "max_depth"
            leaves.append(node)
            nodes.append(node)
            continue

        if region_area < args.min_region_patches:
            node["status"] = "leaf"
            node["stop_reason"] = "small_region"
            leaves.append(node)
            nodes.append(node)
            continue

        result = solve_region(
            features, grid_h, grid_w, region_mask, args.tau,
            eig_index=getattr(args, "eig_index", 1),
            split_method=getattr(args, "split_method", "laplacian"),
            pca_index=getattr(args, "pca_index", 0),
        )
        if result is None:
            node["status"] = "leaf"
            node["stop_reason"] = "unsolved_region"
            leaves.append(node)
            nodes.append(node)
            continue

        split_area_ratio = result["split_area_ratio"]
        child_a = result["component"]
        child_b = result["complement"]
        min_child_area = min(int(child_a.sum()), int(child_b.sum()))
        node.update(
            {
                "fiedler_values": result["region_values"],
                "split_mask": child_a,
                "noise_score": result["noise_score"],
                "split_area_ratio": split_area_ratio,
                "min_child_area_patches": min_child_area,
            }
        )

        too_noisy = False if getattr(args, "unlimited_noise", False) else result["noise_score"] >= args.noise_threshold
        imbalanced = False if getattr(args, "unlimited_imbalance", False) else split_area_ratio <= args.min_split_ratio or split_area_ratio >= 1.0 - args.min_split_ratio
        child_too_small = min_child_area < args.min_region_patches

        if too_noisy or imbalanced or child_too_small:
            node["status"] = "leaf"
            if too_noisy:
                node["stop_reason"] = "noisy_divergence"
            elif imbalanced:
                node["stop_reason"] = "imbalanced_split"
            else:
                node["stop_reason"] = "small_child"
            leaves.append(node)
        else:
            node["status"] = "split"
            queue.append((next_id, child_a, depth + 1, node_id))
            next_id += 1
            queue.append((next_id, child_b, depth + 1, node_id))
            next_id += 1

        nodes.append(node)

    return {
        "image": image,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "nodes": nodes,
        "leaves": leaves,
    }


def collect_images_from_summary(summary_json, max_images=None, allow_class_agnostic_labels=False):
    with open(summary_json, "r", encoding="utf-8") as file_handle:
        summary = json.load(file_handle)

    if summary_uses_class_agnostic_source(summary) and not allow_class_agnostic_labels:
        raise ValueError(
            "The source summary was generated from a class-agnostic annotation file. "
            "Its class folders are not reliable for class-specific experiments. "
            "Regenerate the Fiedler report with a class-specific annotation file such as "
            "datasets/coco/annotations/instances_val2017.json, or pass "
            "--allow-class-agnostic-labels if you intentionally want the old class-agnostic grouping."
        )

    samples = []
    for class_summary in summary.get("classes", []):
        class_name = class_summary.get("class_name", "unknown")
        for image_record in class_summary.get("images", []):
            if image_record.get("status") != "ok" or not image_record.get("image_path"):
                continue
            samples.append(
                {
                    "class_name": class_name,
                    "sample_id": image_record.get("sample_id", os.path.splitext(os.path.basename(image_record["image_path"]))[0]),
                    "image_path": image_record["image_path"],
                }
            )
            if max_images is not None and len(samples) >= max_images:
                return samples
    return samples


def run_recursive_fiedler(args, backbone=None):
    import numpy as np
    from PIL import Image, ImageDraw

    os.makedirs(args.out_dir, exist_ok=True)

    if backbone is None:
        dino_version = getattr(args, "dino_version", "v1")
        debiased = getattr(args, "debiased", False)
        backbone = build_backbone(
            args.vit_arch,
            args.vit_feat,
            args.patch_size,
            args.pretrain_path,
            dino_version=dino_version,
            debiased=debiased,
        )
        if not args.cpu and dino_version == "v1":
            backbone.half()
            backbone.cuda()

    tree = collect_recursive_candidates(args, backbone=backbone)
    image = tree["image"]
    grid_h = tree["grid_h"]
    grid_w = tree["grid_w"]
    nodes = tree["nodes"]
    leaves = tree["leaves"]

    palette = [
        (235, 88, 73),
        (70, 146, 230),
        (76, 175, 105),
        (241, 176, 64),
        (153, 101, 221),
        (54, 184, 184),
        (230, 96, 156),
        (150, 150, 72),
    ]
    composite = image.copy()
    for index, leaf in enumerate(leaves):
        mask = upsample_mask(leaf["region_mask"], args.fixed_size)
        composite = overlay_mask(composite, mask, palette[index % len(palette)], alpha=0.42)

    composite_image = Image.fromarray(composite)
    draw = ImageDraw.Draw(composite_image)
    draw.rectangle((0, 0, args.fixed_size, 28), fill=(255, 255, 255))
    draw.text((5, 6), f"recursive leaves={len(leaves)} nodes={len(nodes)} noise_threshold={args.noise_threshold}", fill=(0, 0, 0))
    composite_path = os.path.join(args.out_dir, "recursive_fiedler_overlay.png")
    composite_image.save(composite_path)

    if args.save_nodes:
        node_dir = os.path.join(args.out_dir, "nodes")
        os.makedirs(node_dir, exist_ok=True)
        for node in nodes:
            save_node_visual(
                image,
                node,
                os.path.join(node_dir, f"node_{node['id']:03d}_{node['status']}.png"),
                args.fixed_size,
            )

    serializable_nodes = []
    for node in nodes:
        serializable_nodes.append(
            {
                key: value
                for key, value in node.items()
                if key not in {"region_mask", "split_mask", "fiedler_values"}
            }
        )

    summary = {
        "image_path": os.path.abspath(args.image_path),
        "fixed_size": args.fixed_size,
        "patch_size": args.patch_size,
        "tau": args.tau,
        "grid_shape": [grid_h, grid_w],
        "noise_threshold": args.noise_threshold,
        "min_region_patches": args.min_region_patches,
        "min_split_ratio": args.min_split_ratio,
        "max_depth": args.max_depth,
        "num_nodes": len(nodes),
        "num_leaves": len(leaves),
        "overlay_path": os.path.abspath(composite_path),
        "nodes": serializable_nodes,
    }
    summary_path = os.path.join(args.out_dir, "recursive_fiedler_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2)

    print(f"Saved recursive overlay to {composite_path}")
    print(f"Saved recursive summary to {summary_path}")
    for node in serializable_nodes:
        reason = node.get("stop_reason", "split")
        noise = node.get("noise_score")
        noise_text = "n/a" if noise is None else f"{noise:.3f}"
        print(
            f"node={node['id']} depth={node['depth']} status={node['status']} "
            f"reason={reason} area={node['region_area_patches']} noise={noise_text}"
        )
    return summary


def default_image_from_summary(summary_json, allow_class_agnostic_labels=False):
    samples = collect_images_from_summary(
        summary_json,
        max_images=1,
        allow_class_agnostic_labels=allow_class_agnostic_labels,
    )
    if samples:
        return samples[0]["image_path"]
    raise ValueError(f"No valid image_path found in {summary_json}")


def save_batch_gallery(batch_summary, output_dir, thumb_size=220):
    from PIL import Image, ImageDraw

    images = []
    for record in batch_summary["images"]:
        overlay_path = record.get("overlay_path")
        if not overlay_path or not os.path.exists(overlay_path):
            continue
        thumb = Image.open(overlay_path).convert("RGB")
        thumb.thumbnail((thumb_size, thumb_size))
        canvas = Image.new("RGB", (thumb_size, thumb_size + 42), (255, 255, 255))
        canvas.paste(thumb, ((thumb_size - thumb.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        title = f"{record['class_name']} {record['sample_id']}"
        detail = f"leaves={record['num_leaves']} noisy={record['noisy_leaf_count']}"
        draw.text((4, thumb_size + 5), title[:34], fill=(0, 0, 0))
        draw.text((4, thumb_size + 23), detail, fill=(0, 0, 0))
        images.append(canvas)

    if not images:
        return None

    cols = min(5, len(images))
    rows = (len(images) + cols - 1) // cols
    gallery = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + 42)), (255, 255, 255))
    for index, image in enumerate(images):
        x = (index % cols) * thumb_size
        y = (index // cols) * (thumb_size + 42)
        gallery.paste(image, (x, y))
    gallery_path = os.path.join(output_dir, "recursive_fiedler_gallery.png")
    gallery.save(gallery_path)
    return gallery_path


def run_recursive_batch(args):
    os.makedirs(args.out_dir, exist_ok=True)
    samples = collect_images_from_summary(
        args.from_summary,
        max_images=args.max_images,
        allow_class_agnostic_labels=args.allow_class_agnostic_labels,
    )
    if not samples:
        raise ValueError(f"No valid images found in {args.from_summary}")

    backbone = build_backbone(args.vit_arch, args.vit_feat, args.patch_size, args.pretrain_path)
    if not args.cpu:
        backbone.cuda()

    batch_summary = {
        "source_summary": os.path.abspath(args.from_summary),
        "out_dir": os.path.abspath(args.out_dir),
        "num_images": len(samples),
        "images": [],
    }

    for index, sample in enumerate(samples, start=1):
        sample_args = copy.copy(args)
        sample_args.image_path = sample["image_path"]
        sample_args.out_dir = os.path.join(
            args.out_dir,
            safe_name(sample["class_name"]),
            safe_name(sample["sample_id"]),
        )
        print(f"[{index}/{len(samples)}] {sample['class_name']} {sample['sample_id']}")
        summary = run_recursive_fiedler(sample_args, backbone=backbone)
        noisy_leaf_count = sum(
            1
            for node in summary["nodes"]
            if node.get("status") == "leaf" and node.get("stop_reason") == "noisy_divergence"
        )
        batch_summary["images"].append(
            {
                "class_name": sample["class_name"],
                "sample_id": sample["sample_id"],
                "image_path": os.path.abspath(sample["image_path"]),
                "out_dir": os.path.abspath(sample_args.out_dir),
                "overlay_path": summary["overlay_path"],
                "num_nodes": summary["num_nodes"],
                "num_leaves": summary["num_leaves"],
                "noisy_leaf_count": noisy_leaf_count,
            }
        )

    gallery_path = save_batch_gallery(batch_summary, args.out_dir)
    if gallery_path is not None:
        batch_summary["gallery_path"] = os.path.abspath(gallery_path)

    batch_summary_path = os.path.join(args.out_dir, "recursive_fiedler_batch_summary.json")
    with open(batch_summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(batch_summary, file_handle, indent=2)
    print(f"Saved batch summary to {batch_summary_path}")
    if gallery_path is not None:
        print(f"Saved batch gallery to {gallery_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Fiedler segmentation recursively on one image and stop when divergence becomes noisy."
    )
    parser.add_argument("--image-path", type=str, default=None, help="Image to segment recursively")
    parser.add_argument(
        "--from-summary",
        type=str,
        default=os.path.join("output", "fiedler_report_full", "fiedler_class_transfer_summary.json"),
        help="Use the first valid image from an existing Fiedler report when --image-path is omitted",
    )
    parser.add_argument("--out-dir", type=str, default=os.path.join("output", "recursive_fiedler"), help="Output directory")
    parser.add_argument("--vit-arch", type=str, default="small", choices=["base", "small"], help="Which architecture")
    parser.add_argument("--vit-feat", type=str, default="k", choices=["k", "q", "v", "kqv"], help="Which features")
    parser.add_argument("--patch-size", type=int, default=8, choices=[16, 8], help="Patch size")
    parser.add_argument("--tau", type=float, default=0.15, help="Threshold used for graph construction")
    parser.add_argument("--fixed-size", type=int, default=480, help="Resize image before feature extraction")
    parser.add_argument("--pretrain-path", type=str, default=None, help="Path to pretrained DINO weights")
    parser.add_argument("--cpu", action="store_true", help="Use CPU")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum recursive split depth")
    parser.add_argument("--max-nodes", type=int, default=63, help="Maximum number of solved nodes")
    parser.add_argument("--min-region-patches", type=int, default=36, help="Minimum active patch count to split")
    parser.add_argument("--min-split-ratio", type=float, default=0.05, help="Reject very imbalanced splits")
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=3.0,
        help="Stop when Laplacian/gradient energy of the Fiedler field reaches this value",
    )
    parser.add_argument("--save-nodes", action="store_true", help="Save per-node visual diagnostics")
    parser.add_argument("--batch", action="store_true", help="Run every valid image in --from-summary")
    parser.add_argument("--max-images", type=int, default=None, help="Limit batch image count")
    parser.add_argument(
        "--allow-class-agnostic-labels",
        action="store_true",
        help="Allow summaries made from class-agnostic annotations. Use only for class-agnostic analysis.",
    )
    args = parser.parse_args()

    if args.batch:
        run_recursive_batch(args)
        return

    if args.image_path is None:
        args.image_path = default_image_from_summary(
            args.from_summary,
            allow_class_agnostic_labels=args.allow_class_agnostic_labels,
        )
    run_recursive_fiedler(args)


if __name__ == "__main__":
    main()
