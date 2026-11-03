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

from recursive_fiedler import (  # noqa: E402
    build_backbone,
    load_image_features,
    overlay_mask,
    safe_name,
    solve_region,
    upsample_mask,
)


PERSON_PARTS = ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]


def load_manifest(path, class_name="person", max_images=None):
    with open(path, "r", encoding="utf-8") as file_handle:
        manifest = json.load(file_handle)

    samples = []
    for record in manifest.get("images", []):
        if record.get("class_name", class_name) != class_name:
            continue
        samples.append(
            {
                "class_name": class_name,
                "sample_id": record.get("sample_id", os.path.splitext(os.path.basename(record["image_path"]))[0]),
                "image_path": record["image_path"],
                "image_id": record.get("image_id"),
                "class_id": record.get("class_id"),
            }
        )
        if max_images is not None and len(samples) >= max_images:
            break
    return samples


def run_recursive_candidates(args, backbone):
    import numpy as np

    image, features, grid_h, grid_w = load_image_features(args.image_path, backbone, args.fixed_size, args.cpu)
    root_mask = np.ones((grid_h, grid_w), dtype=bool)
    queue = deque([(0, root_mask, 0, "root")])
    nodes = []
    leaves = []
    next_id = 1

    while queue and len(nodes) < args.max_nodes:
        node_id, region_mask, depth, parent = queue.popleft()
        region_area = int(region_mask.sum())
        node = {
            "id": node_id,
            "parent": parent,
            "depth": depth,
            "region_area_patches": region_area,
            "region_area_ratio": float(region_area / region_mask.size),
            "region_mask": region_mask,
        }

        if depth >= args.max_depth:
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

        result = solve_region(features, grid_h, grid_w, region_mask, args.tau)
        if result is None:
            node["status"] = "leaf"
            node["stop_reason"] = "unsolved_region"
            leaves.append(node)
            nodes.append(node)
            continue

        child_a = result["component"]
        child_b = result["complement"]
        split_area_ratio = result["split_area_ratio"]
        min_child_area = min(int(child_a.sum()), int(child_b.sum()))
        node.update(
            {
                "noise_score": result["noise_score"],
                "split_area_ratio": split_area_ratio,
                "min_child_area_patches": min_child_area,
            }
        )

        too_noisy = result["noise_score"] >= args.noise_threshold
        imbalanced = split_area_ratio <= args.min_split_ratio or split_area_ratio >= 1.0 - args.min_split_ratio
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

    return image, nodes, leaves, grid_h, grid_w


def leaf_geometry(leaf, grid_h, grid_w):
    import numpy as np

    ys, xs = np.where(leaf["region_mask"])
    area = int(xs.size)
    if area == 0:
        return None
    x0 = float(xs.min()) / grid_w
    x1 = float(xs.max() + 1) / grid_w
    y0 = float(ys.min()) / grid_h
    y1 = float(ys.max() + 1) / grid_h
    cx = float(xs.mean() + 0.5) / grid_w
    cy = float(ys.mean() + 0.5) / grid_h
    return {
        "leaf_id": leaf["id"],
        "area_patches": area,
        "area_ratio": float(area / (grid_h * grid_w)),
        "bbox": [x0, y0, x1, y1],
        "centroid": [cx, cy],
        "width": x1 - x0,
        "height": y1 - y0,
    }


def part_score(part, geom):
    cx, cy = geom["centroid"]
    area = geom["area_ratio"]
    width = geom["width"]
    height = geom["height"]
    aspect = width / max(height, 1e-6)

    if part == "head":
        return abs(cx - 0.50) * 1.1 + abs(cy - 0.16) * 1.8 + abs(area - 0.06) * 2.0
    if part == "torso":
        return abs(cx - 0.50) * 1.0 + abs(cy - 0.43) * 1.3 + abs(area - 0.22) * 0.8
    if part == "left_arm":
        return abs(cx - 0.25) * 1.4 + abs(cy - 0.44) * 1.0 + abs(aspect - 0.45) * 0.2
    if part == "right_arm":
        return abs(cx - 0.75) * 1.4 + abs(cy - 0.44) * 1.0 + abs(aspect - 0.45) * 0.2
    if part == "left_leg":
        return abs(cx - 0.40) * 1.0 + abs(cy - 0.76) * 1.7 + abs(aspect - 0.40) * 0.2
    if part == "right_leg":
        return abs(cx - 0.60) * 1.0 + abs(cy - 0.76) * 1.7 + abs(aspect - 0.40) * 0.2
    raise ValueError(f"Unknown part: {part}")


def acceptable_part(part, geom):
    cx, cy = geom["centroid"]
    area = geom["area_ratio"]
    if area < 0.01:
        return False
    if part == "head":
        return cy <= 0.38 and 0.02 <= area <= 0.22
    if part == "torso":
        return 0.18 <= cy <= 0.72 and area >= 0.08
    if part in {"left_arm", "right_arm"}:
        return 0.16 <= cy <= 0.72 and area <= 0.30
    if part in {"left_leg", "right_leg"}:
        return cy >= 0.45 and area <= 0.35
    return False


def assign_person_parts(leaves, grid_h, grid_w):
    geometries = []
    by_leaf_id = {}
    for leaf in leaves:
        geom = leaf_geometry(leaf, grid_h, grid_w)
        if geom is None:
            continue
        geometries.append(geom)
        by_leaf_id[leaf["id"]] = leaf

    assignments = {}
    used_leaf_ids = set()
    for part in PERSON_PARTS:
        candidates = [
            (part_score(part, geom), geom)
            for geom in geometries
            if geom["leaf_id"] not in used_leaf_ids and acceptable_part(part, geom)
        ]
        if not candidates:
            assignments[part] = {"status": "missing_or_uncertain", "leaf_id": None, "score": None}
            continue
        score, geom = min(candidates, key=lambda item: item[0])
        max_score = 0.55 if part in {"head", "torso"} else 0.65
        if score > max_score:
            assignments[part] = {"status": "missing_or_uncertain", "leaf_id": None, "score": float(score)}
            continue
        used_leaf_ids.add(geom["leaf_id"])
        assignments[part] = {
            "status": "assigned",
            "leaf_id": geom["leaf_id"],
            "score": float(score),
            "geometry": geom,
            "area_patches": by_leaf_id[geom["leaf_id"]]["region_area_patches"],
        }
    return assignments, by_leaf_id, geometries


def save_part_overlay(image, assignments, leaves_by_id, output_path, fixed_size):
    from PIL import Image, ImageDraw

    colors = {
        "head": (240, 86, 72),
        "torso": (68, 145, 220),
        "left_arm": (63, 170, 106),
        "right_arm": (75, 190, 185),
        "left_leg": (238, 174, 62),
        "right_leg": (150, 100, 220),
    }
    composite = image.copy()
    for part in PERSON_PARTS:
        assignment = assignments[part]
        leaf_id = assignment.get("leaf_id")
        if leaf_id is None:
            continue
        mask = upsample_mask(leaves_by_id[leaf_id]["region_mask"], fixed_size)
        composite = overlay_mask(composite, mask, colors[part], alpha=0.48)

    canvas = Image.fromarray(composite)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, fixed_size, 28), fill=(255, 255, 255))
    assigned = sum(1 for item in assignments.values() if item["status"] == "assigned")
    draw.text((5, 6), f"logic-guided person parts assigned={assigned}/6", fill=(0, 0, 0))
    canvas.save(output_path)


def run_logic_guided(args, backbone=None):
    os.makedirs(args.out_dir, exist_ok=True)
    if backbone is None:
        backbone = build_backbone(args.vit_arch, args.vit_feat, args.patch_size, args.pretrain_path)
        if not args.cpu:
            backbone.cuda()

    image, nodes, leaves, grid_h, grid_w = run_recursive_candidates(args, backbone)
    assignments, leaves_by_id, geometries = assign_person_parts(leaves, grid_h, grid_w)
    overlay_path = os.path.join(args.out_dir, "logic_guided_overlay.png")
    save_part_overlay(image, assignments, leaves_by_id, overlay_path, args.fixed_size)

    serializable_nodes = [
        {key: value for key, value in node.items() if key != "region_mask"}
        for node in nodes
    ]
    summary = {
        "image_path": os.path.abspath(args.image_path),
        "class_name": "person",
        "expected_parts": PERSON_PARTS,
        "fixed_size": args.fixed_size,
        "patch_size": args.patch_size,
        "tau": args.tau,
        "grid_shape": [grid_h, grid_w],
        "num_nodes": len(nodes),
        "num_leaves": len(leaves),
        "num_assigned_parts": sum(1 for item in assignments.values() if item["status"] == "assigned"),
        "overlay_path": os.path.abspath(overlay_path),
        "assignments": assignments,
        "leaf_geometries": geometries,
        "nodes": serializable_nodes,
    }
    summary_path = os.path.join(args.out_dir, "logic_guided_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2)
    print(f"Saved logic-guided overlay to {overlay_path}")
    print(f"Saved logic-guided summary to {summary_path}")
    return summary


def save_batch_summary(batch_summary, output_dir):
    summary_path = os.path.join(output_dir, "logic_guided_batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(batch_summary, file_handle, indent=2)
    return summary_path


def run_batch(args):
    samples = load_manifest(args.from_manifest, max_images=args.max_images)
    if not samples:
        raise ValueError(f"No person samples found in {args.from_manifest}")

    os.makedirs(args.out_dir, exist_ok=True)
    backbone = build_backbone(args.vit_arch, args.vit_feat, args.patch_size, args.pretrain_path)
    if not args.cpu:
        backbone.cuda()

    batch_summary = {
        "source_manifest": os.path.abspath(args.from_manifest),
        "out_dir": os.path.abspath(args.out_dir),
        "class_name": "person",
        "num_images": len(samples),
        "images": [],
    }
    for index, sample in enumerate(samples, start=1):
        sample_args = copy.copy(args)
        sample_args.image_path = sample["image_path"]
        sample_args.out_dir = os.path.join(args.out_dir, safe_name(sample["class_name"]), safe_name(sample["sample_id"]))
        print(f"[{index}/{len(samples)}] {sample['sample_id']}")
        summary = run_logic_guided(sample_args, backbone=backbone)
        batch_summary["images"].append(
            {
                "sample_id": sample["sample_id"],
                "image_path": os.path.abspath(sample["image_path"]),
                "out_dir": os.path.abspath(sample_args.out_dir),
                "overlay_path": summary["overlay_path"],
                "num_leaves": summary["num_leaves"],
                "num_assigned_parts": summary["num_assigned_parts"],
                "missing_parts": [
                    part
                    for part, assignment in summary["assignments"].items()
                    if assignment["status"] != "assigned"
                ],
            }
        )
    summary_path = save_batch_summary(batch_summary, args.out_dir)
    print(f"Saved batch summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Apply person-part logic to recursive Fiedler leaf candidates.")
    parser.add_argument("--image-path", type=str, default=None, help="Single image to segment")
    parser.add_argument(
        "--from-manifest",
        type=str,
        default=os.path.join("output", "person_instances_manifest", "person_instances_manifest.json"),
        help="Clean person manifest generated from instances_val2017.json",
    )
    parser.add_argument("--out-dir", type=str, default=os.path.join("output", "logic_guided_person"), help="Output directory")
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
    parser.add_argument("--noise-threshold", type=float, default=3.0, help="Reject noisy recursive splits")
    parser.add_argument("--batch", action="store_true", help="Run every valid image in --from-manifest")
    parser.add_argument("--max-images", type=int, default=None, help="Limit batch image count")
    args = parser.parse_args()

    if args.batch:
        run_batch(args)
        return

    if args.image_path is None:
        samples = load_manifest(args.from_manifest, max_images=1)
        if not samples:
            raise ValueError(f"No person samples found in {args.from_manifest}")
        args.image_path = samples[0]["image_path"]
    run_logic_guided(args)


if __name__ == "__main__":
    main()
