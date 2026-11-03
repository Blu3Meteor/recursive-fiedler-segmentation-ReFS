#!/usr/bin/env python3
"""Test whether the recursive Fiedler tree isolates a multi-instance class as a
single group (e.g. "the cars") before any node goes on to separate its individual
instances -- i.e. whether the group's best-matching node is an ANCESTOR (shallower,
same branch) of each instance's own best-matching node, rather than the tree
splitting straight into individual instances or mixing the class with others.

For each image in the class_group_hierarchy dataset:
  1. Build the full, unlimited-depth Fiedler tree (same params as other
     experiments in this project: DINOv2-small, patch14, fixed_size=1022,
     min_region_patches=5 is the only real stopping criterion).
  2. For the designated multi-instance class, find (oracle, full-tree search):
       - the best-matching node for the GROUP mask (union of all its instances)
       - the best-matching node for EACH individual instance
  3. Check ancestry: walk each instance-node's parent chain up to the root and
     see whether the group-node's id appears in it (group-node is a strict
     ancestor), and compare depths.
  4. Visualize: original, GT (all instances colored), group-best-node overlay,
     one overlay per instance-best-node, with ancestry verdict printed on the
     group panel and in the console summary.

No IoU/GT masks are used for tree construction (this is the standard
class-agnostic Fiedler builder) -- GT is used only afterward, to search the
already-built tree, exactly as in visualize_class_vs_instance.py.
"""

import argparse
import gc
import json
import os
import sys
import time

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

from fiedler_class_transfer import build_backbone, overlay_mask, resize_stretch, resize_with_aspect_ratio  # noqa: E402
from recursive_fiedler import collect_recursive_candidates, load_image_features, upsample_mask  # noqa: E402

GROUP_COLOR = (255, 180, 0)
INSTANCE_PALETTE = [
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


def resize_binary_mask(mask, fixed_size, metadata):
    if fixed_size <= 0 or not metadata:
        return mask
    scale = metadata.get("scale", 1.0)
    pad_left = metadata.get("pad_left", 0)
    pad_top = metadata.get("pad_top", 0)
    resized_w = metadata.get("resized_w", int(metadata["orig_w"] * scale))
    resized_h = metadata.get("resized_h", int(metadata["orig_h"] * scale))
    if (resized_w, resized_h) != (mask.shape[1], mask.shape[0]):
        downscaled = Image.fromarray(mask.astype(np.uint8) * 255).resize((resized_w, resized_h), Image.NEAREST)
        mask = np.asarray(downscaled) > 128
    else:
        mask = mask.copy()
    padded = np.zeros((fixed_size, fixed_size), dtype=bool)
    padded[pad_top: pad_top + resized_h, pad_left: pad_left + resized_w] = mask
    return padded


def find_best_matching_node(nodes, gt_mask_grid):
    best_node, best_iou = None, -1.0
    for node in nodes:
        iou = mask_iou(node["region_mask"], gt_mask_grid)
        if iou > best_iou:
            best_iou = iou
            best_node = node
    return best_node, best_iou


def ancestor_chain(nodes_by_id, node):
    """List of node ids from `node` up to (and including) the root."""
    chain = [node["id"]]
    current = node
    while current["parent"] != "root":
        current = nodes_by_id[current["parent"]]
        chain.append(current["id"])
    return chain


def overlay_with_label(image_array, mask, color, label_lines):
    canvas = overlay_mask(image_array, mask, color=color, alpha=0.5)
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    label_h = 16 * len(label_lines) + 8
    draw.rectangle((0, 0, pil_canvas.width, label_h), fill=(255, 255, 255))
    for i, line in enumerate(label_lines):
        draw.text((6, 4 + 16 * i), line, fill=(0, 0, 0))
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
    row_h = max(p.size[1] for p in panels)
    canvas = Image.new("RGB", (display_w * len(panels), row_h + 32), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 32), fill=(240, 240, 240))
    draw.text((8, 8), title, fill=(0, 0, 0))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * display_w, 32))
    return canvas


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


def process_image(record, backbone, coco, categories, args):
    image_path = os.path.join(args.dataset_root, record["local_file_name"])
    image_id = record["image_id"]

    image_pil = Image.open(image_path).convert("RGB")
    if args.resize_mode == "stretch":
        image_resized, resize_metadata = resize_stretch(image_pil, args.fixed_size)
        # Panels are shown at the true aspect ratio (no bars, no squashing).
        display_size = (args.fixed_size, max(1, round(args.fixed_size * image_pil.height / image_pil.width)))
    else:
        image_resized, resize_metadata = resize_with_aspect_ratio(image_pil, args.fixed_size)
        display_size = (args.fixed_size, args.fixed_size)
    image_np = np.asarray(image_resized)
    image_disp = np.asarray(image_resized.resize(display_size, Image.LANCZOS)) if args.resize_mode == "stretch" else image_np

    def disp(mask):
        if args.resize_mode != "stretch":
            return mask
        return np.asarray(Image.fromarray(np.asarray(mask).astype(np.uint8) * 255).resize(display_size, Image.NEAREST)) > 127

    import argparse as _argparse
    tree_args = _argparse.Namespace(
        image_path=image_path, vit_arch=args.vit_arch, vit_feat=args.vit_feat,
        patch_size=args.patch_size, pretrain_path=None, cpu=args.cpu,
        dino_version=args.dino_version, debiased=False, fixed_size=args.fixed_size,
        tau=args.tau, max_depth=args.max_depth, max_nodes=args.max_nodes,
        min_region_patches=args.min_region_patches, min_split_ratio=args.min_split_ratio,
        noise_threshold=args.noise_threshold, unlimited_depth=True, unlimited_nodes=True,
        unlimited_imbalance=True, unlimited_noise=True, resize_metadata=resize_metadata,
        resize_mode=args.resize_mode,
    )
    tree = collect_recursive_candidates(tree_args, backbone=backbone)
    nodes_by_id = {n["id"]: n for n in tree["nodes"]}
    grid_h, grid_w = tree["grid_h"], tree["grid_w"]

    # All prominent non-crowd GT annotations (for the full "GT" panel).
    ann_ids = coco.getAnnIds(imgIds=[image_id])
    all_anns = [a for a in coco.loadAnns(ann_ids) if not a.get("iscrowd", 0)]
    image_area = record["width"] * record["height"]
    min_area = image_area * 0.01
    all_anns = [a for a in all_anns if a.get("area", 0) > min_area]

    group_cat_ids = set(record["group_category_ids"])
    group_anns = [a for a in all_anns if a["category_id"] in group_cat_ids]
    group_name = "+".join(record["group_category_names"])

    # Group GT mask = union of all its instances.
    group_mask_fixed = np.zeros((args.fixed_size, args.fixed_size), dtype=bool)
    for a in group_anns:
        m = resize_binary_mask(coco.annToMask(a).astype(bool), args.fixed_size, resize_metadata)
        group_mask_fixed |= m
    group_mask_grid = downsample_mask_to_grid(group_mask_fixed, grid_h, grid_w)
    group_node, group_iou = find_best_matching_node(tree["nodes"], group_mask_grid)
    group_chain = ancestor_chain(nodes_by_id, group_node) if group_node is not None else []

    # Per-instance best match, within the group class.
    instance_results = []
    for i, ann in enumerate(group_anns):
        inst_mask_fixed = resize_binary_mask(coco.annToMask(ann).astype(bool), args.fixed_size, resize_metadata)
        inst_mask_grid = downsample_mask_to_grid(inst_mask_fixed, grid_h, grid_w)
        inst_node, inst_iou = find_best_matching_node(tree["nodes"], inst_mask_grid)
        inst_chain = ancestor_chain(nodes_by_id, inst_node) if inst_node is not None else []
        is_descendant = group_node is not None and inst_node is not None and group_node["id"] in inst_chain and inst_node["id"] != group_node["id"]
        is_same_node = group_node is not None and inst_node is not None and inst_node["id"] == group_node["id"]
        instance_results.append({
            "ann_id": ann["id"], "iou": inst_iou,
            "node_id": inst_node["id"] if inst_node else None,
            "node_depth": inst_node["depth"] if inst_node else None,
            "is_descendant_of_group": bool(is_descendant),
            "is_same_as_group": bool(is_same_node),
        })

    n_descendant = sum(r["is_descendant_of_group"] for r in instance_results)
    n_same = sum(r["is_same_as_group"] for r in instance_results)
    n_total = len(instance_results)
    if n_total == 0:
        verdict = "no instances"
    elif n_descendant == n_total:
        verdict = "HIERARCHY HOLDS: group isolated first, then all instances split beneath it"
    elif n_descendant + n_same == n_total:
        verdict = "PARTIAL: group isolated, some instances never separated from it (leaf/not-split)"
    else:
        verdict = "HIERARCHY DOES NOT HOLD: some instance nodes are NOT descendants of the group node"

    # ---- Visualization ----
    all_masks_colors, all_labels = [], []
    for i, a in enumerate(all_anns):
        m = resize_binary_mask(coco.annToMask(a).astype(bool), args.fixed_size, resize_metadata)
        color = INSTANCE_PALETTE[i % len(INSTANCE_PALETTE)]
        all_masks_colors.append((disp(m), color))
        all_labels.append(categories.get(a["category_id"], str(a["category_id"])))
    gt_panel = overlay_multi_with_label(image_disp, all_masks_colors, "GT: " + ", ".join(all_labels))

    original_panel = Image.fromarray(image_disp)

    group_label = [
        f"GROUP node: {group_name}",
        f"id={group_node['id']} depth={group_node['depth']} IoU={group_iou:.3f}" if group_node else "no match",
    ]
    group_panel = overlay_with_label(image_disp, disp(upsample_mask(group_node["region_mask"], args.fixed_size)), GROUP_COLOR, group_label) if group_node else Image.fromarray(image_disp)

    instance_panels = []
    for i, (ann, res) in enumerate(zip(group_anns, instance_results)):
        node = nodes_by_id.get(res["node_id"]) if res["node_id"] is not None else None
        color = INSTANCE_PALETTE[i % len(INSTANCE_PALETTE)]
        rel = "descendant of group" if res["is_descendant_of_group"] else ("== group node" if res["is_same_as_group"] else "NOT under group")
        label = [
            f"{group_name} instance #{i+1}",
            f"id={res['node_id']} depth={res['node_depth']} IoU={res['iou']:.3f} [{rel}]",
        ]
        mask = disp(upsample_mask(node["region_mask"], args.fixed_size)) if node else np.zeros_like(image_disp[:, :, 0], dtype=bool)
        instance_panels.append(overlay_with_label(image_disp, mask, color, label))

    panels = [original_panel, gt_panel, group_panel] + instance_panels
    title = f"{record['file_name']} | image_id={image_id} | group={group_name} ({n_total} instances) | {verdict}"
    row = build_panel_row(panels, title)

    report = {
        "image_id": image_id, "file_name": record["file_name"], "group": group_name,
        "group_node_id": group_node["id"] if group_node else None,
        "group_node_depth": group_node["depth"] if group_node else None,
        "group_iou": group_iou, "group_chain": group_chain,
        "num_group_instances": n_total, "num_descendant": n_descendant, "num_same_as_group": n_same,
        "verdict": verdict, "instances": instance_results,
    }

    del image_pil, image_resized, tree
    return row, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="Fiedler Evaluation/data/class_group_hierarchy")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--coco-json", default="CutLER Implementation/TFSS-CutLER/datasets/coco/annotations/instances_val2017.json")
    parser.add_argument("--output-dir", default="Fiedler Evaluation/output/05_reference_vs_gt_and_semantics/class_group_hierarchy_viz")
    parser.add_argument("--vit-arch", default="small")
    parser.add_argument("--vit-feat", default="k")
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--dino-version", default="v2")
    parser.add_argument("--fixed-size", type=int, default=1022)
    parser.add_argument("--resize-mode", default="letterbox", choices=["letterbox", "stretch"])
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=63)
    parser.add_argument("--min-region-patches", type=int, default=5)
    parser.add_argument("--min-split-ratio", type=float, default=0.05)
    parser.add_argument("--noise-threshold", type=float, default=3.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None, help="Comma-separated local_file_name values to process (others skipped)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metadata_path = args.metadata or os.path.join(args.dataset_root, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    if args.limit is not None:
        metadata = metadata[: args.limit]
    if args.only:
        keep = set(args.only.split(","))
        metadata = [m for m in metadata if m["local_file_name"] in keep]

    print("Loading backbone...")
    backbone = build_backbone(args.vit_arch, args.vit_feat, args.patch_size, None, dino_version=args.dino_version, debiased=False)

    print("Loading COCO annotations...")
    coco = COCO(args.coco_json)
    categories = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    rows, reports, failures = [], [], []
    for index, record in enumerate(metadata, start=1):
        start_time = time.time()
        try:
            row, report = process_image(record, backbone, coco, categories, args)
            out_path = os.path.join(args.output_dir, record["local_file_name"].replace(".jpg", ".png"))
            row.save(out_path)
            rows.append(row)
            reports.append(report)
            elapsed = time.time() - start_time
            print(f"[{index}/{len(metadata)}] {record['file_name']} -> {report['verdict']} ({elapsed:.2f}s)", flush=True)
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"ERROR: {record['file_name']} failed after {elapsed:.2f}s: {e}", flush=True)
            failures.append({"record": record, "error": str(e)})

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    gallery_path = os.path.join(args.output_dir, "gallery.png")
    save_gallery(rows, gallery_path, "class-group-hierarchy Fiedler visualization")
    print(f"Saved gallery: {gallery_path}")

    with open(os.path.join(args.output_dir, "reports.json"), "w") as f:
        json.dump(reports, f, indent=2)

    n_full = sum(1 for r in reports if r["verdict"].startswith("HIERARCHY HOLDS"))
    n_partial = sum(1 for r in reports if r["verdict"].startswith("PARTIAL"))
    n_fail = sum(1 for r in reports if r["verdict"].startswith("HIERARCHY DOES NOT"))
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Images processed: {len(reports)} (failures: {len(failures)})")
    print(f"  Hierarchy holds fully:        {n_full}")
    print(f"  Partial (group isolated, some instances un-split): {n_partial}")
    print(f"  Hierarchy does NOT hold:      {n_fail}")
    for r in reports:
        print(f"  {r['file_name']:25s} group={r['group']:12s} depth={r['group_node_depth']} "
              f"descendant={r['num_descendant']}/{r['num_group_instances']} same={r['num_same_as_group']} -> {r['verdict']}")

    if failures:
        with open(os.path.join(args.output_dir, "failures.json"), "w") as f:
            json.dump(failures, f, indent=2)


if __name__ == "__main__":
    main()
