#!/usr/bin/env python3
"""Visualize a recursive Fiedler tree with a distinct color per segment.

Colors are hierarchical: the root owns the whole hue circle, and each split
hands each child half of its parent's hue interval (siblings also alternate
brightness so deep siblings stay distinguishable). A child is therefore
always a shade "inside" its parent's color family, and the same color is used
for a node in every panel and in the dendrogram.

Figure layout:
  top row    -- original image, final leaf partition, then the partition of the
                image at successive depths (depth d = deepest node <= d per pixel)
  bottom row -- dendrogram of the whole tree, nodes colored as above, sized by
                area; oracle-best nodes (if given) are starred and labeled.
"""

import colorsys
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MAX_DEPTH_PANELS = 6
PANEL_WIDTH_PX = 360


def _children_map(nodes):
    children = defaultdict(list)
    for node in nodes:
        children[node["parent"]].append(node["id"])
    return children


def _assign_colors(nodes, children, root_id):
    """node id -> RGB in [0, 1], by recursive hue-interval halving."""
    colors = {}
    stack = [(root_id, 0.0, 1.0, 0)]
    while stack:
        node_id, lo, hi, sibling_index = stack.pop()
        depth_value = 0.92 if sibling_index % 2 == 0 else 0.68
        colors[node_id] = colorsys.hsv_to_rgb((lo + hi) / 2.0, 0.8, depth_value)
        kids = children.get(node_id, [])
        if not kids:
            continue
        step = (hi - lo) / len(kids)
        for k, kid in enumerate(kids):
            stack.append((kid, lo + k * step, lo + (k + 1) * step, k))
    return colors


def _display_image(image_path, fixed_size, resize_metadata):
    """Image resized exactly as the tree saw it, padding cropped away."""
    from fiedler_class_transfer import resize_stretch, resize_with_aspect_ratio

    image = Image.open(image_path).convert("RGB")
    if resize_metadata and "scale_x" in resize_metadata:
        image, meta = resize_stretch(image, fixed_size)
    else:
        image, meta = resize_with_aspect_ratio(image, fixed_size)
    box = _content_box(meta, fixed_size)
    return np.asarray(image.crop(box)), box, meta


def _content_box(meta, fixed_size):
    left, top = meta.get("pad_left", 0), meta.get("pad_top", 0)
    w = meta.get("resized_w", fixed_size)
    h = meta.get("resized_h", fixed_size)
    return (left, top, left + w, top + h)


def _grid_to_display(grid_rgb, fixed_size, box, display_size, orig_size, stretch):
    """Patch-grid color map -> display-resolution image aligned with the photo."""
    img = Image.fromarray((grid_rgb * 255).astype(np.uint8)).resize((fixed_size, fixed_size), Image.NEAREST)
    img = img.crop(box)
    if stretch:
        # Undo the stretch so segments overlay the image at its true proportions.
        img = img.resize(orig_size, Image.NEAREST)
    return np.asarray(img.resize(display_size, Image.NEAREST)).astype(np.float32) / 255.0


def _partition(nodes, colors, grid_shape, max_depth):
    """Per-patch color of the deepest node with depth <= max_depth (None = leaves)."""
    out = np.zeros(grid_shape + (3,), dtype=np.float32)
    label = np.full(grid_shape, -1, dtype=np.int64)
    for node in sorted(nodes, key=lambda n: n["depth"]):
        if max_depth is not None and node["depth"] > max_depth:
            continue
        mask = np.asarray(node["region_mask"], dtype=bool)
        out[mask] = colors[node["id"]]
        label[mask] = node["id"]
    return out, label


def _boundaries(label):
    edge = np.zeros(label.shape, dtype=bool)
    edge[:, 1:] |= label[:, 1:] != label[:, :-1]
    edge[1:, :] |= label[1:, :] != label[:-1, :]
    return edge


def _layout(children, root_id):
    """Leaves at consecutive x in DFS order; parents centered over children."""
    x = {}
    next_leaf = [0]

    def visit(node_id):
        kids = children.get(node_id, [])
        if not kids:
            x[node_id] = next_leaf[0]
            next_leaf[0] += 1
            return
        for kid in kids:
            visit(kid)
        x[node_id] = float(np.mean([x[k] for k in kids]))

    import sys
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    visit(root_id)
    return x, next_leaf[0]


def visualize_tree(image_path, tree, output_path, fixed_size, resize_metadata=None, highlights=None, tree_label="Fiedler", max_depth=None):
    """max_depth: draw only nodes with depth <= max_depth (the tree cut at that level)."""
    nodes = [n for n in tree["nodes"] if n.get("region_mask") is not None
             and (max_depth is None or n["depth"] <= max_depth)]
    if not nodes:
        return None
    by_id = {n["id"]: n for n in nodes}
    root = min(nodes, key=lambda n: n["depth"])
    children = _children_map(nodes)
    children = {k: [c for c in v if c in by_id] for k, v in children.items()}
    colors = _assign_colors(nodes, children, root["id"])
    grid_shape = np.asarray(root["region_mask"]).shape

    photo, box, meta = _display_image(image_path, fixed_size, resize_metadata)
    stretch = "scale_x" in meta
    orig_size = (meta["orig_w"], meta["orig_h"])
    aspect = (orig_size[1] / orig_size[0]) if stretch else (photo.shape[0] / photo.shape[1])
    display_size = (PANEL_WIDTH_PX, max(1, int(round(PANEL_WIDTH_PX * aspect))))
    photo_disp = np.asarray(Image.fromarray(photo).resize(display_size, Image.LANCZOS)).astype(np.float32) / 255.0

    depth_cut = max_depth
    max_depth = max(n["depth"] for n in nodes)
    if max_depth <= MAX_DEPTH_PANELS:
        depths = list(range(1, max_depth + 1))
    else:
        depths = sorted(set(np.linspace(1, max_depth, MAX_DEPTH_PANELS).round().astype(int).tolist()))
    leaves = [n for n in nodes if not children.get(n["id"])]

    panels = [("original", photo_disp)]

    def overlay(depth_limit):
        rgb, label = _partition(nodes, colors, grid_shape, depth_limit)
        color = _grid_to_display(rgb, fixed_size, box, display_size, orig_size, stretch)
        # Boundaries at display resolution so they stay 1px instead of a patch wide.
        label_img = Image.fromarray(label.astype(np.int32), mode="I").resize((fixed_size, fixed_size), Image.NEAREST).crop(box)
        if stretch:
            label_img = label_img.resize(orig_size, Image.NEAREST)
        edge = _boundaries(np.asarray(label_img.resize(display_size, Image.NEAREST)))
        blended = 0.4 * photo_disp + 0.6 * color
        blended[edge] = 0.0
        return blended

    if depth_cut is None:
        panels.append((f"leaves ({len(leaves)})", overlay(None)))
    for d in depths:
        n_at = sum(1 for n in nodes if n["depth"] == d)
        panels.append((f"depth {d} ({n_at} nodes)", overlay(d)))

    n_cols = len(panels)
    panel_in = 3.0
    img_h_in = panel_in * display_size[1] / display_size[0]
    fig = plt.figure(figsize=(panel_in * n_cols, img_h_in + 4.2), dpi=90)
    grid = fig.add_gridspec(2, n_cols, height_ratios=[img_h_in, 4.0], hspace=0.12, wspace=0.03)
    for col, (title, image) in enumerate(panels):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(grid[1, :])
    x, n_leaves = _layout(children, root["id"])
    total_area = max(1, int(np.asarray(root["region_mask"]).sum()))
    for node in nodes:
        for kid in children.get(node["id"], []):
            ax.plot([x[node["id"]], x[kid]], [-node["depth"], -by_id[kid]["depth"]], color="0.75", linewidth=0.6, zorder=1)
    xs = [x[n["id"]] for n in nodes]
    ys = [-n["depth"] for n in nodes]
    sizes = [8 + 220 * np.sqrt(n.get("region_area_patches", np.asarray(n["region_mask"]).sum()) / total_area) for n in nodes]
    ax.scatter(xs, ys, s=sizes, c=[colors[n["id"]] for n in nodes], edgecolors="0.2", linewidths=0.3, zorder=2)

    for node_id, label_text in (highlights or {}).items():
        if node_id not in by_id:
            continue
        hx, hy = x[node_id], -by_id[node_id]["depth"]
        ax.scatter([hx], [hy], marker="*", s=180, c="black", zorder=3)
        ax.annotate(label_text, (hx, hy), xytext=(4, 6), textcoords="offset points", fontsize=8, zorder=4)

    ax.set_xlim(-1, max(n_leaves, 1))
    ax.set_ylim(-max_depth - 0.6, 0.6)
    ax.set_yticks([-d for d in range(max_depth + 1)])
    ax.set_yticklabels([str(d) for d in range(max_depth + 1)], fontsize=8)
    ax.set_ylabel("depth", fontsize=9)
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{tree_label} tree" + (f" cut at depth {depth_cut}" if depth_cut is not None else "") + f": {len(nodes)} nodes, {len(leaves)} leaves, max depth {max_depth}"
                 + ("  (* = oracle-best node per class)" if highlights else ""), fontsize=10)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_candidate_visualization(image_id, image_path, tree, output_dir, resize_metadata=None, fixed_size=None, highlights=None):
    if not tree or not tree.get("nodes"):
        return None
    if fixed_size is None:
        fixed_size = tree["grid_w"] * 14
    output_path = os.path.join(output_dir, f"{image_id}_tree.png")
    return visualize_tree(image_path, tree, output_path, fixed_size, resize_metadata=resize_metadata, highlights=highlights)
