#!/usr/bin/env python3
"""Schematic: generic recursive-bipartition tree -> reference mask -> same tree
with the union of pure nodes highlighted (the --union-taus selection in
fiedler_reference_eval_v2.py: keep every node with purity |n & C| / |n| >= tau,
output their union).

Purely illustrative: leaf purities are made up, parents are the (equal-area)
average of their children so the numbers stay internally consistent.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

TAU = 0.7
# Two pure leaves in different root subtrees: their union is not any single node of
# the tree (unlike two pure siblings, whose union is just their parent). A pure
# internal node would always bring a pure child with it (equal-area averaging), so
# leaves are the only way to show exactly two non-nested selections.
LEAF_PURITY = [0.10, 0.05, 0.90, 0.20, 0.15, 0.00, 0.88, 0.08]
DEPTH = 3

NEUTRAL = "#d9dde3"
NEUTRAL_EDGE = "#6b7280"
SELECTED = "#f59e0b"
SELECTED_EDGE = "#b45309"
MAXIMAL_EDGE = "#7c2d12"
TEXT = "#1f2937"


def build_tree():
    """Heap-indexed complete binary tree: node i has children 2i+1, 2i+2."""
    n_nodes = 2 ** (DEPTH + 1) - 1
    first_leaf = 2 ** DEPTH - 1
    purity = [0.0] * n_nodes
    for k, p in enumerate(LEAF_PURITY):
        purity[first_leaf + k] = p
    for i in range(first_leaf - 1, -1, -1):
        purity[i] = (purity[2 * i + 1] + purity[2 * i + 2]) / 2
    return n_nodes, first_leaf, purity


def positions(n_nodes, x0, width, y_top, y_step):
    pos = {}
    for i in range(n_nodes):
        depth = (i + 1).bit_length() - 1
        index_in_level = i - (2 ** depth - 1)
        slots = 2 ** depth
        pos[i] = (x0 + width * (index_in_level + 0.5) / slots, y_top - depth * y_step)
    return pos


def draw_tree(ax, pos, n_nodes, purity, selected=None, maximal=None, show_purity=False):
    selected = selected or set()
    maximal = maximal or set()
    for i in range(n_nodes):
        for c in (2 * i + 1, 2 * i + 2):
            if c < n_nodes:
                ax.plot(*zip(pos[i], pos[c]), color="#9ca3af", lw=1.4, zorder=1)
    for i in range(n_nodes):
        x, y = pos[i]
        is_sel = i in selected
        ax.scatter([x], [y], s=620, zorder=3,
                   c=SELECTED if is_sel else NEUTRAL,
                   edgecolors=MAXIMAL_EDGE if i in maximal else (SELECTED_EDGE if is_sel else NEUTRAL_EDGE),
                   linewidths=3.0 if i in maximal else 1.2)
        if show_purity:
            ax.text(x, y, f"{purity[i]:.2f}", ha="center", va="center", fontsize=8.5, color=TEXT, zorder=4,
                    fontweight="bold" if is_sel else "normal")


def draw_strip(ax, x0, width, y, first_leaf, highlight, label):
    """The image as the leaf partition (one cell per leaf region)."""
    n = len(LEAF_PURITY)
    cell = width / n
    for k in range(n):
        on = (first_leaf + k) in highlight
        ax.add_patch(Rectangle((x0 + k * cell, y), cell, 0.42, facecolor=SELECTED if on else "white",
                               edgecolor=NEUTRAL_EDGE, lw=1.0, zorder=2))
    ax.text(x0 + width / 2, y - 0.22, label, ha="center", va="top", fontsize=9, color=TEXT)


def draw_mask_icon(ax, cx, cy, w, h):
    """Small binary-mask pictogram: dark frame with a white blob."""
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor="#1f2937", edgecolor="#1f2937", zorder=3))
    ax.add_patch(Ellipse((cx - 0.12 * w, cy - 0.05 * h), 0.55 * w, 0.62 * h, angle=-15, facecolor="white", zorder=4))
    ax.add_patch(Ellipse((cx + 0.2 * w, cy + 0.12 * h), 0.32 * w, 0.4 * h, facecolor="white", zorder=4))


def render_minimal(n_nodes, purity, selected, out_dir):
    """Idea-only version: tree -> reference mask -> same tree with selected nodes."""
    fig, ax = plt.subplots(figsize=(12.5, 4.0))
    ax.set_xlim(-0.3, 12.8)
    ax.set_ylim(-0.45, 3.85)
    ax.axis("off")

    tree_w, y_top, y_step = 4.2, 3.0, 0.95
    left_x, right_x = 0.0, 8.5
    draw_tree(ax, positions(n_nodes, left_x, tree_w, y_top, y_step), n_nodes, purity)
    draw_tree(ax, positions(n_nodes, right_x, tree_w, y_top, y_step), n_nodes, purity, selected)
    ax.text(left_x + tree_w / 2, 3.62, "Tree", ha="center", fontsize=13, fontweight="bold", color=TEXT)
    ax.text(right_x + tree_w / 2, 3.62, "Selected nodes", ha="center", fontsize=13, fontweight="bold", color=TEXT)

    box_x, box_w, box_y, box_h = 5.05, 2.4, 0.45, 2.55
    ax.add_patch(FancyBboxPatch((box_x, box_y), box_w, box_h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                facecolor="#eef2ff", edgecolor="#4f46e5", lw=1.6, zorder=2))
    cx = box_x + box_w / 2
    ax.text(cx, box_y + box_h - 0.35, "Reference mask", ha="center", fontsize=12.5, fontweight="bold", color="#312e81")
    draw_mask_icon(ax, cx, box_y + 0.95, 1.5, 1.1)
    for (x_from, x_to) in ((left_x + tree_w + 0.1, box_x - 0.12), (box_x + box_w + 0.12, right_x - 0.1)):
        ax.add_patch(FancyArrowPatch((x_from, 1.72), (x_to, 1.72), arrowstyle="-|>", mutation_scale=22,
                                     lw=2.0, color="#4f46e5", zorder=2))

    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"union_of_nodes_schematic_minimal.{ext}")
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)


def main():
    n_nodes, first_leaf, purity = build_tree()
    selected = {i for i in range(n_nodes) if purity[i] >= TAU}
    # A selected node whose parent is not selected: the union equals the union of these.
    maximal = {i for i in selected if i == 0 or (i - 1) // 2 not in selected}

    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set_xlim(-0.3, 13.3)
    ax.set_ylim(-1.75, 3.95)
    ax.axis("off")

    tree_w, y_top, y_step = 4.2, 3.0, 0.95
    left_x, right_x = 0.0, 9.0
    left_pos = positions(n_nodes, left_x, tree_w, y_top, y_step)
    right_pos = positions(n_nodes, right_x, tree_w, y_top, y_step)

    # Left: generic tree.
    draw_tree(ax, left_pos, n_nodes, purity)
    ax.text(left_x + tree_w / 2, 3.72, "Recursive bipartition tree", ha="center", fontsize=12, fontweight="bold", color=TEXT)
    draw_strip(ax, left_x, tree_w, -0.95, first_leaf, set(), "leaf regions (image partition)")

    # Middle: reference-mask process.
    box_x, box_w, box_y, box_h = 4.85, 3.5, 0.55, 2.35
    ax.add_patch(FancyBboxPatch((box_x, box_y), box_w, box_h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                facecolor="#eef2ff", edgecolor="#4f46e5", lw=1.6, zorder=2))
    cx = box_x + box_w / 2
    ax.text(cx, box_y + box_h - 0.3, "Reference mask", ha="center", fontsize=12.5, fontweight="bold", color="#312e81")
    steps = [
        "1. localize target patches $C$",
        "    (forward $\\cap$ backward matching)",
        "2. node purity",
        "    $p(n) = |n \\cap C|\\,/\\,|n|$",
        f"3. keep nodes with $p(n) \\geq \\tau$ ($\\tau$ = {TAU})",
        "4. output their union",
    ]
    for k, s in enumerate(steps):
        ax.text(box_x + 0.18, box_y + box_h - 0.72 - k * 0.27, s, ha="left", va="center", fontsize=9.2, color=TEXT)
    for (x_from, x_to) in ((left_x + tree_w + 0.1, box_x - 0.12), (box_x + box_w + 0.12, right_x - 0.1)):
        ax.add_patch(FancyArrowPatch((x_from, 1.72), (x_to, 1.72), arrowstyle="-|>", mutation_scale=22,
                                     lw=2.0, color="#4f46e5", zorder=2))

    # Right: same tree, purities shown, pure nodes highlighted.
    draw_tree(ax, right_pos, n_nodes, purity, selected, maximal, show_purity=True)
    ax.text(right_x + tree_w / 2, 3.72, "Same tree, union of pure nodes", ha="center", fontsize=12, fontweight="bold", color=TEXT)
    covered = {first_leaf + k for k in range(len(LEAF_PURITY)) if any(
        (first_leaf + k + 1) >> s == (m + 1) for m in maximal for s in range(DEPTH + 1))}
    draw_strip(ax, right_x, tree_w, -0.95, first_leaf, covered, "predicted mask = union of highlighted nodes")

    # Legend.
    legend_y = -1.62
    items = [(NEUTRAL, NEUTRAL_EDGE, 1.2, f"$p(n) < \\tau$"),
             (SELECTED, SELECTED_EDGE, 1.2, f"$p(n) \\geq \\tau$ (selected)"),
             (SELECTED, MAXIMAL_EDGE, 3.0, "top-most selected node (defines the union)")]
    x = 3.2
    for face, edge, lw, label in items:
        ax.scatter([x], [legend_y], s=160, c=face, edgecolors=edge, linewidths=lw)
        ax.text(x + 0.2, legend_y, label, va="center", fontsize=9, color=TEXT)
        x += 2.0 if "selected)" not in label else 2.6

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "figures")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"union_of_nodes_schematic.{ext}")
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)

    render_minimal(n_nodes, purity, selected, out_dir)


if __name__ == "__main__":
    main()
