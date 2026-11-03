#!/usr/bin/env python3
"""fig_fiedler_bipartition.png: toy illustration of one Fiedler split.

The toy graph follows the method's own construction (fiedler_class_transfer.get_affinity_matrix
with tau = 0): a FULLY CONNECTED graph over all patches, W_ij = cosine similarity of the patch
features, with non-positive similarities replaced by a tiny epsilon. The split solves the same
generalized eigenproblem as second_smallest_eigenvector: (D - W) v = lambda D v.
Patch features are synthetic: a 6 x 9 grid with an object region and a background region.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.linalg import eigh

ROWS, COLS, DIM, EPS = 6, 9, 32, 1e-5
BG = "#fafaf7"
ORANGE, BLUE = "#ec6a36", "#2a74d4"
TEXT, SUB = "#111111", "#555555"

# Object region (part B in the original figure).
OBJECT = np.zeros((ROWS, COLS), bool)
OBJECT[1, 1:5] = OBJECT[2, 1:6] = OBJECT[3, 1:6] = OBJECT[4, 1:5] = True


def synthetic_features(seed=3):
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=DIM)
    obj = shared + 1.6 * rng.normal(size=DIM)
    bg = shared + 1.6 * rng.normal(size=DIM)
    feats = np.where(OBJECT.reshape(-1, 1), obj, bg) + 0.45 * rng.normal(size=(ROWS * COLS, DIM))
    return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def fiedler(feats):
    W = feats @ feats.T
    W = np.where(W > 0.0, W, EPS)  # tau = 0, as in the method
    D = np.diag(W.sum(1))
    _, vecs = eigh(D - W, D)
    v = vecs[:, 1]
    if v[OBJECT.reshape(-1)].mean() < 0:  # orient so the object is positive (part B)
        v = -v
    return W, v / np.linalg.norm(v) * 0.4  # display scale only; the split depends on sign


def main():
    feats = synthetic_features()
    W, v = fiedler(feats)
    side_b = v > 0
    n = ROWS * COLS

    fig = plt.figure(figsize=(20, 9), facecolor=BG)
    fig.text(0.03, 0.94, "Fiedler vector and bipartitioning", fontsize=30, fontweight="bold", color=TEXT, va="center")
    fig.text(0.03, 0.895, "One eigenvector of the graph Laplacian splits the patches into two groups.",
             fontsize=17, color=SUB, va="center")

    # ---- 1: fully connected graph ----
    ax1 = fig.add_axes([0.03, 0.27, 0.30, 0.50], facecolor=BG)
    fig.text(0.03, 0.83, "1  Patches become a graph", fontsize=20, fontweight="bold", color=TEXT, va="center")
    ys, xs = np.divmod(np.arange(n), COLS)
    xs, ys = xs.astype(float), -ys.astype(float)
    iu, ju = np.triu_indices(n, k=1)
    w = W[iu, ju]
    wn = (w - w.min()) / (w.max() - w.min())
    order = np.argsort(wn)  # weak edges first, strong edges on top
    grey = np.array(to_rgb("#9a9a9a"))
    for k in order:
        i, j = iu[k], ju[k]
        same = side_b[i] == side_b[j]
        col = (np.array(to_rgb(BLUE if side_b[i] else ORANGE)) * 0.55 + grey * 0.45) if same else grey
        strength = wn[k] ** 4
        # gentle arc so edges between collinear patches do not sit on top of each other
        (x0, y0), (x1, y1) = (xs[i], ys[i]), (xs[j], ys[j])
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        bend = 0.12
        cx, cy = mx - bend * dy, my + bend * dx
        ax1.add_patch(PathPatch(Path([(x0, y0), (cx, cy), (x1, y1)], [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                                fill=False, edgecolor=col, lw=0.2 + 1.0 * strength, alpha=0.015 + 0.5 * strength, zorder=1))
    ax1.scatter(xs, ys, s=330, c=[BLUE if b else ORANGE for b in side_b], zorder=3, edgecolors=BG, linewidths=1.2)
    ax1.set_xlim(-0.6, COLS - 0.4); ax1.set_ylim(-ROWS + 0.4, 0.6); ax1.axis("off")
    fig.text(0.03, 0.205, "Every pair of patches is connected; darker = more similar.\n"
                          "Edges across the two groups are weak, so the cut falls there.",
             fontsize=14.5, color=SUB, va="center", linespacing=1.4)

    # ---- 2: Fiedler vector as a patch grid ----
    ax2 = fig.add_axes([0.375, 0.29, 0.30, 0.46], facecolor=BG)
    fig.text(0.375, 0.83, "2  Fiedler vector: one value per patch", fontsize=20, fontweight="bold", color=TEXT, va="center")
    vmax = np.abs(v).max()
    grid = v.reshape(ROWS, COLS)
    img = np.zeros((ROWS, COLS, 3))
    for r in range(ROWS):
        for c in range(COLS):
            t = min(1.0, abs(grid[r, c]) / vmax) ** 0.8
            base = np.array(to_rgb(BLUE if grid[r, c] > 0 else ORANGE))
            img[r, c] = (1 - (0.18 + 0.82 * t)) * np.ones(3) + (0.18 + 0.82 * t) * base
            img[r, c] = 0.25 * np.array(to_rgb(BG)) + 0.75 * img[r, c]
            ax2.text(c, r, f"{grid[r, c]:+.2f}", ha="center", va="center", fontsize=10.5,
                     color="white" if t > 0.6 else TEXT)
    ax2.imshow(img, interpolation="nearest")
    ax2.axis("off")
    fig.text(0.375, 0.205, "The smoothest non-trivial labelling of the graph:\nsimilar patches get similar values.",
             fontsize=14.5, color=SUB, va="center", linespacing=1.4)

    # ---- 3: sorted values, sign = cut ----
    ax3 = fig.add_axes([0.765, 0.27, 0.22, 0.48], facecolor=BG)
    fig.text(0.735, 0.83, "3  Its sign is the cut", fontsize=20, fontweight="bold", color=TEXT, va="center")
    sv = np.sort(v)
    ax3.bar(np.arange(n), sv, width=0.8, color=[BLUE if x > 0 else ORANGE for x in sv])
    ax3.axhline(0, color=TEXT, lw=1.6)
    ax3.set_ylabel("Fiedler value", fontsize=14, color=SUB)
    ax3.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax3.spines[s].set_visible(False)
    ax3.tick_params(axis="y", labelsize=11, colors=SUB)
    ax3.text(0.05, 0.62, "v < 0: part A", transform=ax3.transAxes, fontsize=16, fontweight="bold", color=ORANGE)
    ax3.text(0.97, 0.06, "v > 0: part B", transform=ax3.transAxes, fontsize=16, fontweight="bold", color=BLUE, ha="right")
    fig.text(0.735, 0.205, "Negative values form one part, positive values\nthe other. Patches are sorted by value.",
             fontsize=14.5, color=SUB, va="center", linespacing=1.4)

    fig.text(0.03, 0.105, r"Graph Laplacian  $L = D - W$.   Relaxed normalised cut:  $L\,v = \lambda\,D\,v$.   "
                          r"$v_1$ is constant (trivial), $v_2$ is the Fiedler vector.   Partition:  $v_2 > 0$  versus  $v_2 < 0$.",
             fontsize=17, color=TEXT, va="center")
    fig.text(0.03, 0.045, "Toy example: 6 x 9 grid of synthetic patch features, fully connected graph with cosine-similarity weights. "
                          "In the method, the nodes are DINOv2 patch features and the cut is applied recursively.",
             fontsize=12, color=SUB, va="center")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_fiedler_bipartition.png")
    fig.savefig(out, dpi=160, facecolor=BG)
    print(out, "| part B patches:", int(side_b.sum()), "| matches object region:", bool((side_b == OBJECT.reshape(-1)).all()))


if __name__ == "__main__":
    main()
