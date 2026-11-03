#!/usr/bin/env python3
"""Visualize the COCO-20i reference images and their annotated masks (one per class)."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

REFERENCE_JSON = "Fiedler Evaluation/data/coco20i/coco20i_reference.json"
IMAGE_ROOT = "Fiedler Evaluation/datasets/val2017"
OUTPUT_PATH = "Fiedler Evaluation/output/04_reference_matching/coco20i_reference_gallery.png"


def overlay_mask(image_array, mask, color=(70, 200, 110), alpha=0.5):
    overlay = image_array.copy()
    mask = np.asarray(mask, dtype=bool)
    color_arr = np.array(color, dtype=np.float32)
    region = overlay[mask].astype(np.float32)
    overlay[mask] = (region * (1 - alpha) + color_arr * alpha).astype(np.uint8)
    return overlay


def main():
    coco = COCO(REFERENCE_JSON)
    categories = {cat["id"]: cat["name"] for cat in coco.loadCats(sorted(coco.getCatIds()))}

    image_ids = sorted(coco.getImgIds())
    n = len(image_ids)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.flatten()

    for idx, image_id in enumerate(image_ids):
        ax = axes[idx]
        image_info = coco.loadImgs([image_id])[0]
        image_path = os.path.join(IMAGE_ROOT, image_info["file_name"])

        ann_ids = coco.getAnnIds(imgIds=[image_id])
        annotations = coco.loadAnns(ann_ids)
        annotation = annotations[0]  # one class annotation per reference image
        category_name = categories.get(annotation["category_id"], "unknown")

        try:
            image = np.asarray(Image.open(image_path).convert("RGB"))
            mask = coco.annToMask(annotation).astype(bool)
            overlaid = overlay_mask(image, mask)
            ax.imshow(overlaid)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "image not found", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(f"{category_name}\n(img {image_id})", fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("COCO-20i reference set: 1 image + mask per class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reference gallery to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
