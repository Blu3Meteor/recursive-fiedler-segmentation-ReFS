#!/usr/bin/env python
"""
Generate visualizations for random COCO val2017 images using CutLER.
"""

import argparse
import glob
import os
import random
import sys

import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
CUTLER_ROOT = os.path.join(REPO_ROOT, "cutler")
DEMO_ROOT = os.path.join(CUTLER_ROOT, "demo")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, CUTLER_ROOT)
sys.path.insert(0, DEMO_ROOT)

from config import add_cutler_config
from predictor import VisualizationDemo


def _require_path(path, description):
    if not os.path.exists(path):
        print(f"Missing {description}: {path}")
        return False
    return True


def _setup_cfg(config_file, weights, device, confidence_threshold):
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    cfg.DATASETS.TEST = ("cls_agnostic_coco",)

    if cfg.MODEL.DEVICE == "cpu" and cfg.MODEL.RESNETS.NORM == "SyncBN":
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.FPN.NORM = "BN"

    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold

    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser(description="CutLER COCO val2017 visualizations")
    parser.add_argument("--count", type=int, default=10, help="Number of random images to visualize")
    parser.add_argument(
        "--output",
        default=os.path.join("output", "coco_val2017_random10"),
        help="Output directory for visualizations",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument(
        "--config-file",
        default="cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--weights",
        default="http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth",
        help="Model weights",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.35,
        help="Minimum score for predictions",
    )
    args = parser.parse_args()

    os.environ["DETECTRON2_DATASETS"] = os.environ.get("DETECTRON2_DATASETS", "./datasets")
    dataset_root = os.path.abspath(os.environ["DETECTRON2_DATASETS"])
    coco_images = os.path.join(dataset_root, "coco", "val2017")

    print(f"DETECTRON2_DATASETS: {dataset_root}")
    print(f"Config file: {args.config_file}")
    print(f"Weights: {args.weights}")
    print(f"Device: {args.device}")
    print(f"Output dir: {args.output}")

    if not _require_path(coco_images, "COCO val2017 images folder"):
        return 2

    images = sorted(glob.glob(os.path.join(coco_images, "*.jpg")))
    if not images:
        print(f"No images found in {coco_images}")
        return 2

    count = min(args.count, len(images))
    if args.seed is not None:
        random.seed(args.seed)
    selected = random.sample(images, count)

    os.makedirs(args.output, exist_ok=True)

    cfg = _setup_cfg(args.config_file, args.weights, args.device, args.confidence_threshold)
    demo = VisualizationDemo(cfg)

    for path in selected:
        img = read_image(path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        out_path = os.path.join(args.output, os.path.basename(path))
        visualized_output.save(out_path)
        print(f"{os.path.basename(path)} -> {out_path} ({len(predictions.get('instances', []))} instances)")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
