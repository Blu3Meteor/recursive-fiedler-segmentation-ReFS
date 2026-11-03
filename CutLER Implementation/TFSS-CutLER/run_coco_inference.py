#!/usr/bin/env python
"""
Script to run CutLER inference on COCO val2017 dataset.
This script sets up the environment and runs inference using the train_net.py script.
"""

import os
import sys
import subprocess

# Add current directory to path
sys.path.insert(0, './')

def _require_path(path, description):
    if not os.path.exists(path):
        print(f"Missing {description}: {path}")
        return False
    return True


def run_inference():
    """
    Run CutLER inference on COCO val2017.
    """
    
    # Set environment variables
    os.environ['DETECTRON2_DATASETS'] = os.environ.get('DETECTRON2_DATASETS', './datasets')
    dataset_root = os.path.abspath(os.environ['DETECTRON2_DATASETS'])
    
    # Configuration and model settings
    config_file = "cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml"
    model_weights = "http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth"
    test_dataset = "cls_agnostic_coco"  # COCO val2017 in class-agnostic format
    num_gpus = 1  # Adjust based on your setup
    
    print(f"DETECTRON2_DATASETS set to: {dataset_root}")
    print(f"Config file: {config_file}")
    print(f"Model weights: {model_weights}")
    print(f"Test dataset: {test_dataset}")
    print(f"Number of GPUs: {num_gpus}")
    print()

    coco_images = os.path.join(dataset_root, "coco", "val2017")
    coco_ann = os.path.join(
        dataset_root, "coco", "annotations", "coco_cls_agnostic_instances_val2017.json"
    )
    if not _require_path(coco_images, "COCO val2017 images folder"):
        return 2
    if not _require_path(coco_ann, "COCO class-agnostic val2017 annotation file"):
        return 2
    
    # Build command
    cmd = [
        sys.executable, "cutler/train_net.py",
        "--num-gpus", str(num_gpus),
        "--config-file", config_file,
        "--test-dataset", test_dataset,
        "--eval-only",
        "MODEL.WEIGHTS", model_weights
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Run the inference
    result = subprocess.run(cmd, cwd=".")
    return result.returncode

if __name__ == "__main__":
    exit_code = run_inference()
    sys.exit(exit_code)
