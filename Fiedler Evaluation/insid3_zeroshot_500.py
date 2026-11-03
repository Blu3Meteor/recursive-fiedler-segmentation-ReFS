#!/usr/bin/env python3
"""Zero-shot INSID3 evaluation on 500-image COCO subset (clustering only, no reference)."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Setup path for INSID3
import sys
insid3_path = str(Path(__file__).parent.parent)
if insid3_path not in sys.path:
    sys.path.insert(0, insid3_path)

from pycocotools.coco import COCO
from INSID3.utils.clustering import agglomerative_clustering

def load_dinov3_model(model_size='large', device='cuda'):
    """Load DINOv3 model from INSID3."""
    # Import DINOv3 model builder
    from INSID3.models import _build_encoder

    model = _build_encoder(f'vit_{model_size}16', image_size=224, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_image(image_path, size=224):
    """Load and preprocess image for DINOv3."""
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

def extract_features(model, image_tensor, device='cuda'):
    """Extract DINOv3 patch features."""
    with torch.no_grad():
        features = model(image_tensor.to(device))
    # features: (1, N, 384) or (1, N, 1024) depending on model size
    return features.squeeze(0).cpu().numpy()  # (N, D)

def compute_candidates_from_clustering(features, tau=0.6):
    """Generate candidates using agglomerative clustering."""
    # Cluster features with cosine distance and tau threshold
    labels = agglomerative_clustering(features, threshold=tau, linkage='average', metric='cosine')

    # Convert cluster labels to candidate masks (each cluster is a candidate)
    num_clusters = len(np.unique(labels))
    candidates = []
    for cluster_id in np.unique(labels):
        mask = (labels == cluster_id).astype(np.uint8)
        candidates.append(mask)

    return candidates

def compute_best_iou(candidates, gt_mask):
    """Find best IoU between candidates and GT."""
    best_iou = 0.0
    for candidate in candidates:
        intersection = np.logical_and(candidate, gt_mask).sum()
        union = np.logical_or(candidate, gt_mask).sum()
        if union > 0:
            iou = intersection / union
            best_iou = max(best_iou, iou)
    return best_iou

def evaluate_dataset(args):
    """Evaluate on COCO subset."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations
    coco = COCO(args.annotations)
    image_ids = coco.getImgIds()
    image_ids = image_ids[:args.max_images] if args.max_images else image_ids

    print(f"Evaluating on {len(image_ids)} images...")

    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Loading DINOv3-{args.model_size} on {device}...")
    model = load_dinov3_model(args.model_size, device)

    # Open CSV for incremental writing
    csv_path = output_dir / "instance_results.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=['image_id', 'instance_id', 'best_iou', 'num_candidates'])
    csv_writer.writeheader()
    csv_file.flush()

    results = []
    failures = []
    elapsed_times = []

    for img_id in tqdm(image_ids, desc="INSID3 Eval"):
        img_start = time.time()
        try:
            image_info = coco.loadImgs([img_id])[0]
            img_path = Path(args.image_root) / image_info['file_name']

            if not img_path.exists():
                failures.append({'image_id': img_id, 'error': 'image_not_found'})
                continue

            # Load and extract features
            img_tensor = load_image(str(img_path))
            features = extract_features(model, img_tensor, device)

            # Cluster to get candidates
            candidates = compute_candidates_from_clustering(features, tau=args.tau)
            num_candidates = len(candidates)

            # Get GT masks and compute best IoU per instance
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            annotations = coco.loadAnns(ann_ids)

            for ann_idx, ann in enumerate(annotations):
                gt_mask = coco.annToMask(ann)
                best_iou = compute_best_iou(candidates, gt_mask)

                result = {
                    'image_id': img_id,
                    'instance_id': ann['id'],
                    'best_iou': best_iou,
                    'num_candidates': num_candidates,
                }
                results.append(result)
                csv_writer.writerow(result)

            csv_file.flush()

            elapsed = time.time() - img_start
            elapsed_times.append(elapsed)
            if len(elapsed_times) > 0:
                eta = np.mean(elapsed_times) * (len(image_ids) - len(results))
                # tqdm.write(f"  elapsed={elapsed:.2f}s ETA={eta/60:.1f}m")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if device == 'cuda':
                    torch.cuda.empty_cache()
                failures.append({'image_id': img_id, 'error': 'oom'})
            else:
                failures.append({'image_id': img_id, 'error': str(e)})
        except Exception as e:
            failures.append({'image_id': img_id, 'error': str(e)})

    csv_file.close()

    # Compute summary statistics
    if results:
        ious = [r['best_iou'] for r in results]
        summary = {
            'num_images': len(set(r['image_id'] for r in results)),
            'num_instances': len(results),
            'num_candidates_mean': np.mean([r['num_candidates'] for r in results]),
            'mean_best_iou': float(np.mean(ious)),
            'median_best_iou': float(np.median(ious)),
            'std_best_iou': float(np.std(ious)),
            'min_best_iou': float(np.min(ious)),
            'max_best_iou': float(np.max(ious)),
            'num_failures': len(failures),
            'model_size': args.model_size,
            'tau': args.tau,
        }
    else:
        summary = {'error': 'no_results', 'num_failures': len(failures)}

    # Save results
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    if failures:
        with open(output_dir / "failures.jsonl", 'w') as f:
            for failure in failures:
                f.write(json.dumps(failure) + '\n')

    print("\n" + "="*60)
    print("INSID3 ZERO-SHOT EVALUATION SUMMARY")
    print("="*60)
    print(json.dumps(summary, indent=2))
    print("="*60)
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', default='Fiedler Evaluation/data/coco_500_random.json')
    parser.add_argument('--image-root', default='CutLER Implementation/TFSS-CutLER/datasets/coco/val2017')
    parser.add_argument('--output-dir', default='Fiedler Evaluation/output/baselines_insid3/coco_500_random_insid3')
    parser.add_argument('--max-images', type=int, default=500)
    parser.add_argument('--tau', type=float, default=0.6)
    parser.add_argument('--model-size', choices=['small', 'large'], default='large')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    evaluate_dataset(args)
