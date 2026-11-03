#!/usr/bin/env python3
"""Drive fiedler_reference_eval_v2.py in small batches, each a fresh subprocess,
to avoid the Windows heap-fragmentation OOM-kill seen when running all 410 images
in one long-lived process. Each batch appends to the same instance_results.csv /
failures.jsonl; a separate finalize step (finalize_reference_v2_report.py)
recomputes summary.json/plots/galleries from the merged CSV afterward.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-exe", default=str(Path(".venv/Scripts/python.exe")))
    parser.add_argument("--script", default="Fiedler Evaluation/fiedler_reference_eval_v2.py")
    parser.add_argument("--dataset-name", default="fiedler_reference_guided_v2_500")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--reference-image-root", required=True)
    parser.add_argument("--reference-annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-images", type=int, required=True, help="Total filtered COCO-20i images expected")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--vit-arch", default="small")
    parser.add_argument("--vit-feat", default="k")
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--dino-version", default="v2")
    parser.add_argument("--pretrain-path", default=None, help="Local checkpoint path (required for dino-version v3 to avoid network downloads of an untested variant)")
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--fixed-size", type=int, default=1022)
    parser.add_argument("--resize-mode", default="letterbox", choices=["letterbox", "stretch"])
    parser.add_argument("--svd-components", type=int, default=100)
    parser.add_argument("--debias-tree", action="store_true")
    parser.add_argument("--min-region-patches", type=int, default=5)
    parser.add_argument("--class-level", action="store_true")
    parser.add_argument("--union-taus", default="")
    parser.add_argument("--class-level-reference", action="store_true")
    parser.add_argument("--mask-interp", default="nearest", choices=["nearest", "bilinear"], help="Patch-grid <-> pixel mask resampling (see fiedler_reference_eval_v2.py --mask-interp)")
    args = parser.parse_args()

    progress_path = Path(args.output_dir) / "_batch_progress.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    start = 0
    if progress_path.exists():
        with open(progress_path) as f:
            start = json.load(f)["next_skip"]
        print(f"Resuming from skip={start}")

    while start < args.total_images:
        batch_start_time = time.time()
        cmd = [
            args.python_exe, args.script,
            "--dataset-name", args.dataset_name,
            "--image-root", args.image_root,
            "--annotations", args.annotations,
            "--reference-image-root", args.reference_image_root,
            "--reference-annotations", args.reference_annotations,
            "--output-dir", args.output_dir,
            "--vit-arch", args.vit_arch,
            "--vit-feat", args.vit_feat,
            "--patch-size", str(args.patch_size),
            "--tau", str(args.tau),
            "--fixed-size", str(args.fixed_size),
            "--resize-mode", args.resize_mode,
            "--dino-version", args.dino_version,
            "--svd-components", str(args.svd_components),
            "--min-region-patches", str(args.min_region_patches),
            "--unlimited-depth", "--unlimited-nodes", "--unlimited-imbalance", "--unlimited-noise",
            "--skip-images", str(start),
            "--max-images", str(args.batch_size),
        ]
        if args.class_level:
            cmd.append("--class-level")
        if args.union_taus:
            cmd += ["--union-taus", args.union_taus]
        if args.class_level_reference:
            cmd.append("--class-level-reference")
        if args.pretrain_path:
            cmd += ["--pretrain-path", args.pretrain_path]
        if args.debias_tree:
            cmd.append("--debias-tree")
        if args.mask_interp != "nearest":
            cmd += ["--mask-interp", args.mask_interp]
        if start > 0:
            cmd.append("--append")

        print(f"\n{'='*70}\nBatch: skip={start}, size={args.batch_size} ({start}-{start+args.batch_size} of {args.total_images})\n{'='*70}", flush=True)
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"WARNING: batch at skip={start} exited with code {result.returncode}. Retrying once...", flush=True)
            time.sleep(5)
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"ERROR: batch at skip={start} failed twice. Stopping. Re-run this script to resume from skip={start}.", flush=True)
                sys.exit(1)

        start += args.batch_size
        with open(progress_path, "w") as f:
            json.dump({"next_skip": start}, f)

        batch_elapsed = time.time() - batch_start_time
        print(f"Batch completed in {batch_elapsed:.1f}s. Progress: {min(start, args.total_images)}/{args.total_images}", flush=True)

    print("\nAll batches complete.")


if __name__ == "__main__":
    main()
