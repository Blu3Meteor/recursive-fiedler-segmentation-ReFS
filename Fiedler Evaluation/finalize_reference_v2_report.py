#!/usr/bin/env python3
"""Compute summary.json + plots from the merged batched-run instance_results.csv
for fiedler_reference_eval_v2.py (positional debiasing + localization + IoU-based
node selection), and print a comparison against the v1 baseline."""

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from fiedler_candidate_eval import plot_overall_histogram, plot_class_overview, plot_class_histograms  # noqa: E402

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "Fiedler Evaluation/output/04_reference_matching/fiedler_reference_guided_v2_500"
V1_SUMMARY = "Fiedler Evaluation/output/04_reference_matching/fiedler_reference_guided_500/summary.json"


def main():
    csv_path = os.path.join(OUTPUT_DIR, "instance_results.csv")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} instance rows")

    all_ious = [float(r["best_iou"]) for r in rows]
    class_scores = defaultdict(list)
    per_image_ious = defaultdict(list)
    for r in rows:
        class_scores[r["category_name"]].append(float(r["best_iou"]))
        per_image_ious[r["image_id"]].append(float(r["best_iou"]))

    per_class_summary = {}
    for cls, scores in class_scores.items():
        per_class_summary[cls] = {
            "count": len(scores),
            "mean_iou": float(np.mean(scores)),
            "median_iou": float(np.median(scores)),
            "min_iou": float(np.min(scores)),
            "max_iou": float(np.max(scores)),
            "scores": scores,
        }

    per_image_best = [max(v) for v in per_image_ious.values()]

    # Class-level runs record per-row intersection/union pixel counts; compute the
    # paper-style cumulative mIoU (per-class sum(inter)/sum(union), averaged over
    # classes), identical to INSID3's AverageMeter.
    class_level = bool(rows) and rows[0].get("intersection") not in (None, "")
    cumulative_miou = None
    cumulative_per_class = {}
    if class_level:
        inter_sum, union_sum = defaultdict(float), defaultdict(float)
        for r in rows:
            inter_sum[r["category_name"]] += float(r["intersection"])
            union_sum[r["category_name"]] += float(r["union"])
        cumulative_per_class = {c: (inter_sum[c] / union_sum[c]) if union_sum[c] > 0 else 0.0 for c in union_sum}
        cumulative_miou = float(np.mean(list(cumulative_per_class.values())))

    # Raw localization (candidate_patch_mask vs GT, before any tree-node selection) --
    # isolates forward/backward-matching quality from the Fiedler tree's contribution.
    candidate_cumulative_miou = None
    candidate_cumulative_per_class = {}
    has_candidate_cols = bool(rows) and rows[0].get("candidate_intersection") not in (None, "")
    if class_level and has_candidate_cols:
        c_inter_sum, c_union_sum = defaultdict(float), defaultdict(float)
        for r in rows:
            c_inter_sum[r["category_name"]] += float(r["candidate_intersection"])
            c_union_sum[r["category_name"]] += float(r["candidate_union"])
        candidate_cumulative_per_class = {c: (c_inter_sum[c] / c_union_sum[c]) if c_union_sum[c] > 0 else 0.0 for c in c_union_sum}
        candidate_cumulative_miou = float(np.mean(list(candidate_cumulative_per_class.values())))

    # "Mean of node and candidate" check (AND / OR combination of the selected node
    # and candidate_patch_mask) -- see fiedler_reference_eval_v2.py for why this is
    # not expected to beat either input alone.
    combo_cumulative = {}
    has_combo_cols = bool(rows) and rows[0].get("combo_and_intersection") not in (None, "")
    if class_level and has_combo_cols:
        for combo in ("and", "or"):
            i_sum, u_sum = defaultdict(float), defaultdict(float)
            for r in rows:
                i_sum[r["category_name"]] += float(r[f"combo_{combo}_intersection"])
                u_sum[r["category_name"]] += float(r[f"combo_{combo}_union"])
            per_cls = {c: (i_sum[c] / u_sum[c]) if u_sum[c] > 0 else 0.0 for c in u_sum}
            combo_cumulative[combo] = {"cumulative_miou": float(np.mean(list(per_cls.values()))), "cumulative_iou_per_class": per_cls}

    # Union-of-nodes variants (one set of columns per purity threshold tau).
    union_variants = {}
    tau_labels = sorted({k[len("inter_tau"):] for k in (rows[0].keys() if rows else []) if k.startswith("inter_tau")}, key=float)
    for tl in tau_labels:
        i_sum, u_sum, n_sel = defaultdict(float), defaultdict(float), []
        for r in rows:
            i_sum[r["category_name"]] += float(r[f"inter_tau{tl}"])
            u_sum[r["category_name"]] += float(r[f"union_tau{tl}"])
            n_sel.append(float(r[f"nsel_tau{tl}"]))
        per_cls = {c: (i_sum[c] / u_sum[c]) if u_sum[c] > 0 else 0.0 for c in u_sum}
        union_variants[tl] = {
            "cumulative_miou": float(np.mean(list(per_cls.values()))),
            "mean_nodes_selected": float(np.mean(n_sel)),
            "cumulative_iou_per_class": per_cls,
        }

    # Optional tune/held-out split (argv[2] = number of images used to pick tau).
    # Uses only numbers already in the CSV, so it costs no extra compute.
    tune_n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    split_report = None
    if tune_n and tau_labels:
        ids_sorted = sorted({int(r["image_id"]) for r in rows})
        tune_ids = set(ids_sorted[:tune_n])
        tune_rows = [r for r in rows if int(r["image_id"]) in tune_ids]
        held_rows = [r for r in rows if int(r["image_id"]) not in tune_ids]

        def cum(rs, ik, uk):
            i_s, u_s = defaultdict(float), defaultdict(float)
            for r in rs:
                i_s[r["category_name"]] += float(r[ik])
                u_s[r["category_name"]] += float(r[uk])
            vals = [i_s[c] / u_s[c] for c in u_s if u_s[c] > 0]
            return float(np.mean(vals)) if vals else 0.0

        split_report = {
            "n_tune_images": len(tune_ids), "n_heldout_images": len(ids_sorted) - len(tune_ids),
            "single": (cum(tune_rows, "intersection", "union"), cum(held_rows, "intersection", "union")),
        }
        for tl in tau_labels:
            split_report[tl] = (cum(tune_rows, f"inter_tau{tl}", f"union_tau{tl}"), cum(held_rows, f"inter_tau{tl}", f"union_tau{tl}"))
        split_report["best_tau"] = max(tau_labels, key=lambda t: split_report[t][0])

    summary = {
        "tune_heldout_split": split_report,
        "union_variants": union_variants,
        "dataset_name": os.path.basename(os.path.normpath(OUTPUT_DIR)),
        "class_level": class_level,
        "cumulative_miou": cumulative_miou,
        "cumulative_iou_per_class": cumulative_per_class,
        "candidate_cumulative_miou": candidate_cumulative_miou,
        "candidate_cumulative_iou_per_class": candidate_cumulative_per_class,
        "combo_cumulative": combo_cumulative,
        "selection_method": "reference-guided v2 (positional debiasing + forward/backward localization + IoU-based node selection, no GT)",
        "num_images": len(per_image_ious),
        "num_instances": len(rows),
        "mean_best_iou": float(np.mean(all_ious)),
        "median_best_iou": float(np.median(all_ious)),
        "min_best_iou": float(np.min(all_ious)),
        "max_best_iou": float(np.max(all_ious)),
        "mean_best_iou_per_image": float(np.mean(per_image_best)),
        "median_best_iou_per_image": float(np.median(per_image_best)),
        "per_class": per_class_summary,
    }

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_overall_histogram(all_ious, os.path.join(OUTPUT_DIR, "overall_iou_histogram.png"), "fiedler_reference_guided_v2_500 best IoU distribution")
    plot_class_overview(per_class_summary, os.path.join(OUTPUT_DIR, "class_mean_iou.png"), "fiedler_reference_guided_v2_500 mean IoU by class")
    plot_class_histograms(class_scores, os.path.join(OUTPUT_DIR, "class_histograms"), "fiedler_reference_guided_v2_500")

    print("\n" + "=" * 60)
    print("V2 (localization + debiasing) RESULTS")
    print("=" * 60)
    print(f"Images: {summary['num_images']}")
    print(f"Instances: {summary['num_instances']}")
    print(f"Mean IoU: {summary['mean_best_iou']:.4f}")
    print(f"Median IoU: {summary['median_best_iou']:.4f}")
    if class_level:
        print(f"Cumulative mIoU (paper-style, class-level): {cumulative_miou:.4f}")
    if candidate_cumulative_miou is not None:
        print(f"Raw localization mIoU (candidate_patch_mask vs GT, pre-tree-selection): {candidate_cumulative_miou:.4f}")
        print("  Per-class:")
        for c in sorted(candidate_cumulative_per_class, key=lambda c: -candidate_cumulative_per_class[c]):
            node_val = cumulative_per_class.get(c, 0.0)
            cand_val = candidate_cumulative_per_class[c]
            print(f"    {c:<14} candidate={cand_val*100:6.1f}  selected-node={node_val*100:6.1f}  ({(node_val-cand_val)*100:+.1f})")
    if combo_cumulative:
        print(f"\nMean-of-node-and-candidate check:")
        print(f"  single node (baseline):              {cumulative_miou*100:6.2f}")
        print(f"  candidate_patch_mask (baseline):      {candidate_cumulative_miou*100:6.2f}")
        print(f"  AND(node, candidate):                 {combo_cumulative['and']['cumulative_miou']*100:6.2f}")
        print(f"  OR(node, candidate):                  {combo_cumulative['or']['cumulative_miou']*100:6.2f}")
    if union_variants:
        print("\nSingle best node vs. union of nodes (purity >= tau), cumulative mIoU:")
        print(f"  {'single best node':<22} {cumulative_miou:.4f}")
        for tl, v in union_variants.items():
            print(f"  union, tau={tl:<12} {v['cumulative_miou']:.4f}   (mean nodes selected: {v['mean_nodes_selected']:.1f})")
        if split_report:
            print(f"\nTune on first {split_report['n_tune_images']} images, report on remaining {split_report['n_heldout_images']} (cumulative mIoU):")
            print(f"  {'':<22} {'tune':>8} {'held-out':>9}")
            print(f"  {'single best node':<22} {split_report['single'][0]:8.4f} {split_report['single'][1]:9.4f}")
            for tl in tau_labels:
                mark = "  <- best on tune set" if tl == split_report["best_tau"] else ""
                print(f"  union, tau={tl:<12} {split_report[tl][0]:8.4f} {split_report[tl][1]:9.4f}{mark}")
            print(f"\n  Honest held-out number: tau={split_report['best_tau']} -> {split_report[split_report['best_tau']][1]:.4f}")
        best_tl = max(union_variants, key=lambda k: union_variants[k]["cumulative_miou"])
        print(f"\nPer-class, single node vs. union tau={best_tl}:")
        for c in sorted(cumulative_per_class, key=lambda c: -cumulative_per_class[c]):
            u = union_variants[best_tl]["cumulative_iou_per_class"].get(c, 0.0)
            print(f"  {c:<14} {cumulative_per_class[c]*100:6.1f} -> {u*100:6.1f}  ({(u-cumulative_per_class[c])*100:+.1f})")

    if not class_level and os.path.exists(V1_SUMMARY):
        with open(V1_SUMMARY) as f:
            v1 = json.load(f)
        print("\n" + "=" * 60)
        print("COMPARISON: v1 (prototype-only) vs v2 (debiasing + localization)")
        print("=" * 60)
        print(f"{'Metric':<20} {'v1':>10} {'v2':>10} {'delta':>10}")
        print(f"{'Mean IoU':<20} {v1['mean_best_iou']*100:>9.2f}% {summary['mean_best_iou']*100:>9.2f}% {(summary['mean_best_iou']-v1['mean_best_iou'])*100:>+9.2f}%")
        print(f"{'Median IoU':<20} {v1['median_best_iou']*100:>9.2f}% {summary['median_best_iou']*100:>9.2f}% {(summary['median_best_iou']-v1['median_best_iou'])*100:>+9.2f}%")

        print(f"\n{'Class':<15} {'v1 IoU':>8} {'v2 IoU':>8} {'delta':>8}")
        v1_classes = v1.get("per_class", {})
        all_class_names = sorted(set(v1_classes.keys()) | set(per_class_summary.keys()))
        for cls in all_class_names:
            v1_iou = v1_classes.get(cls, {}).get("mean_iou", None)
            v2_iou = per_class_summary.get(cls, {}).get("mean_iou", None)
            if v1_iou is not None and v2_iou is not None:
                print(f"{cls:<15} {v1_iou*100:>7.2f}% {v2_iou*100:>7.2f}% {(v2_iou-v1_iou)*100:>+7.2f}%")

        comparison = {
            "v1_mean_iou": v1["mean_best_iou"],
            "v2_mean_iou": summary["mean_best_iou"],
            "delta": summary["mean_best_iou"] - v1["mean_best_iou"],
            "per_class_v1": {k: v.get("mean_iou") for k, v in v1_classes.items()},
            "per_class_v2": {k: v.get("mean_iou") for k, v in per_class_summary.items()},
        }
        with open(os.path.join(OUTPUT_DIR, "v1_vs_v2_comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)

    print(f"\nSaved to: {OUTPUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
