#!/usr/bin/env python3
"""Compare two Fiedler evaluation runs and report differences in IoU metrics and distributions."""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summary(summary_json_path):
    """Load summary.json from an evaluation run."""
    with open(summary_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_scalars(summary_a, summary_b):
    """Extract and compare headline scalar metrics."""
    metrics = [
        "mean_best_iou",
        "median_best_iou",
        "min_best_iou",
        "max_best_iou",
        "mean_best_iou_per_image",
        "median_best_iou_per_image",
        "num_images",
        "num_instances",
    ]

    print("\n=== Scalar Metrics Comparison ===\n")
    print(f"{'Metric':<35} {'Run A':<12} {'Run B':<12} {'Delta':<12} {'% Change':<12}")
    print("-" * 80)

    for metric in metrics:
        val_a = summary_a.get(metric, "N/A")
        val_b = summary_b.get(metric, "N/A")

        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            delta = val_b - val_a
            pct_change = (delta / val_a * 100) if val_a != 0 else 0.0
            print(f"{metric:<35} {val_a:<12.4f} {val_b:<12.4f} {delta:<12.4f} {pct_change:<12.1f}%")
        else:
            print(f"{metric:<35} {str(val_a):<12} {str(val_b):<12} N/A          N/A")


def compare_per_class(summary_a, summary_b):
    """Compare per-class mean IoU statistics."""
    print("\n=== Per-Class Mean IoU Comparison ===\n")
    print(f"{'Class':<30} {'Run A':<12} {'Run B':<12} {'Delta':<12}")
    print("-" * 65)

    per_class_a = summary_a.get("per_class", {})
    per_class_b = summary_b.get("per_class", {})
    all_classes = set(per_class_a.keys()) | set(per_class_b.keys())

    deltas = []
    for class_name in sorted(all_classes):
        stats_a = per_class_a.get(class_name, {})
        stats_b = per_class_b.get(class_name, {})
        mean_a = stats_a.get("mean_iou", "N/A")
        mean_b = stats_b.get("mean_iou", "N/A")

        if isinstance(mean_a, (int, float)) and isinstance(mean_b, (int, float)):
            delta = mean_b - mean_a
            deltas.append((class_name, delta, mean_a, mean_b))
            print(f"{class_name:<30} {mean_a:<12.4f} {mean_b:<12.4f} {delta:<12.4f}")
        else:
            print(f"{class_name:<30} {str(mean_a):<12} {str(mean_b):<12} N/A")

    # Summary: show classes with largest improvements and regressions
    if deltas:
        deltas.sort(key=lambda x: x[1], reverse=True)
        print("\nLargest improvements (Run B vs Run A):")
        for class_name, delta, _, mean_b in deltas[:3]:
            print(f"  +{delta:>7.4f} → {class_name} (new mean: {mean_b:.4f})")

        print("\nLargest regressions (Run B vs Run A):")
        for class_name, delta, _, mean_b in deltas[-3:]:
            print(f"  {delta:>7.4f} → {class_name} (new mean: {mean_b:.4f})")


def extract_all_scores(summary):
    """Extract all per-instance best_iou scores from summary.json."""
    all_scores = []
    for image_summary in summary.get("image_summaries", []):
        # Note: image_summary only has best_instance_iou, not per-instance details.
        # We'd need instance_results.csv for full granularity, but use what we have here.
        # For now, just report that we're working with image-level bests.
        pass

    # Alternative: work directly with per_class score lists if available
    # (the structure stores scores per class in per_class[class_name]["scores"])
    per_class = summary.get("per_class", {})
    for class_name, stats in per_class.items():
        all_scores.extend(stats.get("scores", []))
    return all_scores


def plot_comparison_histogram(summary_a, summary_b, output_dir):
    """Plot overlaid histograms comparing Run A and Run B IoU distributions."""
    all_scores_a = extract_all_scores(summary_a)
    all_scores_b = extract_all_scores(summary_b)

    if not all_scores_a or not all_scores_b:
        print("WARNING: Could not extract score distributions for histogram comparison.")
        return None

    plt.figure(figsize=(10, 6))
    bins = np.linspace(0.0, 1.0, 21)

    plt.hist(all_scores_a, bins=bins, alpha=0.6, label=f"Run A (n={len(all_scores_a)})", color="#4c78a8", edgecolor="white")
    plt.hist(all_scores_b, bins=bins, alpha=0.6, label=f"Run B (n={len(all_scores_b)})", color="#e15759", edgecolor="white")

    plt.xlim(0.0, 1.0)
    plt.xlabel("Best IoU")
    plt.ylabel("Count")
    plt.title("IoU Distribution Comparison")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, "comparison_histogram.png")
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_per_class_delta_bar(summary_a, summary_b, output_dir):
    """Plot per-class mean IoU deltas as a bar chart."""
    per_class_a = summary_a.get("per_class", {})
    per_class_b = summary_b.get("per_class", {})
    all_classes = sorted(set(per_class_a.keys()) | set(per_class_b.keys()))

    if not all_classes:
        return None

    deltas = []
    labels = []
    for class_name in all_classes:
        stats_a = per_class_a.get(class_name, {})
        stats_b = per_class_b.get(class_name, {})
        mean_a = stats_a.get("mean_iou", 0.0)
        mean_b = stats_b.get("mean_iou", 0.0)
        delta = mean_b - mean_a
        deltas.append(delta)
        labels.append(class_name)

    # Sort by delta for visibility
    sorted_indices = np.argsort(deltas)
    deltas = [deltas[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    plt.figure(figsize=(max(10, len(labels) * 0.3), 6))
    colors = ["#e15759" if d < 0 else "#59a14f" for d in deltas]
    plt.bar(range(len(labels)), deltas, color=colors)
    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    plt.xlabel("Object Class")
    plt.ylabel("Mean IoU Delta (Run B - Run A)")
    plt.title("Per-Class Mean IoU Difference")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "per_class_delta.png")
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def write_comparison_report(summary_a, summary_b, output_dir, run_a_name, run_b_name):
    """Write a markdown comparison report."""
    report_path = os.path.join(output_dir, "comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Comparison: {run_a_name} vs {run_b_name}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Run A dataset:** {summary_a.get('dataset_name', 'unknown')}\n")
        f.write(f"- **Run B dataset:** {summary_b.get('dataset_name', 'unknown')}\n")
        f.write(f"- **Run A images:** {summary_a.get('num_images', 'N/A')}\n")
        f.write(f"- **Run B images:** {summary_b.get('num_images', 'N/A')}\n")
        f.write(f"- **Run A instances:** {summary_a.get('num_instances', 'N/A')}\n")
        f.write(f"- **Run B instances:** {summary_b.get('num_instances', 'N/A')}\n\n")

        f.write("## Headline Metrics\n\n")
        f.write(f"| Metric | Run A | Run B | Delta |\n")
        f.write(f"|--------|-------|-------|-------|\n")

        metrics = [
            ("Mean best IoU", "mean_best_iou"),
            ("Median best IoU", "median_best_iou"),
            ("Mean best IoU per image", "mean_best_iou_per_image"),
        ]
        for label, key in metrics:
            val_a = summary_a.get(key, 0.0)
            val_b = summary_b.get(key, 0.0)
            delta = val_b - val_a
            f.write(f"| {label} | {val_a:.4f} | {val_b:.4f} | {delta:+.4f} |\n")

        f.write("\n## Outputs\n\n")
        f.write("- Overlaid IoU histogram: `comparison_histogram.png`\n")
        f.write("- Per-class mean IoU delta bar chart: `per_class_delta.png`\n")

    return report_path


def build_parser():
    parser = argparse.ArgumentParser(description="Compare two Fiedler evaluation run outputs.")
    parser.add_argument("--run-a-dir", type=str, required=True, help="Output directory of Run A")
    parser.add_argument("--run-b-dir", type=str, required=True, help="Output directory of Run B")
    parser.add_argument("--output-dir", type=str, default="output/comparison", help="Output directory for comparison results")
    parser.add_argument("--run-a-name", type=str, default="Run A", help="Label for Run A (default: 'Run A')")
    parser.add_argument("--run-b-name", type=str, default="Run B", help="Label for Run B (default: 'Run B')")
    return parser


def main():
    args = build_parser().parse_args()

    # Load summaries
    summary_a_path = os.path.join(args.run_a_dir, "summary.json")
    summary_b_path = os.path.join(args.run_b_dir, "summary.json")

    if not os.path.exists(summary_a_path):
        raise FileNotFoundError(f"Run A summary not found: {summary_a_path}")
    if not os.path.exists(summary_b_path):
        raise FileNotFoundError(f"Run B summary not found: {summary_b_path}")

    print(f"Loading Run A from {summary_a_path}")
    summary_a = load_summary(summary_a_path)
    print(f"Loading Run B from {summary_b_path}")
    summary_b = load_summary(summary_b_path)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Print comparisons
    compare_scalars(summary_a, summary_b)
    compare_per_class(summary_a, summary_b)

    # Generate plots
    print("\n=== Generating Comparison Plots ===\n")
    histogram_path = plot_comparison_histogram(summary_a, summary_b, args.output_dir)
    if histogram_path:
        print(f"Saved comparison histogram to {histogram_path}")

    delta_bar_path = plot_per_class_delta_bar(summary_a, summary_b, args.output_dir)
    if delta_bar_path:
        print(f"Saved per-class delta bar chart to {delta_bar_path}")

    # Write report
    report_path = write_comparison_report(summary_a, summary_b, args.output_dir, args.run_a_name, args.run_b_name)
    print(f"Saved comparison report to {report_path}")

    print(f"\nAll comparison outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
