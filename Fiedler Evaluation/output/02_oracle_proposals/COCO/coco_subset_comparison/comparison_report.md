# Comparison: Default (depth=5, patches=36) vs Patch-level (depth=9, patches=6)

## Summary

- **Run A dataset:** coco_subset_default
- **Run B dataset:** coco_subset_patchlevel
- **Run A images:** 300
- **Run B images:** 300
- **Run A instances:** 2147
- **Run B instances:** 2147

## Headline Metrics

| Metric | Run A | Run B | Delta |
|--------|-------|-------|-------|
| Mean best IoU | 0.1909 | 0.2039 | +0.0130 |
| Median best IoU | 0.0371 | 0.0451 | +0.0080 |
| Mean best IoU per image | 0.6193 | 0.6239 | +0.0046 |

## Outputs

- Overlaid IoU histogram: `comparison_histogram.png`
- Per-class mean IoU delta bar chart: `per_class_delta.png`
