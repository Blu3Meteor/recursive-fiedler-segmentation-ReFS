# fiedler_v2_dinov3_1024_stretch_debiastree_union_clref reference-guided recursive Fiedler evaluation (v2)

Selection method: positional debiasing + forward/backward localization + IoU-based node selection (no GT used for selection, GT used only for scoring)

- annotations: D:\Coding\Recursive Segmentation Master's Project\Fiedler Evaluation\data\coco_500_from_1000.json
- reference annotations: D:\Coding\Recursive Segmentation Master's Project\Fiedler Evaluation\data\coco20i\coco20i_reference.json
- image root: D:\Coding\Recursive Segmentation Master's Project\Fiedler Evaluation\datasets\val2017
- svd components (positional basis): 100
- tree features: debiased
- resize mode: stretch
- images: 10
- instances: 21
- mean best IoU: 0.2925
- median best IoU: 0.1770
- mean best IoU per image: 0.4901

- results CSV: instance_results.csv
- summary JSON: summary.json
- overall histogram: overall_iou_histogram.png
- class mean chart: class_mean_iou.png
- class histograms: class_histograms/index.md
- top gallery: top_10_gallery.png
- bottom gallery: bottom_10_gallery.png
