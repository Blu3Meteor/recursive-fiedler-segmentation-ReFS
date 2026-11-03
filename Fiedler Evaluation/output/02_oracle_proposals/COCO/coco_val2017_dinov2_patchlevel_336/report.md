# COCO val2017 DINOv2 Patch-Level Recursive Fiedler Evaluation

## Configuration
- Dataset: COCO val2017
- Model: DINOv2-small
- Fixed size: 336
- Patch size: 14
- Max depth: 9
- Max nodes: 255
- Min region patches: 6

## Results
- Images: 4952
- Instances: 36335
- Mean best IoU: 0.1533
- Median best IoU: 0.0349
- Min best IoU: 0.0000
- Max best IoU: 0.9949
- Mean best IoU per image: 0.5220
- Median best IoU per image: 0.5672

## Outputs
- Results CSV: instance_results.csv
- Summary JSON: summary.json
- Overall histogram: overall_iou_histogram.png
- Class histogram: class_mean_iou.png
- Top 10 gallery: top_10_gallery.png
- Bottom 10 gallery: bottom_10_gallery.png
