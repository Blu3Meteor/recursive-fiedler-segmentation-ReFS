# Fiedler Candidate Evaluation

> See the project-level `README.md` one folder up for the full repo guide and the map of `output/` folders by experiment.

This folder contains a standalone harness for the first test we discussed: recursively splitting images with the Fiedler-vector splitter, then scoring every candidate region mask against every COCO-format ground-truth instance mask.

The current scoring rule is instance-level:
- run the recursive splitter once per image
- collect every candidate region mask from the recursive tree
- resize each GT instance mask to the same square input size used by the splitter
- report the best IoU per GT instance
- export the best and worst examples plus class-wise IoU histograms

## Recommended environment

Use the `.venv` environment from `requirements.txt`; see "Setup" (§5) in the project-level `README.md`.

## Example usage

COCO val2017:

```powershell
.\.venv\Scripts\Activate.ps1
python "d:\Coding\Recursive Segmentation Master's Project\Fiedler Evaluation\fiedler_candidate_eval.py" ^
  --dataset-name coco_val2017 ^
  --image-root "d:\Coding\Recursive Segmentation Master's Project\CutLER Implementation\TFSS-CutLER\datasets\coco\val2017" ^
  --annotations "d:\Coding\Recursive Segmentation Master's Project\CutLER Implementation\TFSS-CutLER\datasets\coco\annotations\instances_val2017.json" ^
  --output-dir "d:\Coding\Recursive Segmentation Master's Project\Fiedler Evaluation\output\coco_val2017" ^
  --vit-arch small --patch-size 8 --tau 0.15 --fixed-size 480
```

COCO20K, PASCAL VOC, and PACO Part can be evaluated the same way as long as they are stored in COCO format: point `--image-root` and `--annotations` at the dataset directory and JSON file you want to score.

If you do not pass `--pretrain-path`, the DINO backbone loader will use its default pretrained weights source.

## Outputs

The script writes:
- `instance_results.csv`
- `summary.json`
- `report.md`
- `overall_iou_histogram.png`
- `class_mean_iou.png`
- `class_histograms/`
- `top_10_gallery.png`
- `bottom_10_gallery.png`

The class histogram directory also contains an `index.md` file.
