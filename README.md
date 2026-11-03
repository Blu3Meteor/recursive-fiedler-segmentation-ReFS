# Recursive Fiedler Segmentation (Master's Project)

Training-free segmentation by recursively splitting an image with the Fiedler vector
(second eigenvector of the graph Laplacian) over self-supervised ViT patch features,
then picking the right region with a reference image instead of ground truth.

This README is a map of the repo: where the code lives, which output folder belongs to
which experiment, how to set it up (external repos, weights, datasets), and how to rerun
the main pipeline.

---

## 1. Repo layout

```
Recursive Segmentation Master's Project/
├── README.md                     ← this guide
├── LICENSE                       licence for this project's own code (see §9)
├── requirements.txt              Python environment (see §5)
├── patches/                      local change to INSID3, applied after cloning (see §5)
├── CutLER Implementation/        CutLER / MaskCut reimplementation (experiment 1)
│   └── TFSS-CutLER/
│       ├── maskcut/              ★ core Fiedler code lives here (see §2)
│       ├── datasets/coco/        COCO val2017 images + annotations       [not in git]
│       └── detectron2/           upstream detectron2, CutLER only         [not in git]
├── Fiedler Evaluation/           all evaluation scripts, data splits, and results
│   ├── *.py                      evaluation / visualisation scripts (see §3)
│   ├── data/                     COCO-20i splits and small curated subsets
│   ├── datasets/                 class-level annotation JSONs, plus PASCAL VOC 2012 and
│   │                             a COCO val2017 copy                      [images not in git]
│   ├── output/                   results, grouped by experiment (see §4)
│   └── Ref Papers/               INSID3, "Deep ViT Features" papers       [not in git, see §8]
├── INSID3/                       upstream INSID3 repo (baseline)          [not in git]
├── dinov3_repo/                  upstream DINOv3 repo + local weights     [not in git]
├── presentation_figures/         presentation figures + the script that draws them
├── logs/                         console logs from long runs
├── *.py (root)                   small one-off helpers (dataset checks, COCO-20i filtering)
└── .venv/                        Python environment                       [not in git]
```

Folders marked `[not in git]` are listed in `.gitignore`. §5 says where to get each one.

## 2. Core code

The algorithm itself is not in `Fiedler Evaluation/`. It lives next to MaskCut so it can
reuse the DINO loaders:

| File (in `CutLER Implementation/TFSS-CutLER/maskcut/`) | What it does |
|---|---|
| `recursive_fiedler.py` | Builds the recursive split tree (`collect_recursive_candidates`), feature loading, mask upsampling |
| `fiedler_class_transfer.py` | Backbone construction (`build_backbone`), resizing, overlay helpers |
| `dinov2_feat.py`, `dinov3_feat.py` | DINOv2 / DINOv3 patch-feature extractors |

Every evaluation script adds `TFSS-CutLER` and `maskcut/` to `sys.path` itself, so run
scripts **from the project root**.

## 3. Main scripts (in `Fiedler Evaluation/`)

| Script | Purpose |
|---|---|
| `fiedler_candidate_eval.py` | **Oracle evaluation.** Builds the tree, scores every node against GT, reports best IoU per instance/class. |
| `fiedler_reference_eval.py` | Reference matching v1: picks the node with highest cosine similarity to a reference prototype. |
| `fiedler_reference_eval_v2.py` | **Current pipeline.** Positional debiasing + forward/backward localisation + IoU-based node selection; optional union of nodes (`--union-taus`). GT is only used to score. |
| `batch_run_reference_v2.py` | Runs v2 in resumable batches of 25 images (writes `_batch_progress.json`). Use this for long runs. |
| `finalize_reference_v2_report.py` | Rebuilds plots/report for a finished v2 run. |
| `insid3_reference_eval.py`, `insid3_zeroshot_eval*.py` | INSID3 baseline runs. |
| `extract_backbone_features.py`, `visualize_backbone_features.py` | DINOv1/v2/v3 feature comparison. |
| `visualize_class_vs_instance.py`, `visualize_class_group_hierarchy.py` | Qualitative class-vs-instance and semantic-vs-spatial visualisations. |
| `create_coco20i_split.py`, `create_class_level_annotations.py`, `pascal_to_coco.py` | Dataset preparation. |

Scripts named `generate_*`, `identify_failed_*`, `retry_failed_images.py`,
`merge_and_finalize*.py`, `debug_*`, `test_*` are one-off helpers from earlier runs.

## 4. Output folders ↔ experiments

All results are in `Fiedler Evaluation/output/`, numbered in the order the experiments
were run. Each run folder contains `summary.json`, `report.md`, `instance_results.csv`
(or `class_level_results.csv`), histograms, and top/bottom-10 galleries.

In git, each run folder keeps its text results and small plots (`class_mean_iou.png`,
`overall_iou_histogram.png`, `class_histograms/`). Galleries and other large images are
in the results release (see "Results archive" below). The per-image candidate trees
(`candidate_visualizations/`, about 2.2 GB per oracle run) are not published; rerun
`fiedler_candidate_eval.py` with `--visualize-candidates` to regenerate them.

| Folder | Experiment | Notable runs |
|---|---|---|
| `01_backbone_comparison/` | DINOv1 vs v2 vs v3 features | `backbone_comparison_features/` |
| `02_oracle_proposals/COCO/`, `.../PASCAL/` | Recursive Fiedler scored against GT (best node per object). Covers backbone choice, resolution, threshold = 0 / min-patches-only, and instance vs class level. | `coco20i_dinov2_min_patches_only` (class level), `coco_20i_dinov2_min_patches_only_instance_complete` (instance level), `coco20i_dinov3_vitl16_pretrain`, `pascal_voc_dinov2_*` |
| `03_class_vs_instance/` | Qualitative class-level vs instance-level comparison | `class_vs_instance_viz/` |
| `04_reference_matching/` | GT replaced by a reference image (forward + backward pass) | `fiedler_reference_guided_500` (v1), `fiedler_reference_guided_v2_1022_classlevel`, `fiedler_v2_1022_union_clref_hi`, `fiedler_v2_1022_stretch_union_clref`, `fiedler_v2_dinov3_1024_union_clref`, `fiedler_v2_dinov3_1024_stretch_debiastree_union_clref` (debiased tree), `fiedler_v2_1022_bilinear_clref`, `combo_check_100` |
| `05_reference_vs_gt_and_semantics/` | Reference-selected mask vs best-possible mask, and whether splits are semantic or feature/spatial | `fiedler_v2_1022_candiou_check/`, `class_group_hierarchy_viz/` |
| `06_alternate_dims/` | Split on a different Laplacian eigenvector or PCA component instead of the Fiedler vector (oracle, 100 images) | `eig_ablation/eig0`–`eig4` (`eig1` = Fiedler), `pca_ablation/pca0`–`pca4`, `pca_vs_fiedler_viz/`, `eigenvector_ablation_comparison.csv` |
| `07_resize_ablation/` | Stretch vs letterbox resizing (class-level oracle) | `stretch` (100 images), `stretch_500` (DINOv2), `stretch_500_dinov3_p14_debiased` (DINOv3, debiased tree) |
| `baselines_insid3/` | INSID3 zero-shot and reference-guided baselines | `coco_500_insid3_zeroshot`, `insid3_lvd_small_clref` |
| `figures/` | Figures that combine several runs (thesis/presentation) | `four_rows_*`, `union_of_nodes_schematic*`, `ordering_examples/` |
| `_debug/` | Debug images | |

`output/_move_manifest.tsv` lists the old and new path of every folder moved during the
reorganisations on 2026-10-08 and 2026-11-03. Paths written inside older `summary.json`
files and logs still show the pre-reorganisation locations.

### Results archive

The images that are not in git are attached to the
[`results-v1` release](https://github.com/Blu3Meteor/recursive-fiedler-segmentation-ReFS/releases/tag/results-v1).
Each zip holds repo-relative paths, so unzipping it at the repo root puts every file back
in its run folder (next to the `report.md` that links to it).

| Asset | Contents | Size |
|---|---|---|
| `output-01-03.zip` | galleries for `01_backbone_comparison/` to `03_class_vs_instance/` | 582 MB |
| `output-04-05.zip` | galleries for `04_reference_matching/` and `05_reference_vs_gt_and_semantics/` | 397 MB |
| `output-06-07-baselines-figures.zip` | galleries for `06_alternate_dims/`, `07_resize_ablation/` and `baselines_insid3/`, plus `figures/ordering_examples/` and `_debug/` | 649 MB |
| `presentation-qualitative.zip` | `presentation_figures/qualitative*/` | 120 MB |
| `cutler-outputs.zip` | `CutLER Implementation/TFSS-CutLER/output/`: CutLER inference on COCO val2017 and the first Fiedler test runs | 241 MB |

### Headline numbers (COCO-20i, DINOv2 ViT-S/14, from each run's `summary.json`)

Metrics differ between runs, so compare within a column, not across.

| Run | Selection | Metric | Value |
|---|---|---|---|
| `02_oracle_proposals/COCO/coco20i_dinov2_min_patches_only` | GT oracle, class level | mean best IoU | 0.459 |
| `02_oracle_proposals/COCO/coco_20i_dinov2_min_patches_only_instance_complete` | GT oracle, instance level | mean best IoU | 0.271 |
| `04_reference_matching/fiedler_reference_guided_500` | Reference v1 | mean IoU | 0.081 |
| `04_reference_matching/fiedler_reference_guided_v2_1022_classlevel` | Reference v2, class level | mean IoU | 0.191 |
| `04_reference_matching/fiedler_v2_1022_union_clref_hi` | Reference v2, union of nodes (τ = 0.5) | cumulative mIoU | 0.332 |
| `04_reference_matching/fiedler_v2_dinov3_1024_union_clref` | Same, DINOv3 | cumulative mIoU | 0.305 |
| `baselines_insid3/coco_500_insid3_zeroshot` | INSID3 zero-shot, GT oracle | mean best IoU | 0.163 |

## 5. Setup

Run everything from the repo root (PowerShell).

### 5.1 Python environment

Python 3.13, CUDA 11.8 build of PyTorch (the extra index is in `requirements.txt`):

```powershell
git clone https://github.com/Blu3Meteor/recursive-fiedler-segmentation-ReFS.git
cd recursive-fiedler-segmentation-ReFS
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` covers the Fiedler pipeline and the INSID3 baseline. CutLER training
and inference (experiment 1) use `CutLER Implementation/TFSS-CutLER/requirements.txt`
plus detectron2 instead.

### 5.2 External repos (pinned to the commits used for the thesis)

INSID3 and DINOv3 must sit at the repo root under these folder names. The code imports
the DINOv3 package from `dinov3_repo/dinov3_weights/` (see `maskcut/dinov3_feat.py`),
and the patch makes INSID3 load DINOv3 from the same folder.

```powershell
git clone https://github.com/visinf/INSID3.git INSID3
git -C INSID3 checkout 0c165a1
git -C INSID3 apply ../patches/insid3-load-local-dinov3.patch

git clone https://github.com/facebookresearch/dinov3.git dinov3_repo
git -C dinov3_repo checkout 6876159
Rename-Item dinov3_repo\dinov3 dinov3_weights
```

Only for CutLER training/inference (experiment 1), also clone detectron2 and follow
`CutLER Implementation/TFSS-CutLER/INSTALL.md`:

```powershell
git clone https://github.com/facebookresearch/detectron2.git "CutLER Implementation\TFSS-CutLER\detectron2"
git -C "CutLER Implementation\TFSS-CutLER\detectron2" checkout e0ec4e1
```

### 5.3 Weights

| Backbone | How to get it |
|---|---|
| DINOv3 | Request access on Meta's [DINOv3 downloads page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/), then put the `.pth` files in `dinov3_repo/dinov3_weights/`. Required: `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` (ViT-S/16, also used by INSID3 small). Only for the ViT-L runs: `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`, `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`. Pass the file with `--pretrain-path`. |
| DINOv2 | Downloaded automatically through `torch.hub` (`facebookresearch/dinov2`) on first use. |
| DINO v1 | Downloaded automatically from `dl.fbaipublicfiles.com` when no `--pretrain-path` is given. |
| CutLER | Checkpoint links are in `CutLER Implementation/TFSS-CutLER/README.md` (experiment 1 only). |

### 5.4 Datasets

| Dataset | Download | Put it in |
|---|---|---|
| COCO 2017 val images | [val2017.zip](http://images.cocodataset.org/zips/val2017.zip) | `CutLER Implementation/TFSS-CutLER/datasets/coco/val2017/`, and a copy in `Fiedler Evaluation/datasets/val2017/` (the ablation runs in `06_` and `07_` read from there) |
| COCO 2017 annotations | [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) | `instances_val2017.json` into `CutLER Implementation/TFSS-CutLER/datasets/coco/annotations/` |
| COCO val, class-agnostic (CutLER) | [coco_cls_agnostic_instances_val2017.json](http://dl.fbaipublicfiles.com/cutler/coco/coco_cls_agnostic_instances_val2017.json) | same `annotations/` folder |
| PASCAL VOC 2012 trainval | [VOCtrainval_11-May-2012.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | extract to `Fiedler Evaluation/datasets/VOCtrainval_11-May-2012/` |
| ADE20K (experiment 1 only) | [ade20k.csail.mit.edu](https://ade20k.csail.mit.edu/) (registration) | `CutLER Implementation/TFSS-CutLER/datasets/ade_20k/{training,validation}/{img,ann}/` |

The derived files are already in git: the COCO-20i split, reference set and image subsets
in `Fiedler Evaluation/data/`, and the class-level COCO-20i / PASCAL VOC annotations in
`Fiedler Evaluation/datasets/*.json`.

## 6. How to run

Activate the environment and run from the project root:

```powershell
cd "D:\Coding\Recursive Segmentation Master's Project"
.\.venv\Scripts\Activate.ps1
```

**Oracle evaluation** (best node vs GT):

```powershell
python "Fiedler Evaluation\fiedler_candidate_eval.py" `
  --dataset-name coco20i_oracle `
  --image-root "CutLER Implementation\TFSS-CutLER\datasets\coco\val2017" `
  --annotations "Fiedler Evaluation\datasets\coco20i_val_class_level.json" `
  --output-dir "Fiedler Evaluation\output\02_oracle_proposals\COCO\<run_name>" `
  --dino-version v2 --patch-size 14 --tau 0 --fixed-size 476
```

**Reference-guided evaluation (current pipeline), batched and resumable:**

```powershell
python "Fiedler Evaluation\batch_run_reference_v2.py" `
  --dataset-name <run_name> `
  --image-root "CutLER Implementation\TFSS-CutLER\datasets\coco\val2017" `
  --annotations "Fiedler Evaluation\data\coco_500_from_1000.json" `
  --reference-image-root "CutLER Implementation\TFSS-CutLER\datasets\coco\val2017" `
  --reference-annotations "Fiedler Evaluation\data\coco20i\coco20i_reference.json" `
  --output-dir "Fiedler Evaluation\output\04_reference_matching\<run_name>" `
  --total-images 410 --dino-version v2 --fixed-size 1022 `
  --class-level --class-level-reference --union-taus 0.5
```

If a batch run stops, rerun the same command. It skips finished batches using
`_batch_progress.json`. Use `--pretrain-path` with a local checkpoint when
`--dino-version v3` is used.

The exact flags of past runs were not saved in their `summary.json`. The commands above
are built from each script's argument list and the run folder names, so check them
against `python <script> --help` before reproducing a specific result.

### Conventions for new runs

- Put each new run under the numbered folder of the experiment it belongs to, or start a
  new numbered folder (next free: `08_<experiment>/`).
- Name runs `<method>_<backbone>_<resolution>_<variant>`, e.g. `fiedler_v2_1022_union_clref_hi`.
- Send console output to `logs/<run_name>.log`.
- Images under `Fiedler Evaluation/output/` are git-ignored except the small per-run
  plots. Ship new galleries by attaching a zip to a new release.

## 7. Experiment history

1. Implemented CutLER / MaskCut.
2. Implemented recursive Fiedler splitting and compared DINOv1, v2 and v3. DINOv2 was best.
3. Tested threshold = 0 with only a minimum-patch stopping rule.
4. Moved from instance-level to class-level evaluation.
5. Replaced the GT oracle with reference matching (forward and backward pass).
6. Compared the reference-selected mask's IoU with GT, and tested whether the splits are
   semantic or follow feature/spatial similarity.

## 8. Papers

- INSID3: Training-Free In-Context Segmentation with DINOv3 ([arXiv:2603.28480](https://arxiv.org/abs/2603.28480))
- Deep ViT Features as Dense Visual Descriptors ([arXiv:2112.05814](https://arxiv.org/abs/2112.05814))

## 9. Licences

- **This project's own code** (`Fiedler Evaluation/`, root scripts, `presentation_figures/`):
  MIT, see `LICENSE`.
- **`CutLER Implementation/TFSS-CutLER/`**: CC BY-NC-SA 4.0, see the `LICENSE` file in
  that folder. This includes the Fiedler code added to `maskcut/`, which builds on MaskCut.
  The files in `third_party/TokenCut/` come from
  [TokenCut](https://github.com/YangtaoWANG95/TokenCut) and keep its licence.
- **Data**: the COCO annotations, and the splits derived from them, are CC BY 4.0
  (COCO Consortium). COCO and PASCAL VOC images, including the samples in
  `Fiedler Evaluation/data/` and the images in the release galleries, remain under their
  original Flickr licences.
- **Not redistributed**: DINOv3 code and weights (DINOv3 License), INSID3 (Apache 2.0) and
  detectron2 (Apache 2.0) are cloned or downloaded separately (§5).
