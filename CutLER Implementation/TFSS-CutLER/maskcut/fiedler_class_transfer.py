#!/usr/bin/env python3

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
THIRD_PARTY_ROOT = os.path.join(REPO_ROOT, "third_party")
if THIRD_PARTY_ROOT not in sys.path:
    sys.path.insert(0, THIRD_PARTY_ROOT)


def build_backbone(vit_arch, vit_feat, patch_size, pretrain_path=None, dino_version="v1", debiased=False):
    """
    Build backbone for feature extraction.

    Args:
        vit_arch: "base" or "small"
        vit_feat: "k", "q", "v", or "kqv"
        patch_size: 8, 14, or 16
        pretrain_path: custom checkpoint path (overrides default)
        dino_version: "v1", "v2", or "v3"
        debiased: if True and dino_version=="v3", apply positional debiasing

    Returns:
        backbone: feature extractor object
    """
    if dino_version == "v1":
        import dino

        if pretrain_path is not None:
            url = pretrain_path
        elif vit_arch == "base" and patch_size == 8:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif vit_arch == "small" and patch_size == 8:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        else:
            raise ValueError(f"Unsupported backbone config: vit_arch={vit_arch}, patch_size={patch_size}")

        feat_dim = 768 if vit_arch == "base" else 384
        backbone = dino.ViTFeat(url, feat_dim, vit_arch, vit_feat, patch_size)
        backbone.eval()
        return backbone

    elif dino_version == "v2":
        import dinov2_feat

        # Map vit_arch + patch_size to DINOv2 model ID
        if vit_arch == "small" and patch_size == 14:
            model_id = "dinov2_vits14"
        elif vit_arch == "base" and patch_size == 14:
            model_id = "dinov2_vitb14"
        else:
            raise ValueError(f"Unsupported DINOv2 config: vit_arch={vit_arch}, patch_size={patch_size}")

        if pretrain_path is not None:
            model_id = pretrain_path

        backbone = dinov2_feat.DINOv2Feat(model_id, feat_type=vit_feat, device="cuda")
        return backbone

    elif dino_version == "v3":
        import dinov3_feat

        # If custom pretrain_path provided, use it directly
        if pretrain_path is not None:
            model_id = pretrain_path
        else:
            # Map vit_arch + patch_size to DINOv3 model ID
            if vit_arch == "small" and patch_size == 14:
                model_id = "dinov3_vits14_pretrain_lvd1689m"
            elif vit_arch == "small" and patch_size == 16:
                model_id = "dinov3_vits16plus_pretrain_lvd1689m"
            elif vit_arch == "base" and patch_size == 14:
                model_id = "dinov3_vitb14_pretrain_lvd1689m"
            else:
                raise ValueError(f"Unsupported DINOv3 config: vit_arch={vit_arch}, patch_size={patch_size}")

        backbone = dinov3_feat.DINOv3Feat(model_id, feat_type=vit_feat, device="cuda")
        backbone.debiased = debiased
        return backbone

    else:
        raise ValueError(f"Unsupported DINO version: {dino_version}")


def make_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )


def resize_square(image, fixed_size):
    from PIL import Image

    return image.resize((fixed_size, fixed_size), Image.LANCZOS)


def resize_stretch(image, fixed_size):
    """Resize image to fixed_size x fixed_size by stretching -- no aspect-ratio
    preservation, no padding. Every patch in the output grid corresponds to real
    image content (no wasted black-padding patches), at the cost of distorting
    the image's proportions. Metadata uses the same keys as
    resize_with_aspect_ratio for drop-in compatibility with resize_binary_mask
    etc.; pad_left/pad_top are always 0 and resized_w/resized_h are always
    fixed_size (no single uniform 'scale' exists since x/y scale independently,
    so scale_x/scale_y are provided instead).
    """
    from PIL import Image

    w, h = image.size
    metadata = {"orig_w": w, "orig_h": h}

    if fixed_size <= 0:
        metadata["scale"] = 1.0
        metadata["pad_left"] = 0
        metadata["pad_top"] = 0
        return image, metadata

    resized = image.resize((fixed_size, fixed_size), Image.LANCZOS)
    metadata["scale_x"] = fixed_size / w
    metadata["scale_y"] = fixed_size / h
    metadata["pad_left"] = 0
    metadata["pad_top"] = 0
    metadata["resized_w"] = fixed_size
    metadata["resized_h"] = fixed_size
    return resized, metadata


def resize_with_aspect_ratio(image, fixed_size, pad_color=(0, 0, 0)):
    """
    Resize image while preserving aspect ratio via letterboxing.

    Args:
        image: PIL Image
        fixed_size: target square size (0 to preserve original size)
        pad_color: RGB color for padding

    Returns:
        resized_image: PIL Image
        metadata: dict with 'scale', 'pad_left', 'pad_top', 'orig_w', 'orig_h'
    """
    from PIL import Image

    w, h = image.size
    metadata = {"orig_w": w, "orig_h": h}

    if fixed_size <= 0:
        metadata["scale"] = 1.0
        metadata["pad_left"] = 0
        metadata["pad_top"] = 0
        return image, metadata

    aspect = w / h

    # Compute scaled dimensions to fit within fixed_size
    if aspect > 1:  # wider than tall
        new_w = fixed_size
        new_h = int(fixed_size / aspect)
    else:  # taller than wide
        new_h = fixed_size
        new_w = int(fixed_size * aspect)

    # Resize image
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    metadata["scale"] = new_w / w

    # Letterbox with padding
    padded = Image.new("RGB", (fixed_size, fixed_size), pad_color)
    pad_top = (fixed_size - new_h) // 2
    pad_left = (fixed_size - new_w) // 2
    padded.paste(resized, (pad_left, pad_top))

    metadata["pad_left"] = pad_left
    metadata["pad_top"] = pad_top
    metadata["resized_w"] = new_w
    metadata["resized_h"] = new_h

    return padded, metadata


def upsample_mask_with_metadata(mask, metadata):
    """
    Upsample mask back to original image size using resize metadata.

    Args:
        mask: binary mask array (resized_h, resized_w)
        metadata: dict from resize_with_aspect_ratio()

    Returns:
        mask_original: binary mask in original image coordinates
    """
    import numpy as np
    from PIL import Image

    if not metadata or metadata.get("scale", 1.0) == 1.0:
        return mask

    scale = metadata["scale"]
    pad_left = metadata.get("pad_left", 0)
    pad_top = metadata.get("pad_top", 0)
    orig_w = metadata["orig_w"]
    orig_h = metadata["orig_h"]
    resized_w = metadata.get("resized_w", int(orig_w * scale))
    resized_h = metadata.get("resized_h", int(orig_h * scale))

    # Crop padding from mask
    if pad_left > 0 or pad_top > 0:
        mask = mask[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w]

    # Upscale to original size
    if scale != 1.0:
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
        mask = np.asarray(mask_pil) > 128

    return mask


def get_affinity_matrix(feats, tau, eps=1e-5):
    import numpy as np
    from scipy import sparse
    import torch.nn.functional as F

    feats = F.normalize(feats, p=2, dim=0)
    affinity = (feats.transpose(0, 1) @ feats).cpu().numpy().astype(np.float32)
    affinity = np.where(affinity > tau, affinity, eps).astype(np.float32)
    degree_values = np.asarray(affinity.sum(axis=1)).reshape(-1).astype(np.float32)
    affinity = sparse.csr_matrix(affinity)
    degree_matrix = sparse.diags(degree_values, format="csr")
    return affinity, degree_matrix


def second_smallest_eigenvector(affinity, degree_matrix):
    import numpy as np
    from scipy.sparse.linalg import eigsh, ArpackNoConvergence

    laplacian = degree_matrix - affinity
    region_size = laplacian.shape[0]
    # Add convergence bounds: maxiter ~ 5 * matrix size, capped at 2000 for very large matrices
    maxiter = min(5 * region_size, 2000)

    try:
        eigenvalues, eigenvectors = eigsh(laplacian, k=2, M=degree_matrix, which="SM", maxiter=maxiter)
    except ArpackNoConvergence as e:
        # If convergence fails, return None to signal unsolvable region
        return None, None

    order = np.argsort(eigenvalues)
    eigenvec = np.copy(eigenvectors[:, order[0]])
    second_smallest_vec = eigenvectors[:, order[1]]
    return eigenvec, second_smallest_vec


def nth_eigenvector(affinity, degree_matrix, eig_index=1):
    """Generalization of second_smallest_eigenvector: returns the eigenvector at
    `eig_index` in ascending-eigenvalue order of the generalized eigenproblem
    L v = lambda D v (eig_index=0 is the trivial ~constant eigenvector, eig_index=1
    is the classic Fiedler vector used by default everywhere else in this codebase).
    Used only for the eigenvector-choice ablation -- every other call site keeps
    using second_smallest_eigenvector unchanged.
    """
    import numpy as np
    from scipy.sparse.linalg import eigsh, ArpackNoConvergence

    laplacian = degree_matrix - affinity
    region_size = laplacian.shape[0]
    k = eig_index + 1
    if k >= region_size:
        return None, None
    maxiter = min(5 * region_size, 2000)

    try:
        eigenvalues, eigenvectors = eigsh(laplacian, k=k, M=degree_matrix, which="SM", maxiter=maxiter)
    except ArpackNoConvergence:
        return None, None

    order = np.argsort(eigenvalues)
    eigenvec = np.copy(eigenvectors[:, order[0]])
    selected_vec = eigenvectors[:, order[eig_index]]
    return eigenvec, selected_vec


def pca_component_vector(region_features, pca_index=0):
    """Alternative to nth_eigenvector for the splitting-criterion ablation: instead
    of an eigenvector of the patch-affinity graph Laplacian, use a principal
    component of the region's raw patch features. region_features is (C, N_region)
    -- each of the N_region patches is one C-dim sample. Returns an (N_region,)
    vector of each patch's projection onto the pca_index-th principal axis
    (pca_index=0 = top/most-variance component, matching the normal PCA
    convention -- there's no trivial "index 0" to skip here unlike the Laplacian,
    since PCA has no analogue of the all-ones null eigenvector).

    This is a different matrix entirely from the Laplacian: PCA's covariance
    matrix captures variance of raw features across patches (unconstrained by
    which patches are spatially/graph-adjacent), whereas the Laplacian's
    eigenvectors capture smoothness over the affinity graph. Used only for the
    PCA-vs-Laplacian splitting-criterion ablation.
    """
    import numpy as np

    feats = np.asarray(region_features, dtype=np.float32)  # (C, N)
    n_region = feats.shape[1]
    if pca_index >= min(feats.shape[0], n_region) or n_region < 3:
        return None

    mean = feats.mean(axis=1, keepdims=True)
    centered = feats - mean
    # SVD of the (C, N) centered matrix: columns of U are the principal axes in
    # feature space (eigenvectors of the C x C covariance matrix); each patch's
    # projection onto axis k is U[:, k] @ centered[:, i], i.e. (U.T @ centered)[k].
    try:
        u, _, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if pca_index >= u.shape[1]:
        return None

    projections = u[:, pca_index] @ centered  # (N,)
    return projections


def get_salient_areas(second_smallest_vec):
    import numpy as np

    average = np.sum(second_smallest_vec) / len(second_smallest_vec)
    return second_smallest_vec > average


def largest_connected_component(mask):
    import numpy as np
    from scipy import ndimage

    labeled, num_features = ndimage.label(mask.astype(np.uint8))
    if num_features <= 1:
        return mask.astype(np.uint8)
    counts = ndimage.sum(mask, labeled, index=np.arange(1, num_features + 1))
    largest_label = int(np.argmax(counts) + 1)
    return (labeled == largest_label).astype(np.uint8)


def orient_vector(vec):
    import numpy as np

    vec = np.asarray(vec, dtype=np.float32)
    if vec.size == 0:
        return vec
    seed = int(np.argmax(np.abs(vec)))
    if vec.flat[seed] < 0:
        vec = -vec
    return vec


def normalize_vector(vec):
    import numpy as np

    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec


def vector_to_mask(vec):
    import numpy as np

    vec = np.asarray(vec)
    if vec.size == 0:
        return vec
    return (vec > vec.mean()).astype(np.uint8)


def mask_iou(mask_a, mask_b):
    import numpy as np

    mask_a = np.asarray(mask_a).astype(bool)
    mask_b = np.asarray(mask_b).astype(bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection) / float(union)


def resize_mask(mask, size):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)).astype(np.uint8) // 255


def load_folder_groups(dataset_path, max_classes=None, max_images_per_class=None):
    import os

    groups = defaultdict(list)
    class_names = [d for d in sorted(os.listdir(dataset_path)) if os.path.isdir(os.path.join(dataset_path, d))]
    if max_classes is not None:
        class_names = class_names[:max_classes]

    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        image_files = []
        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith((".jpeg", ".jpg", ".png", ".bmp", ".webp")):
                image_files.append(os.path.join(class_dir, file_name))
        if max_images_per_class is not None:
            image_files = image_files[:max_images_per_class]
        for image_path in image_files:
            groups[class_name].append(
                {
                    "sample_id": os.path.splitext(os.path.basename(image_path))[0],
                    "class_id": class_name,
                    "class_name": class_name,
                    "image_path": image_path,
                    "gt_mask": None,
                }
            )
    return groups


def preflight_class_specific_coco_json(coco_json):
    with open(coco_json, "r", encoding="utf-8") as file_handle:
        dataset = json.load(file_handle)

    categories = {cat["id"]: cat["name"] for cat in dataset.get("categories", [])}
    annotation_category_ids = {ann["category_id"] for ann in dataset.get("annotations", [])}
    validate_class_specific_coco_ids(categories, annotation_category_ids)


def load_coco_groups(image_root, coco_json, max_classes=None, max_images_per_class=None, allow_class_agnostic_labels=False):
    import numpy as np

    if not allow_class_agnostic_labels:
        preflight_class_specific_coco_json(coco_json)

    try:
        from pycocotools.coco import COCO
    except ModuleNotFoundError:
        return load_coco_groups_without_masks(
            image_root,
            coco_json,
            max_classes=max_classes,
            max_images_per_class=max_images_per_class,
        )

    coco = COCO(coco_json)
    categories = {cat["id"]: cat["name"] for cat in coco.loadCats(sorted(coco.getCatIds()))}
    category_ids = sorted(categories.keys())
    if max_classes is not None:
        category_ids = category_ids[:max_classes]

    groups = defaultdict(list)
    seen_per_class = defaultdict(int)
    for image_id in sorted(coco.getImgIds()):
        image_info = coco.loadImgs([image_id])[0]
        image_path = os.path.join(image_root, image_info["file_name"])
        ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=category_ids)
        annotations = coco.loadAnns(ann_ids)
        anns_by_category = defaultdict(list)
        for annotation in annotations:
            anns_by_category[annotation["category_id"]].append(annotation)

        for category_id, category_annotations in anns_by_category.items():
            if max_images_per_class is not None and seen_per_class[category_id] >= max_images_per_class:
                continue
            category_name = categories[category_id]
            gt_mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
            for annotation in category_annotations:
                gt_mask = np.maximum(gt_mask, coco.annToMask(annotation).astype(np.uint8))

            groups[category_name].append(
                {
                    "sample_id": f"{image_id}_{category_id}",
                    "image_id": image_id,
                    "class_id": category_id,
                    "class_name": category_name,
                    "image_path": image_path,
                    "gt_mask": gt_mask,
                    "image_size": (image_info["width"], image_info["height"]),
                }
            )
            seen_per_class[category_id] += 1

    return groups


def load_coco_groups_without_masks(image_root, coco_json, max_classes=None, max_images_per_class=None):
    with open(coco_json, "r", encoding="utf-8") as file_handle:
        dataset = json.load(file_handle)

    categories = {cat["id"]: cat["name"] for cat in dataset.get("categories", [])}
    category_ids = sorted(categories.keys())
    if max_classes is not None:
        category_ids = category_ids[:max_classes]
    allowed_category_ids = set(category_ids)

    images = {image["id"]: image for image in dataset.get("images", [])}
    anns_by_image_category = defaultdict(list)
    for annotation in dataset.get("annotations", []):
        category_id = annotation["category_id"]
        if category_id in allowed_category_ids:
            anns_by_image_category[(annotation["image_id"], category_id)].append(annotation)

    groups = defaultdict(list)
    seen_per_class = defaultdict(int)
    for image_id in sorted(images):
        image_info = images[image_id]
        for category_id in category_ids:
            category_annotations = anns_by_image_category.get((image_id, category_id), [])
            if not category_annotations:
                continue
            if max_images_per_class is not None and seen_per_class[category_id] >= max_images_per_class:
                continue
            category_name = categories[category_id]
            groups[category_name].append(
                {
                    "sample_id": f"{image_id}_{category_id}",
                    "image_id": image_id,
                    "class_id": category_id,
                    "class_name": category_name,
                    "image_path": os.path.join(image_root, image_info["file_name"]),
                    "gt_mask": None,
                    "image_size": (image_info["width"], image_info["height"]),
                    "num_annotations": len(category_annotations),
                }
            )
            seen_per_class[category_id] += 1
    return groups


def validate_class_specific_coco_ids(categories, annotation_category_ids):
    if len(annotation_category_ids) == 1 and len(categories) > 1:
        only_category_id = next(iter(annotation_category_ids))
        only_category_name = categories.get(only_category_id, str(only_category_id))
        raise ValueError(
            "This looks like a class-agnostic annotation file: every annotation has "
            f"category_id={only_category_id} ({only_category_name}), but the file still lists "
            f"{len(categories)} category names. That will put bears, cats, signs, etc. under "
            "the first category folder. Use a class-specific COCO annotation file such as "
            "datasets/coco/annotations/instances_val2017.json, or pass "
            "--allow-class-agnostic-labels if you intentionally want class-agnostic grouping."
        )


def pairwise_cosine(vectors):
    import numpy as np

    if len(vectors) < 2:
        return None
    scores = []
    for first, second in combinations(vectors, 2):
        denom = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denom == 0:
            continue
        scores.append(float(np.dot(first, second) / denom))
    if not scores:
        return None
    return float(np.mean(scores))


def colorize_heatmap(array):
    import numpy as np

    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return np.zeros((*array.shape, 3), dtype=np.uint8)
    low = float(np.min(array))
    high = float(np.max(array))
    if high > low:
        scaled = (array - low) / (high - low)
    else:
        scaled = np.zeros_like(array)
    red = (255.0 * scaled).astype(np.uint8)
    green = (255.0 * (1.0 - np.abs(scaled - 0.5) * 2.0)).clip(0, 255).astype(np.uint8)
    blue = (255.0 * (1.0 - scaled)).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    import numpy as np

    image = np.asarray(image).copy()
    mask = np.asarray(mask).astype(bool)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    overlay = image.astype(np.float32)
    overlay[mask] = overlay[mask] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_binary_mask(mask, output_path):
    from PIL import Image
    import numpy as np

    Image.fromarray((np.asarray(mask).astype(np.uint8) * 255)).save(output_path)


def save_triptych(image, heatmap, self_mask, prototype_mask, output_path, title):
    from PIL import Image, ImageDraw

    panels = [
        (image, "image"),
        (heatmap, "fiedler heatmap"),
        (overlay_mask(image, self_mask), "self mask"),
        (overlay_mask(image, prototype_mask), "prototype mask"),
    ]
    pil_panels = [Image.fromarray(panel) for panel, _ in panels]
    widths = [panel.width for panel in pil_panels]
    heights = [panel.height for panel in pil_panels]
    label_height = 28
    canvas = Image.new("RGB", (sum(widths), max(heights) + label_height + 30), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for (panel, panel_title), panel_image in zip(panels, pil_panels):
        canvas.paste(panel_image, (x, label_height + 20))
        draw.text((x + 5, 5), panel_title, fill=(0, 0, 0))
        x += panel_image.width
    draw.text((5, label_height + 2), title, fill=(0, 0, 0))
    canvas.save(output_path)


def save_gallery(records, output_path, title, score_key, top_k=3):
    from PIL import Image, ImageDraw

    ranked = [record for record in records if record.get("status") == "ok" and record.get(score_key) is not None]
    if not ranked:
        return
    ranked = sorted(ranked, key=lambda record: record[score_key])
    lowest = ranked[:top_k]
    highest = list(reversed(ranked[-top_k:]))
    selection = lowest + highest

    panels = []
    for record in selection:
        image = Image.fromarray(overlay_mask(record["image"], record["prototype_mask"]))
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, image.width, 26], fill=(255, 255, 255))
        draw.text((4, 4), f"{record['sample_id']} {score_key}={record[score_key]:.3f}", fill=(0, 0, 0))
        panels.append(image)

    if len(panels) < top_k * 2:
        blank = Image.new("RGB", panels[0].size if panels else (256, 256), (240, 240, 240))
        while len(panels) < top_k * 2:
            panels.append(blank.copy())

    widths = [panel.width for panel in panels]
    heights = [panel.height for panel in panels]
    canvas = Image.new("RGB", (sum(widths[:top_k]), max(heights) * 2), (255, 255, 255))
    x = 0
    for panel in panels[:top_k]:
        canvas.paste(panel, (x, 0))
        x += panel.width
    x = 0
    for panel in panels[top_k:top_k * 2]:
        canvas.paste(panel, (x, max(heights)))
        x += panel.width
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill=(0, 0, 0))
    canvas.save(output_path)


def run_sample(sample, backbone, patch_size, tau, fixed_size, N, cpu):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from torchvision import transforms

    image = Image.open(sample["image_path"]).convert("RGB")
    image = resize_square(image, fixed_size)
    image_np = np.array(image)
    transform = make_transform()
    tensor = transform(image).unsqueeze(0)
    if not cpu:
        tensor = tensor.cuda()

    features = backbone(tensor)[0]
    if features.dim() == 4:
        _, channels, height, width = features.shape
        features = features.reshape(channels, height * width)
        grid_h, grid_w = height, width
    elif features.dim() == 3:
        channels, grid_h, grid_w = features.shape
        features = features.reshape(channels, grid_h * grid_w)
    elif features.dim() == 2:
        channels, num_patches = features.shape
        grid = int(round(num_patches ** 0.5))
        if grid * grid != num_patches:
            raise RuntimeError(f"Cannot infer square patch grid from feature shape: {tuple(features.shape)}")
        grid_h, grid_w = grid, grid
    else:
        raise RuntimeError(f"Unexpected feature shape: {tuple(features.shape)}")

    affinity, degree_matrix = get_affinity_matrix(features, tau)
    _, second_smallest_vec = second_smallest_eigenvector(affinity, degree_matrix)
    bipartition = get_salient_areas(second_smallest_vec)
    bipartition = np.asarray(bipartition).reshape(grid_h, grid_w)
    bipartition = largest_connected_component(bipartition)
    self_mask = np.asarray(Image.fromarray(bipartition * 255).resize((fixed_size, fixed_size), Image.NEAREST)) > 0
    self_mask = self_mask.astype(np.uint8)

    first_vector = orient_vector(second_smallest_vec)
    first_vector = normalize_vector(first_vector)
    heatmap = colorize_heatmap(first_vector.reshape(grid_h, grid_w))
    heatmap = np.asarray(Image.fromarray(heatmap).resize((fixed_size, fixed_size), Image.NEAREST))

    gt_mask = sample.get("gt_mask")
    if gt_mask is not None:
        gt_mask = resize_mask(gt_mask, (fixed_size, fixed_size))

    return {
        "image": image_np,
        "heatmap": heatmap,
        "fiedler_vector": first_vector,
        "self_mask": self_mask,
        "gt_mask": gt_mask,
        "grid_shape": (grid_h, grid_w),
        "self_mask_area_ratio": float(self_mask.mean()),
    }


def analyze_group(class_name, samples, backbone, patch_size, tau, fixed_size, N, cpu, output_dir, save_visuals, save_masks):
    import numpy as np

    class_dir = os.path.join(output_dir, class_name)
    if save_visuals or save_masks:
        os.makedirs(class_dir, exist_ok=True)

    records = []
    vectors = []
    self_masks = []
    gt_masks = []

    for sample in samples:
        try:
            sample_result = run_sample(sample, backbone, patch_size, tau, fixed_size, N, cpu)
        except Exception as exc:
            records.append({
                "sample_id": sample["sample_id"],
                "image_path": sample["image_path"],
                "class_name": class_name,
                "status": "error",
                "error": str(exc),
            })
            continue

        vectors.append(sample_result["fiedler_vector"])
        self_masks.append(sample_result["self_mask"])
        if sample_result["gt_mask"] is not None:
            gt_masks.append(sample_result["gt_mask"])

        records.append({
            "sample_id": sample["sample_id"],
            "image_path": sample["image_path"],
            "class_name": class_name,
            "status": "ok",
            "fiedler_mean": float(np.mean(sample_result["fiedler_vector"])),
            "fiedler_std": float(np.std(sample_result["fiedler_vector"])),
            "self_mask_area_ratio": sample_result["self_mask_area_ratio"],
            "grid_shape": sample_result["grid_shape"],
            "image": sample_result["image"],
            "heatmap": sample_result["heatmap"],
            "self_mask": sample_result["self_mask"],
            "gt_mask": sample_result["gt_mask"],
        })

    class_summary = {
        "class_name": class_name,
        "num_samples": len(samples),
        "num_valid": len(vectors),
        "num_errors": sum(1 for record in records if record.get("status") != "ok"),
        "images": [{key: value for key, value in record.items() if key not in {"image", "heatmap", "self_mask", "gt_mask"}} for record in records],
    }

    if len(vectors) < 1:
        class_summary["status"] = "empty"
        return class_summary

    prototype_vector = normalize_vector(np.mean(np.stack(vectors, axis=0), axis=0))
    first_valid_record = next(record for record in records if record.get("status") == "ok")
    prototype_grid_mask = vector_to_mask(prototype_vector).reshape(first_valid_record["grid_shape"])
    prototype_mask = resize_mask(prototype_grid_mask, (fixed_size, fixed_size))
    proto_to_self_ious = [mask_iou(mask, prototype_mask) for mask in self_masks]
    vector_consistency = pairwise_cosine(vectors)

    class_summary.update(
        {
            "status": "ok",
            "vector_consistency_mean_cosine": vector_consistency,
            "prototype_mask_area_ratio": float(prototype_mask.mean()),
            "mean_self_to_prototype_iou": float(np.mean(proto_to_self_ious)),
            "median_self_to_prototype_iou": float(np.median(proto_to_self_ious)),
            "min_self_to_prototype_iou": float(np.min(proto_to_self_ious)),
            "max_self_to_prototype_iou": float(np.max(proto_to_self_ious)),
        }
    )

    if gt_masks:
        gt_proto_ious = []
        gt_self_ious = []
        for self_mask, gt_mask in zip(self_masks, gt_masks):
            gt_proto_ious.append(mask_iou(prototype_mask, gt_mask))
            gt_self_ious.append(mask_iou(self_mask, gt_mask))
        class_summary.update(
            {
                "mean_self_to_gt_iou": float(np.mean(gt_self_ious)),
                "mean_proto_to_gt_iou": float(np.mean(gt_proto_ious)),
                "median_proto_to_gt_iou": float(np.median(gt_proto_ious)),
            }
        )

    for index, record in enumerate(records):
        if record.get("status") != "ok":
            continue
        record["prototype_mask"] = prototype_mask
        record["prototype_iou"] = mask_iou(record["self_mask"], prototype_mask)
        if record.get("gt_mask") is not None:
            record["gt_iou"] = mask_iou(record["gt_mask"], prototype_mask)
            record["self_gt_iou"] = mask_iou(record["gt_mask"], record["self_mask"])

        if save_masks:
            stem = record["sample_id"]
            save_binary_mask(record["self_mask"], os.path.join(class_dir, f"{stem}_self_mask.png"))
            save_binary_mask(prototype_mask, os.path.join(class_dir, f"{stem}_prototype_mask.png"))

        if save_visuals:
            title = f"{class_name} | {record['sample_id']}"
            if record.get("gt_iou") is not None:
                title += f" | proto_gt_iou={record['gt_iou']:.3f}"
            save_triptych(
                record["image"],
                record["heatmap"],
                record["self_mask"],
                prototype_mask,
                os.path.join(class_dir, f"{record['sample_id']}_triptych.png"),
                title,
            )

    transfer_key = "gt_iou" if any(record.get("gt_iou") is not None for record in records) else "prototype_iou"
    class_summary["transfer_key"] = transfer_key

    ranked = [record for record in records if record.get("status") == "ok" and record.get(transfer_key) is not None]
    if ranked:
        ranked = sorted(ranked, key=lambda record: record[transfer_key])
        class_summary["best_transfer_sample"] = ranked[-1]["sample_id"]
        class_summary["worst_transfer_sample"] = ranked[0]["sample_id"]
        class_summary["best_transfer_score"] = float(ranked[-1][transfer_key])
        class_summary["worst_transfer_score"] = float(ranked[0][transfer_key])

    if save_visuals and ranked:
        save_gallery(
            ranked,
            os.path.join(class_dir, f"{class_name}_best_worst_gallery.png"),
            f"{class_name} transfer gallery",
            transfer_key,
        )

    class_summary["images"] = [
        {key: value for key, value in record.items() if key not in {"image", "heatmap", "self_mask", "gt_mask", "prototype_mask"}}
        for record in records
    ]
    return class_summary


def save_summary_plot(class_summaries, output_dir):
    from PIL import Image, ImageDraw

    valid = [summary for summary in class_summaries if summary.get("status") == "ok"]
    if not valid:
        return

    names = [summary["class_name"] for summary in valid]
    vector_scores = [summary.get("vector_consistency_mean_cosine", 0.0) or 0.0 for summary in valid]
    proto_scores = [summary.get("mean_self_to_prototype_iou", 0.0) or 0.0 for summary in valid]
    gt_scores = [summary.get("mean_proto_to_gt_iou") for summary in valid]

    width_per_class = 90
    canvas_width = max(800, width_per_class * len(names) + 100)
    canvas_height = 900
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    sections = [
        (vector_scores, "Class-wise Fiedler consistency", (47, 111, 219)),
        (proto_scores, "Prototype-to-self-mask agreement", (217, 79, 112)),
        ([0.0 if score is None else score for score in gt_scores] if any(score is not None for score in gt_scores) else None, "Prototype-to-ground-truth agreement", (45, 157, 116)),
    ]

    for section_index, (scores, label, color) in enumerate(sections):
        top = 40 + section_index * 280
        draw.text((20, top - 25), label, fill=(0, 0, 0))
        draw.line((20, top + 200, canvas_width - 20, top + 200), fill=(220, 220, 220), width=1)
        if scores is None:
            draw.text((20, top + 80), "No ground-truth annotations available", fill=(100, 100, 100))
            continue
        for idx, score in enumerate(scores):
            x0 = 40 + idx * width_per_class
            bar_height = int(180 * max(0.0, min(1.0, float(score))))
            y0 = top + 200 - bar_height
            draw.rectangle((x0, y0, x0 + 50, top + 200), fill=color)
            draw.text((x0, top + 205), names[idx][:12], fill=(0, 0, 0))
            draw.text((x0, y0 - 15), f"{float(score):.2f}", fill=(0, 0, 0))

    canvas.save(os.path.join(output_dir, "class_metrics_summary.png"))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze class-wise Fiedler vectors and test prototype-mask transfer across same-class images."
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Image folder root for folder-based datasets")
    parser.add_argument("--coco-json", type=str, default=None, help="COCO annotations file for labeled experiments")
    parser.add_argument("--image-root", type=str, default=None, help="Image root for COCO annotations")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for summaries and visualizations")
    parser.add_argument("--vit-arch", type=str, default="small", choices=["base", "small"], help="Which architecture")
    parser.add_argument("--vit-feat", type=str, default="k", choices=["k", "q", "v", "kqv"], help="Which features")
    parser.add_argument("--patch-size", type=int, default=8, choices=[16, 8], help="Patch size")
    parser.add_argument("--tau", type=float, default=0.15, help="Threshold used for graph construction")
    parser.add_argument("--fixed-size", type=int, default=480, help="Resize images before feature extraction")
    parser.add_argument("--pretrain-path", type=str, default=None, help="Path to pretrained DINO weights")
    parser.add_argument("--N", type=int, default=1, help="Number of pseudo-masks per image")
    parser.add_argument("--cpu", action="store_true", help="Use CPU")
    parser.add_argument("--max-classes", type=int, default=None, help="Limit the number of classes to process")
    parser.add_argument("--max-images-per-class", type=int, default=8, help="Limit the number of images per class")
    parser.add_argument(
        "--allow-class-agnostic-labels",
        action="store_true",
        help="Allow COCO annotations where all objects share one category id. Use only for class-agnostic analysis.",
    )
    parser.add_argument("--save-visuals", action="store_true", help="Save heatmaps, overlays, and galleries")
    parser.add_argument("--save-masks", action="store_true", help="Save binary self/prototype masks")
    args = parser.parse_args()

    if args.coco_json and not args.image_root:
        raise ValueError("--image-root is required when --coco-json is set")
    if not args.coco_json and not args.dataset_path:
        raise ValueError("Provide either --dataset-path or --coco-json/--image-root")

    if args.coco_json:
        groups = load_coco_groups(
            args.image_root,
            args.coco_json,
            max_classes=args.max_classes,
            max_images_per_class=args.max_images_per_class,
            allow_class_agnostic_labels=args.allow_class_agnostic_labels,
        )
        source = {
            "type": "coco",
            "coco_json": os.path.abspath(args.coco_json),
            "image_root": os.path.abspath(args.image_root),
        }
    else:
        groups = load_folder_groups(
            args.dataset_path,
            max_classes=args.max_classes,
            max_images_per_class=args.max_images_per_class,
        )
        source = {
            "type": "folders",
            "dataset_path": os.path.abspath(args.dataset_path),
        }

    os.makedirs(args.out_dir, exist_ok=True)
    backbone = build_backbone(args.vit_arch, args.vit_feat, args.patch_size, args.pretrain_path)
    if not args.cpu:
        backbone.cuda()

    summary = {
        "source": source,
        "fixed_size": args.fixed_size,
        "patch_size": args.patch_size,
        "tau": args.tau,
        "N": args.N,
        "classes": [],
    }

    for class_name in sorted(groups.keys()):
        class_summary = analyze_group(
            class_name=class_name,
            samples=groups[class_name],
            backbone=backbone,
            patch_size=args.patch_size,
            tau=args.tau,
            fixed_size=args.fixed_size,
            N=args.N,
            cpu=args.cpu,
            output_dir=args.out_dir,
            save_visuals=args.save_visuals,
            save_masks=args.save_masks,
        )
        summary["classes"].append(class_summary)
        message = f"{class_name}: {class_summary['num_valid']}/{class_summary['num_samples']} valid"
        if class_summary.get("status") == "ok":
            message += f", mean cosine={class_summary.get('vector_consistency_mean_cosine', 0.0):.3f}, proto IoU={class_summary.get('mean_self_to_prototype_iou', 0.0):.3f}"
            if class_summary.get("mean_proto_to_gt_iou") is not None:
                message += f", gt IoU={class_summary.get('mean_proto_to_gt_iou', 0.0):.3f}"
        print(message)

    summary_path = os.path.join(args.out_dir, "fiedler_class_transfer_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2)

    save_summary_plot(summary["classes"], args.out_dir)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
