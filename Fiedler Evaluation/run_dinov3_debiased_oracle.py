#!/usr/bin/env python3
"""Oracle evaluation with a positionally debiased, patch-14 DINOv3 backbone.

Runs fiedler_candidate_eval.py unchanged, with its build_backbone swapped for a DINOv3
wrapper that:
  1. resizes the ViT-S/16 patch embedding to 14x14 with the FlexiViT pseudo-inverse
     resize (DINOv3 ships patch-16 weights only), so the patch grid matches DINOv2/14;
  2. projects the positional subspace out of every patch feature BEFORE the Fiedler
     tree is built (blank-image basis, 100 components, re-L2-normalised -- the same
     build_positional_basis / debias_and_normalize as fiedler_reference_eval_v2.py).

The existing --debiased flag is not used: DINOv3Feat never computes its positional
basis, so that flag leaves features unchanged.

All other arguments are passed straight through to fiedler_candidate_eval.py, e.g. to
match 07_resize_ablation/stretch_500 (DINOv2):
  --patch-size 14 --tau 0 --fixed-size 1022 --resize-mode stretch --eval-mode class
  --min-region-patches 5 --unlimited-depth --unlimited-nodes --unlimited-noise
  --unlimited-imbalance
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_V3_WEIGHTS = os.path.join(WORKSPACE_ROOT, "dinov3_repo", "dinov3_weights", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import fiedler_candidate_eval  # noqa: E402  (sets up the maskcut import paths)
from fiedler_class_transfer import build_backbone  # noqa: E402
from fiedler_reference_eval_v2 import build_positional_basis, debias_and_normalize  # noqa: E402

N_DEBIAS_COMPONENTS = 100


def pi_resize_patch_embed(backbone, new_patch):
    """FlexiViT PI-resize: w_new = pinv(R) w_old, so <x, w_old> ~= <resize(x), w_new>."""
    model = backbone.model
    proj = model.patch_embed.proj
    old_patch = proj.kernel_size[0]
    basis = torch.eye(old_patch * old_patch).reshape(-1, 1, old_patch, old_patch)
    R = F.interpolate(basis, size=(new_patch, new_patch), mode="bilinear", align_corners=False, antialias=True)
    R = R.reshape(old_patch * old_patch, new_patch * new_patch)
    pinv = torch.linalg.pinv(R.double())
    w = proj.weight.data.double().cpu()
    out_ch, in_ch = w.shape[:2]
    w_new = (pinv @ w.reshape(out_ch * in_ch, -1).T).T.reshape(out_ch, in_ch, new_patch, new_patch)
    new_proj = nn.Conv2d(in_ch, out_ch, kernel_size=new_patch, stride=new_patch).to(proj.weight.device)
    new_proj.weight.data = w_new.float().to(proj.weight.device)
    new_proj.bias.data = proj.bias.data.clone()
    model.patch_embed.proj = new_proj
    model.patch_embed.patch_size = (new_patch, new_patch)
    model.patch_size = new_patch
    backbone.patch_size = new_patch
    return backbone


class DebiasedBackbone:
    """Callable with the DINOv3Feat interface; returns debiased (N, C) features."""

    def __init__(self, inner, basis):
        self.inner = inner
        self.basis = basis

    def __call__(self, tensor):
        features, grid = self.inner(tensor)  # (N, C)
        deb = debias_and_normalize(features.float().cpu().numpy().T, self.basis)  # (C, N)
        return torch.from_numpy(np.ascontiguousarray(deb.T)), grid


def make_build_backbone(fixed_size):
    def build(vit_arch, vit_feat, patch_size, pretrain_path=None, dino_version="v1", debiased=False):
        if dino_version != "v3":
            raise ValueError("run_dinov3_debiased_oracle.py is for --dino-version v3 only")
        inner = build_backbone(vit_arch, vit_feat, 16, pretrain_path or DEFAULT_V3_WEIGHTS, dino_version="v3")
        if patch_size != 16:
            inner = pi_resize_patch_embed(inner, patch_size)
        basis = build_positional_basis(inner, fixed_size, False, N_DEBIAS_COMPONENTS)
        print(f"DINOv3 patch {inner.patch_size}, positional basis {basis.shape} removed before tree building", flush=True)
        return DebiasedBackbone(inner, basis)
    return build


def main():
    args = fiedler_candidate_eval.build_parser().parse_args()
    fiedler_candidate_eval.build_backbone = make_build_backbone(args.fixed_size)
    fiedler_candidate_eval.evaluate_dataset(args)


if __name__ == "__main__":
    main()
