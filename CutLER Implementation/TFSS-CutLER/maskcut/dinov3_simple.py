#!/usr/bin/env python3
"""Simplified DINOv3 feature extraction without full module imports."""

import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision.transforms import Normalize
from PIL import Image, ImageOps


class SimpleDINOv3:
    """Simplified DINOv3 feature extractor using timm or torch hub."""

    def __init__(self, model_id="dinov3_vits16", weights_path=None, device="cuda"):
        """
        Args:
            model_id: Model name (e.g., "dinov3_vits16", "dinov3_vitb14")
            weights_path: Path to checkpoint file
            device: "cuda" or "cpu"
        """
        self.device = device
        self.model_id = model_id
        self.weights_loaded = False

        self.model = None

        # Try to load via torch.hub
        print(f"Loading {model_id} from torch.hub...")
        try:
            self.model = torch.hub.load('facebookresearch/dinov3:main', model_id)
            self.weights_loaded = True
            print(f"Successfully loaded {model_id} from torch.hub")
        except Exception as e:
            print(f"Failed to load from torch.hub: {e}")

        # If torch.hub failed, try loading from local checkpoint
        if self.model is None and weights_path and Path(weights_path).exists():
            print(f"Attempting to load from local checkpoint: {weights_path}")
            try:
                checkpoint = torch.load(weights_path, map_location="cpu")
                print(f"Loaded checkpoint, building model from scratch...")
                # For now, just load the state dict as a simple model wrapper
                # This is a fallback - ideally we'd construct the proper model
                self.model = checkpoint
                self.weights_loaded = True
                print(f"Loaded weights from {weights_path}")
            except Exception as e2:
                print(f"Failed to load checkpoint: {e2}")
                print(f"ERROR: Could not load model")
                raise

        if self.model is None:
            raise RuntimeError(f"Failed to load DINOv3 model {model_id}")

        self.model = self.model.to(device)
        self.model.eval()

        # Feature dimension and patch size
        self.feat_dim = self.model.embed_dim
        self.patch_size = self.model.patch_embed.patch_size[0]

        # Normalize transform
        self.normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # Positional debiasing
        self.positional_basis = None
        if self.weights_loaded:
            self._compute_positional_basis()

    def _compute_positional_basis(self, rank=500):
        """Compute positional subspace from noise image."""
        print("Computing positional bias subspace from noise image...")

        with torch.no_grad():
            # Generate noise tensor
            noise_tensor = torch.randn(1, 3, 512, 512).to(self.device)
            noise_tensor = self.normalize(noise_tensor)

            # Forward pass
            output = self.model.forward_features(noise_tensor)
            patch_features = output["x_norm_patchtokens"]
            F_noise = patch_features.squeeze(0)

        # SVD to find positional subspace
        U, S, Vh = torch.linalg.svd(F_noise, full_matrices=False)
        B = Vh[:min(rank, len(S)), :].T
        self.positional_basis = B.to(self.device)
        print(f"Positional basis computed: shape {B.shape}, rank={min(rank, len(S))}")

    def debias_features(self, features):
        """Project features onto orthogonal complement of positional subspace."""
        if self.positional_basis is None or not self.weights_loaded:
            return features

        B = self.positional_basis
        I_D = torch.eye(self.feat_dim, device=self.device)
        proj = I_D - (B @ B.T)

        return features @ proj

    def extract_features(self, image, debiased=False):
        """
        Extract DINOv3 features from image.

        Args:
            image: PIL Image or torch tensor
            debiased: if True, apply positional debiasing

        Returns:
            features: (num_patches, feat_dim)
        """
        with torch.no_grad():
            if isinstance(image, torch.Tensor):
                tensor = image.to(self.device)
            else:
                # PIL Image input
                img_array = image.convert("RGB")
                img_array = ImageOps.expand(img_array, border=0, fill=(255, 255, 255))
                tensor = torch.tensor(
                    [[c / 255.0 for c in img_array.getdata(i)] for i in range(3)],
                    dtype=torch.float32,
                ).view(3, img_array.height, img_array.width).unsqueeze(0)
                tensor = tensor.to(self.device)

            # Forward pass
            output = self.model.forward_features(tensor)
            patch_features = output["x_norm_patchtokens"]
            features = patch_features.squeeze(0)

            # Apply debiasing if requested
            if debiased:
                features = self.debias_features(features)

            return features

    def __call__(self, tensor):
        """Compatibility with old interface."""
        return self.extract_features(tensor)
