#!/usr/bin/env python3
"""DINOv2 feature extraction (no debiasing)."""

import sys
import torch
from pathlib import Path
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image


class DINOv2Feat:
    """DINOv2 feature extractor."""

    def __init__(self, model_id, feat_type="k", device="cuda"):
        """
        Args:
            model_id: Model name (e.g., "dinov2_vits14", "dinov2_vitb14")
            feat_type: "k", "q", "v", or "kqv" (which features to extract)
            device: "cuda" or "cpu"
        """
        self.model_id = model_id
        self.feat_type = feat_type
        self.device = device
        self.weights_loaded = False

        # Load model via torch.hub
        self.model = self._load_dinov2_model(model_id, device)
        self.model.eval()

        # Feature dimension and patch size
        self.feat_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size

        # Normalize transform (ImageNet)
        self.normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def _load_dinov2_model(self, model_id, device):
        """Load DINOv2 model via torch.hub."""
        try:
            print(f"Loading DINOv2 model: {model_id}...")
            # DINOv2 models available via torch.hub from facebookresearch/dinov2
            model = torch.hub.load("facebookresearch/dinov2", model_id)
            self.weights_loaded = True
            print(f"Successfully loaded {model_id} from torch.hub")
            return model.to(device)
        except Exception as e:
            print(f"ERROR: Could not load DINOv2 model ({type(e).__name__}: {e})")
            print(f"DINOv2 models require the facebookresearch/dinov2 repo")
            print(f"Clone it with: git clone https://github.com/facebookresearch/dinov2.git")
            self.weights_loaded = False
            raise

    def extract_features(self, image, debiased=False):
        """
        Extract DINOv2 features from image.

        Args:
            image: PIL Image or torch tensor
            debiased: ignored for DINOv2 (not supported)

        Returns:
            features: (num_patches, feat_dim)
            patch_grid: (grid_h, grid_w) indicating spatial layout
        """
        from PIL import Image as PILImage

        with torch.no_grad():
            # Handle torch tensor input
            if isinstance(image, torch.Tensor):
                tensor = image.to(self.device)
                img_h, img_w = image.shape[-2:]
            else:
                # PIL Image input
                img_array = image.convert("RGB")
                img_w, img_h = img_array.size

                # Ensure dimensions are divisible by patch_size
                # Round down to nearest multiple of patch_size
                h_aligned = (img_h // self.patch_size) * self.patch_size
                w_aligned = (img_w // self.patch_size) * self.patch_size

                if h_aligned != img_h or w_aligned != img_w:
                    img_array = img_array.resize((w_aligned, h_aligned), PILImage.LANCZOS)
                    img_h, img_w = h_aligned, w_aligned

                # Convert to tensor and normalize
                tensor = ToTensor()(img_array).unsqueeze(0).to(self.device)
                tensor = self.normalize(tensor)

            # Forward pass
            # DINOv2 forward_features returns dict with normalized patch tokens
            output = self.model.forward_features(tensor)

            # Extract patch features (x_norm_patchtokens has no class token, already separated)
            if isinstance(output, dict):
                features = output["x_norm_patchtokens"]  # (1, num_patches, feat_dim)
            else:
                features = output

            features = features.squeeze(0)  # (num_patches, feat_dim)

            # Infer patch grid from aligned image size
            grid_h = img_h // self.patch_size
            grid_w = img_w // self.patch_size

            return features, (grid_h, grid_w)

    def __call__(self, tensor):
        """
        Callable interface matching DINO v1 API.
        Called as: features = backbone(tensor)[0]

        Args:
            tensor: (1, 3, H, W) torch tensor

        Returns:
            tuple: (features, patch_grid) for compatibility with recursive_fiedler.py
        """
        features, grid_info = self.extract_features(tensor, debiased=False)
        return (features, grid_info)
