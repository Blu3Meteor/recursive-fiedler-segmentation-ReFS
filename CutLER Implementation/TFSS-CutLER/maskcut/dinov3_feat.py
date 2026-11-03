#!/usr/bin/env python3
"""DINOv3 feature extraction with positional debiasing."""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

# Add dinov3 repo to path
DINOV3_ROOT = Path(__file__).parent.parent.parent.parent / "dinov3_repo"
DINOV3_WEIGHTS = DINOV3_ROOT / "dinov3_weights"

if str(DINOV3_WEIGHTS) not in sys.path:
    sys.path.insert(0, str(DINOV3_WEIGHTS))

# Create module alias so dinov3_weights can be imported as dinov3
import importlib.util
import types

if "dinov3" not in sys.modules:
    dinov3_spec = importlib.util.spec_from_file_location("dinov3", str(DINOV3_WEIGHTS / "__init__.py"))
    if dinov3_spec and dinov3_spec.loader:
        dinov3 = importlib.util.module_from_spec(dinov3_spec)
        sys.modules["dinov3"] = dinov3


class DINOv3Feat:
    """DINOv3 feature extractor with positional debiasing."""

    def __init__(self, model_id, feat_type="k", device="cuda"):
        """
        Args:
            model_id: Model name (e.g., "dinov3_vits14_pretrain", "dinov3_vits16plus_pretrain")
                     or path to checkpoint file
            feat_type: "k", "q", "v", or "kqv" (which features to extract)
            device: "cuda" or "cpu"
        """
        self.model_id = model_id
        self.feat_type = feat_type
        self.device = device
        self.weights_loaded = False  # Track if pretrained weights were loaded

        # Load model from official DINOv3 repo
        self.model = self._load_dinov3_model(model_id, device)
        self.model.eval()

        # Feature dimension and patch size
        self.feat_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size

        # Normalize transform
        self.normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # Positional subspace (disabled by default - not beneficial for this task)
        self.positional_basis = None

    # Explicit architecture kwargs matching how each checkpoint was actually
    # trained (mirrors INSID3/models/__init__.py's _MODEL_CONFIGS). A bare
    # vits.__dict__[arch](patch_size=..., num_classes=0) call omits LayerScale
    # (layerscale_init), storage tokens (n_storage_tokens), and the QKV bias
    # mask (mask_k_bias) -- confirmed via check_dinov3_load.py to silently
    # drop 37/188 checkpoint tensors (all LayerScale gammas, storage_tokens,
    # and attn.qkv.bias_mask) under strict=False with zero visible symptom.
    _ARCH_KWARGS = {
        "vit_small": {
            "img_size": 224, "in_chans": 3,
            "pos_embed_rope_base": 100, "pos_embed_rope_normalize_coords": "separate",
            "pos_embed_rope_rescale_coords": 2, "pos_embed_rope_dtype": "fp32",
            "qkv_bias": True, "drop_path_rate": 0.0, "layerscale_init": 1.0e-05,
            "norm_layer": "layernormbf16", "ffn_layer": "mlp", "ffn_bias": True, "proj_bias": True,
            "n_storage_tokens": 4, "mask_k_bias": True,
        },
        "vit_large": {
            "img_size": 224, "in_chans": 3,
            "pos_embed_rope_base": 100, "pos_embed_rope_normalize_coords": "separate",
            "pos_embed_rope_rescale_coords": 2, "pos_embed_rope_dtype": "fp32",
            "qkv_bias": True, "drop_path_rate": 0.0, "layerscale_init": 1.0e-05,
            "norm_layer": "layernormbf16", "ffn_layer": "mlp", "ffn_bias": True, "proj_bias": True,
            "n_storage_tokens": 4, "mask_k_bias": True, "untie_global_and_local_cls_norm": True,
        },
    }

    def _load_dinov3_model(self, model_id, device):
        """Load DINOv3 model from official repo or HuggingFace."""
        from dinov3.models import vision_transformer as vits

        # Parse model_id to get architecture and checkpoint URL
        # Format: dinov3_vits14_pretrain_lvd1689m -> "vit_small", patch_size=14
        if "vits14" in model_id:
            arch = "vit_small"
            patch_size = 14
        elif "vits16" in model_id:
            arch = "vit_small"
            patch_size = 16
        elif "vitb14" in model_id:
            arch = "vit_base"
            patch_size = 14
        elif "vitl16" in model_id:
            arch = "vit_large"
            patch_size = 16
        else:
            raise ValueError(f"Unknown DINOv3 model: {model_id}")

        # Build model with the real training-time architecture kwargs when
        # known (vit_small/vit_large, patch16 -- our local checkpoints). Other
        # combinations (patch14 variants, reachable only via network download)
        # have no confirmed kwargs here, so fall back to the bare call with a
        # loud warning rather than silently guessing.
        kwargs_key = arch if patch_size == 16 else None
        if kwargs_key in self._ARCH_KWARGS:
            model = vits.__dict__[arch](patch_size=patch_size, num_classes=0, **self._ARCH_KWARGS[kwargs_key])
        else:
            print(f"WARNING: no confirmed architecture kwargs for {arch}/patch{patch_size} -- "
                  f"building with bare defaults, which previously caused 37/188 checkpoint "
                  f"tensors (LayerScale, storage tokens, QKV bias mask) to silently fail to load.")
            model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)

        # Load pretrained weights if checkpoint path provided
        if Path(model_id).exists():
            checkpoint = torch.load(model_id, map_location=device)
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            strict = kwargs_key in self._ARCH_KWARGS
            result = model.load_state_dict(state_dict, strict=strict)
            if not strict:
                print(f"  missing_keys={len(result.missing_keys)} unexpected_keys={len(result.unexpected_keys)}")
            self.weights_loaded = True
            print(f"Loaded pretrained weights from {model_id} (strict={strict})")
        else:
            # Try Meta's official checkpoint first
            checkpoint_url = self._get_checkpoint_url(model_id)
            if checkpoint_url:
                try:
                    print(f"Downloading DINOv3 checkpoint from Meta's servers...")
                    checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location=device, progress=True)
                    if "model" in checkpoint:
                        state_dict = checkpoint["model"]
                    else:
                        state_dict = checkpoint
                    model.load_state_dict(state_dict, strict=False)
                    self.weights_loaded = True
                    print(f"Successfully loaded pretrained weights from Meta")
                except Exception as e:
                    print(f"Failed to download from Meta ({type(e).__name__}): {e}")
                    print(f"Trying HuggingFace Transformers as fallback...")
                    # Try HuggingFace as fallback
                    self._load_from_huggingface(model_id, model, device)
            else:
                self.weights_loaded = False
                print(f"Warning: Could not find checkpoint for {model_id}, using random initialization")

        return model.to(device)

    def _load_from_huggingface(self, model_id, model, device):
        """Load DINOv3 model from HuggingFace Transformers library."""
        try:
            from transformers import AutoModel

            # Map model IDs to HuggingFace model names
            hf_model_map = {
                "dinov3_vits14_pretrain_lvd1689m": "facebook/dinov3-vits14-pretrain-lvd1689m",
                "dinov3_vits16plus_pretrain_lvd1689m": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
                "dinov3_vitb14_pretrain_lvd1689m": "facebook/dinov3-vitb14-pretrain-lvd1689m",
            }

            if model_id in hf_model_map:
                hf_model_name = hf_model_map[model_id]
                print(f"Loading {hf_model_name} from HuggingFace...")
                hf_model = AutoModel.from_pretrained(hf_model_name)

                # Load state dict from HuggingFace model to our model
                state_dict = hf_model.state_dict()
                model.load_state_dict(state_dict, strict=False)
                self.weights_loaded = True
                print(f"Successfully loaded pretrained weights from HuggingFace")
            else:
                print(f"No HuggingFace mapping for {model_id}")
                self.weights_loaded = False
        except ImportError:
            print("ERROR: transformers library not installed")
            print("Install with: pip install transformers")
            self.weights_loaded = False
        except Exception as e:
            print(f"ERROR: Failed to load from HuggingFace ({type(e).__name__}: {e})")
            print("Proceeding with RANDOM INITIALIZATION (metrics will be invalid)")
            self.weights_loaded = False

    def _get_checkpoint_url(self, model_id):
        """Get official checkpoint URL for DINOv3 model."""
        # Meta's official checkpoint URLs
        urls = {
            "dinov3_vits14_pretrain_lvd1689m": "https://dl.fbaipublicfiles.com/dinov3/dinov3_vits14_pretrain_lvd1689m.pth",
            "dinov3_vits16plus_pretrain_lvd1689m": "https://dl.fbaipublicfiles.com/dinov3/dinov3_vits16plus_pretrain_lvd1689m.pth",
            "dinov3_vitb14_pretrain_lvd1689m": "https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb14_pretrain_lvd1689m.pth",
        }
        return urls.get(model_id)

    def _compute_positional_basis(self, rank=500):
        """Compute positional subspace from noise image (Section 3.1 of INSID3)."""
        print("Computing positional bias subspace from noise image...")

        with torch.no_grad():
            # Generate noise tensor
            noise_tensor = torch.randn(1, 3, 512, 512).to(self.device)
            noise_tensor = self.normalize(noise_tensor)

            # Forward pass using DINOv3's forward_features
            output = self.model.forward_features(noise_tensor)

            # Extract patch features from output dict
            # x_norm_patchtokens already has only patch tokens, no class token
            patch_features = output["x_norm_patchtokens"]  # (1, num_patches, feat_dim)
            F_noise = patch_features.squeeze(0)  # (num_patches, feat_dim)

        # SVD to find positional subspace
        U, S, Vh = torch.linalg.svd(F_noise, full_matrices=False)

        # Take top rank right singular vectors as positional basis
        B = Vh[:min(rank, len(S)), :].T  # (feat_dim, rank), handle case where fewer than rank singular values
        self.positional_basis = B.to(self.device)

        print(f"Positional basis computed: shape {B.shape}, rank={min(rank, len(S))}")

    def debias_features(self, features):
        """
        Project features onto orthogonal complement of positional subspace.
        Equation (4) from INSID3: F_tilde = F * (I_D - B*B^T)

        Args:
            features: (num_patches, feat_dim)

        Returns:
            debiased_features: (num_patches, feat_dim)
        """
        if self.positional_basis is None or not self.weights_loaded:
            return features

        B = self.positional_basis
        I_D = torch.eye(self.feat_dim, device=self.device)
        proj = I_D - (B @ B.T)

        return features @ proj

    def extract_features(self, image, debiased=False):
        """
        Extract DINOv3 features from image or tensor.

        Args:
            image: PIL Image or torch tensor
            debiased: if True, apply positional debiasing

        Returns:
            features: (num_patches, feat_dim)
            patch_grid: (grid_h, grid_w) indicating spatial layout
        """
        with torch.no_grad():
            # Handle torch tensor input (from recursive_fiedler.py)
            if isinstance(image, torch.Tensor):
                # Tensor shape: (1, 3, H, W)
                tensor = image.to(self.device)
                img_h, img_w = image.shape[-2:]
            else:
                # PIL Image input
                img_array = image.convert("RGB")
                img_h, img_w = img_array.size
                # Convert to tensor and normalize
                tensor = ToTensor()(img_array).unsqueeze(0).to(self.device)
                tensor = self.normalize(tensor)

            # Forward pass using DINOv3's forward_features
            output = self.model.forward_features(tensor)

            # Extract patch features from output dict
            # x_norm_patchtokens already has only patch tokens, no class token
            patch_features = output["x_norm_patchtokens"]  # (1, num_patches, feat_dim)
            features = patch_features.squeeze(0)  # (num_patches, feat_dim)

            # Infer patch grid from image size
            grid_h = img_h // self.patch_size
            grid_w = img_w // self.patch_size

            # Apply debiasing if requested
            if debiased or getattr(self, "debiased", False):
                features = self.debias_features(features)

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
        features, grid_info = self.extract_features(tensor, debiased=getattr(self, "debiased", False))
        # Return as tuple to match [0] indexing in recursive_fiedler.py
        return (features, grid_info)
