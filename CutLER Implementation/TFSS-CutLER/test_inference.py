import sys
sys.path.insert(0, './')

import torch
import os
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from cutler.config.cutler_config import add_cutler_config
from cutler.demo.predictor import VisualizationDemo

# Monitor GPU memory
def get_gpu_memory_mb():
    return torch.cuda.memory_allocated() / 1024**2

# Setup
cfg = get_cfg()
add_cutler_config(cfg)
cfg.merge_from_file("model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml")
cfg.MODEL.DEVICE = 'cuda'
cfg.freeze()

print(f"Config loaded. GPU memory before init: {get_gpu_memory_mb():.1f} MB")

# Initialize demo (loads model)
demo = VisualizationDemo(cfg)
print(f"Model loaded. GPU memory after init: {get_gpu_memory_mb():.1f} MB")

# Test on one image
img_path = "./cutler/demo/imgs/input1.jpg"
if os.path.exists(img_path):
    img = read_image(img_path, format="BGR")
    print(f"Image shape: {img.shape}")
    
    print(f"GPU memory before inference: {get_gpu_memory_mb():.1f} MB")
    predictions, visualized_output = demo.run_on_image(img)
    print(f"GPU memory after inference: {get_gpu_memory_mb():.1f} MB")
    print(f"Detected {len(predictions['instances'])} instances")
else:
    print(f"Image not found at {img_path}")
    print("Available images:")
    for f in os.listdir("./cutler/demo/imgs/"):
        print(f"  {f}")