from PIL import Image


def resize_pil(image, patch_size):
    width, height = image.size
    feat_w = width // patch_size
    feat_h = height // patch_size
    new_width = feat_w * patch_size
    new_height = feat_h * patch_size
    resized = image.resize((new_width, new_height), Image.LANCZOS)
    return resized, new_width, new_height, feat_w, feat_h

