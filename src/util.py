import numpy as np
from PIL import Image


def load_rgb_image(path):
    """
    Load image as RGB float32 array in range [0, 255].
    """
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32)


def save_rgb_image(array, path):
    """
    Save float image array as uint8 PNG.
    """
    array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def resize_residual_to_target(residual, target_shape):
    """
    Resize residual if source and target image sizes differ.
    """
    if residual.shape == target_shape:
        return residual

    residual_img = Image.fromarray(np.clip(residual + 128, 0, 255).astype(np.uint8))
    residual_img = residual_img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    resized = np.array(residual_img).astype(np.float32) - 128
    return resized