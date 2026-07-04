# src/residual_template/denoisers.py

from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageFilter


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from residual_template.config_residual import (
    GAUSSIAN_RADIUS,
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    NLM_H,
    NLM_H_COLOR,
    NLM_TEMPLATE_WINDOW_SIZE,
    NLM_SEARCH_WINDOW_SIZE,
    MEDIAN_KERNEL_SIZE,
)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def from_uint8(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def gaussian_denoise(image: np.ndarray, radius: float = GAUSSIAN_RADIUS) -> np.ndarray:
    """
    Gaussian blur clean estimator.
    This matches the style of the script that got your best baseline.
    """

    pil_image = Image.fromarray(to_uint8(image))
    blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))

    return from_uint8(np.asarray(blurred))


def bilateral_denoise(
    image: np.ndarray,
    d: int = BILATERAL_D,
    sigma_color: float = BILATERAL_SIGMA_COLOR,
    sigma_space: float = BILATERAL_SIGMA_SPACE,
) -> np.ndarray:
    """
    Bilateral clean estimator.
    Smooths weak details while preserving strong edges.
    """

    image_u8 = to_uint8(image)

    filtered = cv2.bilateralFilter(
        image_u8,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    return from_uint8(filtered)


def nlm_denoise(
    image: np.ndarray,
    h: float = NLM_H,
    h_color: float = NLM_H_COLOR,
    template_window_size: int = NLM_TEMPLATE_WINDOW_SIZE,
    search_window_size: int = NLM_SEARCH_WINDOW_SIZE,
) -> np.ndarray:
    """
    Non-local means clean estimator.
    Useful if the watermark behaves like weak noise.
    """

    image_u8 = to_uint8(image)

    # OpenCV expects RGB/BGR-like 3-channel uint8; color ordering is not critical
    # for denoising consistency here.
    denoised = cv2.fastNlMeansDenoisingColored(
        image_u8,
        None,
        h=h,
        hColor=h_color,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size,
    )

    return from_uint8(denoised)


def median_denoise(
    image: np.ndarray,
    kernel_size: int = MEDIAN_KERNEL_SIZE,
) -> np.ndarray:
    """
    Median filter clean estimator.
    """

    if kernel_size % 2 == 0:
        kernel_size += 1

    image_u8 = to_uint8(image)

    filtered = cv2.medianBlur(image_u8, ksize=kernel_size)

    return from_uint8(filtered)


def ensemble_denoise(image: np.ndarray) -> np.ndarray:
    """
    Ensemble clean estimator.
    Averages multiple clean estimates.
    """

    estimates = [
        gaussian_denoise(image),
        bilateral_denoise(image),
        nlm_denoise(image),
        median_denoise(image),
    ]

    return np.mean(np.stack(estimates, axis=0), axis=0).astype(np.float32)


def denoise_image(image: np.ndarray, method: str) -> np.ndarray:
    """
    Dispatch denoising method.
    """

    method = method.lower().strip()

    if method == "gaussian":
        return gaussian_denoise(image)

    if method == "bilateral":
        return bilateral_denoise(image)

    if method == "nlm":
        return nlm_denoise(image)

    if method == "median":
        return median_denoise(image)

    if method == "ensemble":
        return ensemble_denoise(image)

    raise ValueError(
        f"Unknown denoiser method: {method}. "
        "Use gaussian, bilateral, nlm, median, or ensemble."
    )


def signed_gaussian_blur(
    template: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Gaussian blur for signed residual templates.
    """

    channels = []

    for c in range(template.shape[2]):
        channel = template[:, :, c].astype(np.float32)

        ch_min = float(channel.min())
        ch_max = float(channel.max())

        if ch_max - ch_min < 1e-8:
            channels.append(channel)
            continue

        normalized = (channel - ch_min) / (ch_max - ch_min)

        pil_image = Image.fromarray(to_uint8(normalized))
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
        blurred = from_uint8(np.asarray(blurred))

        restored = blurred * (ch_max - ch_min) + ch_min

        channels.append(restored)

    return np.stack(channels, axis=2).astype(np.float32)