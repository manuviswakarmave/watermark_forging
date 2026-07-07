

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



# Gaussian denoising


def gaussian_denoise(
    image: np.ndarray,
    radius: float = GAUSSIAN_RADIUS,
) -> np.ndarray:
    """
    Gaussian blur clean estimator.

    This matches the style of the original baseline code.
    """

    pil_image = Image.fromarray(to_uint8(image))

    blurred = pil_image.filter(
        ImageFilter.GaussianBlur(radius=radius)
    )

    return from_uint8(np.asarray(blurred))



# Bilateral denoising


def bilateral_denoise(
    image: np.ndarray,
    d: int = BILATERAL_D,
    sigma_color: float = BILATERAL_SIGMA_COLOR,
    sigma_space: float = BILATERAL_SIGMA_SPACE,
) -> np.ndarray:
    """
    Bilateral clean estimator.

    It smooths weak/noise-like details while preserving stronger edges.
    """

    image_u8 = to_uint8(image)

    filtered = cv2.bilateralFilter(
        image_u8,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    return from_uint8(filtered)



# Non-local means denoising


def nlm_denoise(
    image: np.ndarray,
    h: float = NLM_H,
    h_color: float = NLM_H_COLOR,
    template_window_size: int = NLM_TEMPLATE_WINDOW_SIZE,
    search_window_size: int = NLM_SEARCH_WINDOW_SIZE,
) -> np.ndarray:
    """
    Non-local means clean estimator.

    Useful if the watermark behaves like weak image noise.

    Default:
        h = NLM_H
        h_color = NLM_H_COLOR

    In your config this is currently h=5, h_color=5.
    """

    image_u8 = to_uint8(image)


    denoised = cv2.fastNlMeansDenoisingColored(
        image_u8,
        None,
        h=h,
        hColor=h_color,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size,
    )

    return from_uint8(denoised)


def nlm_denoise_h3(image: np.ndarray) -> np.ndarray:
    """
    Weaker NLM denoising than default h=5.

    This extracts a weaker residual:
        residual = image - D_h3(image)

    It may preserve better visual quality but may also reduce detector strength.
    """

    return nlm_denoise(
        image=image,
        h=3,
        h_color=3,
        template_window_size=NLM_TEMPLATE_WINDOW_SIZE,
        search_window_size=NLM_SEARCH_WINDOW_SIZE,
    )


def nlm_denoise_h7(image: np.ndarray) -> np.ndarray:
    """
    Stronger NLM denoising than default h=5.

    This can make:
        residual = image - D_h7(image)

    stronger than the default residual, which may improve watermark bit copying.
    """

    return nlm_denoise(
        image=image,
        h=7,
        h_color=7,
        template_window_size=NLM_TEMPLATE_WINDOW_SIZE,
        search_window_size=NLM_SEARCH_WINDOW_SIZE,
    )


def nlm_denoise_h9(image: np.ndarray) -> np.ndarray:
    """
    Even stronger NLM denoising.

    This may extract stronger watermark-like residuals, but it may also leak
    more image content into the residual.
    """

    return nlm_denoise(
        image=image,
        h=9,
        h_color=9,
        template_window_size=NLM_TEMPLATE_WINDOW_SIZE,
        search_window_size=NLM_SEARCH_WINDOW_SIZE,
    )



# Median denoising


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

    filtered = cv2.medianBlur(
        image_u8,
        ksize=kernel_size,
    )

    return from_uint8(filtered)



# Ensemble denoising


def ensemble_denoise(image: np.ndarray) -> np.ndarray:
    """
    Ensemble clean estimator.

    Averages multiple clean estimates:

        gaussian
        bilateral
        nlm default h=5
        median
    """

    estimates = [
        gaussian_denoise(image),
        bilateral_denoise(image),
        nlm_denoise(image),
        median_denoise(image),
    ]

    return np.mean(
        np.stack(estimates, axis=0),
        axis=0,
    ).astype(np.float32)


def ensemble_nlm_strong_denoise(image: np.ndarray) -> np.ndarray:
    """
    Ensemble variant using stronger NLM.

    This is optional for future experiments.
    """

    estimates = [
        gaussian_denoise(image),
        bilateral_denoise(image),
        nlm_denoise_h7(image),
        median_denoise(image),
    ]

    return np.mean(
        np.stack(estimates, axis=0),
        axis=0,
    ).astype(np.float32)

def nlm_denoise_h10(image: np.ndarray) -> np.ndarray:
    return nlm_denoise(
        image=image,
        h=10,
        h_color=10,
        template_window_size=NLM_TEMPLATE_WINDOW_SIZE,
        search_window_size=NLM_SEARCH_WINDOW_SIZE,
    )


def nlm_denoise_h11(image: np.ndarray) -> np.ndarray:
    return nlm_denoise(
        image=image,
        h=11,
        h_color=11,
        template_window_size=NLM_TEMPLATE_WINDOW_SIZE,
        search_window_size=NLM_SEARCH_WINDOW_SIZE,
    )


def nlm_denoise_h12(image: np.ndarray) -> np.ndarray:
    return nlm_denoise(
        image=image,
        h=12,
        h_color=12,
        template_window_size=NLM_TEMPLATE_WINDOW_SIZE,
        search_window_size=NLM_SEARCH_WINDOW_SIZE,
    )
def nlm_denoise_h13(image: np.ndarray) -> np.ndarray:
    return nlm_denoise(
        image=image,
        h=13,
        h_color=13,
        template_window_size=NLM_TEMPLATE_WINDOW_SIZE,
        search_window_size=NLM_SEARCH_WINDOW_SIZE,
    )


def nlm_denoise_h14(image: np.ndarray) -> np.ndarray:
    return nlm_denoise(
        image=image,
        h=14,
        h_color=14,
        template_window_size=NLM_TEMPLATE_WINDOW_SIZE,
        search_window_size=NLM_SEARCH_WINDOW_SIZE,
    )


def nlm_denoise_h15(image: np.ndarray) -> np.ndarray:
    return nlm_denoise(
        image=image,
        h=15,
        h_color=15,
        template_window_size=NLM_TEMPLATE_WINDOW_SIZE,
        search_window_size=NLM_SEARCH_WINDOW_SIZE,
    )

# =============================================================================
# Denoiser dispatcher
# =============================================================================

def denoise_image(
    image: np.ndarray,
    method: str,
) -> np.ndarray:
    """
    Dispatch denoising method.

    Existing methods:
        gaussian
        bilateral
        nlm
        median
        ensemble

    New NLM variants:
        nlm_h3
        nlm_h7
        nlm_h9

    Optional ensemble variant:
        ensemble_nlm_strong
    """

    method = method.lower().strip()

    if method == "gaussian":
        return gaussian_denoise(image)

    if method == "bilateral":
        return bilateral_denoise(image)

    if method == "nlm":
        return nlm_denoise(image)

    if method == "nlm_h3":
        return nlm_denoise_h3(image)

    if method == "nlm_h7":
        return nlm_denoise_h7(image)

    if method == "nlm_h9":
        return nlm_denoise_h9(image)

    if method == "median":
        return median_denoise(image)

    if method == "ensemble":
        return ensemble_denoise(image)

    if method == "ensemble_nlm_strong":
        return ensemble_nlm_strong_denoise(image)

    if method == "nlm_h10":
        return nlm_denoise_h10(image)

    if method == "nlm_h11":
        return nlm_denoise_h11(image)

    if method == "nlm_h12":
        return nlm_denoise_h12(image)

    if method == "nlm_h13":
        return nlm_denoise_h13(image)

    if method == "nlm_h14":
        return nlm_denoise_h14(image)

    if method == "nlm_h15":
        return nlm_denoise_h15(image)

    raise ValueError(
        f"Unknown denoiser method: {method}. "
        "Use gaussian, bilateral, nlm, nlm_h3, nlm_h7, nlm_h9, "
        "median, ensemble, or ensemble_nlm_strong."
    )



# Signed template smoothing


def signed_gaussian_blur(
    template: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Gaussian blur for signed residual templates.

    PIL cannot directly blur signed float residuals safely, so each channel is:

        1. normalized to [0, 1]
        2. blurred
        3. mapped back to its original signed range
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

        blurred = pil_image.filter(
            ImageFilter.GaussianBlur(radius=radius)
        )

        blurred = from_uint8(np.asarray(blurred))

        restored = blurred * (ch_max - ch_min) + ch_min

        channels.append(restored)

    return np.stack(channels, axis=2).astype(np.float32)