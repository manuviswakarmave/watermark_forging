# src/residual_template/masking.py

from pathlib import Path
import sys

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from residual_template.config_residual import (
    USE_DETAIL_MASK,
    MASK_FLOOR,
    MASK_CEILING,
    MASK_DETAIL_QUANTILE,
)

from residual_template.denoisers import denoise_image


def detail_mask(
    target: np.ndarray,
    experiment: dict,
) -> np.ndarray:
    """
    Create target-aware detail mask.

    More injection in textured areas, less in smooth regions.
    """

    if not USE_DETAIL_MASK:
        return np.ones(target.shape[:2], dtype=np.float32)

    mask_floor = float(experiment.get("mask_floor", MASK_FLOOR))
    mask_ceiling = float(experiment.get("mask_ceiling", MASK_CEILING))
    mask_denoiser = experiment.get("mask_denoiser", "gaussian")

    gray = target.mean(axis=2, keepdims=True).astype(np.float32)
    gray_rgb = np.repeat(gray, 3, axis=2)

    clean_est = denoise_image(gray_rgb, method=mask_denoiser)
    clean_gray = clean_est.mean(axis=2)

    detail = np.abs(gray[:, :, 0] - clean_gray)

    q = float(np.quantile(detail, MASK_DETAIL_QUANTILE))

    if not np.isfinite(q) or q < 1e-8:
        q = 1.0

    m = np.clip(detail / q, 0.0, 1.0)

    m = mask_floor + (mask_ceiling - mask_floor) * m

    return m.astype(np.float32)