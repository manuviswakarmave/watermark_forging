# src/residual_template/template_extraction.py

from pathlib import Path
import sys
from typing import Dict, List

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from residual_template.config_residual import (
    GROUPS,
    TARGET_MAPPING,
    TEMPLATE_AGGREGATION,
    REMOVE_TEMPLATE_CHANNEL_MEAN,
    TEMPLATE_NORMALIZATION_QUANTILE,
    RESIDUAL_SCALE,
    TEMPLATE_SHARPEN_MODE,
    TEMPLATE_SHARPEN_RADIUS,
    TEMPLATE_SHARPEN_MIX_ORIGINAL,
    TEMPLATE_SHARPEN_MIX_HIGHPASS,
    CLIP_TEMPLATE_MIN,
    CLIP_TEMPLATE_MAX,
    SAVE_TEMPLATE_NUMPY,
    get_experiment_template_dir,
)

from residual_template.image_ops import (
    load_rgb,
    save_rgb,
    list_source_group_images,
)

from residual_template.denoisers import (
    denoise_image,
    gaussian_denoise,
    signed_gaussian_blur,
)


def aggregate_residuals(residuals: List[np.ndarray]) -> np.ndarray:
    """
    Aggregate residuals across 25 watermarked source images.
    """

    stack = np.stack(residuals, axis=0).astype(np.float32)

    if TEMPLATE_AGGREGATION == "median":
        return np.median(stack, axis=0).astype(np.float32)

    if TEMPLATE_AGGREGATION == "mean":
        return np.mean(stack, axis=0).astype(np.float32)

    raise ValueError(f"Unknown TEMPLATE_AGGREGATION={TEMPLATE_AGGREGATION}")


def original_shifted_template_sharpen(template: np.ndarray) -> np.ndarray:
    """
    Sharpening mode closest to the script that got 0.32.

    It blurs the shifted/clipped template, then mixes a high-pass term.
    """

    shifted = np.clip((template + 1.0) * 0.5, 0.0, 1.0)

    shifted_blur = gaussian_denoise(
        shifted,
        radius=TEMPLATE_SHARPEN_RADIUS,
    )

    template_hp = template - shifted_blur

    sharpened = (
        TEMPLATE_SHARPEN_MIX_ORIGINAL * template
        + TEMPLATE_SHARPEN_MIX_HIGHPASS * (template_hp * 2.0)
    )

    return sharpened.astype(np.float32)


def signed_template_sharpen(template: np.ndarray) -> np.ndarray:
    """
    Cleaner signed high-pass sharpening.
    """

    smooth = signed_gaussian_blur(
        template=template,
        radius=TEMPLATE_SHARPEN_RADIUS,
    )

    highpass = template - smooth

    sharpened = (
        TEMPLATE_SHARPEN_MIX_ORIGINAL * template
        + TEMPLATE_SHARPEN_MIX_HIGHPASS * highpass
    )

    return sharpened.astype(np.float32)


def sharpen_template(template: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower().strip()

    if mode == "none":
        return template.astype(np.float32)

    if mode == "original_shifted":
        return original_shifted_template_sharpen(template)

    if mode == "signed":
        return signed_template_sharpen(template)

    raise ValueError(
        f"Unknown sharpen mode: {mode}. "
        "Use none, original_shifted, or signed."
    )


def normalize_template(template: np.ndarray) -> np.ndarray:
    """
    Robustly normalize template magnitude.
    """

    template = template.astype(np.float32)

    if REMOVE_TEMPLATE_CHANNEL_MEAN:
        template = template - template.mean(axis=(0, 1), keepdims=True)

    scale = float(np.quantile(np.abs(template), TEMPLATE_NORMALIZATION_QUANTILE))

    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0

    template = template / scale

    template = np.clip(
        template * RESIDUAL_SCALE,
        CLIP_TEMPLATE_MIN,
        CLIP_TEMPLATE_MAX,
    )

    return template.astype(np.float32)


def extract_group_template(
    group_name: str,
    experiment: dict,
) -> Dict:
    """
    Extract watermark template for one group:

        r_i = x_wm_i - D(x_wm_i)
        template = median_i(r_i)
    """

    source_paths = list_source_group_images(group_name)

    if len(source_paths) == 0:
        raise FileNotFoundError(f"No source images found for {group_name}")

    residuals = []

    denoiser_name = experiment.get("denoiser", "gaussian")
    sharpen_mode = experiment.get("sharpen_mode", TEMPLATE_SHARPEN_MODE)

    for source_path in source_paths:
        x = load_rgb(source_path)
        x_clean_est = denoise_image(x, method=denoiser_name)
        residual = x - x_clean_est
        residuals.append(residual.astype(np.float32))

    raw_template = aggregate_residuals(residuals)
    normalized_template = normalize_template(raw_template)
    sharpened_template = sharpen_template(normalized_template, mode=sharpen_mode)

    final_template = np.clip(
        sharpened_template,
        CLIP_TEMPLATE_MIN,
        CLIP_TEMPLATE_MAX,
    ).astype(np.float32)

    return {
        "group_name": group_name,
        "source_paths": source_paths,
        "template": final_template,
        "raw_template": raw_template,
        "denoiser": denoiser_name,
        "sharpen_mode": sharpen_mode,
    }


def extract_all_templates(experiment: dict) -> Dict[str, Dict]:
    """
    Extract templates for all WM groups.
    """

    results = {}

    print()
    print("=" * 80)
    print(f"Extracting templates for experiment: {experiment['name']}")
    print("=" * 80)

    for group_name in GROUPS:
        print(f"  Extracting {group_name}...")

        result = extract_group_template(
            group_name=group_name,
            experiment=experiment,
        )

        results[group_name] = result

        print_template_stats(
            result["template"],
            label=f"{experiment['name']} {group_name}",
        )

    return results


def save_template_numpy(
    template: np.ndarray,
    experiment_name: str,
    group_name: str,
) -> Path:
    """
    Save template as .npy.
    """

    save_dir = get_experiment_template_dir(experiment_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{group_name}_template.npy"

    np.save(save_path, template.astype(np.float32))

    return save_path


def save_all_templates_numpy(
    experiment_name: str,
    template_results: Dict[str, Dict],
) -> None:
    if not SAVE_TEMPLATE_NUMPY:
        return

    for group_name, result in template_results.items():
        save_path = save_template_numpy(
            template=result["template"],
            experiment_name=experiment_name,
            group_name=group_name,
        )

        print(f"  Saved template: {save_path}")


def template_centered_visual(template: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(template)))

    if max_abs < 1e-8:
        return np.ones_like(template, dtype=np.float32) * 0.5

    visual = 0.5 + template / (2.0 * max_abs)

    return np.clip(visual, 0.0, 1.0).astype(np.float32)


def template_abs_visual(template: np.ndarray) -> np.ndarray:
    abs_template = np.abs(template)

    max_value = float(abs_template.max())

    if max_value < 1e-8:
        return np.zeros_like(template, dtype=np.float32)

    visual = abs_template / max_value

    return np.clip(visual, 0.0, 1.0).astype(np.float32)


def save_template_visuals(
    experiment_name: str,
    group_name: str,
    template: np.ndarray,
) -> None:
    save_dir = get_experiment_template_dir(experiment_name) / "visuals" / group_name
    save_dir.mkdir(parents=True, exist_ok=True)

    save_rgb(
        template_centered_visual(template),
        save_dir / "template_centered.png",
    )

    save_rgb(
        template_abs_visual(template),
        save_dir / "template_absolute.png",
    )


def save_all_template_visuals(
    experiment_name: str,
    template_results: Dict[str, Dict],
) -> None:
    for group_name, result in template_results.items():
        save_template_visuals(
            experiment_name=experiment_name,
            group_name=group_name,
            template=result["template"],
        )


def print_template_stats(template: np.ndarray, label: str) -> None:
    print(
        f"    {label}: "
        f"shape={template.shape}, "
        f"min={template.min():.5f}, "
        f"max={template.max():.5f}, "
        f"mean={template.mean():.5f}, "
        f"std={template.std():.5f}, "
        f"mean_abs={np.mean(np.abs(template)):.5f}"
    )