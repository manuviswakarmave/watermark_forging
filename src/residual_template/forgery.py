
from pathlib import Path
import sys
from typing import Dict

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from residual_template.config_residual import (
    GROUPS,
    TARGET_MAPPING,
    ALPHA_MODE,
    BASE_ALPHA,
    MIN_ALPHA,
    MAX_ALPHA,
    ALPHA_TANH_BASE,
    ALPHA_TANH_SCALE,
    get_experiment_output_dir,
)

from residual_template.image_ops import (
    load_clean_target,
    save_rgb,
    resize_to_shape,
    clear_png_folder,
)

from residual_template.masking import detail_mask


def pick_alpha(template: np.ndarray, experiment: dict) -> float:
    """
    Pick alpha using the same dynamic formula as the baseline code.
    """

    base_alpha = float(experiment.get("base_alpha", BASE_ALPHA))
    min_alpha = float(experiment.get("min_alpha", MIN_ALPHA))
    max_alpha = float(experiment.get("max_alpha", MAX_ALPHA))

    if ALPHA_MODE == "fixed":
        return float(np.clip(base_alpha, min_alpha, max_alpha))

    strength = float(np.mean(np.abs(template)))

    alpha = base_alpha * (
        ALPHA_TANH_BASE + ALPHA_TANH_SCALE * np.tanh(strength)
    )

    return float(np.clip(alpha, min_alpha, max_alpha))


def inject_template(
    target: np.ndarray,
    template: np.ndarray,
    alpha: float,
    experiment: dict,
) -> np.ndarray:
    """
    Forge one image.

    I_forged = clip(I_target + sign * alpha * mask * template)
    """

    if template.shape != target.shape:
        template = resize_to_shape(template, target.shape)

    sign = float(experiment.get("sign", 1.0))

    mask = detail_mask(target, experiment=experiment)

    forged = target + sign * float(alpha) * mask[:, :, None] * template

    return np.clip(forged, 0.0, 1.0).astype(np.float32)


def forge_group(
    group_name: str,
    template: np.ndarray,
    experiment: dict,
    output_dir: Path,
) -> Dict:
    """
    Forge all target images for one group.
    """

    start_idx, end_idx = TARGET_MAPPING[group_name]

    alpha = pick_alpha(template, experiment)

    saved_paths = []
    sample_pairs = []

    for idx in range(start_idx, end_idx + 1):
        target = load_clean_target(idx)

        forged = inject_template(
            target=target,
            template=template,
            alpha=alpha,
            experiment=experiment,
        )

        save_path = Path(output_dir) / f"{idx}.png"

        save_rgb(forged, save_path)

        saved_paths.append(save_path)

        if len(sample_pairs) < 3:
            sample_pairs.append(
                {
                    "index": idx,
                    "clean": target,
                    "forged": forged,
                }
            )

    print(
        f"  {group_name}: alpha={alpha:.5f}, "
        f"saved {len(saved_paths)} images"
    )

    return {
        "group_name": group_name,
        "alpha": alpha,
        "saved_paths": saved_paths,
        "sample_pairs": sample_pairs,
    }


def forge_experiment(
    experiment: dict,
    template_results: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Forge all 200 targets for one experiment.
    """

    experiment_name = experiment["name"]
    output_dir = get_experiment_output_dir(experiment_name)

    clear_png_folder(output_dir)

    print()
    print("=" * 80)
    print(f"Forging experiment: {experiment_name}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)

    results = {}

    for group_name in GROUPS:
        template = template_results[group_name]["template"]

        group_result = forge_group(
            group_name=group_name,
            template=template,
            experiment=experiment,
            output_dir=output_dir,
        )

        results[group_name] = group_result

    return results