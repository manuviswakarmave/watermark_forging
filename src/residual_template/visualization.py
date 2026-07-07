

from pathlib import Path
import sys
from typing import Dict

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from residual_template.config_residual import (
    SAVE_COMPARISON_VISUALS,
    NUM_VISUALIZE_PER_GROUP,
    DIFF_VISUAL_SCALE,
    get_experiment_visualization_dir,
)

from residual_template.image_ops import save_rgb


def diff_visual(clean: np.ndarray, forged: np.ndarray) -> np.ndarray:
    diff = forged.astype(np.float32) - clean.astype(np.float32)

    visual = 0.5 + DIFF_VISUAL_SCALE * diff

    return np.clip(visual, 0.0, 1.0).astype(np.float32)


def hconcat(images: list[np.ndarray], sep_width: int = 8) -> np.ndarray:
    images = [np.clip(img, 0.0, 1.0).astype(np.float32) for img in images]

    h = min(img.shape[0] for img in images)

    resized = []

    for img in images:
        if img.shape[0] != h:
            scale = h / img.shape[0]
            w = int(round(img.shape[1] * scale))

            import cv2

            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

            if img.ndim == 2:
                img = img[:, :, None]

        resized.append(img)

    pieces = []

    for idx, img in enumerate(resized):
        pieces.append(img)

        if idx != len(resized) - 1:
            pieces.append(np.ones((h, sep_width, 3), dtype=np.float32))

    return np.concatenate(pieces, axis=1).astype(np.float32)


def save_comparison_visuals(
    experiment_name: str,
    forge_results: Dict[str, Dict],
) -> None:
    """
    Save clean | forged | amplified difference comparisons.
    """

    if not SAVE_COMPARISON_VISUALS:
        return

    vis_root = get_experiment_visualization_dir(experiment_name) / "comparisons"
    vis_root.mkdir(parents=True, exist_ok=True)

    for group_name, group_result in forge_results.items():
        group_dir = vis_root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        sample_pairs = group_result["sample_pairs"][:NUM_VISUALIZE_PER_GROUP]

        for sample in sample_pairs:
            clean = sample["clean"]
            forged = sample["forged"]

            row = hconcat(
                [
                    clean,
                    forged,
                    diff_visual(clean, forged),
                ]
            )

            save_rgb(
                row,
                group_dir / f"{sample['index']}_clean_forged_diff.png",
            )