import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    CLEAN_TARGETS_DIR,
    RESIDUAL_DIR,
    FORGED_IMAGES_DIR,
    SUBMISSION_ZIP,
    CATEGORIES,
    FORGE_ALPHA_VALUES,
    FORGE_MAX_PERTURBATION,
)


def clear_output_dir(output_dir: Path):
    """
    Remove old forged images so stale files do not remain from previous runs.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def load_image_as_array(image_path: Path) -> np.ndarray:
    """
    Load image as float32 RGB array in [0, 1].

    Output shape:
        [H, W, 3]
    """
    image = Image.open(image_path).convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    return arr


def save_image_from_array(arr: np.ndarray, out_path: Path):
    """
    Save float32 RGB array in [0, 1] as PNG.
    """
    arr = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(arr_uint8).save(out_path)


def load_residual_chw(residual_path: Path) -> np.ndarray:
    """
    Load residual saved by extract_preference_residuals.py.

    Expected shape:
        [3, H, W]
    """
    residual = np.load(residual_path).astype(np.float32)

    if residual.ndim != 3:
        raise ValueError(f"Expected residual shape [3,H,W], got {residual.shape}")

    if residual.shape[0] != 3:
        raise ValueError(f"Expected residual first dimension to be 3, got {residual.shape}")

    return residual


def resize_residual_to_target(
    residual_chw: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """
    Resize residual from preference-model resolution to target image resolution.

    Input:
        residual_chw: [3, H, W]

    Output:
        residual_hwc: [target_height, target_width, 3]
    """
    residual_tensor = torch.from_numpy(residual_chw).unsqueeze(0)  # [1, 3, H, W]

    resized = F.interpolate(
        residual_tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    resized_chw = resized.squeeze(0).numpy()
    resized_hwc = np.transpose(resized_chw, (1, 2, 0))

    return resized_hwc.astype(np.float32)


def forge_single_image(
    clean_arr: np.ndarray,
    residual_hwc: np.ndarray,
    alpha: float,
    max_perturbation: float,
) -> np.ndarray:
    """
    Apply residual to one clean target image.

    Formula:
        forged = clean + alpha * residual

    We clip the applied perturbation to protect image quality.
    """
    perturbation = alpha * residual_hwc

    perturbation = np.clip(
        perturbation,
        -max_perturbation,
        max_perturbation,
    )

    forged = clean_arr + perturbation
    forged = np.clip(forged, 0.0, 1.0)

    return forged


def save_perturbation_preview(
    perturbation_hwc: np.ndarray,
    out_path: Path,
    scale: float = 10.0,
):
    """
    Save a visible preview of the perturbation.

    This is only for debugging.
    Not used in submission.
    """
    preview = perturbation_hwc * scale + 0.5
    preview = np.clip(preview, 0.0, 1.0)
    preview_uint8 = (preview * 255.0).round().astype(np.uint8)
    Image.fromarray(preview_uint8).save(out_path)


def create_submission_zip():
    """
    Create flat zip containing exactly the forged 200 PNG files.

    The assignment requires:
        - no subfolders
        - original filenames unchanged
        - only the 200 images
    """
    image_paths = sorted(
        FORGED_IMAGES_DIR.glob("*.png"),
        key=lambda p: int(p.stem),
    )

    if len(image_paths) != 200:
        raise RuntimeError(
            f"Expected 200 forged images, but found {len(image_paths)} in {FORGED_IMAGES_DIR}"
        )

    if SUBMISSION_ZIP.exists():
        SUBMISSION_ZIP.unlink()

    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for image_path in image_paths:
            zipf.write(image_path, arcname=image_path.name)

    print(f"Created submission zip: {SUBMISSION_ZIP}")


def main():
    print("Starting residual-based forging...")

    clear_output_dir(FORGED_IMAGES_DIR)

    total_forged = 0

    for wm_name, target_start, target_stop in CATEGORIES:
        print("\n==============================")
        print(f"Forging targets using {wm_name}")
        print("==============================")

        residual_path = RESIDUAL_DIR / f"{wm_name}_residual.npy"

        if not residual_path.exists():
            raise FileNotFoundError(f"Missing residual file: {residual_path}")

        residual_chw = load_residual_chw(residual_path)

        alpha = FORGE_ALPHA_VALUES.get(wm_name, 0.30)

        print(f"Residual file: {residual_path}")
        print(f"Alpha: {alpha}")
        print(f"Max perturbation: {FORGE_MAX_PERTURBATION}")

        for target_number in range(target_start, target_stop + 1):
            target_path = CLEAN_TARGETS_DIR / f"{target_number}.png"

            if not target_path.exists():
                raise FileNotFoundError(f"Missing target image: {target_path}")

            clean_arr = load_image_as_array(target_path)
            height, width, _ = clean_arr.shape

            residual_hwc = resize_residual_to_target(
                residual_chw=residual_chw,
                target_height=height,
                target_width=width,
            )

            forged_arr = forge_single_image(
                clean_arr=clean_arr,
                residual_hwc=residual_hwc,
                alpha=alpha,
                max_perturbation=FORGE_MAX_PERTURBATION,
            )

            out_path = FORGED_IMAGES_DIR / f"{target_number}.png"
            save_image_from_array(forged_arr, out_path)

            # Save a few debug previews only for the first image of each group.
            if target_number == target_start:
                perturbation = np.clip(
                    alpha * residual_hwc,
                    -FORGE_MAX_PERTURBATION,
                    FORGE_MAX_PERTURBATION,
                )

                preview_path = FORGED_IMAGES_DIR / f"debug_{wm_name}_perturbation_preview.png"
                save_perturbation_preview(perturbation, preview_path)

            total_forged += 1

        print(f"Finished {wm_name}: targets {target_start}.png to {target_stop}.png")

    print(f"\nTotal forged images: {total_forged}")

    if total_forged != 200:
        raise RuntimeError(f"Expected 200 forged images, got {total_forged}")

    # Remove debug previews before zipping.
    for debug_file in FORGED_IMAGES_DIR.glob("debug_*.png"):
        debug_file.unlink()

    create_submission_zip()

    print("\nForging finished successfully.")


if __name__ == "__main__":
    main()