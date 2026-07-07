
from pathlib import Path
import sys
from typing import List

import cv2
import numpy as np
from PIL import Image


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from residual_template.config_residual import (
    WATERMARKED_SOURCES_DIR,
    CLEAN_TARGETS_DIR,
    IMAGE_EXTENSIONS,
)


def load_rgb(path: Path) -> np.ndarray:
    """
    Load image as RGB float32 in [0, 1].
    """

    path = Path(path)

    image = Image.open(path).convert("RGB")

    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb(image: np.ndarray, path: Path) -> None:
    """
    Save RGB float image in [0, 1].
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.clip(image, 0.0, 1.0)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    Image.fromarray(arr).save(path)


def resize_to_shape(image: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Resize image to target H x W x C.
    """

    target_h, target_w, target_c = target_shape

    if image.shape == target_shape:
        return image.astype(np.float32)

    resized = cv2.resize(
        image.astype(np.float32),
        dsize=(target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )

    if resized.ndim == 2:
        resized = resized[:, :, None]

    if resized.shape[2] == 1 and target_c == 3:
        resized = np.repeat(resized, 3, axis=2)

    return resized.astype(np.float32)


def sort_by_numeric_stem(paths: List[Path]) -> List[Path]:
    """
    Sort paths as 1.png, 2.png, ..., 10.png.
    """

    def key(path: Path):
        try:
            return int(path.stem)
        except ValueError:
            return path.stem

    return sorted(paths, key=key)


def list_image_paths(folder: Path) -> List[Path]:
    """
    List image files from a folder.
    """

    folder = Path(folder)

    paths = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sort_by_numeric_stem(paths)


def get_source_group_dir(group_name: str) -> Path:
    return WATERMARKED_SOURCES_DIR / group_name


def list_source_group_images(group_name: str) -> List[Path]:
    source_dir = get_source_group_dir(group_name)
    return list_image_paths(source_dir)


def get_clean_target_path(index: int) -> Path:
    return CLEAN_TARGETS_DIR / f"{index}.png"


def load_clean_target(index: int) -> np.ndarray:
    return load_rgb(get_clean_target_path(index))


def clear_png_folder(folder: Path) -> None:
    """
    Create folder and delete old .png files.
    """

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    for path in folder.glob("*.png"):
        path.unlink()


def expected_filenames_from_mapping(groups, target_mapping) -> list[str]:
    names = []

    for group_name in groups:
        start_idx, end_idx = target_mapping[group_name]

        for idx in range(start_idx, end_idx + 1):
            names.append(f"{idx}.png")

    return sorted(names, key=lambda name: int(Path(name).stem))


def verify_png_folder(folder: Path, groups, target_mapping) -> dict:
    """
    Verify folder contains exactly expected PNG files.
    """

    folder = Path(folder)

    expected_names = set(expected_filenames_from_mapping(groups, target_mapping))

    if not folder.exists():
        return {
            "exists": False,
            "count": 0,
            "missing": sorted(expected_names),
            "extra": [],
            "ok": False,
        }

    actual_names = {path.name for path in folder.glob("*.png")}

    missing = sorted(
        expected_names - actual_names,
        key=lambda name: int(Path(name).stem),
    )

    extra = sorted(
        actual_names - expected_names,
        key=lambda name: int(Path(name).stem) if Path(name).stem.isdigit() else 10**9,
    )

    ok = len(missing) == 0 and len(extra) == 0 and len(actual_names) == len(expected_names)

    return {
        "exists": True,
        "count": len(actual_names),
        "missing": missing,
        "extra": extra,
        "ok": ok,
    }