

from pathlib import Path
import sys
import argparse
import csv
import math
from typing import List, Optional, Dict

import numpy as np
import torch


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from residual_template.config_residual import (
    CLEAN_TARGETS_DIR,
    EXPERIMENTS,
    RESIDUAL_QUALITY_REPORT_ROOT,
    get_experiment_output_dir,
)

from residual_template.image_ops import (
    load_rgb,
    resize_to_shape,
    sort_by_numeric_stem,
)


try:
    import lpips
except ImportError:
    lpips = None


def s_qlt_from_lpips(value: float) -> float:
    return float(math.exp(-8.0 * value))


def image_to_lpips_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    image = np.clip(image.astype(np.float32), 0.0, 1.0)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0

    return tensor.to(device=device, dtype=torch.float32)


def create_lpips_model(net: str, device: torch.device):
    if lpips is None:
        raise ImportError(
            "lpips is not installed. Install it with:\n"
            "  pip install lpips"
        )

    model = lpips.LPIPS(net=net)
    model = model.to(device)
    model.eval()

    return model


def compute_mse(clean: np.ndarray, forged: np.ndarray) -> float:
    return float(np.mean((clean.astype(np.float32) - forged.astype(np.float32)) ** 2))


def compute_psnr(clean: np.ndarray, forged: np.ndarray) -> float:
    mse = compute_mse(clean, forged)

    if mse <= 1e-12:
        return 99.0

    return float(-10.0 * math.log10(mse))


@torch.no_grad()
def evaluate_pair(
    clean_path: Path,
    forged_path: Path,
    model,
    device: torch.device,
) -> Dict:
    clean = load_rgb(clean_path)
    forged = load_rgb(forged_path)

    if forged.shape != clean.shape:
        forged = resize_to_shape(forged, clean.shape)

    clean_tensor = image_to_lpips_tensor(clean, device)
    forged_tensor = image_to_lpips_tensor(forged, device)

    lpips_value = float(model(clean_tensor, forged_tensor).item())

    diff = forged - clean

    return {
        "filename": forged_path.name,
        "lpips": lpips_value,
        "s_qlt": s_qlt_from_lpips(lpips_value),
        "mse": compute_mse(clean, forged),
        "psnr": compute_psnr(clean, forged),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }


@torch.no_grad()
def evaluate_experiment(
    experiment_name: str,
    model,
    device: torch.device,
    max_images: Optional[int] = None,
) -> tuple[Dict, List[Dict]]:
    folder = get_experiment_output_dir(experiment_name)

    if not folder.exists():
        raise FileNotFoundError(f"Experiment folder does not exist: {folder}")

    forged_paths = sort_by_numeric_stem(list(folder.glob("*.png")))

    if max_images is not None:
        forged_paths = forged_paths[:max_images]

    per_image = []

    print()
    print("=" * 80)
    print(f"Evaluating experiment: {experiment_name}")
    print(f"Folder: {folder}")
    print(f"Images: {len(forged_paths)}")
    print("=" * 80)

    for idx, forged_path in enumerate(forged_paths, start=1):
        clean_path = CLEAN_TARGETS_DIR / forged_path.name

        if not clean_path.exists():
            raise FileNotFoundError(f"Missing clean target: {clean_path}")

        row = evaluate_pair(
            clean_path=clean_path,
            forged_path=forged_path,
            model=model,
            device=device,
        )

        row["experiment"] = experiment_name
        row["folder"] = str(folder)

        per_image.append(row)

        if idx % 25 == 0 or idx == len(forged_paths):
            print(
                f"  {idx}/{len(forged_paths)} "
                f"| LPIPS={row['lpips']:.6f} "
                f"| S_qlt={row['s_qlt']:.6f}"
            )

    lpips_values = np.array([r["lpips"] for r in per_image], dtype=np.float64)
    s_values = np.array([r["s_qlt"] for r in per_image], dtype=np.float64)
    psnr_values = np.array([r["psnr"] for r in per_image], dtype=np.float64)
    mean_abs_values = np.array([r["mean_abs_diff"] for r in per_image], dtype=np.float64)

    summary = {
        "experiment": experiment_name,
        "num_images": len(per_image),
        "mean_lpips": float(lpips_values.mean()),
        "median_lpips": float(np.median(lpips_values)),
        "max_lpips": float(lpips_values.max()),
        "mean_s_qlt": float(s_values.mean()),
        "s_qlt_from_mean_lpips": s_qlt_from_lpips(float(lpips_values.mean())),
        "mean_psnr": float(psnr_values.mean()),
        "mean_abs_diff": float(mean_abs_values.mean()),
        "folder": str(folder),
    }

    return summary, per_image


def save_summary_csv(summaries: List[Dict]) -> Path:
    save_path = RESIDUAL_QUALITY_REPORT_ROOT / "quality_summary.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment",
        "num_images",
        "mean_lpips",
        "median_lpips",
        "max_lpips",
        "mean_s_qlt",
        "s_qlt_from_mean_lpips",
        "mean_psnr",
        "mean_abs_diff",
        "folder",
    ]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summaries:
            writer.writerow(row)

    print(f"Saved summary CSV: {save_path}")

    return save_path


def print_ranked_summary(summaries: List[Dict]) -> None:
    ranked = sorted(summaries, key=lambda row: row["mean_lpips"])

    print()
    print("=" * 100)
    print("Quality ranking")
    print("=" * 100)
    print(
        f"{'rank':>4}  {'experiment':<25}  "
        f"{'LPIPS':>10}  {'S_qlt':>10}  {'PSNR':>10}  {'mean_abs_diff':>14}"
    )
    print("-" * 100)

    for idx, row in enumerate(ranked, start=1):
        print(
            f"{idx:>4}  "
            f"{row['experiment']:<25}  "
            f"{row['mean_lpips']:>10.6f}  "
            f"{row['s_qlt_from_mean_lpips']:>10.6f}  "
            f"{row['mean_psnr']:>10.3f}  "
            f"{row['mean_abs_diff']:>14.6f}"
        )

    print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LPIPS quality for residual-template outputs."
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Experiment names to evaluate.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu.",
    )

    parser.add_argument(
        "--lpips-net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional quick test limit.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.experiments is None:
        experiment_names = [experiment["name"] for experiment in EXPERIMENTS]
    else:
        experiment_names = args.experiments

    device = torch.device(args.device)

    print()
    print("=" * 80)
    print("Residual-template quality evaluation")
    print("=" * 80)
    print(f"Experiments: {experiment_names}")
    print(f"Device:      {device}")
    print(f"LPIPS net:   {args.lpips_net}")
    print("=" * 80)

    model = create_lpips_model(args.lpips_net, device)

    summaries = []

    for experiment_name in experiment_names:
        summary, _ = evaluate_experiment(
            experiment_name=experiment_name,
            model=model,
            device=device,
            max_images=args.max_images,
        )

        summaries.append(summary)

    save_summary_csv(summaries)
    print_ranked_summary(summaries)


if __name__ == "__main__":
    main()