from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import (
    WATERMARKED_SOURCES_DIR,
    RESIDUAL_DIR,
    PREFERENCE_CKPT_PATH,
    CATEGORIES,
    DEVICE,
    PREFERENCE_BACKBONE,
    PREFERENCE_INPUT_SIZE,
    DROPOUT,
    EXTRACT_STEPS,
    EXTRACT_LR,
    EXTRACT_L2_WEIGHT,
    EXTRACT_MAX_DELTA,
    RESIDUAL_AGGREGATION,
)

from preference_model import PreferenceModel


def get_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)


def load_preference_model(device: torch.device) -> PreferenceModel:
    """
    Load the trained preference model.

    Important:
    We set pretrained=False here because the checkpoint already contains
    the trained weights. This avoids downloading the backbone again.
    """

    checkpoint = torch.load(PREFERENCE_CKPT_PATH, map_location=device)

    ckpt_config = checkpoint.get("config", {})

    backbone = ckpt_config.get("preference_backbone", PREFERENCE_BACKBONE)
    input_size = ckpt_config.get("preference_input_size", PREFERENCE_INPUT_SIZE)

    model = PreferenceModel(
        backbone=backbone,
        pretrained=False,
        freeze_backbone=False,
        dropout=DROPOUT,
        input_size=input_size,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {PREFERENCE_CKPT_PATH}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint val loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"Checkpoint val acc: {checkpoint.get('val_acc', 'unknown')}")
    print(f"Backbone: {backbone}")
    print(f"Input size: {input_size}")

    return model


def load_image_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    """
    Load image as tensor in [0, 1].

    Output shape:
        [1, 3, PREFERENCE_INPUT_SIZE, PREFERENCE_INPUT_SIZE]
    """

    transform = transforms.Compose(
        [
            transforms.Resize((PREFERENCE_INPUT_SIZE, PREFERENCE_INPUT_SIZE)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    return tensor


def save_image_tensor(tensor: torch.Tensor, out_path: Path):
    """
    Save [1, 3, H, W] or [3, H, W] tensor in [0, 1] as PNG.
    """

    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    array = np.transpose(array, (1, 2, 0))
    array = (array * 255.0).round().astype(np.uint8)

    Image.fromarray(array).save(out_path)


def save_residual_preview(
    residual_chw: np.ndarray,
    out_path: Path,
    scale: float = 8.0,
):
    """
    Save an amplified preview of the residual.

    Residual values are small and centered around 0.
    For viewing, we map:

        residual * scale + 0.5

    This preview is only for debugging.
    """

    residual_hwc = np.transpose(residual_chw, (1, 2, 0))

    preview = residual_hwc * scale + 0.5
    preview = np.clip(preview, 0.0, 1.0)
    preview = (preview * 255.0).round().astype(np.uint8)

    Image.fromarray(preview).save(out_path)


def extract_residual_from_image(
    model: PreferenceModel,
    x_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract watermark-like residual from one watermarked image.

    We optimize delta such that:

        x_hat = x_w - delta

    gets a higher preference score.

    Objective:

        minimize -R(x_hat) + lambda * ||delta||^2

    Returns:
        delta: extracted residual
        x_hat: cleaned image estimate
    """

    model.eval()

    x_w = x_w.detach()

    delta = torch.zeros_like(x_w, requires_grad=True)

    optimizer = torch.optim.Adam([delta], lr=EXTRACT_LR)

    for step in range(1, EXTRACT_STEPS + 1):
        optimizer.zero_grad()

        x_hat = torch.clamp(x_w - delta, 0.0, 1.0)

        preference_score = model(x_hat).mean()

        l2_penalty = delta.pow(2).mean()

        loss = -preference_score + EXTRACT_L2_WEIGHT * l2_penalty

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.clamp_(-EXTRACT_MAX_DELTA, EXTRACT_MAX_DELTA)

        if step == 1 or step % 10 == 0 or step == EXTRACT_STEPS:
            print(
                f"      Step [{step}/{EXTRACT_STEPS}] "
                f"Score: {preference_score.item():.4f} "
                f"L2: {l2_penalty.item():.6f} "
                f"Loss: {loss.item():.4f} "
                f"Delta mean abs: {delta.abs().mean().item():.6f}"
            )

    with torch.no_grad():
        x_hat = torch.clamp(x_w - delta, 0.0, 1.0)

    return delta.detach(), x_hat.detach()


def aggregate_residuals(residuals: np.ndarray) -> np.ndarray:
    """
    residuals shape:
        [N, 3, H, W]

    Returns:
        [3, H, W]
    """

    if RESIDUAL_AGGREGATION == "mean":
        return residuals.mean(axis=0).astype(np.float32)

    if RESIDUAL_AGGREGATION == "median":
        return np.median(residuals, axis=0).astype(np.float32)

    raise ValueError(f"Unknown RESIDUAL_AGGREGATION: {RESIDUAL_AGGREGATION}")


def main():
    device = get_device()
    print(f"Using device: {device}")

    RESIDUAL_DIR.mkdir(parents=True, exist_ok=True)

    model = load_preference_model(device)

    for wm_name, _, _ in CATEGORIES:
        print(f"\n==============================")
        print(f"Extracting residuals for {wm_name}")
        print(f"==============================")

        source_dir = WATERMARKED_SOURCES_DIR / wm_name
        source_paths = sorted(source_dir.glob("*.png"))

        if len(source_paths) == 0:
            print(f"[WARNING] No PNG source images found in {source_dir}")
            continue

        print(f"Found {len(source_paths)} source images")

        wm_out_dir = RESIDUAL_DIR / wm_name
        wm_out_dir.mkdir(parents=True, exist_ok=True)

        residual_list = []

        for idx, image_path in enumerate(source_paths, start=1):
            print(f"\n  Processing {idx}/{len(source_paths)}: {image_path.name}")

            x_w = load_image_tensor(image_path, device)

            delta, x_hat = extract_residual_from_image(model, x_w)

            residual_np = delta.squeeze(0).cpu().numpy().astype(np.float32)
            residual_list.append(residual_np)

            individual_residual_path = wm_out_dir / f"{image_path.stem}_residual.npy"
            np.save(individual_residual_path, residual_np)

            preview_path = wm_out_dir / f"{image_path.stem}_residual_preview.png"
            save_residual_preview(residual_np, preview_path)

            cleaned_path = wm_out_dir / f"{image_path.stem}_cleaned_preview.png"
            save_image_tensor(x_hat, cleaned_path)

        residuals = np.stack(residual_list, axis=0)

        all_residuals_path = RESIDUAL_DIR / f"{wm_name}_all_residuals.npy"
        np.save(all_residuals_path, residuals)

        aggregated = aggregate_residuals(residuals)

        aggregated_path = RESIDUAL_DIR / f"{wm_name}_residual.npy"
        np.save(aggregated_path, aggregated)

        aggregated_preview_path = RESIDUAL_DIR / f"{wm_name}_residual_preview.png"
        save_residual_preview(aggregated, aggregated_preview_path)

        print(f"\nSaved all residuals: {all_residuals_path}")
        print(f"Saved aggregated residual: {aggregated_path}")
        print(f"Saved aggregated preview: {aggregated_preview_path}")

    print("\nResidual extraction finished.")


if __name__ == "__main__":
    main()