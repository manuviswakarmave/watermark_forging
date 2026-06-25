import random
from typing import Literal, Tuple

import torch
import torch.nn.functional as F


ArtifactType = Literal["random", "wave", "line", "noise"]


def _ensure_batch(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """
    Accepts [C, H, W] or [B, C, H, W].
    Returns [B, C, H, W] and whether a batch dimension was added.
    """
    if x.dim() == 3:
        return x.unsqueeze(0), True
    if x.dim() == 4:
        return x, False
    raise ValueError(f"Expected tensor shape [C,H,W] or [B,C,H,W], got {x.shape}")


def _remove_batch(x: torch.Tensor, added_batch: bool) -> torch.Tensor:
    if added_batch:
        return x.squeeze(0)
    return x


def _normalize_artifact(artifact: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize artifact per image to roughly [-1, 1].
    """
    b = artifact.shape[0]
    flat = artifact.view(b, -1)
    max_abs = flat.abs().max(dim=1)[0].view(b, 1, 1, 1)
    return artifact / (max_abs + eps)


def _maybe_grayscale(artifact: torch.Tensor, grayscale_prob: float = 0.5) -> torch.Tensor:
    """
    With some probability, make artifact identical across RGB channels.
    """
    b, c, h, w = artifact.shape

    if c != 3:
        return artifact

    masks = []
    for _ in range(b):
        masks.append(random.random() < grayscale_prob)

    for i, use_gray in enumerate(masks):
        if use_gray:
            gray = artifact[i].mean(dim=0, keepdim=True)
            artifact[i] = gray.repeat(c, 1, 1)

    return artifact


def generate_wave_artifact(
    x: torch.Tensor,
    min_waves: int = 2,
    max_waves: int = 6,
    grayscale_prob: float = 0.5,
) -> torch.Tensor:
    """
    Generate sinusoidal wave-like artifacts.

    This is a lightweight approximation of the paper's Fourier wave artifacts.
    """
    x, added_batch = _ensure_batch(x)
    b, c, h, w = x.shape
    device = x.device
    dtype = x.dtype

    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h, device=device, dtype=dtype),
        torch.linspace(0, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )

    artifact = torch.zeros_like(x)

    for i in range(b):
        num_waves = random.randint(min_waves, max_waves)

        for _ in range(num_waves):
            freq = random.uniform(4.0, 30.0)
            theta = random.uniform(0.0, 2.0 * 3.14159265)
            phase = random.uniform(0.0, 2.0 * 3.14159265)

            direction_x = torch.cos(torch.tensor(theta, device=device, dtype=dtype))
            direction_y = torch.sin(torch.tensor(theta, device=device, dtype=dtype))

            pattern = torch.sin(
                2.0 * 3.14159265 * freq * (direction_x * xx + direction_y * yy)
                + phase
            )

            channel_scale = torch.randn(c, device=device, dtype=dtype).view(c, 1, 1)
            artifact[i] += channel_scale * pattern.unsqueeze(0)

    artifact = _normalize_artifact(artifact)
    artifact = _maybe_grayscale(artifact, grayscale_prob)

    return _remove_batch(artifact, added_batch)


def generate_line_artifact(
    x: torch.Tensor,
    min_lines: int = 3,
    max_lines: int = 12,
    grayscale_prob: float = 0.5,
) -> torch.Tensor:
    """
    Generate horizontal/vertical line-style artifacts.
    """
    x, added_batch = _ensure_batch(x)
    b, c, h, w = x.shape
    device = x.device
    dtype = x.dtype

    artifact = torch.zeros_like(x)

    for i in range(b):
        num_lines = random.randint(min_lines, max_lines)

        for _ in range(num_lines):
            orientation = random.choice(["horizontal", "vertical"])
            thickness = random.uniform(1.0, 3.0)
            strength = random.uniform(-1.0, 1.0)

            if orientation == "horizontal":
                center = random.uniform(0, h - 1)
                coords = torch.arange(h, device=device, dtype=dtype).view(h, 1)
                line = torch.exp(-((coords - center) ** 2) / (2 * thickness ** 2))
                line = line.repeat(1, w)
            else:
                center = random.uniform(0, w - 1)
                coords = torch.arange(w, device=device, dtype=dtype).view(1, w)
                line = torch.exp(-((coords - center) ** 2) / (2 * thickness ** 2))
                line = line.repeat(h, 1)

            channel_scale = torch.randn(c, device=device, dtype=dtype).view(c, 1, 1)
            artifact[i] += strength * channel_scale * line.unsqueeze(0)

    artifact = _normalize_artifact(artifact)
    artifact = _maybe_grayscale(artifact, grayscale_prob)

    return _remove_batch(artifact, added_batch)


def generate_fourier_noise_artifact(
    x: torch.Tensor,
    grayscale_prob: float = 0.5,
) -> torch.Tensor:
    """
    Generate smooth/random Fourier-style artifact.

    This creates random noise in image space, transforms it to Fourier space,
    applies a frequency envelope, and transforms it back.
    """
    x, added_batch = _ensure_batch(x)
    b, c, h, w = x.shape
    device = x.device
    dtype = x.dtype

    noise = torch.randn_like(x)

    fft = torch.fft.fft2(noise, dim=(-2, -1))
    fft = torch.fft.fftshift(fft, dim=(-2, -1))

    fy = torch.linspace(-1, 1, h, device=device, dtype=dtype).view(h, 1)
    fx = torch.linspace(-1, 1, w, device=device, dtype=dtype).view(1, w)

    radius = torch.sqrt(fx ** 2 + fy ** 2)

    sigma = random.uniform(0.15, 0.75)
    envelope = torch.exp(-(radius ** 2) / (2 * sigma ** 2))

    # Randomly emphasize either lower/mid frequencies or higher frequencies.
    if random.random() < 0.5:
        envelope = 1.0 - envelope

    envelope = envelope.view(1, 1, h, w)

    filtered_fft = fft * envelope
    filtered_fft = torch.fft.ifftshift(filtered_fft, dim=(-2, -1))

    artifact = torch.fft.ifft2(filtered_fft, dim=(-2, -1)).real

    artifact = _normalize_artifact(artifact)
    artifact = _maybe_grayscale(artifact, grayscale_prob)

    return _remove_batch(artifact, added_batch)


def generate_synthetic_artifact(
    x: torch.Tensor,
    artifact_type: ArtifactType = "random",
    grayscale_prob: float = 0.5,
) -> torch.Tensor:
    """
    Generate one synthetic artifact with the same shape as x.

    x should be in [0, 1].
    """
    if artifact_type == "random":
        artifact_type = random.choice(["wave", "line", "noise"])

    if artifact_type == "wave":
        return generate_wave_artifact(x, grayscale_prob=grayscale_prob)

    if artifact_type == "line":
        return generate_line_artifact(x, grayscale_prob=grayscale_prob)

    if artifact_type == "noise":
        return generate_fourier_noise_artifact(x, grayscale_prob=grayscale_prob)

    raise ValueError(f"Unknown artifact_type: {artifact_type}")


def add_synthetic_artifact(
    x: torch.Tensor,
    artifact_type: ArtifactType = "random",
    strength_range: Tuple[float, float] = (0.01, 0.05),
    grayscale_prob: float = 0.5,
    clamp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add synthetic artifact to image tensor.

    Returns:
        corrupted image x_minus
        actual artifact added

    Example:
        x_clean = ...
        x_corrupt, artifact = add_synthetic_artifact(x_clean)

    x_clean and x_corrupt are expected in [0, 1].
    """
    x_batch, added_batch = _ensure_batch(x)

    artifact = generate_synthetic_artifact(
        x_batch,
        artifact_type=artifact_type,
        grayscale_prob=grayscale_prob,
    )

    artifact_batch, _ = _ensure_batch(artifact)

    b = x_batch.shape[0]
    strengths = torch.empty(
        b, 1, 1, 1,
        device=x_batch.device,
        dtype=x_batch.dtype,
    ).uniform_(strength_range[0], strength_range[1])

    artifact_batch = artifact_batch * strengths
    x_corrupt = x_batch + artifact_batch

    if clamp:
        x_corrupt = torch.clamp(x_corrupt, 0.0, 1.0)

    return (
        _remove_batch(x_corrupt, added_batch),
        _remove_batch(artifact_batch, added_batch),
    )


def ranking_pair_from_clean(
    x_clean: torch.Tensor,
    artifact_type: ArtifactType = "random",
    strength_range: Tuple[float, float] = (0.01, 0.05),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a preference training pair.

    Returns:
        x_plus  = clean image
        x_minus = synthetically corrupted image

    The preference model should learn:
        R(x_plus) > R(x_minus)
    """
    x_minus, _ = add_synthetic_artifact(
        x_clean,
        artifact_type=artifact_type,
        strength_range=strength_range,
        clamp=True,
    )

    x_plus = x_clean

    return x_plus, x_minus


if __name__ == "__main__":
    # Quick sanity check
    dummy = torch.rand(4, 3, 224, 224)

    x_plus, x_minus = ranking_pair_from_clean(dummy)

    print("Clean shape:", x_plus.shape)
    print("Corrupted shape:", x_minus.shape)
    print("Clean range:", x_plus.min().item(), x_plus.max().item())
    print("Corrupted range:", x_minus.min().item(), x_minus.max().item())