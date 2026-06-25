import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from config import (
    CLEAN_TARGETS_DIR,
    CHECKPOINT_DIR,
    SEED,
    DEVICE,
    PREFERENCE_BACKBONE,
    PREFERENCE_INPUT_SIZE,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DROPOUT,
    PRETRAINED,
    FREEZE_BACKBONE,
    MIN_ARTIFACT_STRENGTH,
    MAX_ARTIFACT_STRENGTH,
    USE_ADV_NEGATIVE,
    ADV_STEPS,
    ADV_STEP_SIZE,
    ADV_WEIGHT,
    VAL_FRACTION,
    NUM_WORKERS,
    PRINT_EVERY,
)

from preference_model import PreferenceModel, preference_ranking_loss
from synthetic_artifacts import ranking_pair_from_clean


class CleanImageDataset(Dataset):
    """
    Dataset of clean images.

    Each clean image x is later converted into a preference pair:

        x_plus  = clean image
        x_minus = clean image + synthetic artifact

    The preference model learns:

        R(x_plus) > R(x_minus)
    """

    def __init__(self, image_dir: Path, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(self.image_dir.glob("*.png"))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No PNG images found in {self.image_dir}")

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


def set_seed(seed: int):
    """
    Make training more reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Select device from config.py.
    """

    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(DEVICE)


def make_train_val_datasets() -> Tuple[Subset, Subset]:
    """
    Build train and validation datasets.

    We use the same train/validation image split, but different transforms:

        train: random crop, flip, color jitter
        val: deterministic center crop
    """

    resize_size = PREFERENCE_INPUT_SIZE + 32

    train_transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomResizedCrop(
                size=PREFERENCE_INPUT_SIZE,
                scale=(0.70, 1.0),
                ratio=(0.90, 1.10),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(PREFERENCE_INPUT_SIZE),
            transforms.ToTensor(),
        ]
    )

    train_base_dataset = CleanImageDataset(
        image_dir=CLEAN_TARGETS_DIR,
        transform=train_transform,
    )

    val_base_dataset = CleanImageDataset(
        image_dir=CLEAN_TARGETS_DIR,
        transform=val_transform,
    )

    total_size = len(train_base_dataset)
    val_size = max(1, int(total_size * VAL_FRACTION))
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(total_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(val_base_dataset, val_indices)

    return train_dataset, val_dataset


def make_adversarial_negative(
    model: torch.nn.Module,
    x_minus: torch.Tensor,
    steps: int = 1,
    step_size: float = 0.005,
) -> torch.Tensor:
    """
    Optional paper-inspired adversarial negative generation.

    Starting from a corrupted image x_minus, move it slightly in the direction
    that increases the preference score R(x_minus).

    Then the model is still trained to prefer the clean image over this stronger
    corrupted image.

    This is useful later, but for the first ConvNeXt run you can keep:

        USE_ADV_NEGATIVE = False
    """

    x_adv = x_minus.detach().clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        score = model(x_adv).mean()

        grad = torch.autograd.grad(
            outputs=score,
            inputs=x_adv,
            create_graph=False,
            retain_graph=False,
        )[0]

        grad_norm = grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8
        grad = grad / grad_norm

        x_adv = x_adv + step_size * grad
        x_adv = torch.clamp(x_adv.detach(), 0.0, 1.0)

    return x_adv


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the preference model.

    Validation preference accuracy means:

        how often R(clean image) > R(corrupted image)
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_clean in val_loader:
        x_clean = x_clean.to(device)

        x_plus, x_minus = ranking_pair_from_clean(
            x_clean,
            artifact_type="random",
            strength_range=(MIN_ARTIFACT_STRENGTH, MAX_ARTIFACT_STRENGTH),
        )

        x_plus = x_plus.to(device)
        x_minus = x_minus.to(device)

        loss = preference_ranking_loss(model, x_plus, x_minus)

        score_plus = model(x_plus)
        score_minus = model(x_minus)

        correct = (score_plus > score_minus).sum().item()
        batch_size = x_clean.size(0)

        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    pref_accuracy = total_correct / max(total_samples, 1)

    return avg_loss, pref_accuracy


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
):
    """
    Save training checkpoint.

    This saves:
        - model weights
        - optimizer state
        - epoch
        - validation metrics
        - important config values
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "config": {
            "seed": SEED,
            "device": DEVICE,
            "preference_backbone": PREFERENCE_BACKBONE,
            "preference_input_size": PREFERENCE_INPUT_SIZE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "pretrained": PRETRAINED,
            "freeze_backbone": FREEZE_BACKBONE,
            "min_artifact_strength": MIN_ARTIFACT_STRENGTH,
            "max_artifact_strength": MAX_ARTIFACT_STRENGTH,
            "use_adv_negative": USE_ADV_NEGATIVE,
            "adv_steps": ADV_STEPS,
            "adv_step_size": ADV_STEP_SIZE,
            "adv_weight": ADV_WEIGHT,
            "val_fraction": VAL_FRACTION,
        },
    }

    torch.save(checkpoint, path)


def train():
    """
    Main training function.
    """

    set_seed(SEED)

    device = get_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = make_train_val_datasets()

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Backbone: {PREFERENCE_BACKBONE}")
    print(f"Input size: {PREFERENCE_INPUT_SIZE}")
    print(f"Pretrained: {PRETRAINED}")
    print(f"Freeze backbone: {FREEZE_BACKBONE}")
    print(f"Artifact strength: {MIN_ARTIFACT_STRENGTH} to {MAX_ARTIFACT_STRENGTH}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = PreferenceModel(
        backbone=PREFERENCE_BACKBONE,
        pretrained=PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,
        dropout=DROPOUT,
        input_size=PREFERENCE_INPUT_SIZE,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for step, x_clean in enumerate(train_loader, start=1):
            x_clean = x_clean.to(device)

            x_plus, x_minus = ranking_pair_from_clean(
                x_clean,
                artifact_type="random",
                strength_range=(MIN_ARTIFACT_STRENGTH, MAX_ARTIFACT_STRENGTH),
            )

            x_plus = x_plus.to(device)
            x_minus = x_minus.to(device)

            loss = preference_ranking_loss(model, x_plus, x_minus)

            if USE_ADV_NEGATIVE:
                x_minus_adv = make_adversarial_negative(
                    model=model,
                    x_minus=x_minus,
                    steps=ADV_STEPS,
                    step_size=ADV_STEP_SIZE,
                )

                adv_loss = preference_ranking_loss(model, x_plus, x_minus_adv)
                loss = loss + ADV_WEIGHT * adv_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                score_plus = model(x_plus)
                score_minus = model(x_minus)
                correct = (score_plus > score_minus).sum().item()

            batch_size = x_clean.size(0)

            running_loss += loss.item() * batch_size
            running_correct += correct
            running_samples += batch_size

            if step % PRINT_EVERY == 0 or step == len(train_loader):
                avg_loss = running_loss / max(running_samples, 1)
                avg_acc = running_correct / max(running_samples, 1)

                print(
                    f"Epoch [{epoch}/{EPOCHS}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"Train Loss: {avg_loss:.4f} "
                    f"Train Pref Acc: {avg_acc:.4f}"
                )

        train_loss = running_loss / max(running_samples, 1)
        train_acc = running_correct / max(running_samples, 1)

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"\nEpoch [{epoch}/{EPOCHS}] completed | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Pref Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Pref Acc: {val_acc:.4f}\n"
        )

        last_checkpoint_path = CHECKPOINT_DIR / "preference_model_last.pt"

        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_checkpoint_path = CHECKPOINT_DIR / "preference_model_best.pt"

            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
            )

            print(f"Saved best checkpoint to {best_checkpoint_path}")

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()