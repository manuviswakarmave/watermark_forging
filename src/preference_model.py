import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PreferenceModel(nn.Module):
    """
    Preference model R(x).

    Higher score = cleaner / more natural image.
    Lower score  = image contains artificial artifacts.

    Supports:
        - convnextv2_tiny  : closer to the paper, requires timm
        - convnext_tiny    : torchvision fallback
        - resnet18         : simple baseline
    """

    def __init__(
        self,
        backbone: str = "convnextv2_tiny",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        input_size: int = 256,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.input_size = input_size

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        if backbone == "convnextv2_tiny":
            try:
                import timm
            except ImportError as exc:
                raise ImportError(
                    "ConvNeXt V2-Tiny requires timm. Install it with: pip install timm"
                ) from exc

            # This is close to the model family used in the paper.
            # num_classes=0 makes the model output feature vectors instead of class logits.
            self.encoder = timm.create_model(
                "convnextv2_tiny.fcmae_ft_in22k_in1k",
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )

            num_features = self.encoder.num_features

            self.preference_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

            self.uses_imagenet_norm = True

        elif backbone == "convnext_tiny":
            if pretrained:
                from torchvision.models import ConvNeXt_Tiny_Weights
                weights = ConvNeXt_Tiny_Weights.DEFAULT
            else:
                weights = None

            base_model = models.convnext_tiny(weights=weights)

            num_features = base_model.classifier[-1].in_features

            # Keep ConvNeXt feature extractor and remove final classifier.
            base_model.classifier[-1] = nn.Identity()

            self.encoder = base_model

            self.preference_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

            self.uses_imagenet_norm = True

        elif backbone == "resnet18":
            if pretrained:
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT
            else:
                weights = None

            base_model = models.resnet18(weights=weights)

            num_features = base_model.fc.in_features
            base_model.fc = nn.Identity()

            self.encoder = base_model

            self.preference_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

            self.uses_imagenet_norm = True

        else:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                f"Use 'convnextv2_tiny', 'convnext_tiny', or 'resnet18'."
            )

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        added_batch = False

        if x.dim() == 3:
            x = x.unsqueeze(0)
            added_batch = True

        if x.dim() != 4:
            raise ValueError(f"Expected [B, 3, H, W] or [3, H, W], got {x.shape}")

        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 RGB channels, got {x.shape[1]}")

        x = torch.clamp(x, 0.0, 1.0)

        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        if self.uses_imagenet_norm:
            x = self.normalize(x)

        features = self.encoder(x)

        # Some torchvision/timm models may return [B, C, 1, 1].
        # Convert that to [B, C].
        if features.dim() == 4:
            features = features.flatten(1)

        score = self.preference_head(features).squeeze(-1)

        if added_batch:
            score = score.squeeze(0)

        return score


def preference_ranking_loss(
    model: nn.Module,
    x_plus: torch.Tensor,
    x_minus: torch.Tensor,
) -> torch.Tensor:
    """
    Ranking loss:

        -log sigmoid(R(x_plus) - R(x_minus))

    We want:
        R(clean image) > R(corrupted image)
    """

    score_plus = model(x_plus)
    score_minus = model(x_minus)

    loss = -torch.log(torch.sigmoid(score_plus - score_minus) + 1e-8).mean()

    return loss


@torch.no_grad()
def score_images(
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    return model(images)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PreferenceModel(
        backbone="convnextv2_tiny",
        pretrained=False,
        freeze_backbone=False,
        input_size=256,
    ).to(device)

    x_clean = torch.rand(4, 3, 256, 256).to(device)
    x_corrupt = torch.clamp(x_clean + 0.05 * torch.randn_like(x_clean), 0.0, 1.0)

    scores_clean = model(x_clean)
    scores_corrupt = model(x_corrupt)

    loss = preference_ranking_loss(model, x_clean, x_corrupt)

    print("Clean scores shape:", scores_clean.shape)
    print("Corrupt scores shape:", scores_corrupt.shape)
    print("Clean scores:", scores_clean)
    print("Corrupt scores:", scores_corrupt)
    print("Ranking loss:", loss.item())