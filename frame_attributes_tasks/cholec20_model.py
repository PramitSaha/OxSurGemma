"""
CholecTrack20 frame-level multilabel model (inference only).

Model definition for loading checkpoint and running inference.
Outputs: operator (4-dim multi-hot), 10 binary flags (visibility, occluded, etc.).
"""
import torch.nn as nn
import torchvision.models as models

NUM_OPERATORS = 4  # MSLH=0, MSRH=1, ASRH=2, NULL=3

BINARY_ATTRS = [
    "visibility",
    "crowded",
    "visible",
    "occluded",
    "bleeding",
    "smoke",
    "blurred",
    "undercoverage",
    "reflection",
    "stainedlens",
]


def get_backbone(name: str, pretrained: bool = True):
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = 512
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        feat_dim = 2048
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = 1280
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return m, feat_dim


class Cholec20MultilabelModel(nn.Module):
    """Shared backbone + operator (multi-hot) + binary heads."""

    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        backbone, feat_dim = get_backbone(backbone_name, pretrained)
        if hasattr(backbone, "fc"):
            backbone.fc = nn.Identity()
        elif hasattr(backbone, "classifier"):
            backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head_operator = nn.Linear(feat_dim, NUM_OPERATORS)
        self.head_binary = nn.Linear(feat_dim, len(BINARY_ATTRS))

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "operator": self.head_operator(feat),
            "binary": self.head_binary(feat),
        }
