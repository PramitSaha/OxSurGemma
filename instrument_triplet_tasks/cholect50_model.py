"""
CholecT50 multi-head model (inference only).

Shared backbone + task-specific heads for tool, verb, target, phase, triplet.
Used by triplet_recognition tool; load checkpoints from cholect50_checkpoints/.
"""
import torch.nn as nn
import torchvision.models as models

# CholecT50 class counts (must match checkpoint training)
NUM_TRIPLETS = 100
NUM_TOOLS = 6
NUM_VERBS = 10
NUM_TARGETS = 15
NUM_PHASES = 7


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


class MultiHeadModel(nn.Module):
    """Shared backbone + task-specific heads (tool, verb, target, phase, triplet)."""

    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True, tasks: list = None):
        super().__init__()
        if tasks is None:
            tasks = ["phase"]
        self.tasks = tasks

        backbone, feat_dim = get_backbone(backbone_name, pretrained)
        if hasattr(backbone, "fc"):
            backbone.fc = nn.Identity()
        elif hasattr(backbone, "classifier"):
            backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.heads = nn.ModuleDict()
        if "phase" in tasks:
            self.heads["phase"] = nn.Linear(feat_dim, NUM_PHASES)
        if "tool" in tasks:
            self.heads["tool"] = nn.Linear(feat_dim, NUM_TOOLS)
        if "verb" in tasks:
            self.heads["verb"] = nn.Linear(feat_dim, NUM_VERBS)
        if "target" in tasks:
            self.heads["target"] = nn.Linear(feat_dim, NUM_TARGETS)
        if "triplet" in tasks:
            self.heads["triplet"] = nn.Linear(feat_dim, NUM_TRIPLETS)

    def forward(self, x):
        feat = self.backbone(x)
        out = {}
        for t, head in self.heads.items():
            out[t] = head(feat)
        return out
