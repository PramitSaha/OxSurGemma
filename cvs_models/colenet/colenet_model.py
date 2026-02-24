import torch.nn as nn
import torchvision.models as models


class ColeNet(nn.Module):
    def __init__(self, backbone):
        super(ColeNet, self).__init__()
        self.model = self.get_backbone(backbone=backbone)

    def get_backbone(self, backbone):
        if backbone == "vgg":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, 3)
        elif backbone == "resnet":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, 3)
        elif backbone == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, 3)
        elif backbone == "densenet":
            model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
            model.classifier = nn.Linear(model.classifier.in_features, 3)
        elif backbone == "inception":
            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, 3)
        elif backbone == "efficientnet":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        else:
            raise NotImplementedError(f"Unknown backbone: {backbone}")
        return model

    def forward(self, x):
        if hasattr(self.model, "fc") and "inception" in str(type(self.model)).lower():
            x, _ = self.model(x)
            return x
        return self.model(x)
