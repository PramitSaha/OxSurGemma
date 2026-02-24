#!/usr/bin/env python3
"""
Run CVS (Critical View of Safety) inference on an image.
Uses the four backbones (vgg, resnet, resnet18, densenet) and prints the three criteria.

Usage:
  python run_cvs_inference.py [--image PATH] [--log_root PATH]
  Default image: ../cholec20track/Training/VID02/Frames/006701.png
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# Add parent so we can import colenet
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from colenet.colenet_model import ColeNet

CRITERIA = [
    "two_structures_visible",
    "cystic_plate_dissected",
    "hepatocystic_triangle_cleared",
]


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model(backbone: str, log_dir: Path, device: torch.device) -> ColeNet:
    ckpt_path = log_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model = ColeNet(backbone=backbone)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="CVS inference (three criteria)")
    parser.add_argument(
        "--image",
        type=Path,
        default=_ROOT.parent / "cholec20track" / "Training" / "VID02" / "Frames" / "006701.png",
        help="Path to input image",
    )
    parser.add_argument(
        "--log_root",
        type=Path,
        default=_ROOT / "log",
        help="Log dir containing colenet_vgg, colenet_resnet, etc.",
    )
    args = parser.parse_args()

    image_path = args.image.resolve()
    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        sys.exit(1)
    log_root = args.log_root.resolve()
    if not log_root.is_dir():
        print(f"Error: log_root not found: {log_root}")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = get_transform()
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    backbones = ["vgg", "resnet", "resnet18", "densenet"]
    all_scores = []
    for backbone in backbones:
        model = load_model(backbone, log_root / f"colenet_{backbone}", device)
        with torch.no_grad():
            logits = model(x.float())
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            pred = torch.sigmoid(logits).cpu().numpy().squeeze()
        all_scores.append(pred)
    scores = sum(all_scores) / len(all_scores)

    print(f"Image: {image_path}")
    print("Critical View of Safety (CVS) – aggregate of vgg, resnet, resnet18, densenet:")
    threshold = 0.5
    for i, name in enumerate(CRITERIA):
        score = float(scores[i]) if i < len(scores) else 0.0
        achieved = "achieved" if score >= threshold else "not achieved"
        print(f"  - {name.replace('_', ' ').title()}: {achieved} (score {score:.3f})")
    print(f"\nRaw scores: {[round(float(s), 3) for s in scores]}")


if __name__ == "__main__":
    main()
