"""
Critical View of Safety (CVS) tool using Cholec80-CVS ColeNet.
Two modes: --cvs in terminal → single ResNet18 (log/new_cvs_model_1); no flag → old ensemble (vgg, resnet, resnet18, densenet).
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pydantic import BaseModel, Field, PrivateAttr

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from surgical_copilot.registry import register
from surgical_copilot.tools.base import SurgicalImageInput


# Path to CVS models (sibling of surgical_copilot)
_COLE_CVS_ROOT = Path(__file__).resolve().parent.parent.parent / "cvs_models"
_OLD_LOG_ROOT = _COLE_CVS_ROOT / "log"  # four backbones: colenet_vgg, colenet_resnet, colenet_resnet18, colenet_densenet
_NEW_CVS_MODEL_DIR = _COLE_CVS_ROOT / "log" / "new_cvs_model_1"  # single ResNet18
_CVS_CRITERIA = [
    "two_structures_visible",      # cystic duct and cystic artery
    "cystic_plate_dissected",
    "hepatocystic_triangle_cleared",
]


def _ensure_colenet_path():
    if str(_COLE_CVS_ROOT) not in sys.path:
        sys.path.insert(0, str(_COLE_CVS_ROOT))


def _get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _load_model(backbone: str, checkpoint_dir: Path, device: torch.device):
    _ensure_colenet_path()
    from colenet.colenet_model import ColeNet
    model = ColeNet(backbone=backbone)
    ckpt_path = checkpoint_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


class _CVSModelCacheSingle:
    """Single ResNet18 from model_dir (e.g. log/new_cvs_model_1)."""
    def __init__(self, model_dir: Path, backbone: str = "resnet18"):
        self.model_dir = model_dir
        self.backbone = backbone
        self._model: Optional[Any] = None
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_model(self):
        if self._model is None:
            self._model = _load_model(self.backbone, self.model_dir, self._device)
        return self._model

    def run_inference(self, image_tensor: torch.Tensor) -> np.ndarray:
        model = self.get_model()
        with torch.no_grad():
            x = image_tensor.unsqueeze(0).to(self._device)
            logits = model(x.float())
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            pred = torch.sigmoid(logits).cpu().numpy().squeeze()
        return pred.astype(np.float64)


class _CVSModelCacheEnsemble:
    """Four backbones (vgg, resnet, resnet18, densenet) from log_root, mean of scores."""
    def __init__(self, log_root: Path):
        self.log_root = log_root
        self._models: Dict[str, Any] = {}
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._backbones: List[str] = ["vgg", "resnet", "resnet18", "densenet"]

    def get(self, backbone: str):
        if backbone not in self._models:
            log_dir = self.log_root / f"colenet_{backbone}"
            self._models[backbone] = _load_model(backbone, log_dir, self._device)
        return self._models[backbone]

    def run_inference(self, image_tensor: torch.Tensor) -> np.ndarray:
        scores_list = []
        for backbone in self._backbones:
            model = self.get(backbone)
            with torch.no_grad():
                x = image_tensor.unsqueeze(0).to(self._device)
                logits = model(x.float())
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                pred = torch.sigmoid(logits).cpu().numpy().squeeze()
            scores_list.append(pred)
        return np.mean(scores_list, axis=0).astype(np.float64)


def _use_new_cvs_from_env() -> bool:
    """True if terminal was started with --cvs (new single model); else False (old ensemble)."""
    return os.environ.get("SURGICAL_CVS_USE_NEW_MODEL", "0").strip().lower() in ("1", "true", "yes")


@register("critical_view_of_safety")
def make_critical_view_of_safety_tool():
    use_new = _use_new_cvs_from_env()
    return CriticalViewOfSafetyTool(use_new_cvs_model=use_new)


class CriticalViewOfSafetyTool(BaseTool):
    """Assess Critical View of Safety (CVS) from a surgical/endoscopy image using Cholec80-CVS models."""

    name: str = "critical_view_of_safety"
    description: str = (
        "critical_view_of_safety: assesses CVS criteria (two structures visible, cystic plate dissected, hepatocystic triangle cleared). "
        "Use when user asks: critical view of safety, CVS, critical view, safety criteria, whether CVS is achieved. "
        "NOT for surgical phase—use phase_detection for workflow steps."
    )
    args_schema: Type[BaseModel] = SurgicalImageInput
    use_new_cvs_model: bool = Field(default=False, description="True: single ResNet18 (new_cvs_model_1); False: old 4-backbone ensemble. Set by --cvs CLI or env SURGICAL_CVS_USE_NEW_MODEL=1.")
    model_dir: Path = Field(default_factory=lambda: _NEW_CVS_MODEL_DIR, description="For new model: dir with best_model.pth")
    log_root: Path = Field(default_factory=lambda: _OLD_LOG_ROOT, description="For old model: log root with colenet_vgg, etc.")
    _cache: Optional[Any] = PrivateAttr(default=None)

    def __init__(
        self,
        use_new_cvs_model: Optional[bool] = None,
        model_dir: Optional[Path] = None,
        log_root: Optional[Path] = None,
        **kwargs,
    ):
        use_new = use_new_cvs_model if use_new_cvs_model is not None else _use_new_cvs_from_env()
        init_dir = Path(model_dir) if model_dir else _NEW_CVS_MODEL_DIR
        init_log = Path(log_root) if log_root else _OLD_LOG_ROOT
        super().__init__(use_new_cvs_model=use_new, model_dir=init_dir, log_root=init_log, **kwargs)

    def _get_cache(self):
        if self._cache is None:
            if self.use_new_cvs_model:
                self._cache = _CVSModelCacheSingle(self.model_dir, backbone="resnet18")
            else:
                self._cache = _CVSModelCacheEnsemble(self.log_root)
        return self._cache

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        path = Path(image_path)
        if not path.exists():
            return f"Error: image not found: {image_path}"

        if self.use_new_cvs_model:
            ckpt = self.model_dir / "best_model.pth"
            if not ckpt.exists():
                return (
                    f"Error: CVS checkpoint not found: {ckpt}. "
                    "Place CVS model weights in cvs_models/log/ (e.g. log/new_cvs_model_1/best_model.pth or log/colenet_*/best_model.pth)."
                )
        else:
            if not self.log_root.exists():
                return f"Error: CVS model weights not found: {self.log_root}. Place trained weights in cvs_models/log/ (see README)."
            for b in ["vgg", "resnet", "resnet18", "densenet"]:
                if not (self.log_root / f"colenet_{b}" / "best_model.pth").exists():
                    return f"Error: checkpoint not found: {self.log_root / f'colenet_{b}'}/best_model.pth"

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            return f"Error loading image: {e}"

        transform = _get_transform()
        try:
            image_tensor = transform(image)
        except Exception as e:
            return f"Error preprocessing image: {e}"

        try:
            cache = self._get_cache()
            scores = cache.run_inference(image_tensor)
        except Exception as e:
            return f"Error running CVS model: {e}"

        threshold = 0.5
        if self.use_new_cvs_model:
            header = "Critical View of Safety (CVS) assessment (ResNet18, new_cvs_model_1):"
        else:
            header = "Critical View of Safety (CVS) assessment (aggregate of vgg, resnet, resnet18, densenet):"
        lines = [header]
        for i, name in enumerate(_CVS_CRITERIA):
            score = float(scores[i]) if i < len(scores) else 0.0
            achieved = "achieved" if score >= threshold else "not achieved"
            lines.append(f"  - {name.replace('_', ' ').title()}: {achieved} (score {score:.2f})")
        return "\n".join(lines)
