"""
Triplet recognition tool using CholecT50 models.
1. Tool/verb/target presence: best_tool.pt, best_verb.pt, best_target.pt
2. Full triplet detection: YOLO model (100 classes) from runs/triplet
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr
from torchvision import transforms

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from surgical_copilot.registry import register
from surgical_copilot.tools.base import SurgicalImageInput


_MICCAI_ROOT = Path(__file__).resolve().parent.parent.parent
_INSTRUMENT_TRIPLET_ROOT = _MICCAI_ROOT / "instrument_triplet_tasks"
_CHOLECT50_CHECKPOINTS = _INSTRUMENT_TRIPLET_ROOT / "cholect50_checkpoints"
_TRIPLET_YOLO_PATHS = [
    _INSTRUMENT_TRIPLET_ROOT / "runs" / "triplet" / "runs" / "detect" / "train" / "weights" / "best.pt",
    _INSTRUMENT_TRIPLET_ROOT / "runs" / "triplet" / "best_detection.pt",
]

# CholecT50 class names (order matches dataloader indices)
TOOL_NAMES = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]
VERB_NAMES = [
    "grasp", "retract", "dissect", "coagulate", "clip", "cut",
    "aspirate", "irrigate", "pack", "null",
]
TARGET_NAMES = [
    "gallbladder", "cystic_duct", "cystic_artery", "hepatic_duct",
    "hepato_cystic_triangle", "liver_ligament", "blood", "hepatocystic_triangle",
    "gallbladder_fundus", "gallbladder_body", "gallbladder_infundibulum",
    "cystic_duct_stump", "liver", "connective_tissue", "abdominal_wall",
]


def _ensure_instrument_triplet_path():
    if str(_INSTRUMENT_TRIPLET_ROOT) not in sys.path:
        sys.path.insert(0, str(_INSTRUMENT_TRIPLET_ROOT))


def _get_transform():
    return transforms.Compose([
        transforms.Resize((256, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _load_cholect50_model(checkpoint_path: Path, device: torch.device):
    """Load a CholecT50 single-task model (tool, verb, or target)."""
    _ensure_instrument_triplet_path()
    from cholect50_model import MultiHeadModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tasks = ckpt.get("tasks", ["tool"])
    backbone = ckpt.get("backbone", "resnet50")
    model = MultiHeadModel(backbone_name=backbone, tasks=tasks, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _get_triplet_yolo_model() -> Any:
    """Load YOLO triplet detection model (100 classes)."""
    writable = os.environ.get("TMPDIR") or os.environ.get("SCRATCH") or os.getcwd()
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(writable, "matplotlib_cache"))
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(writable, "ultralytics_config"))
    from ultralytics import YOLO
    for p in _TRIPLET_YOLO_PATHS:
        if p.exists():
            return YOLO(str(p))
    return None


def _run_triplet_detection(image_path: str, conf: float = 0.25, iou: float = 0.4) -> List[tuple]:
    """Run YOLO triplet detection; return list of (triplet_id, triplet_name, conf)."""
    model = _get_triplet_yolo_model()
    if model is None:
        return []
    results = model.predict(image_path, conf=conf, iou=iou, verbose=False)
    if not results or results[0].orig_img is None:
        return []
    result = results[0]
    detections = []
    if result.boxes is not None and len(result.boxes.cls) > 0:
        for j in range(len(result.boxes.cls)):
            cls_id = int(result.boxes.cls[j])
            conf_val = float(result.boxes.conf[j])
            name = f"triplet_{cls_id}" if 0 <= cls_id < 100 else f"class_{cls_id}"
            detections.append((cls_id, name, conf_val))
    return detections


@register("triplet_recognition")
def make_triplet_recognition_tool(checkpoint_dir: Optional[Path] = None):
    return TripletRecognitionTool(checkpoint_dir=checkpoint_dir)


class TripletRecognitionTool(BaseTool):
    """Recognize tools, verbs, and targets in surgical images using CholecT50 models."""

    name: str = "triplet_recognition"
    description: str = (
        "Recognizes structured triplets in a surgical image (e.g. who does what to what). "
        "Input: path to image. "
        "Output: list of tools, verbs, targets, and possible action triplets. "
        "Use when the user asks about actions, relations, or structured scene description."
    )
    args_schema: Type[BaseModel] = SurgicalImageInput
    checkpoint_dir: Path = Field(default_factory=lambda: _CHOLECT50_CHECKPOINTS)
    threshold: float = Field(default=0.5, ge=0.1, le=0.99)
    triplet_conf: float = Field(default=0.25, ge=0.01, le=0.99)
    _models: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _device: Optional[torch.device] = PrivateAttr(default=None)

    def __init__(self, checkpoint_dir: Optional[Path] = None, **kwargs):
        init_ckpt = Path(checkpoint_dir) if checkpoint_dir else _CHOLECT50_CHECKPOINTS
        super().__init__(checkpoint_dir=init_ckpt, **kwargs)

    def _get_models(self) -> Dict[str, Any]:
        """Lazy-load tool, verb, target models."""
        if self._device is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for task, fname in [("tool", "best_tool.pt"), ("verb", "best_verb.pt"), ("target", "best_target.pt")]:
            if task not in self._models:
                path = self.checkpoint_dir / fname
                if not path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {path}")
                self._models[task] = _load_cholect50_model(path, self._device)
        return self._models

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not Path(image_path).exists():
            return f"Error: image not found: {image_path}"

        for fname in ["best_tool.pt", "best_verb.pt", "best_target.pt"]:
            if not (self.checkpoint_dir / fname).exists():
                return (
                    f"Error: CholecT50 checkpoint not found: {self.checkpoint_dir / fname}. "
                    "Place best_tool.pt, best_verb.pt, best_target.pt in instrument_triplet_tasks/cholect50_checkpoints/."
                )

        try:
            models = self._get_models()
        except Exception as e:
            return f"Error loading CholecT50 models: {e}"

        transform = _get_transform()
        try:
            image = Image.open(image_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(self._device)
        except Exception as e:
            return f"Error loading image: {e}"

        tools_present: List[Tuple[str, float]] = []
        verbs_present: List[Tuple[str, float]] = []
        targets_present: List[Tuple[str, float]] = []

        with torch.no_grad():
            for task, names in [
                ("tool", TOOL_NAMES),
                ("verb", VERB_NAMES),
                ("target", TARGET_NAMES),
            ]:
                logits = models[task](img_tensor)[task]
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                probs = torch.sigmoid(logits).cpu().numpy().squeeze()
                for i, p in enumerate(probs.ravel()[: len(names)]):
                    if p >= self.threshold:
                        p_val = float(p)
                        if task == "tool":
                            tools_present.append((names[i], p_val))
                        elif task == "verb":
                            verbs_present.append((names[i], p_val))
                        else:
                            targets_present.append((names[i], p_val))

        def _fmt(items: List[Tuple[str, float]]) -> str:
            return ", ".join(f"{n} (conf {c:.2f})" for n, c in items) if items else "none detected"

        lines = [
            "Triplet recognition (CholecT50 tool/verb/target, with confidence):",
            f"  Tools: {_fmt(tools_present)}",
            f"  Verbs: {_fmt(verbs_present)}",
            f"  Targets: {_fmt(targets_present)}",
        ]

        if tools_present and (verbs_present or targets_present):
            triplets = []
            tool_names = [n for n, _ in tools_present]
            verb_names = [n for n, _ in verbs_present]
            target_names = [n for n, _ in targets_present]
            for t in tool_names:
                for v in verb_names or ["acting"]:
                    for tg in target_names or ["scene"]:
                        triplets.append(f"{t} {v} {tg}")
            if len(triplets) <= 10:
                lines.append("  Possible triplets (from presence): " + "; ".join(triplets))
            else:
                lines.append(f"  Possible triplets (from presence): {len(triplets)} combinations (e.g. {triplets[0]}; {triplets[1]}; ...)")

        # Full triplet detection (100-class YOLO)
        triplet_detections = _run_triplet_detection(image_path, conf=self.triplet_conf)
        if triplet_detections:
            seen = set()
            unique = []
            for _tid, tname, conf_val in triplet_detections:
                if tname not in seen:
                    seen.add(tname)
                    unique.append(f"{tname}({conf_val:.2f})")
            lines.append("  Full triplets (100-class detection): " + ", ".join(unique))
        elif any(p.exists() for p in _TRIPLET_YOLO_PATHS):
            lines.append("  Full triplets (100-class detection): none detected above threshold")
        else:
            lines.append("  Full triplets (100-class): model not found (place YOLO triplet weights in instrument_triplet_tasks/runs/triplet/...)")

        return "\n".join(lines)
