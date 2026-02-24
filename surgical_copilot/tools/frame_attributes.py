"""
Frame-level attribute prediction using CholecTrack20 multilabel model.

Predicts operator presence (MSLH, MSRH, ASRH, NULL) and visual conditions
(visibility, crowded, occluded, bleeding, smoke, etc.) from a surgical frame.
"""
import sys
from pathlib import Path
from typing import Any, Optional, Type

import torch
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from surgical_copilot.registry import register

_MICCAI_ROOT = Path(__file__).resolve().parent.parent.parent


class FrameAttributesInput(BaseModel):
    """Accept path or image_path (agent may output either)."""

    image_path: Optional[str] = Field(None, description="Path to the surgical/endoscopy image")
    path: Optional[str] = Field(None, description="Path to the image (alias for image_path)")

    @model_validator(mode="before")
    @classmethod
    def ensure_path(cls, data: Any) -> Any:
        if isinstance(data, dict):
            p = data.get("path") or data.get("image_path")
            if p:
                return {"image_path": p, "path": p}
        return data


if str(_MICCAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_MICCAI_ROOT))
_FRAME_ATTRIBUTES_TASKS = _MICCAI_ROOT / "frame_attributes_tasks"
if str(_FRAME_ATTRIBUTES_TASKS) not in sys.path:
    sys.path.insert(0, str(_FRAME_ATTRIBUTES_TASKS))

from PIL import Image

OPERATOR_NAMES = ["MSLH", "MSRH", "ASRH", "NULL"]
OPERATOR_FULL_NAMES = {
    "MSLH": "main surgeon left hand (MSLH)",
    "MSRH": "main surgeon right hand (MSRH)",
    "ASRH": "assistant surgeon right hand (ASRH)",
    "NULL": "null operator (NULL)",
}
# Conditions we skip in display (don't show visible/visibility)
CONDITIONS_HIDE = {"visibility", "visible"}
# Alarming conditions shown with ✗; others with ✓
CONDITION_ALARMING = {"occluded", "bleeding", "smoke", "stainedlens", "blurred", "reflection"}

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

_DEFAULT_CHECKPOINT = _MICCAI_ROOT / "frame_attributes_tasks" / "cholec20_multilabel_checkpoints" / "best_cholec20_multilabel.pt"


def is_frame_attributes_available() -> bool:
    """Return True if the multilabel model checkpoint exists (tool can run)."""
    return _DEFAULT_CHECKPOINT.exists()


def _load_model(checkpoint_path: Path, device: torch.device):
    """Lazy-load the multilabel model."""
    import torchvision.transforms as transforms
    from cholec20_model import Cholec20MultilabelModel

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "resnet50")
    model = Cholec20MultilabelModel(backbone_name=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model


def _get_transform(img_size: int = 224):
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _predict(image_path: str, checkpoint_path: Path, device: torch.device) -> dict:
    transform = _get_transform()
    model = _load_model(checkpoint_path, device)
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    op_logits = out["operator"][0]
    bin_logits = out["binary"][0]
    op_probs = torch.sigmoid(op_logits).cpu().numpy()
    bin_probs = torch.sigmoid(bin_logits).cpu().numpy()
    op_pred = (op_probs > 0.5)
    bin_pred = (bin_probs > 0.5)
    operators = [OPERATOR_NAMES[i] for i in range(4) if op_pred[i]]
    binary = {BINARY_ATTRS[i]: bool(bin_pred[i]) for i in range(len(BINARY_ATTRS))}
    op_probs_dict = {OPERATOR_NAMES[i]: float(op_probs[i]) for i in range(4)}
    bin_probs_dict = {BINARY_ATTRS[i]: float(bin_probs[i]) for i in range(len(BINARY_ATTRS))}
    return {
        "operators": operators,
        "binary": binary,
        "operator_probs": op_probs_dict,
        "binary_probs": bin_probs_dict,
    }


@register("frame_attributes")
def make_frame_attributes_tool():
    return FrameAttributesTool()


class FrameAttributesTool(BaseTool):
    """Predict frame-level surgical attributes: operator presence and visual conditions."""

    name: str = "frame_attributes"
    description: str = (
        "frame_attributes: operator presence (MSLH, MSRH, ASRH, NULL) and visual conditions (visibility, occlusion, bleeding, smoke). "
        "Use ONLY for: who is operating, operators visible, visibility, occlusion, bleeding, smoke, lens quality. "
        "Do NOT use for: surgical phase (use phase_detection), segmentation (use surgical_scene_segmentation), "
        "instruments (use instrument_tracking), or any workflow/phase question."
    )
    args_schema: Type[BaseModel] = FrameAttributesInput

    def __init__(self, checkpoint_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self._checkpoint_path = checkpoint_path

    def _get_checkpoint_path(self) -> Path:
        if self._checkpoint_path is not None and Path(self._checkpoint_path).exists():
            return Path(self._checkpoint_path)
        if _DEFAULT_CHECKPOINT.exists():
            return _DEFAULT_CHECKPOINT
        return _DEFAULT_CHECKPOINT

    def _run(
        self,
        image_path: Optional[str] = None,
        path: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        img_path = image_path or path or kwargs.get("path") or kwargs.get("image_path")
        if not img_path:
            return "Error: image_path or path required"
        path_obj = Path(img_path)
        if not path_obj.exists():
            return f"Error: image not found: {img_path}"

        ckpt = self._get_checkpoint_path()
        if not ckpt.exists():
            return (
                "Error: Frame attributes model checkpoint not found. "
                "This tool is unavailable. Do NOT retry. Proceed using other tools and inform the user "
                "that frame-level attribute prediction requires training the multilabel model first."
            )

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            result = _predict(str(path_obj), ckpt, device)
        except Exception as e:
            return f"Error running frame attributes: {e}"

        lines = ["Frame attributes prediction (with confidence):"]
        op_probs = result.get("operator_probs", {})
        # Never show NULL; show full form (e.g. main surgeon right hand (MSRH)) with confidence
        operators_display = [op for op in result["operators"] if op != "NULL"]
        op_parts = [f"{OPERATOR_FULL_NAMES.get(op, op)} ({op_probs.get(op, 0):.2f})" for op in operators_display]
        lines.append(f"  Operators present: {', '.join(op_parts) or 'none'}")
        bin_probs = result.get("binary_probs", {})
        active = [k for k, v in result["binary"].items() if v and k not in CONDITIONS_HIDE]
        # Output condition name + confidence (UI will add ✓/✗)
        cond_parts = [f"{k} ({bin_probs.get(k, 0):.2f})" for k in active]
        lines.append(f"  Active conditions: {', '.join(cond_parts) if cond_parts else 'none'}")
        return "\n".join(lines)
