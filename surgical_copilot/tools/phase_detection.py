"""
Surgical phase detection tool using ResNet50 (8 phases).
Default: Standalone checkpoint best_phase.pt in phase_detection_workflow/.
Alternative: M2CAI16 workflow (set SURGICAL_PHASE_USE_STANDALONE=0).

- Standalone: raw state_dict checkpoint best_phase.pt in phase_detection_workflow/.
- M2CAI16: checkpoints from phase_detection_workflow/workflow_codes (best_model.pth).
"""
import os
from pathlib import Path
from typing import Optional, Type

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from surgical_copilot.registry import register
from surgical_copilot.tools.base import SurgicalVideoInput


# Project root (repository root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Path to phase detection workflow (sibling of surgical_copilot under repo root)
_PHASE_DETECTION_WORKFLOW_ROOT = _PROJECT_ROOT / "phase_detection_workflow"
# Default for standalone (best_phase.pt) mode — lives in phase_detection_workflow/
_DEFAULT_STANDALONE_CHECKPOINT = _PHASE_DETECTION_WORKFLOW_ROOT / "best_phase.pt"

_WORKFLOW_CODES = _PHASE_DETECTION_WORKFLOW_ROOT / "workflow_codes"
_DEFAULT_MODEL_PATHS = [
    _WORKFLOW_CODES / "m2cai16_workflow_output_fast" / "best_model.pth",
    _WORKFLOW_CODES / "m2cai16_workflow_output" / "best_model.pth",
    _WORKFLOW_CODES / "best_model.pth",
]

NUM_CLASSES = 8
PHASE_LABELS = [
    "TrocarPlacement",
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]


def _use_standalone_from_env() -> bool:
    """Default True (standalone / best_phase.pt). Set SURGICAL_PHASE_USE_STANDALONE=0 for M2CAI16."""
    v = os.environ.get("SURGICAL_PHASE_USE_STANDALONE", "").strip().lower()
    if v in ("0", "false", "no"):
        return False
    return True


def _standalone_checkpoint_from_env() -> Path:
    """Default best_phase.pt in phase_detection_workflow/. Override with SURGICAL_PHASE_CHECKPOINT."""
    p = os.environ.get("SURGICAL_PHASE_CHECKPOINT", "").strip()
    if p:
        return Path(p)
    return _DEFAULT_STANDALONE_CHECKPOINT


def _get_transform(standalone: bool = False):
    """Standalone (best_phase.pt) uses Resize(224,224) only; M2CAI16 uses Resize(256,256)+CenterCrop(224)."""
    if standalone:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _load_model(checkpoint_path: Path, device: torch.device, standalone: bool = False):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if standalone:
        state = checkpoint  # standalone format is raw state_dict
    else:
        state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _extract_frame(video_or_frames_path: str, frame_index: Optional[int]) -> Optional[np.ndarray]:
    """Extract a single frame from a video file, image file, or frames directory. Returns RGB numpy array or None."""
    path = Path(video_or_frames_path)
    if not path.exists():
        return None
    idx = frame_index if frame_index is not None else 0

    if path.is_file():
        # Single image file (e.g. from Gradio upload)
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            frame = cv2.imread(str(path))
            if frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Video file
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if path.is_dir():
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for ext in exts:
            files.extend(path.glob(ext))
        files = sorted(files, key=lambda p: p.stem)
        if not files:
            return None
        idx = min(idx, len(files) - 1)
        frame = cv2.imread(str(files[idx]))
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return None


def _run_phase_inference(
    model_path: Path,
    frame_rgb: np.ndarray,
    device: torch.device,
    model_cache: dict,
    standalone: bool = False,
):
    """Run ResNet50 phase prediction on one frame. Returns (phase_name, confidence)."""
    from PIL import Image
    cache_key = f"{model_path.resolve()}:standalone={standalone}"
    if cache_key not in model_cache:
        model_cache[cache_key] = _load_model(model_path, device, standalone=standalone)
    model = model_cache[cache_key]
    transform = _get_transform(standalone=standalone)
    pil = Image.fromarray(frame_rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return PHASE_LABELS[pred], conf


@register("phase_detection")
def make_phase_detection_tool():
    use_standalone = _use_standalone_from_env()
    checkpoint = _standalone_checkpoint_from_env() if use_standalone else None
    return PhaseDetectionTool(use_standalone_inference=use_standalone, model_path=checkpoint)


class PhaseDetectionTool(BaseTool):
    """Detect current surgical phase from video or frame(s) using ResNet50 (8 phases). Supports M2CAI16 or standalone best_phase.pt checkpoint."""

    name: str = "phase_detection"
    description: str = (
        "phase_detection: surgical phase / workflow step (TrocarPlacement, Preparation, CalotTriangleDissection, etc.). "
        "Use when user asks: what phase, surgical phase, current phase, previous phase, next phase, workflow step. "
        "Returns current, previous, and next phase for the frame. NOT for anatomies—use surgical_scene_segmentation. NOT for CVS—use critical_view_of_safety."
    )
    args_schema: Type[BaseModel] = SurgicalVideoInput

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_standalone_inference: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_path = Path(model_path) if model_path is not None else None
        self._use_standalone = (
            use_standalone_inference if use_standalone_inference is not None else _use_standalone_from_env()
        )
        self._model_cache = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_model_path(self) -> Optional[Path]:
        if self._use_standalone:
            if self._model_path is not None and self._model_path.exists():
                return self._model_path
            return _standalone_checkpoint_from_env()  # default: phase_detection_workflow/best_phase.pt
        if self._model_path is not None and self._model_path.exists():
            return self._model_path
        for p in _DEFAULT_MODEL_PATHS:
            if p.exists():
                return p
        return _DEFAULT_MODEL_PATHS[0]

    def _run(
        self,
        video_or_frames_path: str,
        frame_index: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        path = Path(video_or_frames_path)
        if not path.exists():
            return f"Error: path not found: {video_or_frames_path}"

        model_path = self._get_model_path()
        if model_path is None or not model_path.exists():
            if self._use_standalone:
                return (
                    f"Error: Standalone phase checkpoint not found: {model_path}. "
                    f"Place best_phase.pt in phase_detection_workflow/ or set SURGICAL_PHASE_CHECKPOINT=/path/to/best_phase.pt"
                )
            return (
                f"Error: M2CAI16 phase model not found. "
                "Train with: cd phase_detection_workflow/workflow_codes && python resnet50_surgical_workflow_fast.py"
            )

        frame = _extract_frame(video_or_frames_path, frame_index)
        if frame is None:
            return f"Error: could not extract a frame from {video_or_frames_path} (invalid file or empty directory)."

        try:
            phase_name, confidence = _run_phase_inference(
                model_path, frame, self._device, self._model_cache, standalone=self._use_standalone
            )
        except Exception as e:
            return f"Error running phase detection: {e}"

        idx = PHASE_LABELS.index(phase_name)
        previous_phase = PHASE_LABELS[idx - 1] if idx > 0 else None
        next_phase = PHASE_LABELS[idx + 1] if idx < len(PHASE_LABELS) - 1 else None

        backend = "standalone" if self._use_standalone else "M2CAI16"
        lines = [
            f"Phase detection ({backend}): current phase is **{phase_name}** (confidence {confidence:.2f}).",
        ]
        if previous_phase:
            lines.append(f"Previous phase in workflow: **{previous_phase}**.")
        else:
            lines.append("This is the first phase.")
        if next_phase:
            lines.append(f"Next phase in workflow: **{next_phase}**.")
        else:
            lines.append("This is the final phase.")
        return " ".join(lines)
