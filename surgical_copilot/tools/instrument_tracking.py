"""
Instrument tracking tool using CholecTrack20 YOLOv8 detection (7 tool classes).
Uses the saved trained model from frame_attributes_tasks or instrument_triplet_tasks.
"""
import os
from pathlib import Path
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from surgical_copilot.registry import register
from surgical_copilot.tools.base import SurgicalImageInput


# Paths to instrument detection models (YOLO)
_MICCAI_ROOT = Path(__file__).resolve().parent.parent.parent
_FRAME_ATTRIBUTES_TASKS_ROOT = _MICCAI_ROOT / "frame_attributes_tasks"
_INSTRUMENT_TRIPLET_ROOT = _MICCAI_ROOT / "instrument_triplet_tasks"
# CholecT50 tool detection (6 classes) checked first; CholecTrack20 (7 classes) as fallback
_DEFAULT_MODEL_PATHS = [
    _INSTRUMENT_TRIPLET_ROOT / "runs" / "tool" / "runs" / "detect" / "train" / "weights" / "best.pt",
    _FRAME_ATTRIBUTES_TASKS_ROOT / "runs" / "cholectrack20" / "best_cholectrack20.pt",
    _FRAME_ATTRIBUTES_TASKS_ROOT / "runs" / "cholectrack20" / "runs" / "detect" / "train" / "weights" / "best.pt",
]

# CholecT50: 6 tools; CholecTrack20: 7 (adds specimenbag)
TOOL_NAMES_CHOLECT50 = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]
TOOL_NAMES_CHOLECTRACK20 = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator", "specimenbag"]


def _writable_dir():
    cand = os.environ.get("TMPDIR") or os.environ.get("SCRATCH") or os.getcwd()
    if not cand or not os.path.isdir(cand):
        return os.getcwd()
    return cand


def _load_model(model_path: Path):
    writable = _writable_dir()
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(writable, "matplotlib_cache"))
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(writable, "ultralytics_config"))
    from ultralytics import YOLO
    return YOLO(str(model_path))


def _run_instrument_detection(model_path: Path, image_path: str, conf: float = 0.15, iou: float = 0.45, imgsz: int = 640):
    """Run YOLOv8 detection on one image; return list of (class_id, tool_name, conf), model_source."""
    model = _load_model(model_path)
    results = model.predict(image_path, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    if not results or results[0].orig_img is None:
        return [], None
    result = results[0]
    # CholecT50 has 6 classes; CholecTrack20 has 7
    use_cholect50 = "cholect50" in str(model_path).lower() or "instrument_triplet_tasks" in str(model_path)
    tool_names = TOOL_NAMES_CHOLECT50 if use_cholect50 else TOOL_NAMES_CHOLECTRACK20
    detections = []
    if result.boxes is not None and len(result.boxes.cls) > 0:
        for j in range(len(result.boxes.cls)):
            cls_id = int(result.boxes.cls[j])
            conf_val = float(result.boxes.conf[j])
            name = tool_names[cls_id] if cls_id < len(tool_names) else f"class_{cls_id}"
            detections.append((cls_id, name, conf_val))
    return detections, "CholecT50" if use_cholect50 else "CholecTrack20"


@register("instrument_tracking")
def make_instrument_tracking_tool():
    return InstrumentTrackingTool()


class InstrumentTrackingTool(BaseTool):
    """Detect surgical instruments in an image using CholecTrack20 saved YOLOv8 model (7 tools)."""

    name: str = "instrument_tracking"
    description: str = (
        "instrument_tracking: list which instruments are visible (grasper, bipolar, hook, scissors, clipper, irrigator, specimenbag). "
        "Use when user asks what instruments, which tools, instruments visible. "
        "NOT for segmentation—use surgical_scene_segmentation for segment/segmentation requests."
    )
    args_schema: Type[BaseModel] = SurgicalImageInput

    def __init__(self, model_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self._model_path = model_path

    def _get_model_path(self) -> Path:
        if self._model_path is not None and Path(self._model_path).exists():
            return Path(self._model_path)
        for p in _DEFAULT_MODEL_PATHS:
            if p.exists():
                return p
        return _DEFAULT_MODEL_PATHS[0]

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        path = Path(image_path)
        if not path.exists():
            return f"Error: image not found: {image_path}"

        model_path = self._get_model_path()
        if not model_path.exists():
            return (
                f"Error: Instrument model not found: {model_path}. "
                "Place YOLO weights at instrument_triplet_tasks/runs/tool/.../best.pt or frame_attributes_tasks/runs/cholectrack20/... for instrument detection."
            )

        try:
            detections, model_source = _run_instrument_detection(model_path, image_path)
        except Exception as e:
            return f"Error running instrument detection: {e}"

        use_cholect50 = model_source == "CholecT50" if model_source else "cholect50" in str(model_path).lower()
        tool_list = ", ".join(TOOL_NAMES_CHOLECT50 if use_cholect50 else TOOL_NAMES_CHOLECTRACK20)
        if not detections:
            return (
                f"Instrument tracking: no instruments detected above confidence threshold. "
                f"Classes: {tool_list}."
            )

        lines = [f"Instrument tracking ({model_source}): detected {len(detections)} instrument(s)."]
        for _cid, name, conf_val in detections:
            lines.append(f"  - {name} (confidence {conf_val:.2f})")
        return "\n".join(lines)
