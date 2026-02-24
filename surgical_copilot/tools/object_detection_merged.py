"""
Object detection tool using the merged detector (anatomy + instruments).
Uses YOLOv8 weights from object_detection: best_detector_balanced.pt.
Classes: anatomy (cystic_plate, calot_triangle, cystic_artery, cystic_duct, gallbladder),
         generic tool, and 7 instruments (grasper, bipolar, hook, scissors, clipper, irrigator, specimen_bag).
Returns text + overlay image path (boxes drawn on frame) for UI display.
"""
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from surgical_copilot.registry import register
from surgical_copilot.tools.base import SurgicalImageInput


_MICCAI_ROOT = Path(__file__).resolve().parent.parent.parent
_OBJECT_DETECTION_ROOT = _MICCAI_ROOT / "object_detection"

# 13 classes: 0–4 anatomy, 5 tool (generic), 6–12 instruments (from prepare_detector_dataset.py)
MERGED_CLASS_NAMES = [
    "cystic_plate", "calot_triangle", "cystic_artery", "cystic_duct", "gallbladder",
    "tool",
    "grasper", "bipolar", "hook", "scissors", "clipper", "irrigator", "specimen_bag",
]

# Default paths for best_detector_balanced.pt (check object_detection/, project root, then cwd)
def _default_weights_paths() -> List[Path]:
    cwd = Path.cwd()
    return [
        _OBJECT_DETECTION_ROOT / "best_detector_balanced.pt",
        _MICCAI_ROOT / "best_detector_balanced.pt",
        cwd / "best_detector_balanced.pt",
        cwd / "object_detection" / "best_detector_balanced.pt",
        _OBJECT_DETECTION_ROOT / "runs" / "detect_balanced" / "weights" / "best.pt",
        _OBJECT_DETECTION_ROOT / "runs" / "detect_balanced" / "best.pt",
    ]


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


def _run_merged_detection(
    model_path: Path,
    image_path: str,
    conf: float = 0.2,
    iou: float = 0.45,
    imgsz: int = 640,
) -> Tuple[List[Tuple[int, str, float, float, float, float, float]], Optional[object]]:
    """Run merged YOLOv8 detection; return (detections, result). result is used for result.plot() overlay."""
    model = _load_model(model_path)
    results = model.predict(image_path, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    if not results or results[0].orig_img is None:
        return [], None
    result = results[0]
    detections = []
    if result.boxes is not None and len(result.boxes.cls) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        for j in range(len(result.boxes.cls)):
            cls_id = int(result.boxes.cls[j])
            conf_val = float(result.boxes.conf[j])
            name = MERGED_CLASS_NAMES[cls_id] if cls_id < len(MERGED_CLASS_NAMES) else f"class_{cls_id}"
            x1, y1, x2, y2 = float(xyxy[j][0]), float(xyxy[j][1]), float(xyxy[j][2]), float(xyxy[j][3])
            detections.append((cls_id, name, conf_val, x1, y1, x2, y2))
    return detections, result


def is_object_detection_merged_available() -> bool:
    """Return True if the merged detector weights exist."""
    for p in _default_weights_paths():
        if p.exists():
            return True
    return False


@register("object_detection_merged")
def make_object_detection_merged_tool():
    return ObjectDetectionMergedTool()


class ObjectDetectionMergedTool(BaseTool):
    """Detect anatomy and instruments in a surgical frame using the merged detector (best_detector_balanced.pt)."""

    name: str = "object_detection_merged"
    description: str = (
        "object_detection_merged: detect anatomy and instruments with bounding boxes (x1,y1,x2,y2 in pixels). "
        "Classes: cystic_plate, calot_triangle, cystic_artery, cystic_duct, gallbladder, tool, grasper, bipolar, hook, scissors, clipper, irrigator, specimen_bag. "
        "Use when the user asks what anatomy/instruments are visible, object detection, or detect objects."
    )
    args_schema: Type[BaseModel] = SurgicalImageInput

    def __init__(self, model_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self._model_path = model_path

    def _get_model_path(self) -> Path:
        if self._model_path is not None and Path(self._model_path).exists():
            return Path(self._model_path)
        paths = _default_weights_paths()
        for p in paths:
            if p.exists():
                return p
        return paths[0]

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
                "Error: Merged detector weights not found (best_detector_balanced.pt). "
                "Train with: python object_detection/train_detector_balanced.py --data_dir /path/to/detector_merged_work --save_weights object_detection/best_detector_balanced.pt"
            )

        try:
            detections, result = _run_merged_detection(model_path, image_path)
        except Exception as e:
            return f"Error running object detection: {e}"

        if not detections:
            return (
                "Object detection (merged): no anatomy or instruments detected above confidence threshold. "
                f"Classes: {', '.join(MERGED_CLASS_NAMES)}."
            )

        # Save overlay (boxes on frame) for UI to display on top of video
        overlay_path = None
        if result is not None:
            try:
                import cv2
                overlay_dir = _MICCAI_ROOT / "temp" / "detection_overlays"
                overlay_dir.mkdir(parents=True, exist_ok=True)
                overlay_path = overlay_dir / f"detection_{int(time.time())}_{path.stem}.png"
                # result.plot() returns BGR numpy array with boxes/labels drawn on original image
                plotted = result.plot()
                cv2.imwrite(str(overlay_path), plotted)
            except Exception:
                overlay_path = None

        # Group by anatomy vs instruments and include bounding boxes (x1, y1, x2, y2 pixels)
        anatomy = [n for _, n, *_ in detections if n in ("cystic_plate", "calot_triangle", "cystic_artery", "cystic_duct", "gallbladder")]
        tools = [n for _, n, *_ in detections if n in ("tool", "grasper", "bipolar", "hook", "scissors", "clipper", "irrigator", "specimen_bag")]
        lines = [f"Object detection (merged): {len(detections)} detection(s) with bounding boxes (x1, y1, x2, y2 in pixels)."]
        if anatomy:
            lines.append(f"  Anatomy: {', '.join(anatomy)}")
        if tools:
            lines.append(f"  Tools/instruments: {', '.join(tools)}")
        for cid, name, conf_val, x1, y1, x2, y2 in detections:
            lines.append(f"  - {name} box=[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}] conf={conf_val:.2f}")
        if overlay_path and overlay_path.exists():
            lines.append(f"Overlay path: {overlay_path}")
        return "\n".join(lines)
