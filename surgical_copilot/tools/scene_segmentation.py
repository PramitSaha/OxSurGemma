"""
Surgical scene segmentation tool using CholecSeg8k YOLOv8-seg.
Segments anatomy/tools in cholecystectomy images (13 classes).
Overlay is drawn in memory and written only to a temp path for display in the main window (no persistent save).
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from surgical_copilot.registry import register
from surgical_copilot.tools.base import SurgicalImageInput
from pydantic import BaseModel, Field, model_validator
from typing import Any


class SegmentationInput(BaseModel):
    """Input for surgical scene segmentation. Optional object_name limits to that class only."""

    image_path: str = Field(..., description="Path to the surgical or endoscopy image")
    object_name: Optional[str] = Field(
        None,
        description="Optional: segment only this object (e.g. gallbladder, grasper, liver). If omitted, segment all.",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_path_alias(cls, data: Any) -> Any:
        if isinstance(data, dict) and "path" in data and "image_path" not in data:
            data = {**data, "image_path": data["path"]}
        return data


# Scene segmentation: code and weights live under scene_segmentation_utils (sibling of surgical_copilot)
_MICCAI_ROOT = Path(__file__).resolve().parent.parent.parent
_SCENE_SEGMENTATION_UTILS = _MICCAI_ROOT / "scene_segmentation_utils"

# Prefer segmentation-specific run weights over generic root best.pt/best_merged.pt (which may be detection or wrong task)
def _default_model_paths():
    cwd = Path.cwd()
    return [
        # 1. Weights inside scene_segmentation_utils (training output or copied best.pt)
        _SCENE_SEGMENTATION_UTILS / "runs" / "segment" / "train" / "weights" / "best.pt",
        _SCENE_SEGMENTATION_UTILS / "runs" / "segment" / "train" / "weights" / "last.pt",
        _SCENE_SEGMENTATION_UTILS / "cholecseg_work" / "runs" / "segment" / "train" / "weights" / "best.pt",
        _SCENE_SEGMENTATION_UTILS / "cholecseg_work" / "runs" / "segment" / "train" / "weights" / "last.pt",
        _SCENE_SEGMENTATION_UTILS / "best.pt",
        # 2. Named segmentation weights in project root (e.g. cholecseg_best.pt avoids confusion with detector)
        _MICCAI_ROOT / "cholecseg_best.pt",
        _MICCAI_ROOT / "best_merged.pt",
        _MICCAI_ROOT / "best.pt",
        cwd / "cholecseg_best.pt",
        cwd / "best_merged.pt",
        cwd / "best.pt",
    ]


def _writable_dir():
    cand = os.environ.get("TMPDIR") or os.getcwd()
    if not cand or "/path/to" in cand or not os.path.isdir(cand):
        return os.getcwd()
    return cand


def _ensure_scene_segmentation_utils_path():
    if str(_SCENE_SEGMENTATION_UTILS) not in sys.path:
        sys.path.insert(0, str(_SCENE_SEGMENTATION_UTILS))
    # Avoid disk quota when Ultralytics/matplotlib write to home
    writable = _writable_dir()
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(writable, "matplotlib_cache"))
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(writable, "ultralytics_config"))


def _load_model(model_path: Path):
    _ensure_scene_segmentation_utils_path()
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    return model


def _run_scene_segmentation_inference(model_path: Path, image_path: str, conf: float = 0.25, iou: float = 0.4):
    """Run YOLOv8-seg on one image; return list of (class_id, class_name, confidence), result object, and cseg module for overlay."""
    _ensure_scene_segmentation_utils_path()
    import cv2
    import cholecseg8k_yolov8 as cseg
    model = _load_model(model_path)
    results = model.predict(image_path, conf=conf, iou=iou, verbose=False)
    if not results or results[0].orig_img is None:
        return [], None, None
    result = results[0]
    detections = []
    if result.boxes is not None and len(result.boxes.cls) > 0:
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [0.0] * len(result.boxes.cls)
        for i in range(len(result.boxes.cls)):
            cls_id = int(result.boxes.cls[i])
            name = cseg.CLASS_NAMES[cls_id] if cls_id < len(cseg.CLASS_NAMES) else f"class_{cls_id}"
            conf_val = float(confs[i]) if i < len(confs) else 0.0
            detections.append((cls_id, name, conf_val))
    return detections, result, cseg


def _class_matches(user_query: str, class_name: str) -> bool:
    """Check if user query matches class name (case-insensitive, normalizes spaces/hyphens)."""
    if not user_query or not class_name:
        return False
    q = user_query.lower().replace("-", " ").replace("_", " ").strip()
    c = class_name.lower().replace("-", " ").replace("_", " ").strip()
    return q in c or c in q or q == c


def _mask_xy_to_polygon(mask_xy) -> Optional["np.ndarray"]:
    """Convert Ultralytics mask xy to OpenCV fillPoly format: contiguous (N, 2) np.int32, N >= 3. Returns None if invalid."""
    import numpy as np
    if mask_xy is None:
        return None
    try:
        p = np.asarray(mask_xy, dtype=np.float64)
        if p.size < 6:  # need at least 3 points (6 values)
            return None
        if p.ndim == 3:
            p = p.reshape(-1, 2)
        elif p.ndim == 1:
            p = p.reshape(-1, 2)
        if p.shape[0] < 3 or p.shape[1] != 2:
            return None
        poly = np.ascontiguousarray(p.astype(np.int32))
        return poly
    except Exception:
        return None


def _build_filtered_mask(result, cseg_module, object_name: Optional[str]):
    """Build segmentation mask including only detections matching object_name. If object_name is None, include all."""
    import cv2
    import numpy as np

    if result is None or result.boxes is None or len(result.boxes.cls) == 0:
        return np.zeros_like(result.orig_img) if result else None
    orig = result.orig_img
    masks = []
    for i in range(len(result.boxes.cls)):
        cls_id = int(result.boxes.cls[i])
        name = (
            cseg_module.CLASS_NAMES[cls_id]
            if cls_id < len(cseg_module.CLASS_NAMES)
            else f"class_{cls_id}"
        )
        if object_name and not _class_matches(object_name, name):
            continue
        background = np.zeros_like(orig)
        color = cseg_module.CLASS_COLOR_MAPPING_BGR.get(cls_id, (0, 0, 0))
        if result.masks is not None and i < len(result.masks):
            try:
                raw = result.masks[i].xy[0]
                mask_points = _mask_xy_to_polygon(raw)
                if mask_points is not None:
                    cv2.fillPoly(background, [mask_points], color)
            except Exception:
                pass
        masks.append(background)
    if not masks:
        return np.zeros_like(orig)
    out = masks[0].copy()
    for m in masks[1:]:
        zero_mask = np.all(out == [0, 0, 0], axis=2)
        out[zero_mask] = m[zero_mask]
    return out


def _build_overlay(result, cseg_module, object_name: Optional[str] = None):
    """Build overlay image (frame + mask blended) in memory. Returns numpy array (BGR) or None."""
    if result is None or cseg_module is None:
        return None
    import cv2
    orig = result.orig_img
    if object_name:
        seg_mask = _build_filtered_mask(result, cseg_module, object_name)
    else:
        seg_mask = cseg_module.postprocess_prediction(result, cseg_module.CLASS_COLOR_MAPPING_BGR)
    overlay = cv2.addWeighted(orig, 0.6, seg_mask, 0.4, 0)
    return overlay


def _draw_anatomy_labels_on_overlay(overlay_bgr, result, cseg_module, object_name: Optional[str] = None):
    """Draw anatomy/structure name on each segmented region (on the overlay). overlay_bgr is modified in place; also returns it."""
    if overlay_bgr is None or result is None or cseg_module is None or result.boxes is None:
        return overlay_bgr
    import cv2
    import numpy as np
    h, w = overlay_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(w, h) / 800.0)
    thickness = max(1, int(round(font_scale * 2)))
    for i in range(len(result.boxes.cls)):
        cls_id = int(result.boxes.cls[i])
        name = (
            cseg_module.CLASS_NAMES[cls_id]
            if cls_id < len(cseg_module.CLASS_NAMES)
            else f"class_{cls_id}"
        )
        if object_name and not _class_matches(object_name, name):
            continue
        # Position: centroid of mask if available, else center of box
        x_center, y_center = None, None
        if result.masks is not None and i < len(result.masks):
            try:
                raw = result.masks[i].xy[0]
                poly = _mask_xy_to_polygon(raw)
                if poly is not None and len(poly) >= 3:
                    x_center = int(np.mean(poly[:, 0]))
                    y_center = int(np.mean(poly[:, 1]))
            except Exception:
                pass
        if x_center is None and result.boxes.xyxy is not None and i < len(result.boxes.xyxy):
            box = result.boxes.xyxy[i]
            if hasattr(box, "cpu"):
                box = box.cpu().numpy()
            x_center = int((float(box[0]) + float(box[2])) / 2)
            y_center = int((float(box[1]) + float(box[3])) / 2)
        if x_center is None or y_center is None:
            continue
        x_center = max(10, min(w - 10, x_center))
        y_center = max(10, min(h - 10, y_center))
        # Draw text with background for readability (white text, dark outline/background)
        (tw, th), _ = cv2.getTextSize(name, font, font_scale, thickness)
        x1 = max(0, x_center - tw // 2 - 4)
        y1 = max(0, y_center - th - 4)
        x2 = min(w, x_center + tw // 2 + 4)
        y2 = min(h, y_center + 4)
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(overlay_bgr, name, (x1 + 2, y2 - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return overlay_bgr


def _write_overlay_to_temp(overlay_bgr, temp_dir: Path) -> Optional[Path]:
    """Write overlay to a single temp file for display only (no persistent save). Returns path or None."""
    if overlay_bgr is None:
        return None
    import cv2
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"seg_overlay_{int(time.time() * 1000)}.png"
    cv2.imwrite(str(path), overlay_bgr)
    return path if path.exists() else None


@register("surgical_scene_segmentation")
def make_scene_segmentation_tool():
    return SurgicalSceneSegmentationTool()


class SurgicalSceneSegmentationTool(BaseTool):
    """Segment surgical scene / anatomy in an endoscopy image (CholecSeg8k YOLOv8, 13 classes)."""

    name: str = "surgical_scene_segmentation"
    description: str = (
        "surgical_scene_segmentation: anatomy and structures (gallbladder, liver, fat, grasper, cystic duct, etc). "
        "Use when user says: segment, anatomies, what anatomies, what anatomy, what structures, what do you see. "
        "Pass object_name when user asks for a specific object only. NOT for phase—use phase_detection. NOT for instruments only—use instrument_tracking."
    )
    args_schema: Type[BaseModel] = SegmentationInput

    def __init__(self, model_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self._model_path = model_path
        self._model = None

    def _get_model_path(self) -> Path:
        if self._model_path is not None and Path(self._model_path).exists():
            return Path(self._model_path)
        # Allow forcing segmentation weights via env (e.g. export CHOLECSEG_WEIGHTS=/path/to/best.pt)
        env_weights = os.environ.get("CHOLECSEG_WEIGHTS") or os.environ.get("SEGMENTATION_WEIGHTS")
        if env_weights and Path(env_weights).exists():
            return Path(env_weights)
        paths = _default_model_paths()
        for p in paths:
            if p.exists():
                return p
        return paths[0]

    def _run(
        self,
        image_path: str,
        object_name: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        path = Path(image_path)
        if not path.exists():
            return f"Error: image not found: {image_path}"

        model_path = self._get_model_path()
        if not model_path.exists():
            return (
                f"Error: CholecSeg8k model not found: {model_path}. "
                "Train with: cd scene_segmentation_utils && python cholecseg8k_yolov8.py train --data_dir /path/to/cholecseg8k --output_dir cholecseg_work"
            )

        try:
            detections, result, cseg = _run_scene_segmentation_inference(model_path, image_path)
        except Exception as e:
            return f"Error running segmentation: {e}"

        # Filter detections by object_name if specified
        if object_name and object_name.strip():
            detections = [
                (cid, name, c)
                for cid, name, c in detections
                if _class_matches(object_name.strip(), name)
            ]

        # Overlay in memory, write only to project temp for main-window display (no persistent save)
        temp_dir = _MICCAI_ROOT / "temp"
        overlay_img = _build_overlay(result, cseg, object_name=object_name)
        overlay_path = _write_overlay_to_temp(overlay_img, temp_dir)
        overlay_msg = f" Overlay path: {overlay_path}" if overlay_path else ""

        if not detections:
            obj_note = f" matching '{object_name}'." if object_name and object_name.strip() else " above confidence threshold."
            return (
                f"Surgical scene segmentation: no structures detected{obj_note} "
                "Classes: Black Background, Abdominal Wall, Liver, Gastrointestinal Tract, Fat, "
                "Grasper, Connective Tissue, Blood, Cystic Duct, L-hook Electrocautery, Gallbladder, Hepatic Vein, Liver Ligament."
                + overlay_msg
            )

        # Deduplicate by class name, keep max confidence per name, build summary with confidence
        name_to_conf = {}
        for _cid, name, conf_val in detections:
            if name not in name_to_conf or conf_val > name_to_conf[name]:
                name_to_conf[name] = conf_val
        count = len(detections)
        names_with_conf = [f"{name} (conf {name_to_conf[name]:.2f})" for name in name_to_conf]
        obj_note = f" (filtered to {object_name})" if object_name and object_name.strip() else ""
        summary = (
            f"Surgical scene segmentation (CholecSeg8k): detected {count} region(s){obj_note}. "
            f"Structures with confidence: {', '.join(names_with_conf)}."
            + overlay_msg
        )
        return summary
