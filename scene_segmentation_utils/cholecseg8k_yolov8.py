#!/usr/bin/env python3
"""
CholecSeg8k – YOLOv8 segmentation: training and inference.

Converted from cholecseg8k-yolov8.ipynb. Supports:
  - Training: prepare dataset from CholecSeg8k, train YOLOv8-seg, save best.pt
  - Inference: load model, run prediction, save masks/overlays
  - Model save/load and export (ONNX)

Usage:
  # Training (from CholecSeg8k directory layout)
  python cholecseg8k_yolov8.py train --data_dir /path/to/cholecseg8k --output_dir ./cholecseg_work --epochs 30

  # Inference
  python cholecseg8k_yolov8.py run --model path/to/best.pt --source path/to/image_or_dir --output_dir ./predictions

  # Copy/save model after load
  python cholecseg8k_yolov8.py run --model path/to/best.pt --save_model ./my_cholecseg.pt
"""

import argparse
import os
import re

# Avoid "Disk quota exceeded" when writing to home (matplotlib font cache, Ultralytics Arial.ttf)
def _writable_dir():
    cand = os.environ.get("TMPDIR") or os.getcwd()
    if not cand or "/path/to" in cand or not os.path.isdir(cand):
        return os.getcwd()
    return cand

_writable = _writable_dir()
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_writable, "matplotlib_cache"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_writable, "ultralytics_config"))

import random
import shutil
from pathlib import Path

import cv2
import yaml
import numpy as np
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# CholecSeg8k class and color mappings (from notebook)
# -----------------------------------------------------------------------------

COLOR_CLASS_MAPPING = {
    (127, 127, 127): 0,
    (210, 140, 140): 1,
    (255, 114, 114): 2,
    (231, 70, 156): 3,
    (186, 183, 75): 4,
    (170, 255, 0): 5,
    (255, 85, 0): 6,
    (255, 0, 0): 7,
    (255, 255, 0): 8,
    (169, 255, 184): 9,
    (255, 160, 165): 10,
    (0, 50, 128): 11,
    (111, 74, 0): 12,
}

CLASS_COLOR_MAPPING = {idx: color for color, idx in COLOR_CLASS_MAPPING.items()}
CLASS_COLOR_MAPPING_BGR = {idx: (c[2], c[1], c[0]) for idx, c in CLASS_COLOR_MAPPING.items()}

CLASS_NAMES = [
    "Black Background", "Abdominal Wall", "Liver", "Gastrointestinal Tract", "Fat",
    "Grasper", "Connective Tissue", "Blood", "Cystic Duct", "L-hook Electrocautery",
    "Gallbladder", "Hepatic Vein", "Liver Ligament",
]
NUM_CLASSES = 13


# -----------------------------------------------------------------------------
# Training: dataset preparation (from notebook)
# -----------------------------------------------------------------------------

def write_polygon_file(class_contour_mapping, H, W, output_path, img_name):
    """Write YOLO segmentation label file (class_id + normalized polygon points)."""
    coordinates = {}
    for obj in class_contour_mapping:
        polygons = []
        for cnt in class_contour_mapping[obj]:
            if cv2.contourArea(cnt) > 20:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(round(x / W, 4))
                    polygon.append(round(y / H, 4))
                polygons.append(polygon)
        coordinates[obj] = polygons

    out_path = os.path.join(output_path, f"{img_name}.txt")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        for obj in coordinates:
            for polygon in coordinates[obj]:
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon) - 1:
                        f.write(f"{p}\n")
                    elif p_ == 0:
                        f.write(f"{obj} {p} ")
                    else:
                        f.write(f"{p} ")


def _dataset_already_prepared(output_dir):
    """True if output_dir contains a valid prepared dataset (config + train images)."""
    config_path = os.path.join(output_dir, "config.yaml")
    train_img_dir = os.path.join(output_dir, "images", "train")
    if not os.path.isfile(config_path) or not os.path.isdir(train_img_dir):
        return False
    try:
        next(os.scandir(train_img_dir), None)
    except (OSError, StopIteration):
        return False
    return any(1 for _ in os.scandir(train_img_dir))


def prepare_dataset(data_dir, output_dir, force=False):
    """
    Copy CholecSeg8k data to raw_images/mask_images, create label files, split train/val/test,
    resize and move to images/{train,validation,test} and labels/{train,validation,test}.
    Returns paths and config_path for training.
    If output_dir already has a prepared dataset and force is False, skips preparation.
    """
    op_path = output_dir
    if not force and _dataset_already_prepared(op_path):
        config_path = os.path.join(op_path, "config.yaml")
        print("Dataset already prepared; using existing config and splits.", flush=True)
        return config_path, op_path

    rawimages_path = os.path.join(op_path, "raw_images")
    maskimages_path = os.path.join(op_path, "mask_images")
    labels_path = os.path.join(op_path, "labels")
    imgtrainpath = os.path.join(op_path, "images", "train")
    imgvalpath = os.path.join(op_path, "images", "validation")
    imgtestpath = os.path.join(op_path, "images", "test")
    labeltrainpath = os.path.join(op_path, "labels", "train")
    labelvalpath = os.path.join(op_path, "labels", "validation")
    labeltestpath = os.path.join(op_path, "labels", "test")

    for d in [rawimages_path, maskimages_path, labels_path, imgtrainpath, imgvalpath, imgtestpath,
              labeltrainpath, labelvalpath, labeltestpath]:
        os.makedirs(d, exist_ok=True)

    path = data_dir
    path_abs = os.path.abspath(path)
    op_path_abs = os.path.abspath(op_path)
    # Only process CholecSeg8k video dirs (video01, video02, ...) to avoid copying previous outputs (cholecseg, cholecseg_work, etc.)
    VIDEO_DIR_PATTERN = re.compile(r"^video\d+$", re.IGNORECASE)
    copied = 0
    for directory in sorted(os.listdir(path)):
        if not VIDEO_DIR_PATTERN.match(directory):
            continue
        dir_path = os.path.join(path, directory)
        if not os.path.isdir(dir_path):
            continue
        # Skip output dir when it lives inside data_dir (avoids SameFileError)
        dir_path_abs = os.path.abspath(dir_path)
        if dir_path_abs == op_path_abs or dir_path_abs.startswith(op_path_abs + os.sep):
            continue
        for sub_dir in os.listdir(dir_path):
            sub_dir_path = os.path.join(dir_path, sub_dir)
            if not os.path.isdir(sub_dir_path):
                continue
            for image in os.listdir(sub_dir_path):
                src_path = os.path.join(sub_dir_path, image)
                if not os.path.isfile(src_path):
                    continue
                newname = sub_dir + image
                if "mask" not in image.lower():
                    dest = os.path.join(rawimages_path, newname)
                    if os.path.abspath(src_path) != os.path.abspath(dest):
                        shutil.copy2(src_path, dest)
                        copied += 1
                if "color_mask" in image.lower():
                    dest = os.path.join(maskimages_path, newname)
                    if os.path.abspath(src_path) != os.path.abspath(dest):
                        shutil.copy2(src_path, dest)
                        copied += 1
        print(f"  Copied from {directory} ... ({copied} files so far)", flush=True)

    print(f"Copy done: {copied} files. Building label files from masks...", flush=True)
    rawimages_list = sorted([f for f in os.listdir(rawimages_path) if os.path.isfile(os.path.join(rawimages_path, f))])
    maskimages_list = sorted([f for f in os.listdir(maskimages_path) if "color_mask" in f and os.path.isfile(os.path.join(maskimages_path, f))])

    # Build mask -> raw name mapping: mask "xxx_color_mask.png" -> raw "xxx.png" or "xxx.jpg"
    raw_set = set(rawimages_list)
    mask_basename_to_raw = {}
    for m in maskimages_list:
        base = os.path.splitext(m)[0].replace("_color_mask", "")
        for ext in (".png", ".jpg", ".jpeg"):
            cand = base + ext
            if cand in raw_set:
                mask_basename_to_raw[m] = cand
                break

    # Create label files (one per mask; label basename = raw image basename for 1:1 match)
    created = 0
    for img in maskimages_list:
        raw_basename = mask_basename_to_raw.get(img)
        if raw_basename is None:
            continue
        newname = os.path.splitext(raw_basename)[0]

        image = cv2.cvtColor(cv2.imread(os.path.join(maskimages_path, img)), cv2.COLOR_BGR2RGB)
        pixels = image.reshape((-1, 3))
        unique_colors = np.unique(pixels, axis=0)
        unique_colors_defined = [tuple(v) for v in unique_colors if tuple(map(int, v)) in COLOR_CLASS_MAPPING]
        if not unique_colors_defined:
            continue

        H, W = image.shape[:2]
        class_contour_mapping = {}
        for color in unique_colors_defined:
            color = np.array(color, dtype=np.uint8)
            class_code = COLOR_CLASS_MAPPING[tuple(color)]
            mask = cv2.inRange(image, color, color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            class_contour_mapping[class_code] = contours

        write_polygon_file(class_contour_mapping, H, W, labels_path, newname)
        created += 1
        if created % 500 == 0 and created > 0:
            print(f"  Labels: {created} ...", flush=True)

    print(f"Created {created} label files.", flush=True)

    # Split: use raw image list; only keep those that have a label
    label_ext = ".txt"
    raw_with_labels = [r for r in rawimages_list if os.path.isfile(os.path.join(labels_path, os.path.splitext(r)[0] + label_ext))]
    random.shuffle(raw_with_labels)
    n = len(raw_with_labels)
    train_images = raw_with_labels[: int(0.8 * n)]
    val_images = raw_with_labels[int(0.8 * n) : int(0.9 * n)]
    test_images = raw_with_labels[int(0.9 * n) :]

    def change_extension(f):
        return os.path.splitext(f)[0] + label_ext

    train_labels = [change_extension(f) for f in train_images]
    val_labels = [change_extension(f) for f in val_images]
    test_labels = [change_extension(f) for f in test_images]

    image_size = 640

    def move_images(data_list, source_path, destination_path):
        for file in data_list:
            filepath = os.path.join(source_path, file)
            if not os.path.isfile(filepath):
                continue
            img = cv2.imread(filepath)
            if img is None:
                continue
            img_resized = cv2.resize(img, (image_size, image_size))
            cv2.imwrite(os.path.join(destination_path, file), img_resized)

    def move_label_files(data_list, source_path, destination_path):
        for file in data_list:
            filepath = os.path.join(source_path, file)
            if os.path.isfile(filepath):
                shutil.move(filepath, destination_path)

    move_images(train_images, rawimages_path, imgtrainpath)
    move_images(val_images, rawimages_path, imgvalpath)
    move_images(test_images, rawimages_path, imgtestpath)
    move_label_files(train_labels, labels_path, labeltrainpath)
    move_label_files(val_labels, labels_path, labelvalpath)
    move_label_files(test_labels, labels_path, labeltestpath)

    # Write config.yaml (ultralytics format)
    config_path = os.path.join(op_path, "config.yaml")
    config = {
        "path": op_path,
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        "nc": NUM_CLASSES,
        "names": dict(enumerate(CLASS_NAMES)),
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Dataset prepared. Train {len(train_images)}, val {len(val_images)}, test {len(test_images)}.")
    return config_path, op_path


def train_model(config_path, output_dir, epochs=30, imgsz=640, batch=8, model_save_path=None, **train_kwargs):
    """Load YOLOv8m-seg and train. Best weights saved to runs/segment/train/weights/best.pt."""
    model = YOLO("yolov8m-seg.yaml").load("yolov8m-seg.pt")
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        iou=0.4,
        conf=0.01,
        degrees=60,
        shear=30,
        perspective=0.0005,
        project=os.path.join(output_dir, "runs"),
        name="segment/train",
        exist_ok=True,
        **train_kwargs,
    )
    best_pt = os.path.join(output_dir, "runs", "segment", "train", "weights", "best.pt")
    if model_save_path and os.path.isfile(best_pt):
        to_path = model_save_path if model_save_path.endswith(".pt") else model_save_path + ".pt"
        shutil.copy2(best_pt, to_path)
        print(f"Best model copied to {to_path}")
    return results


# -----------------------------------------------------------------------------
# Inference (from notebook + existing script)
# -----------------------------------------------------------------------------

def _mask_xy_to_fillpoly(mask_xy):
    """Convert Ultralytics mask xy to OpenCV fillPoly format: contiguous (N, 2) np.int32, N >= 3. Returns None if invalid."""
    if mask_xy is None:
        return None
    try:
        p = np.asarray(mask_xy, dtype=np.float64)
        if p.size < 6:
            return None
        if p.ndim == 3:
            p = p.reshape(-1, 2)
        elif p.ndim == 1:
            p = p.reshape(-1, 2)
        if p.shape[0] < 3 or p.shape[1] != 2:
            return None
        return np.ascontiguousarray(p.astype(np.int32))
    except Exception:
        return None


def postprocess_prediction(prediction, class_color_mapping_bgr=None):
    """Convert YOLO segmentation result to BGR mask with CholecSeg8k colors."""
    if class_color_mapping_bgr is None:
        class_color_mapping_bgr = CLASS_COLOR_MAPPING_BGR
    masks = []
    if prediction.boxes is None or len(prediction.boxes.cls) == 0:
        return np.zeros_like(prediction.orig_img)
    for i in range(len(prediction.boxes.cls)):
        background = np.zeros_like(prediction.orig_img)
        cls_id = int(prediction.boxes.cls[i])
        color = class_color_mapping_bgr.get(cls_id, (0, 0, 0))
        if prediction.masks is not None and i < len(prediction.masks):
            try:
                raw = prediction.masks[i].xy[0]
                mask_points = _mask_xy_to_fillpoly(raw)
                if mask_points is not None:
                    cv2.fillPoly(background, [mask_points], color)
            except Exception:
                pass
        masks.append(background)
    if len(masks) == 1:
        return masks[0]
    out = masks[0].copy()
    for i in range(1, len(masks)):
        zero_mask = np.all(out == [0, 0, 0], axis=2)
        out[zero_mask] = masks[i][zero_mask]
    return out


def run_inference(model, source, output_dir=None, save_overlay=True, save_mask=True, conf=0.25, iou=0.4):
    """Run segmentation and optionally save masks and overlays."""
    results = model.predict(source, conf=conf, iou=iou, verbose=False)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        masks_dir = os.path.join(output_dir, "masks")
        overlays_dir = os.path.join(output_dir, "overlays")
        if save_mask:
            os.makedirs(masks_dir, exist_ok=True)
        if save_overlay:
            os.makedirs(overlays_dir, exist_ok=True)

    for result in results:
        if result.orig_img is None:
            continue
        orig = result.orig_img
        name = Path(result.path).stem if result.path else "frame"
        seg_mask = postprocess_prediction(result, CLASS_COLOR_MAPPING_BGR)
        if output_dir:
            if save_mask:
                cv2.imwrite(os.path.join(masks_dir, f"{name}_mask.png"), seg_mask)
            if save_overlay:
                overlay = cv2.addWeighted(orig, 0.6, seg_mask, 0.4, 0)
                cv2.imwrite(os.path.join(overlays_dir, f"{name}_overlay.png"), overlay)

    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def cmd_train(args):
    random.seed(args.seed)
    print("Preparing dataset (copying images, building labels)...", flush=True)
    config_path, op_path = prepare_dataset(args.data_dir, args.output_dir, force=args.force_prepare)
    print("Starting YOLOv8 training...", flush=True)
    train_model(
        config_path,
        op_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_save_path=args.model_save,
    )


def cmd_run(args):
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = YOLO(args.model)
    print(f"Loaded model from {args.model}")

    if args.save_model:
        to_path = args.save_model if args.save_model.endswith(".pt") else args.save_model + ".pt"
        shutil.copy2(args.model, to_path)
        print(f"Model copied to {to_path}")

    if args.export_onnx:
        export_path = model.export(format="onnx", imgsz=640)
        print(f"Exported ONNX to {export_path}")

    if args.source:
        output_dir = None if args.no_save else args.output_dir
        run_inference(
            model,
            args.source,
            output_dir=output_dir,
            save_overlay=not args.no_overlay,
            save_mask=True,
            conf=args.conf,
            iou=args.iou,
        )
        if output_dir:
            print(f"Predictions saved to {output_dir}")
    elif not args.save_model and not args.export_onnx:
        print("No --source. Use --source to run inference.")


def main():
    parser = argparse.ArgumentParser(
        description="CholecSeg8k YOLOv8 segmentation: train and inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="train | run")

    # train
    t = subparsers.add_parser("train", help="Prepare dataset and train YOLOv8-seg")
    t.add_argument("--data_dir", type=str, required=True, help="CholecSeg8k dataset root (directory/subdir/images)")
    t.add_argument("--output_dir", type=str, default="./cholecseg_work", help="Working directory for prepared data and runs")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", "--batch_size", type=int, default=8, dest="batch", help="Batch size")
    t.add_argument("--model_save", type=str, default=None, help="Copy best.pt to this path after training")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--force_prepare", action="store_true", help="Re-prepare dataset even if output_dir already has one")

    # run (inference)
    r = subparsers.add_parser("run", help="Load model and run inference")
    r.add_argument("--model", type=str, required=True, help="Path to weights (e.g. best.pt)")
    r.add_argument("--source", type=str, default=None, help="Image, directory, or video")
    r.add_argument("--output_dir", type=str, default="./cholecseg_predictions")
    r.add_argument("--no_save", action="store_true")
    r.add_argument("--no_overlay", action="store_true")
    r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--iou", type=float, default=0.4)
    r.add_argument("--save_model", type=str, default=None, help="Copy loaded model to this path")
    r.add_argument("--export_onnx", action="store_true")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python cholecseg8k_yolov8.py train --data_dir /path/to/cholecseg8k --output_dir ./work --epochs 30")
        print("  python cholecseg8k_yolov8.py run --model ./work/runs/segment/train/weights/best.pt --source ./images --output_dir ./out")


if __name__ == "__main__":
    main()
