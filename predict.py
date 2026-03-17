"""Run a unified PPE+pose checkpoint on images with a cleaner research visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.custom.models import load_ppepose_model, postprocess_ppepose_predictions
from ultralytics.custom.runtime import load_data_config, preprocess_image, scale_unified_result, tensor_to_list


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8), (8, 10)]
PINK = (180, 105, 255)
YELLOW = (0, 255, 255)
GREEN = (40, 220, 40)
ORANGE = (0, 140, 255)

# Simple script-mode defaults. Edit these paths directly if you don't want to pass CLI arguments.
DEFAULT_WEIGHTS = "runs/ppepose/overfit_dup25/weights/best.pt"
DEFAULT_SOURCE = r"C:\Users\dalab\Desktop\azimjaan21\my_PAPERS\AAAI 26\HELMET_AAAI26\DATA\VOC2007_YOLO\images\test\000777.jpg"
DEFAULT_DATA = "ppepose_dataset.yaml"
DEFAULT_MODEL_CFG = "ppepose_unified.yaml"
DEFAULT_PROJECT = r"C:\Users\dalab\Desktop\azimjaan21\my_PAPERS\Unified_Safety_Network\output"
DEFAULT_NAME = "bestpt_clean_demo"
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "cpu"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_MAX_DET = 300
DEFAULT_EXIST_OK = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Path to best.pt or last.pt.")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Image path or directory of images.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Dataset YAML path.")
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG, help="Unified model YAML path.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Inference image size.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device string, e.g. cpu or 0.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET, help="Maximum detections per image.")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="Output project directory.")
    parser.add_argument("--name", type=str, default=DEFAULT_NAME, help="Run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse output directory if it already exists.")
    args = parser.parse_args()
    if DEFAULT_EXIST_OK and not args.exist_ok:
        args.exist_ok = True
    return args


def iter_images(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return [source]


def draw_pose_only(image: np.ndarray, keypoints: np.ndarray, conf_thr: float = 0.35) -> None:
    for a, b in SKELETON:
        if keypoints[a, 2] >= conf_thr and keypoints[b, 2] >= conf_thr:
            pa = tuple(np.round(keypoints[a, :2]).astype(int))
            pb = tuple(np.round(keypoints[b, :2]).astype(int))
            cv2.line(image, pa, pb, PINK, 2)
    for kp in keypoints:
        if kp[2] >= conf_thr:
            cv2.circle(image, tuple(np.round(kp[:2]).astype(int)), 3, YELLOW, -1)


def draw_clean_scene(image: np.ndarray, result: dict[str, torch.Tensor], class_names: dict[int, str]) -> np.ndarray:
    canvas = image.copy()

    person_keypoints = result["person_keypoints"].detach().cpu().numpy()
    for keypoints in person_keypoints:
        draw_pose_only(canvas, keypoints)

    boxes = result["ppe_boxes"].detach().cpu().numpy()
    scores = result["ppe_scores"].detach().cpu().numpy()
    classes = result["ppe_classes"].detach().cpu().numpy()
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = np.round(box).astype(int)
        class_name = class_names.get(int(cls), str(int(cls)))
        color = GREEN if class_name == "vest" else ORANGE
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{class_name} {score:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return canvas


def save_prediction_json(path: Path, raw_result: dict[str, torch.Tensor]) -> None:
    path.write_text(json.dumps({"raw": tensor_to_list(raw_result)}, indent=2), encoding="utf-8")


def run_frame(image: np.ndarray, model, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    tensor, meta = preprocess_image(image, args.imgsz, int(model.stride.max()), next(model.parameters()).device)
    with torch.inference_mode():
        preds = model(tensor)
    result = postprocess_ppepose_predictions(preds, tuple(model.kpt_shape), conf=args.conf, iou=args.iou, max_det=args.max_det)[0]
    return scale_unified_result(result, meta)


def main() -> None:
    args = parse_args()
    data = load_data_config(args.data)
    class_names = data["names"]

    run_dir = Path(args.project) / args.name
    if run_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"{run_dir} already exists. Pass --exist-ok to reuse it.")
    (run_dir / "visuals").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    model = load_ppepose_model(args.weights, model_cfg=args.model_cfg, device=args.device)
    source = Path(args.source)
    for image_path in iter_images(source):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        result = run_frame(image, model, args)
        visual = draw_clean_scene(image, result, class_names)
        cv2.imwrite(str((run_dir / "visuals" / image_path.name)), visual)
        save_prediction_json(run_dir / "predictions" / f"{image_path.stem}.json", result)


if __name__ == "__main__":
    main()
