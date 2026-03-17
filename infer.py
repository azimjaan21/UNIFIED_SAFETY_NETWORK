"""Run unified PPE+pose inference and optional Pose-Guided PPE Consistency Filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.custom.models import load_ppepose_model, postprocess_ppepose_predictions
from ultralytics.custom.postprocess import PoseGuidedPPEConsistencyFilter
from ultralytics.custom.runtime import (
    load_data_config,
    parse_bool,
    preprocess_image,
    scale_unified_result,
    tensor_to_list,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8), (8, 10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified PPE+pose inference with optional PG-PCF filtering.")
    parser.add_argument("--weights", type=str, required=True, help="Unified checkpoint path.")
    parser.add_argument("--source", type=str, required=True, help="Image, directory, or video path.")
    parser.add_argument("--data", type=str, default="ppepose_dataset.yaml", help="Dataset YAML for class metadata.")
    parser.add_argument("--model-cfg", type=str, default="ppepose_unified.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--use_kp_guided_filter", type=parse_bool, default=True)
    parser.add_argument("--helmet-validation", type=parse_bool, default=True)
    parser.add_argument("--vest-validation", type=parse_bool, default=True)
    parser.add_argument("--filter-mode", type=str, default="score_decay", choices=("hard", "score_decay"))
    parser.add_argument("--region-mode", type=str, default="polygon", choices=("rectangle", "polygon"))
    parser.add_argument("--project", type=str, default="runs/ppepose")
    parser.add_argument("--name", type=str, default="infer")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def draw_keypoints(image: np.ndarray, keypoints: np.ndarray, conf_thr: float = 0.35) -> None:
    for a, b in SKELETON:
        if keypoints[a, 2] >= conf_thr and keypoints[b, 2] >= conf_thr:
            pa = tuple(np.round(keypoints[a, :2]).astype(int))
            pb = tuple(np.round(keypoints[b, :2]).astype(int))
            cv2.line(image, pa, pb, (255, 180, 0), 2)
    for kp in keypoints:
        if kp[2] >= conf_thr:
            cv2.circle(image, tuple(np.round(kp[:2]).astype(int)), 3, (0, 220, 255), -1)


def draw_scene(
    image: np.ndarray,
    result: dict[str, torch.Tensor],
    filtered: dict | None,
    class_names: dict[int, str],
) -> np.ndarray:
    canvas = image.copy()
    person_boxes = result["person_boxes"].detach().cpu().numpy()
    person_scores = result["person_scores"].detach().cpu().numpy()
    person_keypoints = result["person_keypoints"].detach().cpu().numpy()

    for person_index, (box, score, keypoints) in enumerate(zip(person_boxes, person_scores, person_keypoints)):
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(
            canvas,
            f"person {score:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 220, 40),
            2,
        )
        draw_keypoints(canvas, keypoints)
        if filtered is not None:
            person_summary = next((p for p in filtered["per_person"] if p["person_index"] == person_index), None)
            if person_summary is not None:
                status = f"H:{int(person_summary['helmet_on'])} V:{int(person_summary['vest_on'])}"
                cv2.putText(
                    canvas,
                    status,
                    (x1, min(canvas.shape[0] - 8, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

    if filtered is None:
        boxes = result["ppe_boxes"].detach().cpu().numpy()
        scores = result["ppe_scores"].detach().cpu().numpy()
        classes = result["ppe_classes"].detach().cpu().numpy()
        dets = [
            {
                "bbox": box.tolist(),
                "score": float(score),
                "class_name": class_names.get(int(cls), str(int(cls))),
            }
            for box, score, cls in zip(boxes, scores, classes)
        ]
    else:
        dets = filtered["validated_detections"]

    for det in dets:
        x1, y1, x2, y2 = np.round(det["bbox"]).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 120, 255), 2)
        label_score = det.get("filtered_score", det.get("score", 0.0))
        cv2.putText(
            canvas,
            f"{det['class_name']} {label_score:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 120, 255),
            2,
        )
        if filtered is not None and det.get("assigned_person") is not None:
            person_box = person_boxes[det["assigned_person"]]
            pcenter = ((person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2)
            dcenter = ((det["bbox"][0] + det["bbox"][2]) / 2, (det["bbox"][1] + det["bbox"][3]) / 2)
            cv2.line(canvas, tuple(np.round(dcenter).astype(int)), tuple(np.round(pcenter).astype(int)), (0, 120, 255), 1)

    return canvas


def run_frame(
    image: np.ndarray,
    model,
    filter_module: PoseGuidedPPEConsistencyFilter | None,
    args: argparse.Namespace,
    class_names: dict[int, str],
) -> tuple[dict, dict | None, np.ndarray, np.ndarray]:
    tensor, meta = preprocess_image(image, args.imgsz, int(model.stride.max()), next(model.parameters()).device)
    with torch.inference_mode():
        preds = model(tensor)
    result = postprocess_ppepose_predictions(preds, tuple(model.kpt_shape), conf=args.conf, iou=args.iou, max_det=args.max_det)[0]
    result = scale_unified_result(result, meta)
    filtered = filter_module.filter(result) if filter_module is not None else None
    raw_vis = draw_scene(image, result, None, class_names)
    filtered_vis = draw_scene(image, result, filtered, class_names) if filtered is not None else raw_vis.copy()
    return result, filtered, raw_vis, filtered_vis


def save_prediction_json(path: Path, raw_result: dict, filtered: dict | None) -> None:
    payload = {
        "raw": tensor_to_list(raw_result),
        "filtered": tensor_to_list(filtered) if filtered is not None else None,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def iter_images(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return [source]


def main() -> None:
    args = parse_args()
    data = load_data_config(args.data)
    class_names = data["names"]
    run_dir = Path(args.project) / args.name
    if run_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"{run_dir} already exists. Pass --exist-ok to reuse it.")
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "filtered").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    model = load_ppepose_model(args.weights, model_cfg=args.model_cfg, device=args.device)
    filter_module = (
        PoseGuidedPPEConsistencyFilter(
            class_names=class_names,
            mode=args.filter_mode,
            region_mode=args.region_mode,
            helmet_validation=args.helmet_validation,
            vest_validation=args.vest_validation,
        )
        if args.use_kp_guided_filter
        else None
    )

    source = Path(args.source)
    if source.suffix.lower() in VIDEO_SUFFIXES:
        cap = cv2.VideoCapture(str(source))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_writer = cv2.VideoWriter(
            str(run_dir / "raw" / f"{source.stem}_raw.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        filtered_writer = cv2.VideoWriter(
            str(run_dir / "filtered" / f"{source.stem}_filtered.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_result, filtered, raw_vis, filtered_vis = run_frame(frame, model, filter_module, args, class_names)
            save_prediction_json(run_dir / "predictions" / f"{source.stem}_{frame_idx:06d}.json", raw_result, filtered)
            raw_writer.write(raw_vis)
            filtered_writer.write(filtered_vis)
            frame_idx += 1
        cap.release()
        raw_writer.release()
        filtered_writer.release()
        return

    for image_path in iter_images(source):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        raw_result, filtered, raw_vis, filtered_vis = run_frame(image, model, filter_module, args, class_names)
        save_prediction_json(run_dir / "predictions" / f"{image_path.stem}.json", raw_result, filtered)
        cv2.imwrite(str(run_dir / "raw" / image_path.name), raw_vis)
        cv2.imwrite(str(run_dir / "filtered" / image_path.name), filtered_vis)


if __name__ == "__main__":
    main()
