"""Evaluate unified PPE+pose checkpoints before and after Pose-Guided PPE Consistency Filtering."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.custom.models import load_ppepose_model, postprocess_ppepose_predictions
from ultralytics.custom.postprocess import PoseGuidedPPEConsistencyFilter
from ultralytics.custom.runtime import (
    load_annotation,
    load_data_config,
    parse_bool,
    preprocess_image,
    resolve_split_entries,
    scale_unified_result,
    tensor_to_list,
)


OKS_SIGMA = np.array(
    [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]
) / 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate unified PPE+pose checkpoints.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, default="ppepose_dataset.yaml")
    parser.add_argument("--model-cfg", type=str, default="ppepose_unified.yaml")
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--filter-on", type=parse_bool, default=True)
    parser.add_argument("--helmet-validation", type=parse_bool, default=True)
    parser.add_argument("--vest-validation", type=parse_bool, default=True)
    parser.add_argument("--filter-mode", type=str, default="score_decay", choices=("hard", "score_decay"))
    parser.add_argument("--region-mode", type=str, default="polygon", choices=("rectangle", "polygon"))
    parser.add_argument("--project", type=str, default="runs/ppepose")
    parser.add_argument("--name", type=str, default="eval")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def xywh_to_xyxy(box: list[float] | np.ndarray) -> np.ndarray:
    x, y, w, h = map(float, box)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area1 = max(box1[2] - box1[0], 0.0) * max(box1[3] - box1[1], 0.0)
    area2 = max(box2[2] - box2[0], 0.0) * max(box2[3] - box2[1], 0.0)
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def compute_oks(pred_kpts: np.ndarray, gt_kpts: np.ndarray, gt_bbox: np.ndarray) -> float:
    vis = gt_kpts[:, 2] > 0 if gt_kpts.shape[1] == 3 else np.ones(len(gt_kpts), dtype=bool)
    if vis.sum() == 0:
        return 0.0
    area = max((gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1]), 1.0)
    diff = pred_kpts[vis, :2] - gt_kpts[vis, :2]
    e = np.sum(diff**2, axis=1) / (2 * (OKS_SIGMA[: len(gt_kpts)][vis] ** 2) * (area + 1e-6))
    return float(np.mean(np.exp(-e)))


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    class_ids: list[int],
    thresholds: np.ndarray,
    matcher: str = "iou",
) -> dict[str, Any]:
    per_class = {}
    confusion = {}
    for class_id in class_ids:
        aps = []
        total_gt = sum(sum(1 for gt in sample["gt"] if gt["class_id"] == class_id) for sample in ground_truth)
        precision_50 = recall_50 = f1_50 = 0.0
        tp50 = fp50 = fn50 = 0
        for threshold in thresholds:
            scored_predictions = []
            for sample in predictions:
                for pred in sample["preds"]:
                    if pred["class_id"] == class_id:
                        scored_predictions.append({"image_id": sample["image_id"], **pred})
            scored_predictions.sort(key=lambda x: x["score"], reverse=True)
            matched = {
                sample["image_id"]: np.zeros(sum(1 for gt in sample["gt"] if gt["class_id"] == class_id), dtype=bool)
                for sample in ground_truth
            }

            tp, fp = [], []
            for pred in scored_predictions:
                gt_sample = next(sample for sample in ground_truth if sample["image_id"] == pred["image_id"])
                candidates = [gt for gt in gt_sample["gt"] if gt["class_id"] == class_id]
                best_score, best_index = -1.0, -1
                for idx, gt in enumerate(candidates):
                    if matched[pred["image_id"]][idx]:
                        continue
                    score = box_iou(pred["bbox"], gt["bbox"]) if matcher == "iou" else compute_oks(pred["keypoints"], gt["keypoints"], gt["bbox"])
                    if score > best_score:
                        best_score, best_index = score, idx
                if best_index >= 0 and best_score >= threshold:
                    matched[pred["image_id"]][best_index] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)

            if total_gt == 0:
                aps.append(0.0)
                continue
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / max(total_gt, 1)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
            aps.append(compute_ap(recalls, precisions))

            if abs(float(threshold) - 0.5) < 1e-9:
                tp50 = int(tp_cum[-1]) if len(tp_cum) else 0
                fp50 = int(fp_cum[-1]) if len(fp_cum) else 0
                fn50 = int(total_gt - tp50)
                precision_50 = tp50 / max(tp50 + fp50, 1)
                recall_50 = tp50 / max(total_gt, 1)
                f1_50 = (2 * precision_50 * recall_50 / max(precision_50 + recall_50, 1e-6)) if total_gt else 0.0

        per_class[class_id] = {
            "AP50": aps[0] if len(aps) else 0.0,
            "mAP50-95": float(np.mean(aps)) if aps else 0.0,
            "precision@50": precision_50,
            "recall@50": recall_50,
            "F1@50": f1_50,
        }
        confusion[class_id] = {"TP": tp50, "FP": fp50, "FN": fn50}

    return {
        "per_class": per_class,
        "mAP50": float(np.mean([v["AP50"] for v in per_class.values()])) if per_class else 0.0,
        "mAP50-95": float(np.mean([v["mAP50-95"] for v in per_class.values()])) if per_class else 0.0,
        "precision@50": float(np.mean([v["precision@50"] for v in per_class.values()])) if per_class else 0.0,
        "recall@50": float(np.mean([v["recall@50"] for v in per_class.values()])) if per_class else 0.0,
        "F1@50": float(np.mean([v["F1@50"] for v in per_class.values()])) if per_class else 0.0,
        "confusion": confusion,
    }


def extract_gt(annotation: dict[str, Any], class_names: dict[int, str], kpt_shape: tuple[int, int]) -> dict[str, Any]:
    name_to_idx = {name: idx for idx, name in class_names.items()}
    gt_ppe = []
    for det in annotation.get("detections", []):
        if det.get("category") is not None:
            class_id = name_to_idx[det["category"]]
        else:
            class_id = int(det["class_id"])
        gt_ppe.append({"bbox": xywh_to_xyxy(det["bbox_xywh"]), "class_id": class_id})
    gt_persons = []
    for person in annotation.get("persons", []):
        keypoints = np.zeros(kpt_shape, dtype=np.float32)
        for idx, kp in enumerate(person.get("keypoints", [])[: kpt_shape[0]]):
            keypoints[idx, : min(len(kp), kpt_shape[1])] = kp[: kpt_shape[1]]
        compliance = person.get("compliance") or person.get("attributes") or {}
        gt_persons.append({"bbox": xywh_to_xyxy(person["bbox_xywh"]), "class_id": 0, "keypoints": keypoints, "compliance": compliance})
    return {"ppe": gt_ppe, "persons": gt_persons}


def run_model_on_image(model, image: np.ndarray, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    tensor, meta = preprocess_image(image, args.imgsz, int(model.stride.max()), next(model.parameters()).device)
    with torch.inference_mode():
        preds = model(tensor)
    result = postprocess_ppepose_predictions(preds, tuple(model.kpt_shape), conf=args.conf, iou=args.iou, max_det=args.max_det)[0]
    return scale_unified_result(result, meta)


def build_prediction_records(result: dict[str, torch.Tensor], filtered: dict | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_ppe = [
        {"bbox": box, "score": float(score), "class_id": int(cls)}
        for box, score, cls in zip(
            result["ppe_boxes"].detach().cpu().numpy(),
            result["ppe_scores"].detach().cpu().numpy(),
            result["ppe_classes"].detach().cpu().numpy(),
        )
    ]
    filtered_ppe = (
        [
            {
                "bbox": np.asarray(det["bbox"], dtype=np.float32),
                "score": float(det["filtered_score"]),
                "class_id": int(det["class_id"]),
            }
            for det in filtered["validated_detections"]
        ]
        if filtered is not None
        else raw_ppe
    )
    persons = [
        {"bbox": box, "score": float(score), "class_id": 0, "keypoints": kpts}
        for box, score, kpts in zip(
            result["person_boxes"].detach().cpu().numpy(),
            result["person_scores"].detach().cpu().numpy(),
            result["person_keypoints"].detach().cpu().numpy(),
        )
    ]
    return raw_ppe, filtered_ppe, persons


def evaluate_compliance(per_image_records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = {"helmet": [], "vest": [], "overall": []}
    for record in per_image_records:
        if not record["gt_persons"]:
            continue
        predicted = record.get("filtered_summary")
        if predicted is None:
            continue
        gt_people = record["gt_persons"]
        pred_people = predicted["per_person"]
        matched = set()
        for gt in gt_people:
            if "helmet" not in gt["compliance"] and "vest" not in gt["compliance"]:
                continue
            best_idx, best_iou = -1, -1.0
            for idx, pred in enumerate(pred_people):
                if idx in matched:
                    continue
                iou = box_iou(np.asarray(pred["bbox"], dtype=np.float32), gt["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx < 0 or best_iou < 0.5:
                continue
            matched.add(best_idx)
            pred = pred_people[best_idx]
            helmet_gt = gt["compliance"].get("helmet")
            vest_gt = gt["compliance"].get("vest")
            if helmet_gt is not None:
                scores["helmet"].append(int(bool(helmet_gt) == bool(pred["helmet_on"])))
            if vest_gt is not None:
                scores["vest"].append(int(bool(vest_gt) == bool(pred["vest_on"])))
            if helmet_gt is not None and vest_gt is not None:
                scores["overall"].append(
                    int(bool(helmet_gt) == bool(pred["helmet_on"]) and bool(vest_gt) == bool(pred["vest_on"]))
                )

    return {
        "helmet_compliance_accuracy": float(np.mean(scores["helmet"])) if scores["helmet"] else None,
        "vest_compliance_accuracy": float(np.mean(scores["vest"])) if scores["vest"] else None,
        "overall_compliance_accuracy": float(np.mean(scores["overall"])) if scores["overall"] else None,
    }


def export_summary_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = []
    for section, values in metrics.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        rows.append((section, key, inner_key, inner_value))
                else:
                    rows.append((section, key, "", value))
        else:
            rows.append(("summary", section, "", values))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "metric", "submetric", "value"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data = load_data_config(args.data)
    run_dir = Path(args.project) / args.name
    if run_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"{run_dir} already exists. Pass --exist-ok to reuse it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = load_ppepose_model(args.weights, model_cfg=args.model_cfg, device=args.device)
    filter_module = (
        PoseGuidedPPEConsistencyFilter(
            class_names=data["names"],
            mode=args.filter_mode,
            region_mode=args.region_mode,
            helmet_validation=args.helmet_validation,
            vest_validation=args.vest_validation,
        )
        if args.filter_on
        else None
    )

    thresholds = np.arange(0.5, 1.0, 0.05)
    entries = resolve_split_entries(args.data, args.split)
    per_image_records = []
    raw_ppe_preds, filtered_ppe_preds, person_preds = [], [], []
    gt_ppe_records, gt_person_records = [], []

    for entry in entries:
        image = cv2.imread(str(entry["image_path"]))
        if image is None:
            continue
        annotation = load_annotation(entry["annotation_path"])
        gt = extract_gt(annotation, data["names"], tuple(data["kpt_shape"]))
        result = run_model_on_image(model, image, args)
        filtered = filter_module.filter(result) if filter_module is not None else None
        raw_ppe, filtered_ppe, persons = build_prediction_records(result, filtered)

        image_id = entry["image_path"].stem
        raw_ppe_preds.append({"image_id": image_id, "preds": raw_ppe})
        filtered_ppe_preds.append({"image_id": image_id, "preds": filtered_ppe})
        person_preds.append({"image_id": image_id, "preds": persons})
        gt_ppe_records.append({"image_id": image_id, "gt": gt["ppe"]})
        gt_person_records.append({"image_id": image_id, "gt": gt["persons"]})
        per_image_records.append(
            {
                "image_id": image_id,
                "gt_persons": gt["persons"],
                "filtered_summary": filtered,
                "raw_result": tensor_to_list(result),
            }
        )

    ppe_classes = sorted(data["names"].keys())
    raw_metrics = evaluate_predictions(raw_ppe_preds, gt_ppe_records, ppe_classes, thresholds, matcher="iou")
    filtered_metrics = evaluate_predictions(filtered_ppe_preds, gt_ppe_records, ppe_classes, thresholds, matcher="iou")
    person_box_metrics = evaluate_predictions(person_preds, gt_person_records, [0], thresholds, matcher="iou")
    pose_metrics = evaluate_predictions(person_preds, gt_person_records, [0], thresholds, matcher="oks")

    compliance_metrics = evaluate_compliance(per_image_records)
    filter_impact = {
        "false_positives_reduced": int(
            sum(v["FP"] for v in raw_metrics["confusion"].values()) - sum(v["FP"] for v in filtered_metrics["confusion"].values())
        ),
        "precision_improvement": filtered_metrics["precision@50"] - raw_metrics["precision@50"],
        "recall_change": filtered_metrics["recall@50"] - raw_metrics["recall@50"],
        "F1_change": filtered_metrics["F1@50"] - raw_metrics["F1@50"],
    }

    metrics = {
        "ppe_detection_raw": raw_metrics,
        "ppe_detection_filtered": filtered_metrics,
        "person_bbox": person_box_metrics,
        "pose_keypoints": pose_metrics,
        "filter_impact": filter_impact,
        "compliance": compliance_metrics,
        "confusion_summary": {
            "raw": raw_metrics["confusion"],
            "filtered": filtered_metrics["confusion"],
        },
    }

    export_summary_csv(run_dir / "metrics_summary.csv", metrics)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(tensor_to_list(metrics), f, indent=2)
    with (run_dir / "per_image_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(tensor_to_list(per_image_records), f, indent=2)


if __name__ == "__main__":
    main()
