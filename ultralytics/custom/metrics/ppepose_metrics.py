"""Reusable metric computation for the unified PPE+pose model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


OKS_SIGMA = np.array(
    [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]
) / 10.0
DEFAULT_IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


@dataclass(frozen=True)
class MetricFieldSpec:
    """Mapping between a metric payload field and the trainer log key."""

    key: str
    section: str
    field: str


def xywh_to_xyxy(box: list[float] | np.ndarray) -> np.ndarray:
    x, y, w, h = map(float, box)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def xywhn_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> np.ndarray:
    if boxes.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x, y, w, h = boxes.unbind(-1)
    x1 = (x - w / 2) * width
    y1 = (y - h / 2) * height
    x2 = (x + w / 2) * width
    y2 = (y + h / 2) * height
    return torch.stack((x1, y1, x2, y2), dim=-1).cpu().numpy().astype(np.float32)


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
                    score = box_iou(pred["bbox"], gt["bbox"]) if matcher == "iou" else compute_oks(
                        pred["keypoints"], gt["keypoints"], gt["bbox"]
                    )
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


def build_batch_ground_truth(
    batch: dict[str, Any], class_names: dict[int, str], kpt_shape: tuple[int, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_ids = [Path(p).stem for p in batch["im_file"]]
    batch_size = len(image_ids)
    height, width = batch["img"].shape[-2:]

    gt_ppe = [{"image_id": image_ids[i], "gt": []} for i in range(batch_size)]
    gt_person = [{"image_id": image_ids[i], "gt": []} for i in range(batch_size)]

    ppe_boxes = xywhn_to_xyxy(batch["bboxes"], width, height)
    ppe_classes = batch["cls"].view(-1).long().cpu().numpy() if batch["cls"].numel() else np.zeros(0, dtype=np.int64)
    ppe_batch_idx = batch["batch_idx"].view(-1).long().cpu().numpy() if batch["batch_idx"].numel() else np.zeros(0, dtype=np.int64)
    for box, cls_id, image_index in zip(ppe_boxes, ppe_classes, ppe_batch_idx):
        gt_ppe[int(image_index)]["gt"].append({"bbox": box.astype(np.float32), "class_id": int(cls_id)})

    person_boxes = xywhn_to_xyxy(batch["person_bboxes"], width, height)
    person_batch_idx = (
        batch["person_batch_idx"].view(-1).long().cpu().numpy()
        if batch["person_batch_idx"].numel()
        else np.zeros(0, dtype=np.int64)
    )
    person_keypoints = batch["keypoints"].detach().cpu().numpy() if batch["keypoints"].numel() else np.zeros((0, *kpt_shape), dtype=np.float32)
    if len(person_keypoints):
        person_keypoints = person_keypoints.copy()
        person_keypoints[..., 0] *= width
        person_keypoints[..., 1] *= height
    for box, keypoints, image_index in zip(person_boxes, person_keypoints, person_batch_idx):
        gt_person[int(image_index)]["gt"].append({"bbox": box.astype(np.float32), "class_id": 0, "keypoints": keypoints.astype(np.float32)})

    return gt_ppe, gt_person


def build_prediction_records(
    results: list[dict[str, torch.Tensor]], image_ids: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ppe_records = []
    person_records = []
    for image_id, result in zip(image_ids, results):
        ppe_records.append(
            {
                "image_id": image_id,
                "preds": [
                    {"bbox": box.astype(np.float32), "score": float(score), "class_id": int(cls_id)}
                    for box, score, cls_id in zip(
                        result["ppe_boxes"].detach().cpu().numpy(),
                        result["ppe_scores"].detach().cpu().numpy(),
                        result["ppe_classes"].detach().cpu().numpy(),
                    )
                ],
            }
        )
        person_records.append(
            {
                "image_id": image_id,
                "preds": [
                    {
                        "bbox": box.astype(np.float32),
                        "score": float(score),
                        "class_id": 0,
                        "keypoints": keypoints.astype(np.float32),
                    }
                    for box, score, keypoints in zip(
                        result["person_boxes"].detach().cpu().numpy(),
                        result["person_scores"].detach().cpu().numpy(),
                        result["person_keypoints"].detach().cpu().numpy(),
                    )
                ],
            }
        )
    return ppe_records, person_records


def summarize_metric_sections(metrics: dict[str, Any], ppe_names: dict[int, str]) -> dict[str, float]:
    summary = {
        "metrics/ppe_precision": metrics["ppe_detection"]["precision@50"],
        "metrics/ppe_recall": metrics["ppe_detection"]["recall@50"],
        "metrics/ppe_mAP50": metrics["ppe_detection"]["mAP50"],
        "metrics/ppe_mAP50-95": metrics["ppe_detection"]["mAP50-95"],
        "metrics/person_precision": metrics["person_bbox"]["precision@50"],
        "metrics/person_recall": metrics["person_bbox"]["recall@50"],
        "metrics/person_mAP50": metrics["person_bbox"]["mAP50"],
        "metrics/person_mAP50-95": metrics["person_bbox"]["mAP50-95"],
        "metrics/pose_precision": metrics["pose_keypoints"]["precision@50"],
        "metrics/pose_recall": metrics["pose_keypoints"]["recall@50"],
        "metrics/pose_mAP50": metrics["pose_keypoints"]["mAP50"],
        "metrics/pose_mAP50-95": metrics["pose_keypoints"]["mAP50-95"],
    }

    for class_id, class_name in ppe_names.items():
        class_metrics = metrics["ppe_detection"]["per_class"].get(class_id, {})
        prefix = f"metrics/{class_name}"
        summary[f"{prefix}_precision"] = class_metrics.get("precision@50", 0.0)
        summary[f"{prefix}_recall"] = class_metrics.get("recall@50", 0.0)
        summary[f"{prefix}_mAP50"] = class_metrics.get("AP50", 0.0)
        summary[f"{prefix}_mAP50-95"] = class_metrics.get("mAP50-95", 0.0)

    return summary


def metric_field_specs(ppe_names: dict[int, str]) -> list[MetricFieldSpec]:
    specs = [
        MetricFieldSpec("metrics/ppe_precision", "ppe_detection", "precision@50"),
        MetricFieldSpec("metrics/ppe_recall", "ppe_detection", "recall@50"),
        MetricFieldSpec("metrics/ppe_mAP50", "ppe_detection", "mAP50"),
        MetricFieldSpec("metrics/ppe_mAP50-95", "ppe_detection", "mAP50-95"),
        MetricFieldSpec("metrics/person_precision", "person_bbox", "precision@50"),
        MetricFieldSpec("metrics/person_recall", "person_bbox", "recall@50"),
        MetricFieldSpec("metrics/person_mAP50", "person_bbox", "mAP50"),
        MetricFieldSpec("metrics/person_mAP50-95", "person_bbox", "mAP50-95"),
        MetricFieldSpec("metrics/pose_precision", "pose_keypoints", "precision@50"),
        MetricFieldSpec("metrics/pose_recall", "pose_keypoints", "recall@50"),
        MetricFieldSpec("metrics/pose_mAP50", "pose_keypoints", "mAP50"),
        MetricFieldSpec("metrics/pose_mAP50-95", "pose_keypoints", "mAP50-95"),
    ]
    for class_id, class_name in ppe_names.items():
        specs.extend(
            [
                MetricFieldSpec(f"metrics/{class_name}_precision", "ppe_detection", f"per_class.{class_id}.precision@50"),
                MetricFieldSpec(f"metrics/{class_name}_recall", "ppe_detection", f"per_class.{class_id}.recall@50"),
                MetricFieldSpec(f"metrics/{class_name}_mAP50", "ppe_detection", f"per_class.{class_id}.AP50"),
                MetricFieldSpec(f"metrics/{class_name}_mAP50-95", "ppe_detection", f"per_class.{class_id}.mAP50-95"),
            ]
        )
    return specs


def build_metric_payload(
    ppe_predictions: list[dict[str, Any]],
    person_predictions: list[dict[str, Any]],
    gt_ppe: list[dict[str, Any]],
    gt_person: list[dict[str, Any]],
    ppe_names: dict[int, str],
    thresholds: np.ndarray = DEFAULT_IOU_THRESHOLDS,
) -> tuple[dict[str, Any], dict[str, float]]:
    ppe_class_ids = sorted(ppe_names.keys())
    ppe_metrics = evaluate_predictions(ppe_predictions, gt_ppe, ppe_class_ids, thresholds, matcher="iou")
    person_box_metrics = evaluate_predictions(person_predictions, gt_person, [0], thresholds, matcher="iou")
    pose_metrics = evaluate_predictions(person_predictions, gt_person, [0], thresholds, matcher="oks")
    metrics = {"ppe_detection": ppe_metrics, "person_bbox": person_box_metrics, "pose_keypoints": pose_metrics}
    return metrics, summarize_metric_sections(metrics, ppe_names)
