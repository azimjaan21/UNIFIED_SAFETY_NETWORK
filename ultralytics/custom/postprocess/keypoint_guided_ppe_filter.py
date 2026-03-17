"""Pose-Guided PPE Consistency Filtering (PG-PCF).

This module implements an anatomically grounded post-prediction validation stage for industrial PPE monitoring.
Predicted human keypoints define adaptive head and torso regions that are used to validate PPE detections, suppress
false alarms, and derive per-person compliance summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AnatomicalRegion:
    """Container for an adaptive person-centric validation region."""

    label: str
    box: np.ndarray
    polygon: np.ndarray
    anchor: np.ndarray
    scale: float
    visibility: float
    reliability: float


class PoseGuidedPPEConsistencyFilter:
    """Validate PPE detections with anatomically plausible regions derived from predicted human pose."""

    def __init__(
        self,
        class_names: dict[int, str] | None = None,
        mode: str = "score_decay",
        region_mode: str = "polygon",
        helmet_validation: bool = True,
        vest_validation: bool = True,
        min_association_score: float = 0.35,
        min_keypoint_conf: float = 0.35,
        head_margin_scale: float = 0.30,
        torso_margin_scale: float = 0.15,
        decay_floor: float = 0.20,
        score_weights: dict[str, float] | None = None,
        one_item_per_person: bool = True,
    ):
        self.class_names = class_names or {0: "helmet", 1: "vest"}
        self.mode = mode
        self.region_mode = region_mode
        self.helmet_validation = helmet_validation
        self.vest_validation = vest_validation
        self.min_association_score = min_association_score
        self.min_keypoint_conf = min_keypoint_conf
        self.head_margin_scale = head_margin_scale
        self.torso_margin_scale = torso_margin_scale
        self.decay_floor = decay_floor
        self.one_item_per_person = one_item_per_person
        self.score_weights = score_weights or {
            "center": 0.35,
            "iou": 0.20,
            "distance": 0.25,
            "visibility": 0.10,
            "scale": 0.10,
        }

    def filter(
        self,
        ppe_detections: dict[str, Any] | list[dict[str, Any]],
        person_predictions: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run PG-PCF on PPE detections and predicted persons."""
        if person_predictions is None and isinstance(ppe_detections, dict) and "person_boxes" in ppe_detections:
            person_predictions = ppe_detections

        detections = self._unpack_ppe(ppe_detections)
        persons = self._unpack_persons(person_predictions)
        regions = [self._build_person_regions(person) for person in persons]

        validated, associations = [], []
        for det_index, det in enumerate(detections):
            if not self._validation_enabled(det["class_name"]):
                det["filtered_score"] = det["score"]
                det["suppressed"] = False
                det["assigned_person"] = None
                det["association_score"] = 1.0
                validated.append(det)
                continue

            candidates = []
            for person_index, person in enumerate(persons):
                region = regions[person_index][det["class_name"]]
                score = self._association_score(det, person, region)
                candidates.append((score, person_index, region))

            best_score, best_person, best_region = max(candidates, default=(0.0, None, None), key=lambda x: x[0])
            det["assigned_person"] = best_person
            det["association_score"] = best_score

            reliable = best_region is not None and best_region.reliability >= 0.4
            if best_person is None or (not reliable and best_score < self.min_association_score):
                det["filtered_score"] = det["score"]
                det["suppressed"] = False
            elif reliable and best_score < self.min_association_score:
                if self.mode == "hard" and reliable:
                    det["filtered_score"] = 0.0
                    det["suppressed"] = True
                else:
                    det["filtered_score"] = det["score"] * max(self.decay_floor, best_score)
                    det["suppressed"] = det["filtered_score"] <= 0.0
            else:
                det["filtered_score"] = det["score"]
                det["suppressed"] = False

            associations.append(
                {
                    "detection_index": det_index,
                    "class_name": det["class_name"],
                    "assigned_person": best_person,
                    "association_score": float(best_score),
                    "region_reliability": float(best_region.reliability if best_region is not None else 0.0),
                }
            )
            if not det["suppressed"] and det["filtered_score"] > 0:
                validated.append(det)

        per_person = self._summarize_person_compliance(persons, validated)
        return {
            "validated_detections": validated,
            "associations": associations,
            "per_person": per_person,
            "suppressed_detections": [d for d in detections if d.get("suppressed", False)],
        }

    def _build_person_regions(self, person: dict[str, Any]) -> dict[str, AnatomicalRegion]:
        keypoints = person["keypoints"]
        bbox = person["bbox"]
        return {
            "helmet": self._build_head_region(keypoints, bbox),
            "vest": self._build_torso_region(keypoints, bbox),
        }

    def _build_head_region(self, keypoints: np.ndarray, bbox: np.ndarray) -> AnatomicalRegion:
        head_indices = [0, 1, 2, 3, 4]
        shoulder_indices = [5, 6]
        head_points = self._visible_points(keypoints, head_indices)
        shoulder_points = self._visible_points(keypoints, shoulder_indices)

        if len(head_points) >= 2:
            points = np.vstack((head_points[:, :2], shoulder_points[:, :2])) if len(shoulder_points) else head_points[:, :2]
            box = self._padded_box(points, self.head_margin_scale)
            anchor = head_points[:, :2].mean(axis=0)
            visibility = float(head_points[:, 2].mean())
            scale = max(np.linalg.norm(points.max(axis=0) - points.min(axis=0)), 1.0)
            reliability = min(1.0, 0.5 + 0.1 * len(head_points))
        else:
            x1, y1, x2, y2 = bbox
            height = y2 - y1
            box = np.array([x1, y1 - 0.05 * height, x2, y1 + 0.30 * height], dtype=np.float32)
            anchor = np.array([(x1 + x2) / 2, y1 + 0.12 * height], dtype=np.float32)
            visibility = 0.0
            scale = max(height * 0.25, 1.0)
            reliability = 0.2

        polygon = self._box_to_polygon(box)
        if self.region_mode == "polygon" and len(head_points) >= 2:
            polygon = self._box_to_polygon(box)
        return AnatomicalRegion("helmet", box, polygon, anchor, scale, visibility, reliability)

    def _build_torso_region(self, keypoints: np.ndarray, bbox: np.ndarray) -> AnatomicalRegion:
        torso_indices = [5, 6, 11, 12]
        torso_points = self._visible_points(keypoints, torso_indices)
        if len(torso_points) >= 3:
            points = torso_points[:, :2]
            box = self._padded_box(points, self.torso_margin_scale)
            anchor = points.mean(axis=0)
            visibility = float(torso_points[:, 2].mean())
            scale = max(np.linalg.norm(points.max(axis=0) - points.min(axis=0)), 1.0)
            reliability = min(1.0, 0.4 + 0.15 * len(torso_points))
            polygon = self._torso_polygon(points) if self.region_mode == "polygon" else self._box_to_polygon(box)
        else:
            x1, y1, x2, y2 = bbox
            height = y2 - y1
            box = np.array([x1 + 0.05 * (x2 - x1), y1 + 0.20 * height, x2 - 0.05 * (x2 - x1), y1 + 0.72 * height])
            anchor = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)
            visibility = 0.0
            scale = max(box[3] - box[1], 1.0)
            reliability = 0.2
            polygon = self._box_to_polygon(box)
        return AnatomicalRegion("vest", box.astype(np.float32), polygon.astype(np.float32), anchor, scale, visibility, reliability)

    def _association_score(self, det: dict[str, Any], person: dict[str, Any], region: AnatomicalRegion) -> float:
        if region is None:
            return 0.0
        center = self._box_center(det["bbox"])
        region_box = region.box
        region_area = self._box_area(region_box)
        det_area = self._box_area(det["bbox"])
        center_score = 1.0 if self._point_in_polygon(center, region.polygon) else 0.0
        iou_score = self._box_iou(det["bbox"], region_box)
        dist = np.linalg.norm(center - region.anchor)
        distance_score = max(0.0, 1.0 - dist / (region.scale + 1e-6))
        scale_ratio = det_area / max(region_area, 1e-6)
        scale_score = float(np.exp(-abs(np.log(max(scale_ratio, 1e-6)))))
        visibility_score = min(1.0, max(region.visibility, region.reliability))

        fused = (
            self.score_weights["center"] * center_score
            + self.score_weights["iou"] * iou_score
            + self.score_weights["distance"] * distance_score
            + self.score_weights["visibility"] * visibility_score
            + self.score_weights["scale"] * scale_score
        )
        return float(min(1.0, fused))

    def _summarize_person_compliance(self, persons: list[dict[str, Any]], detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        per_person: list[dict[str, Any]] = []
        for person_index, person in enumerate(persons):
            assigned = [d for d in detections if d.get("assigned_person") == person_index]
            best_by_class: dict[str, dict[str, Any]] = {}
            for det in assigned:
                name = det["class_name"]
                if name not in best_by_class or det["filtered_score"] > best_by_class[name]["filtered_score"]:
                    best_by_class[name] = det

            per_person.append(
                {
                    "person_index": person_index,
                    "bbox": person["bbox"].tolist(),
                    "score": float(person["score"]),
                    "helmet_on": "helmet" in best_by_class,
                    "helmet_missing": "helmet" not in best_by_class,
                    "vest_on": "vest" in best_by_class,
                    "vest_missing": "vest" not in best_by_class,
                    "assigned_ppe": {
                        name: {
                            "bbox": det["bbox"].tolist(),
                            "score": float(det["filtered_score"]),
                            "association_score": float(det["association_score"]),
                        }
                        for name, det in best_by_class.items()
                    },
                }
            )
        return per_person

    def _validation_enabled(self, class_name: str) -> bool:
        if class_name == "helmet":
            return self.helmet_validation
        if class_name == "vest":
            return self.vest_validation
        return True

    def _unpack_ppe(self, detections: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(detections, list):
            return detections
        boxes = self._to_numpy(detections.get("ppe_boxes"))
        scores = self._to_numpy(detections.get("ppe_scores"))
        classes = self._to_numpy(detections.get("ppe_classes")).astype(int)
        unpacked = []
        for box, score, cls in zip(boxes, scores, classes):
            unpacked.append(
                {
                    "bbox": box.astype(np.float32),
                    "score": float(score),
                    "class_id": int(cls),
                    "class_name": self.class_names.get(int(cls), str(cls)),
                }
            )
        return unpacked

    def _unpack_persons(self, persons: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if persons is None:
            return []
        if isinstance(persons, list):
            return persons
        boxes = self._to_numpy(persons.get("person_boxes"))
        scores = self._to_numpy(persons.get("person_scores"))
        keypoints = self._to_numpy(persons.get("person_keypoints"))
        unpacked = []
        for box, score, kpts in zip(boxes, scores, keypoints):
            unpacked.append({"bbox": box.astype(np.float32), "score": float(score), "keypoints": kpts.astype(np.float32)})
        return unpacked

    def _visible_points(self, keypoints: np.ndarray, indices: list[int]) -> np.ndarray:
        selected = keypoints[indices]
        if selected.shape[-1] == 2:
            visibility = np.ones((selected.shape[0], 1), dtype=selected.dtype)
            selected = np.concatenate((selected, visibility), axis=-1)
        return selected[selected[:, 2] >= self.min_keypoint_conf]

    @staticmethod
    def _padded_box(points: np.ndarray, margin_scale: float) -> np.ndarray:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        size = np.maximum(maxs - mins, 1.0)
        margin = size * margin_scale
        return np.array([mins[0] - margin[0], mins[1] - margin[1], maxs[0] + margin[0], maxs[1] + margin[1]], dtype=np.float32)

    @staticmethod
    def _torso_polygon(points: np.ndarray) -> np.ndarray:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        return np.array(
            [
                [mins[0], mins[1]],
                [maxs[0], mins[1]],
                [maxs[0], maxs[1]],
                [mins[0], maxs[1]],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _box_center(box: np.ndarray) -> np.ndarray:
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)

    @staticmethod
    def _box_to_polygon(box: np.ndarray) -> np.ndarray:
        return np.array(
            [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
            dtype=np.float32,
        )

    @staticmethod
    def _box_area(box: np.ndarray) -> float:
        return float(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0))

    @staticmethod
    def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
        union = PoseGuidedPPEConsistencyFilter._box_area(box1) + PoseGuidedPPEConsistencyFilter._box_area(box2) - inter
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-6) + xi)
            if intersects:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if value is None:
            return np.zeros((0,), dtype=np.float32)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)
