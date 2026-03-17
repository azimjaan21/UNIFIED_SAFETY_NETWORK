"""Render unified PPE+pose ground-truth annotations for quick debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml


DEFAULT_DATA = "ppepose_dataset.yaml"
DEFAULT_SOURCE = "datasets/ppepose/images/train/000620.jpg"
DEFAULT_PROJECT = "runs/ppepose"
DEFAULT_NAME = "gt_debug"
DEFAULT_DRAW_PERSON_BBOX = False

HELMET_COLOR = (0, 165, 255)  # orange
VEST_COLOR = (0, 255, 0)  # green
PERSON_BBOX_COLOR = (255, 255, 255)  # white
KEYPOINT_COLOR = (0, 255, 255)  # yellow
SKELETON_COLOR = (203, 192, 255)  # pink

SKELETON_EDGES = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to unified dataset YAML.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Image path to visualize.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Output project directory.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Output run name.")
    parser.add_argument("--draw-person-bbox", action="store_true", default=DEFAULT_DRAW_PERSON_BBOX)
    return parser.parse_args()


def load_dataset_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_annotation_path(image_path: Path, data_cfg: dict) -> Path:
    dataset_root = Path(data_cfg["path"])
    image_rel = image_path.relative_to(dataset_root)
    split = image_rel.parts[1]
    annotation_dir = dataset_root / data_cfg["annotations"][split]
    return annotation_dir / f"{image_path.stem}.json"


def draw_box(image, bbox_xywh, color, label: str | None = None) -> None:
    x, y, w, h = bbox_xywh
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if label:
        text_y = max(y1 - 6, 16)
        cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_person(image, person: dict, draw_bbox: bool = False) -> None:
    keypoints = person["keypoints"]
    if draw_bbox:
        draw_box(image, person["bbox_xywh"], PERSON_BBOX_COLOR, "person-gt")

    for a, b in SKELETON_EDGES:
        ka = keypoints[a]
        kb = keypoints[b]
        if ka[2] > 0 and kb[2] > 0:
            pt_a = (int(round(ka[0])), int(round(ka[1])))
            pt_b = (int(round(kb[0])), int(round(kb[1])))
            cv2.line(image, pt_a, pt_b, SKELETON_COLOR, 2, cv2.LINE_AA)

    for x, y, v in keypoints:
        if v > 0:
            radius = 4 if v > 1 else 3
            cv2.circle(image, (int(round(x)), int(round(y))), radius, KEYPOINT_COLOR, -1, cv2.LINE_AA)


def render_ground_truth(image_path: Path, annotation_path: Path, draw_bbox: bool) -> tuple[any, dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    with annotation_path.open("r", encoding="utf-8") as f:
        annotation = json.load(f)

    for det in annotation.get("detections", []):
        category = det["category"]
        color = HELMET_COLOR if category == "helmet" else VEST_COLOR
        draw_box(image, det["bbox_xywh"], color, f"{category}-gt")

    for person in annotation.get("persons", []):
        draw_person(image, person, draw_bbox=draw_bbox)

    return image, annotation


def main() -> None:
    args = parse_args()
    data_cfg = load_dataset_config(args.data)
    image_path = Path(args.source)
    annotation_path = resolve_annotation_path(image_path, data_cfg)

    rendered, annotation = render_ground_truth(image_path, annotation_path, draw_bbox=args.draw_person_bbox)

    save_dir = Path(args.project) / args.name
    vis_dir = save_dir / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / image_path.name
    cv2.imwrite(str(out_path), rendered)

    print(f"Saved GT visualization: {out_path}")
    print(
        f"Summary: detections={len(annotation.get('detections', []))}, "
        f"persons={len(annotation.get('persons', []))}"
    )


if __name__ == "__main__":
    main()
