"""Convert a CVAT-for-images export into the unified PPE+pose JSON dataset layout."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
PPE_LABELS = {"helmet", "vest"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CVAT export directory containing images and annotations.xml")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/ppepose"),
        help="Output root for unified dataset layout",
    )
    parser.add_argument("--train-ratio", type=float, default=0.75, help="Fraction of annotated images assigned to train")
    parser.add_argument("--val-ratio", type=float, default=0.25, help="Fraction of annotated images assigned to val")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Fraction of annotated images assigned to test")
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=0.05,
        help="Extra padding added around person boxes derived from keypoints as a fraction of box size",
    )
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of hard-linking when possible")
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def keypoint_visibility(point_node: ET.Element) -> int:
    if point_node.get("outside", "0") == "1":
        return 0
    if point_node.get("occluded", "0") == "1":
        return 1
    return 2


def parse_point_xy(point_node: ET.Element) -> tuple[float, float]:
    xy = point_node.get("points", "0,0").split(",")
    return float(xy[0]), float(xy[1])


def derive_person_bbox(keypoints: list[list[float]], width: int, height: int, padding: float) -> list[float] | None:
    valid_xy = [(kp[0], kp[1]) for kp in keypoints if len(kp) >= 3 and kp[2] > 0]
    if not valid_xy:
        return None

    xs, ys = zip(*valid_xy)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_w = max(x_max - x_min, 1.0)
    box_h = max(y_max - y_min, 1.0)
    pad_x = box_w * padding
    pad_y = box_h * padding

    x_min = clamp(x_min - pad_x, 0.0, float(width - 1))
    y_min = clamp(y_min - pad_y, 0.0, float(height - 1))
    x_max = clamp(x_max + pad_x, x_min + 1.0, float(width))
    y_max = clamp(y_max + pad_y, y_min + 1.0, float(height))
    return [round(x_min, 2), round(y_min, 2), round(x_max - x_min, 2), round(y_max - y_min, 2)]


def parse_image_annotation(image_node: ET.Element, bbox_padding: float) -> dict:
    width = int(image_node.get("width", 0))
    height = int(image_node.get("height", 0))
    image_name = image_node.get("name", "")

    detections: list[dict] = []
    persons: list[dict] = []

    person_index = 0
    for child in image_node:
        if child.tag == "box" and child.get("label") in PPE_LABELS:
            xtl = float(child.get("xtl", 0.0))
            ytl = float(child.get("ytl", 0.0))
            xbr = float(child.get("xbr", 0.0))
            ybr = float(child.get("ybr", 0.0))
            detections.append(
                {
                    "category": child.get("label"),
                    "bbox_xywh": [round(xtl, 2), round(ytl, 2), round(max(xbr - xtl, 1.0), 2), round(max(ybr - ytl, 1.0), 2)],
                }
            )
        elif child.tag == "skeleton" and child.get("label") == "person":
            person_index += 1
            keypoint_map = {name: [0.0, 0.0, 0.0] for name in KEYPOINT_NAMES}
            for point in child.findall("points"):
                label = point.get("label", "")
                if label not in keypoint_map:
                    continue
                x, y = parse_point_xy(point)
                keypoint_map[label] = [round(x, 2), round(y, 2), float(keypoint_visibility(point))]

            keypoints = [keypoint_map[name] for name in KEYPOINT_NAMES]
            bbox_xywh = derive_person_bbox(keypoints, width, height, bbox_padding)
            if bbox_xywh is None:
                continue

            persons.append(
                {
                    "person_id": f"{Path(image_name).stem}_person_{person_index:02d}",
                    "bbox_xywh": bbox_xywh,
                    "keypoints": keypoints,
                }
            )

    return {
        "image": {"file_name": image_name, "width": width, "height": height},
        "detections": detections,
        "persons": persons,
    }


def split_names(names: list[str], train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, list[str]]:
    if not names:
        return {"train": [], "val": [], "test": []}

    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")

    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    n = len(names)
    train_n = int(math.floor(n * train_ratio))
    val_n = int(math.floor(n * val_ratio))
    test_n = int(math.floor(n * test_ratio))

    remainder = n - (train_n + val_n + test_n)
    buckets = [("train", train_ratio), ("val", val_ratio), ("test", test_ratio)]
    buckets.sort(key=lambda item: item[1], reverse=True)
    counts = {"train": train_n, "val": val_n, "test": test_n}
    for split_name, _ in buckets:
        if remainder <= 0:
            break
        counts[split_name] += 1
        remainder -= 1

    if n > 1 and counts["val"] == 0 and val_ratio > 0:
        donor = "train" if counts["train"] > 1 else "test"
        if counts[donor] > 0:
            counts[donor] -= 1
            counts["val"] += 1

    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": names[:train_end],
        "val": names[train_end:val_end],
        "test": names[val_end : val_end + counts["test"]],
    }


def ensure_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "annotations" / split).mkdir(parents=True, exist_ok=True)


def transfer_image(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    xml_path = args.input / "annotations.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing annotations.xml in {args.input}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_nodes = root.findall("image")
    parsed = {}
    for image_node in image_nodes:
        item = parse_image_annotation(image_node, args.bbox_padding)
        # Skip images without any annotations. CVAT exports may include unannotated frames.
        if not item["detections"] and not item["persons"]:
            continue
        parsed[item["image"]["file_name"]] = item

    if not parsed:
        raise RuntimeError("No annotated images found in the CVAT export.")

    names = sorted(parsed)
    splits = split_names(names, args.train_ratio, args.val_ratio, args.test_ratio)
    ensure_dirs(args.output)

    for split_name, split_names_list in splits.items():
        for image_name in split_names_list:
            src_image = args.input / image_name
            if not src_image.exists():
                raise FileNotFoundError(f"Missing image referenced by XML: {src_image}")

            dst_image = args.output / "images" / split_name / image_name
            dst_ann = args.output / "annotations" / split_name / f"{Path(image_name).stem}.json"
            transfer_image(src_image, dst_image, args.copy_images)
            dst_ann.write_text(json.dumps(parsed[image_name], indent=2), encoding="utf-8")

    summary = {split: len(items) for split, items in splits.items()}
    print(json.dumps({"output": str(args.output), "annotated_images": len(parsed), "splits": summary}, indent=2))


if __name__ == "__main__":
    main()
