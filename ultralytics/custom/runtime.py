"""Runtime helpers shared by unified PPE+pose training, inference, and evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.utils import YAML, ops


def parse_bool(value: Any) -> bool:
    """Parse permissive CLI boolean values."""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Unable to parse boolean value from '{value}'.")


def load_data_config(data_yaml: str | Path) -> dict[str, Any]:
    """Load a unified dataset YAML file."""
    data = YAML.load(data_yaml)
    data["path"] = str(Path(data["path"]).resolve())
    return data


def resolve_annotation_path(image_path: Path, data: dict[str, Any], split: str) -> Path:
    """Resolve the paired JSON annotation path for an image path."""
    annotations = data.get("annotations", "annotations")
    if isinstance(annotations, dict):
        ann_root = Path(data["path"]) / annotations.get(split, f"annotations/{split}")
    else:
        ann_root = Path(data["path"]) / annotations / split
    image_root = Path(data["path"]) / data[split]
    try:
        relative = image_path.relative_to(image_root)
    except ValueError:
        relative = Path(image_path.name)
    return ann_root / relative.with_suffix(".json")


def resolve_split_entries(data_yaml: str | Path, split: str = "val") -> list[dict[str, Path]]:
    """Collect image/annotation pairs for a dataset split."""
    data = load_data_config(data_yaml)
    image_root = Path(data["path"]) / data[split]
    image_files = sorted(
        p for p in image_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    return [
        {
            "image_path": image_path,
            "annotation_path": resolve_annotation_path(image_path, data, split),
        }
        for image_path in image_files
    ]


def load_annotation(annotation_path: str | Path) -> dict[str, Any]:
    """Load a unified JSON annotation file."""
    with Path(annotation_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def letterbox(image: np.ndarray, new_shape: int | tuple[int, int], stride: int = 32) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Resize and pad an image while preserving aspect ratio."""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    shape = image.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, (r, r), (dw, dh)


def preprocess_image(image: np.ndarray, imgsz: int, stride: int, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    """Prepare a single image tensor for unified model inference."""
    resized, ratio, pad = letterbox(image, imgsz, stride=stride)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return tensor.to(device), {"input_shape": resized.shape[:2], "ratio": ratio, "pad": pad, "orig_shape": image.shape[:2]}


def scale_unified_result(result: dict[str, torch.Tensor], meta: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Scale predicted boxes and keypoints from the model canvas back to the original image."""
    input_shape = meta["input_shape"]
    orig_shape = meta["orig_shape"]
    ratio_pad = (meta["ratio"], meta["pad"])
    result = {k: v.clone() if hasattr(v, "clone") else v for k, v in result.items()}
    if len(result["ppe_boxes"]):
        result["ppe_boxes"] = ops.scale_boxes(input_shape, result["ppe_boxes"], orig_shape, ratio_pad=ratio_pad)
    if len(result["person_boxes"]):
        result["person_boxes"] = ops.scale_boxes(input_shape, result["person_boxes"], orig_shape, ratio_pad=ratio_pad)
    if len(result["person_keypoints"]):
        result["person_keypoints"] = ops.scale_coords(
            input_shape,
            result["person_keypoints"].clone(),
            orig_shape,
            ratio_pad=ratio_pad,
        )
    return result


def tensor_to_list(value: Any) -> Any:
    """Convert tensors and numpy arrays into JSON-serializable Python objects."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [tensor_to_list(v) for v in value]
    if isinstance(value, dict):
        return {k: tensor_to_list(v) for k, v in value.items()}
    return value
