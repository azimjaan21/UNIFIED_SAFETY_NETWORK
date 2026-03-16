"""Dataset support for unified PPE detection and human pose estimation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ultralytics.data.augment import Compose, Format, LetterBox, v8_transforms
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import colorstr


class PPEPoseFormat(Format):
    """Split jointly augmented instances into PPE and person-specific training targets."""

    def __init__(self, num_ppe_classes: int, person_class_index: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_ppe_classes = num_ppe_classes
        self.person_class_index = person_class_index

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        labels = super().__call__(labels)

        cls = labels["cls"].long()
        bboxes = labels["bboxes"]
        keypoints = labels.get("keypoints", torch.zeros((0, 0, 0), dtype=bboxes.dtype))

        class_ids = cls.squeeze(-1) if cls.numel() else torch.empty(0, dtype=torch.long)
        ppe_mask = class_ids < self.num_ppe_classes
        person_mask = class_ids == self.person_class_index

        labels["cls"] = cls[ppe_mask].float()
        labels["bboxes"] = bboxes[ppe_mask]
        labels["batch_idx"] = torch.zeros(int(ppe_mask.sum()), dtype=torch.float32)

        labels["person_cls"] = torch.zeros((int(person_mask.sum()), 1), dtype=torch.float32)
        labels["person_bboxes"] = bboxes[person_mask]
        labels["person_batch_idx"] = torch.zeros(int(person_mask.sum()), dtype=torch.float32)
        labels["keypoints"] = keypoints[person_mask]
        return labels


class PPEPoseUnifiedDataset(YOLODataset):
    """Unified dataset that reads one JSON annotation per image for PPE boxes and person pose labels."""

    def __init__(self, *args, data: dict | None = None, task: str = "ppepose", **kwargs):
        self.data = data or {}
        self.image_root = Path(kwargs.get("img_path", ""))
        self.num_ppe_classes = len(self.data.get("names", {}))
        self.person_class_index = self.num_ppe_classes
        super().__init__(*args, data=self.data, task="pose", **kwargs)

    def get_labels(self) -> list[dict]:
        """Load JSON annotations and convert them into Ultralytics instance dictionaries."""
        labels: list[dict] = []
        nkpt, ndim = self.data["kpt_shape"]
        for im_file in self.im_files:
            im_path = Path(im_file)
            width, height = self._image_shape(im_path)
            annotation = self._load_annotation(im_path)

            cls, bboxes, keypoints = [], [], []
            for det in annotation.get("detections", []):
                cls.append([self._map_ppe_class(det)])
                bboxes.append(self._xywh_abs_to_norm(det["bbox_xywh"], width, height))
                keypoints.append(np.zeros((nkpt, ndim), dtype=np.float32))

            for person in annotation.get("persons", []):
                cls.append([self.person_class_index])
                bboxes.append(self._xywh_abs_to_norm(person["bbox_xywh"], width, height))
                keypoints.append(self._keypoints_to_array(person.get("keypoints", []), width, height, nkpt, ndim))

            labels.append(
                {
                    "im_file": str(im_path),
                    "shape": (height, width),
                    "cls": np.asarray(cls, dtype=np.float32).reshape(-1, 1),
                    "bboxes": np.asarray(bboxes, dtype=np.float32).reshape(-1, 4),
                    "segments": [],
                    "keypoints": np.asarray(keypoints, dtype=np.float32).reshape(-1, nkpt, ndim),
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """Build augmentations and split mixed-task targets only after geometric transforms."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            PPEPoseFormat(
                self.num_ppe_classes,
                self.person_class_index,
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=True,
                return_obb=False,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate unified PPE and person targets into a single training batch."""
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        cat_keys = {"masks", "keypoints", "bboxes", "cls", "segments", "obb", "person_bboxes", "person_cls"}
        stack_keys = {"img", "text_feats", "sem_masks"}

        for i, key in enumerate(keys):
            value = values[i]
            if key in stack_keys:
                value = torch.stack(value, 0)
            elif key == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            elif key in cat_keys:
                value = torch.cat(value, 0)
            new_batch[key] = value

        for index_key in ("batch_idx", "person_batch_idx"):
            indices = list(new_batch[index_key])
            for i in range(len(indices)):
                indices[i] += i
            new_batch[index_key] = torch.cat(indices, 0) if indices else torch.zeros(0)
        return new_batch

    def _image_shape(self, image_path: Path) -> tuple[int, int]:
        with Image.open(image_path) as image:
            width, height = image.size
        return width, height

    def _load_annotation(self, image_path: Path) -> dict[str, Any]:
        ann_path = self._annotation_path(image_path)
        with ann_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _annotation_path(self, image_path: Path) -> Path:
        annotations = self.data.get("annotations", "annotations")
        split = self.image_root.name
        if isinstance(annotations, dict):
            ann_root = Path(self.data["path"]) / annotations.get(split, f"annotations/{split}")
        else:
            ann_root = Path(self.data["path"]) / annotations / split
        try:
            relative_path = image_path.relative_to(self.image_root)
        except ValueError:
            relative_path = Path(image_path.name)
        return ann_root / relative_path.with_suffix(".json")

    def _map_ppe_class(self, det: dict[str, Any]) -> int:
        category = det.get("category")
        if category is None and "class_id" in det:
            return int(det["class_id"])
        if isinstance(category, int):
            return category
        name_to_idx = {name: idx for idx, name in self.data["names"].items()}
        if category not in name_to_idx:
            raise KeyError(f"Unknown PPE category '{category}' in annotation.")
        return name_to_idx[category]

    @staticmethod
    def _xywh_abs_to_norm(bbox_xywh: list[float], width: int, height: int) -> np.ndarray:
        x, y, w, h = map(float, bbox_xywh)
        return np.array([(x + w / 2) / width, (y + h / 2) / height, w / width, h / height], dtype=np.float32)

    @staticmethod
    def _keypoints_to_array(
        values: list[Any], width: int, height: int, nkpt: int, ndim: int
    ) -> np.ndarray:
        keypoints = np.zeros((nkpt, ndim), dtype=np.float32)
        if not values:
            return keypoints
        if values and not isinstance(values[0], (list, tuple)):
            step = 3 if ndim == 3 else 2
            values = [values[i : i + step] for i in range(0, len(values), step)]
        for index, kp in enumerate(values[:nkpt]):
            if len(kp) < 2:
                continue
            keypoints[index, 0] = float(kp[0]) / width
            keypoints[index, 1] = float(kp[1]) / height
            if ndim == 3:
                keypoints[index, 2] = float(kp[2]) if len(kp) > 2 else 1.0
        return keypoints


def build_ppepose_dataset(
    cfg,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
):
    """Build the unified PPE+pose dataset with Ultralytics-style augmentation settings."""
    return PPEPoseUnifiedDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=getattr(cfg, "task", "ppepose"),
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )
