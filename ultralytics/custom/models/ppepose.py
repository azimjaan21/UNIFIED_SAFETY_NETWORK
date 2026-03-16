"""Unified shared-backbone PPE detection and human pose estimation components."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.data import build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, nms
from ultralytics.utils.loss import v8DetectionLoss, v8PoseLoss
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

from ultralytics.custom.data import build_ppepose_dataset


class _BranchHeadProxy:
    """Small attribute proxy that exposes one branch as a standalone Ultralytics head."""

    def __init__(self, head, branch: str):
        self.stride = head.stride
        self.reg_max = head.reg_max
        self.flow_model = None
        if branch == "det":
            self.nc = head.nc
        else:
            self.nc = head.pose_nc
            self.kpt_shape = head.kpt_shape


class _LossProxy:
    """Adapter that lets stock Ultralytics loss classes operate on a single branch."""

    def __init__(self, model: DetectionModel, branch: str):
        self._model = model
        self.model = [_BranchHeadProxy(model.model[-1], branch)]
        self.args = model.args

    def parameters(self):
        return self._model.parameters()


class PPEPoseUnifiedLoss:
    """Weighted multi-task loss for PPE detection and person pose estimation."""

    def __init__(self, model: DetectionModel):
        self.model = model
        self.det_loss = v8DetectionLoss(_LossProxy(model, "det"))
        self.pose_loss = v8PoseLoss(_LossProxy(model, "pose"))

    def __call__(
        self,
        preds: dict[str, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute weighted branch losses and a compact logging vector."""
        det_batch = {
            "batch_idx": batch["batch_idx"],
            "cls": batch["cls"],
            "bboxes": batch["bboxes"],
        }
        pose_batch = {
            "batch_idx": batch["person_batch_idx"],
            "cls": batch["person_cls"],
            "bboxes": batch["person_bboxes"],
            "keypoints": batch["keypoints"],
        }

        det_vec, _ = self.det_loss.loss(preds["det"], det_batch)
        pose_vec, _ = self.pose_loss.loss(preds["pose"], pose_batch)

        det_branch = det_vec.sum()
        pose_branch = pose_vec[0] + pose_vec[3] + pose_vec[4]
        kpt_branch = pose_vec[1] + pose_vec[2]
        if len(pose_vec) > 5:
            kpt_branch += pose_vec[5]

        det_w = getattr(self.model.args, "det_loss_weight", 1.0)
        pose_w = getattr(self.model.args, "pose_loss_weight", 1.0)
        kpt_w = getattr(self.model.args, "kpt_loss_weight", 1.0)
        total = det_w * det_branch + pose_w * pose_branch + kpt_w * kpt_branch

        items = torch.stack((det_branch.detach(), pose_branch.detach(), kpt_branch.detach(), total.detach()))
        return total, items


class PPEPoseUnifiedModel(DetectionModel):
    """Shared-backbone model with a PPE detection branch and a person pose branch."""

    def __init__(
        self,
        cfg: str | Path | dict[str, Any] = "ppepose_unified.yaml",
        ch: int = 3,
        nc: int | None = None,
        data_kpt_shape: tuple[int | None, int | None] = (None, None),
        verbose: bool = True,
    ):
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)
        cfg["scale"] = cfg.get("scale") or "n"
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = list(data_kpt_shape)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.kpt_shape = cfg["kpt_shape"]
        self.pose_names = {0: cfg.get("person_name", "person")}

    def init_criterion(self):
        """Initialize the unified criterion."""
        return PPEPoseUnifiedLoss(self)


class _MetricStub:
    """Expose the minimal `keys` interface expected by the trainer."""

    keys: tuple[str, ...] = ()


class PPEPoseUnifiedValidator:
    """Validation loop that tracks task losses for model selection without forcing stock task metrics."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.args = args
        self.metrics = _MetricStub()

    def __call__(self, trainer: BaseTrainer) -> dict[str, float]:
        model = unwrap_model(trainer.ema.ema if trainer.ema else trainer.model)
        was_training = model.training
        model.eval()
        mean_items = None
        batches = 0
        with torch.inference_mode():
            for batch in self.dataloader:
                batch = trainer.preprocess_batch(batch)
                _, loss_items = model.loss(batch)
                mean_items = loss_items if mean_items is None else mean_items + loss_items
                batches += 1
        if was_training:
            model.train()
        if batches == 0:
            return {"fitness": 0.0}
        mean_items = mean_items / batches
        metrics = trainer.label_loss_items(mean_items, prefix="val")
        metrics["fitness"] = -float(mean_items[-1])
        return metrics


class PPEPoseUnifiedTrainer(DetectionTrainer):
    """Ultralytics-style trainer for the unified PPE and pose model."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        overrides = overrides or {}
        overrides.setdefault("task", "ppepose")
        overrides.setdefault("det_loss_weight", 1.0)
        overrides.setdefault("pose_loss_weight", 1.0)
        overrides.setdefault("kpt_loss_weight", 1.0)
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_ppepose_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> PPEPoseUnifiedModel:
        model = PPEPoseUnifiedModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        self.model.pose_names = {0: self.data.get("person_name", "person")}
        self.model.kpt_names = self.data.get("kpt_names", {0: list(map(str, range(self.model.kpt_shape[0])))} )

    def get_validator(self):
        self.loss_names = ("det_branch_loss", "pose_branch_loss", "kpt_branch_loss", "total_loss")
        return PPEPoseUnifiedValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def get_dataset(self) -> dict[str, Any]:
        data = super().get_dataset()
        required = ("kpt_shape", "annotations")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing required dataset keys for unified training: {missing}")
        return data


def load_ppepose_model(
    weights: str | Path,
    model_cfg: str | Path = "ppepose_unified.yaml",
    device: str | torch.device = "cpu",
) -> PPEPoseUnifiedModel:
    """Load either a training checkpoint or raw weights into the unified model."""
    weights = str(weights)
    device = torch.device(device)
    ckpt = torch_load(weights, map_location=device)
    if isinstance(ckpt, dict) and ckpt.get("ema") is not None:
        model = ckpt["ema"].float()
    elif isinstance(ckpt, dict) and ckpt.get("model") is not None:
        model = ckpt["model"].float()
    else:
        model = PPEPoseUnifiedModel(model_cfg, verbose=False)
        model.load(weights)
    return model.to(device).eval()


def postprocess_ppepose_predictions(
    preds: dict[str, dict[str, torch.Tensor]],
    kpt_shape: tuple[int, int],
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
) -> list[dict[str, torch.Tensor]]:
    """Apply task-specific NMS and unpack unified outputs into structured tensors."""
    det_preds = nms.non_max_suppression(preds["det"]["decoded"], conf, iou, nc=2, max_det=max_det)
    pose_preds = nms.non_max_suppression(preds["pose"]["decoded"], conf, iou, nc=1, max_det=max_det)
    results: list[dict[str, torch.Tensor]] = []
    for det, pose in zip(det_preds, pose_preds):
        keypoints = pose[:, 6:].view(-1, *kpt_shape) if len(pose) else pose.new_zeros((0, *kpt_shape))
        results.append(
            {
                "ppe_boxes": det[:, :4],
                "ppe_scores": det[:, 4],
                "ppe_classes": det[:, 5].long(),
                "person_boxes": pose[:, :4],
                "person_scores": pose[:, 4],
                "person_classes": pose[:, 5].long(),
                "person_keypoints": keypoints,
            }
        )
    return results
