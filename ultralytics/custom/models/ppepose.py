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
from ultralytics.custom.metrics import DEFAULT_IOU_THRESHOLDS, build_metric_payload, build_batch_ground_truth, build_prediction_records, metric_field_specs


def load_pretrained_unified_weights(model: DetectionModel, weights: str | Path) -> tuple[int, int]:
    """Load a checkpoint into the unified model, remapping stock pose-head weights when available."""
    source_model = None
    if hasattr(weights, "state_dict"):
        source_model = weights
    else:
        ckpt = torch_load(weights, map_location="cpu")
        if isinstance(ckpt, dict):
            source_model = ckpt.get("ema") or ckpt.get("model")
    if source_model is None:
        model.load(weights)
        return 0, 0

    source_state = source_model.float().state_dict()
    target_state = model.state_dict()
    remapped_state = {}

    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            remapped_state[key] = value

        remap_key = key
        if ".cv2." in key:
            remap_key = key.replace(".cv2.", ".pose_cv2.")
        elif ".cv3." in key:
            remap_key = key.replace(".cv3.", ".pose_cv3.")
        elif ".cv4." in key:
            remap_key = key.replace(".cv4.", ".pose_cv4.")

        if remap_key != key and remap_key in target_state and target_state[remap_key].shape == value.shape:
            remapped_state[remap_key] = value

    model.load_state_dict(remapped_state, strict=False)
    return len(remapped_state), len(target_state)


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

    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = keys or []


class PPEPoseUnifiedValidator:
    """Validation loop that tracks task losses for model selection without forcing stock task metrics."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.args = args
        self.metrics = _MetricStub()
        self.thresholds = DEFAULT_IOU_THRESHOLDS
        self.val_conf = 0.001
        self.val_iou = 0.7
        self.max_det = getattr(args, "max_det", 300) if args is not None else 300

    def set_metric_keys(self, names: dict[int, str]) -> None:
        self.metrics = _MetricStub([spec.key for spec in metric_field_specs(names)])

    @staticmethod
    def _format_metric_row(name: str, values: dict[str, float]) -> str:
        return (
            f"{name:<14}"
            f"{values.get('precision@50', 0.0):>11.4f}"
            f"{values.get('recall@50', 0.0):>11.4f}"
            f"{values.get('AP50', values.get('mAP50', 0.0)):>11.4f}"
            f"{values.get('mAP50-95', 0.0):>13.4f}"
        )

    def __call__(self, trainer: BaseTrainer) -> dict[str, float]:
        model = unwrap_model(trainer.ema.ema if trainer.ema else trainer.model)
        was_training = model.training
        head = model.model[-1]
        was_head_training = head.training
        model.eval()
        mean_items = None
        batches = 0
        ppe_predictions = []
        person_predictions = []
        gt_ppe = []
        gt_person = []
        with torch.inference_mode():
            for batch in self.dataloader:
                batch = trainer.preprocess_batch(batch)
                image_ids = [Path(path).stem for path in batch["im_file"]]
                preds_eval = model(batch["img"])
                batch_results = postprocess_ppepose_predictions(
                    preds_eval,
                    tuple(model.kpt_shape),
                    conf=self.val_conf,
                    iou=self.val_iou,
                    max_det=self.max_det,
                )
                batch_ppe_preds, batch_person_preds = build_prediction_records(batch_results, image_ids)
                batch_gt_ppe, batch_gt_person = build_batch_ground_truth(batch, trainer.data["names"], tuple(model.kpt_shape))
                ppe_predictions.extend(batch_ppe_preds)
                person_predictions.extend(batch_person_preds)
                gt_ppe.extend(batch_gt_ppe)
                gt_person.extend(batch_gt_person)

                model.training = True
                head.training = True
                preds = model(batch["img"])
                model.training = False
                head.training = False
                _, loss_items = model.loss(batch, preds)
                mean_items = loss_items if mean_items is None else mean_items + loss_items
                batches += 1
        model.training = was_training
        head.training = was_head_training
        if was_training:
            model.train()
        else:
            model.eval()
        if batches == 0:
            return {"fitness": 0.0}
        mean_items = mean_items / batches
        metrics = trainer.label_loss_items(mean_items, prefix="val")
        metric_payload, scalar_metrics = build_metric_payload(
            ppe_predictions,
            person_predictions,
            gt_ppe,
            gt_person,
            trainer.data["names"],
            self.thresholds,
        )
        metrics.update(scalar_metrics)
        metrics["fitness"] = float(
            (
                metric_payload["ppe_detection"]["mAP50-95"]
                + metric_payload["person_bbox"]["mAP50-95"]
                + metric_payload["pose_keypoints"]["mAP50-95"]
            )
            / 3.0
        )
        helmet = metric_payload["ppe_detection"]["per_class"].get(0, {})
        vest = metric_payload["ppe_detection"]["per_class"].get(1, {})
        person_box = metric_payload["person_bbox"]
        pose = metric_payload["pose_keypoints"]
        person_row = {
            "precision@50": person_box["precision@50"],
            "recall@50": person_box["recall@50"],
            "AP50": person_box["mAP50"],
            "mAP50-95": person_box["mAP50-95"],
        }
        pose_row = {
            "precision@50": pose["precision@50"],
            "recall@50": pose["recall@50"],
            "AP50": pose["mAP50"],
            "mAP50-95": pose["mAP50-95"],
        }
        LOGGER.info(
            "\nval metrics\n"
            f"{'Target':<14}{'Precision':>11}{'Recall':>11}{'mAP50':>11}{'mAP50-95':>13}\n"
            f"{self._format_metric_row('Helmet', helmet)}\n"
            f"{self._format_metric_row('Vest', vest)}\n"
            f"{self._format_metric_row('Person(Box)', person_row)}\n"
            f"{self._format_metric_row('Pose-Kpt', pose_row)}"
        )
        return metrics


class PPEPoseUnifiedTrainer(DetectionTrainer):
    """Ultralytics-style trainer for the unified PPE and pose model."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        overrides = overrides or {}
        overrides.setdefault("task", "ppepose")
        custom_loss_weights = {
            "det_loss_weight": overrides.pop("det_loss_weight", 1.0),
            "pose_loss_weight": overrides.pop("pose_loss_weight", 1.0),
            "kpt_loss_weight": overrides.pop("kpt_loss_weight", 1.0),
        }
        super().__init__(cfg, overrides, _callbacks)
        for name, value in custom_loss_weights.items():
            setattr(self.args, name, value)

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
            loaded, total = load_pretrained_unified_weights(model, weights)
            if loaded and RANK in {-1, 0}:
                LOGGER.info(f"Transferred {loaded}/{total} parameters into unified model (including pose-head remap)")
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        self.model.pose_names = {0: self.data.get("person_name", "person")}
        self.model.kpt_names = self.data.get("kpt_names", {0: list(map(str, range(self.model.kpt_shape[0])))} )

    def get_validator(self):
        self.loss_names = ("det_branch_loss", "pose_branch_loss", "kpt_branch_loss", "total_loss")
        validator = PPEPoseUnifiedValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
        validator.set_metric_keys(self.data["names"])
        return validator

    def get_dataset(self) -> dict[str, Any]:
        data = super().get_dataset()
        required = ("kpt_shape", "annotations")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing required dataset keys for unified training: {missing}")
        return data

    def final_eval(self):
        """Skip the stock best-checkpoint revalidation path, which assumes a task-specific validator API."""
        return


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
    # Ultralytics NMS mutates the input tensor in-place when converting xywh to xyxy.
    det_decoded = preds["det"]["decoded"].clone()
    pose_decoded = preds["pose"]["decoded"].clone()
    det_preds = nms.non_max_suppression(det_decoded, conf, iou, nc=2, max_det=max_det)
    pose_preds = nms.non_max_suppression(pose_decoded, conf, iou, nc=1, max_det=max_det)
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
