"""Train a shared-backbone unified PPE detection and person pose model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics.custom.models import PPEPoseUnifiedTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the unified PPE+pose Ultralytics model.")
    parser.add_argument("--model", type=str, default="ppepose_unified.yaml", help="Unified model YAML path.")
    parser.add_argument("--data", type=str, default="ppepose_dataset.yaml", help="Unified dataset YAML path.")
    parser.add_argument("--pretrained", type=str, default="", help="Optional pretrained checkpoint for initialization.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--det-loss-weight", type=float, default=1.0)
    parser.add_argument("--pose-loss-weight", type=float, default=1.0)
    parser.add_argument("--kpt-loss-weight", type=float, default=1.0)
    parser.add_argument("--mosaic", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--cutmix", type=float, default=None)
    parser.add_argument("--degrees", type=float, default=None)
    parser.add_argument("--translate", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--shear", type=float, default=None)
    parser.add_argument("--perspective", type=float, default=None)
    parser.add_argument("--flipud", type=float, default=None)
    parser.add_argument("--fliplr", type=float, default=None)
    parser.add_argument("--hsv-h", type=float, default=None, dest="hsv_h")
    parser.add_argument("--hsv-s", type=float, default=None, dest="hsv_s")
    parser.add_argument("--hsv-v", type=float, default=None, dest="hsv_v")
    parser.add_argument("--project", type=str, default="runs/ppepose")
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "det_loss_weight": args.det_loss_weight,
        "pose_loss_weight": args.pose_loss_weight,
        "kpt_loss_weight": args.kpt_loss_weight,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "plots": args.plots,
    }
    for aug_key in (
        "mosaic",
        "mixup",
        "cutmix",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
    ):
        value = getattr(args, aug_key)
        if value is not None:
            overrides[aug_key] = value
    if args.pretrained:
        overrides["pretrained"] = str(Path(args.pretrained))

    trainer = PPEPoseUnifiedTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
