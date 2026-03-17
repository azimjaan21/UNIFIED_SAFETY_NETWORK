# 🦺 Unified Safety Network

<p align="center">
  <strong>Shared-Backbone Dual-Head PPE Detection and Human Pose Estimation</strong>
</p>

<p align="center">
  Industrial PPE monitoring with unified detection, pose reasoning, and pose-guided semantic filtering.
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white">
  <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="ultralytics" src="https://img.shields.io/badge/Ultralytics-Based-111111?logo=yolo&logoColor=white">
  <img alt="task" src="https://img.shields.io/badge/Task-Multi--Task-0A7B83">
  <img alt="license" src="https://img.shields.io/badge/License-Research%20Code-6A5ACD">
</p>

## 🔍 Overview

Unified Safety Network is a research-oriented PPE monitoring framework built on top of the official Ultralytics codebase. It targets a practical industrial setting where PPE detection alone is not sufficient: detections should also be anatomically plausible with respect to the worker wearing them.

This repository implements a unified end-to-end model with:

- PPE object detection for `helmet` and `vest`
- human pose estimation for `person bbox + keypoints`
- one shared YOLO-style backbone and neck
- two task-specific heads
- a post-prediction semantic filtering module based on pose geometry

The main method contribution is:

- **PG-PCF: Pose-Guided PPE Consistency Filtering**

PG-PCF uses predicted keypoints to construct adaptive head and torso regions, associate PPE detections to the most plausible person, and suppress or re-score anatomically inconsistent PPE predictions.

## ✨ Why This Repo Exists

Stock YOLO pose format is not a clean fit for this problem. PPE objects and human keypoints are different supervision targets, and forcing them into a single stock pose-class format is unnecessarily restrictive.

This repository instead uses:

- a **PPE detection head** for `helmet` and `vest`
- a **pose head** for `person` box + keypoints
- a **custom unified JSON dataset loader**
- a **shared-backbone multi-task training pipeline**

This keeps the framework practical for experiments, fair for baseline comparison, and clean enough for paper submission.

## 🧠 Method Summary

### Model

- Shared YOLO-style backbone and neck
- PPE branch: `helmet`, `vest`
- Pose branch: `person` bbox + `17 x 3` keypoints
- End-to-end training in one run with weighted multi-task losses

### Post-Processing

- Adaptive head region for helmet validation
- Adaptive torso region for vest validation
- PPE-to-person association scoring
- Hard suppression or score decay
- Per-person PPE compliance summary

See:

- [Method Overview](docs/PPEPOSE_METHOD_OVERVIEW.md)
- [Dataset Format](docs/PPEPOSE_DATASET_FORMAT.md)

## 📁 Repository Layout

```text
UNIFIED_SAFETY_NETWORK/
├─ ultralytics/
│  ├─ custom/
│  │  ├─ data/
│  │  ├─ metrics/
│  │  ├─ models/
│  │  ├─ postprocess/
│  │  └─ runtime.py
│  └─ nn/
├─ datasets/
├─ docs/
├─ examples/
├─ tools/
│  ├─ convert_cvat.py
│  └─ duplicate_dataset.py
├─ train.py
├─ infer.py
├─ val.py
├─ predict.py
├─ vis_gt.py
├─ ppepose_unified.yaml
├─ ppepose_dataset.yaml
└─ ppepose_dataset_overfit.yaml
```

## 🏷️ Annotation Format

This project does **not** use stock YOLO `.txt` labels for the unified task.

It uses one `.json` annotation per image with:

- `detections`: PPE boxes
- `persons`: person bbox + keypoints

Example:

- [Sample Annotation](examples/sample_annotation.json)

Full specification:

- [PPEPOSE_DATASET_FORMAT.md](docs/PPEPOSE_DATASET_FORMAT.md)

## ⚙️ Environment Setup

```powershell
C:\Users\dalab\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.\.venv\Scripts\activate
.\.venv\Scripts\python.exe -m pip install -e .
```

## 🚀 Main Scripts

### `train.py`

Unified multi-task training entry point.

```powershell
.\.venv\Scripts\python.exe train.py --model ppepose_unified.yaml --data ppepose_dataset.yaml --epochs 100 --imgsz 640 --batch 16 --device 0
```

Important loss controls:

- `--det-loss-weight`
- `--pose-loss-weight`
- `--kpt-loss-weight`

Important augmentation controls:

- `--mosaic`
- `--mixup`
- `--cutmix`
- `--degrees`
- `--translate`
- `--scale`
- `--fliplr`
- `--flipud`

### `infer.py`

Unified inference with optional PG-PCF filtering.

```powershell
.\.venv\Scripts\python.exe infer.py --weights runs/ppepose/train/weights/best.pt --source path\to\images_or_video --data ppepose_dataset.yaml --use_kp_guided_filter true
```

### `val.py`

Standalone validation/evaluation for saved checkpoints.

```powershell
.\.venv\Scripts\python.exe val.py --weights runs/ppepose/train/weights/best.pt --data ppepose_dataset.yaml --split val
```

### `predict.py`

Clean best-checkpoint visualization script for qualitative inspection.

### `vis_gt.py`

Ground-truth visualization script for checking annotation correctness.

## 📊 Metrics

The training pipeline logs validation metrics every epoch in a research-comparable format.

Reported metrics include:

- `Helmet`: Precision, Recall, mAP50, mAP50-95
- `Vest`: Precision, Recall, mAP50, mAP50-95
- `Person(Box)`: Precision, Recall, mAP50, mAP50-95
- `Pose-Kpt`: Precision, Recall, mAP50, mAP50-95

Validation output is printed in column format and saved to `results.csv`.

## 🧪 Utilities

### Convert CVAT Export

```powershell
.\.venv\Scripts\python.exe tools\convert_cvat.py --input PPE_POSE --output datasets/ppepose --train-ratio 0.8 --val-ratio 0.2
```

### Duplicate Tiny Dataset for Overfit Debugging

```powershell
.\.venv\Scripts\python.exe tools\duplicate_dataset.py --input datasets/ppepose --output datasets/ppepose_overfit --train-copies 30 --val-copies 1 --test-copies 1
```

## ✅ Current Status

This repository is already functional as a research codebase:

- unified model builds from YAML
- unified JSON dataset loads correctly
- dual-head training runs end-to-end
- validation metrics are logged every epoch
- GT visualization and prediction visualization are available
- PG-PCF is implemented as a modular post-processing component

Current practical note:

- the framework is structurally solid
- final performance still depends on more labeled data and full experiments

## 📌 Recommended Workflow

1. Label images in CVAT with PPE boxes and person keypoints.
2. Convert exports into the unified JSON format.
3. Train with `train.py`.
4. Inspect GT with `vis_gt.py`.
5. Inspect model predictions with `predict.py` or `infer.py`.
6. Evaluate checkpoints with `val.py`.
7. Run ablations on PG-PCF settings for the paper.

## 📚 Paper-Facing Notes

This repository is intended to support a future paper submission around:

- unified shared-backbone PPE + pose modeling
- pose-guided PPE consistency reasoning
- false-alarm reduction through anatomical validation

Recommended references inside this repo:

- [Method Overview](docs/PPEPOSE_METHOD_OVERVIEW.md)
- [Dataset Format](docs/PPEPOSE_DATASET_FORMAT.md)

## 📝 Citation

Citation block will be updated when the paper and preprint are finalized.

```bibtex
@misc{unified_safety_network,
  title        = {Unified Safety Network},
  author       = {Authors TBD},
  year         = {2026},
  note         = {Research code repository}
}
```

## 🙏 Acknowledgment

- Official [Ultralytics](https://github.com/ultralytics/ultralytics) repository for the base framework
- CVAT for mixed PPE + pose annotation workflow

