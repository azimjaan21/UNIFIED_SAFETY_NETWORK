# Unified Safety Network

Unified Safety Network is a research-oriented industrial PPE monitoring project built on top of the official Ultralytics codebase.

It implements a unified multi-task model with:

- PPE object detection for `helmet` and `vest`
- human pose estimation for `person bbox + keypoints`
- one shared backbone and neck
- two task-specific heads
- a pose-guided semantic filtering module for reducing implausible PPE false alarms

## Core Idea

Stock YOLO pose format is not a clean fit for this problem because PPE objects do not share the same keypoint target as humans. This project therefore uses:

- a PPE detection head for PPE classes
- a pose head for person bounding boxes and keypoints
- a custom unified dataset loader that reads one JSON annotation per image

The main paper-facing contribution is:

- **Pose-Guided PPE Consistency Filtering (PG-PCF)**

PG-PCF uses predicted human keypoints to construct adaptive head and torso regions, associate PPE detections to people, and suppress or down-weight anatomically implausible PPE boxes.

## Implemented Components

- unified model config: `ppepose_unified.yaml`
- unified dataset config: `ppepose_dataset.yaml`
- unified trainer entry point: `train_ppepose_unified.py`
- unified inference entry point: `infer_ppepose_unified.py`
- unified evaluation entry point: `evaluate_ppepose_unified.py`
- PG-PCF post-processing module:
  - `ultralytics/custom/postprocess/keypoint_guided_ppe_filter.py`
- dataset format specification:
  - `docs/PPEPOSE_DATASET_FORMAT.md`
- method overview:
  - `docs/PPEPOSE_METHOD_OVERVIEW.md`
- example annotation:
  - `examples/sample_annotation.json`

## Project Structure

```text
Unified_Safety_Network/
├─ ultralytics/
│  ├─ custom/
│  │  ├─ data/
│  │  ├─ models/
│  │  ├─ postprocess/
│  │  └─ runtime.py
│  ├─ nn/
│  │  ├─ modules/
│  │  └─ tasks.py
├─ docs/
├─ examples/
├─ ppepose_unified.yaml
├─ ppepose_dataset.yaml
├─ train_ppepose_unified.py
├─ infer_ppepose_unified.py
└─ evaluate_ppepose_unified.py
```

## Annotation Format

This project does **not** use standard YOLO `.txt` labels for the unified task.

It uses:

- one `.json` file per image
- PPE detections in a `detections` section
- person boxes and keypoints in a `persons` section

See:

- `docs/PPEPOSE_DATASET_FORMAT.md`
- `examples/sample_annotation.json`

## Environment Setup

Create and activate a local virtual environment:

```powershell
C:\Users\dalab\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.\.venv\Scripts\activate
.\.venv\Scripts\python.exe -m pip install -e .
```

## Training

Example training command:

```powershell
.\.venv\Scripts\python.exe train_ppepose_unified.py --model ppepose_unified.yaml --data ppepose_dataset.yaml --epochs 100 --imgsz 640 --batch 16 --device 0
```

Loss weights are exposed through:

- `--det-loss-weight`
- `--pose-loss-weight`
- `--kpt-loss-weight`

## Inference

Run unified inference with optional PG-PCF:

```powershell
.\.venv\Scripts\python.exe infer_ppepose_unified.py --weights runs/ppepose/train/weights/best.pt --source path\to\images_or_video --data ppepose_dataset.yaml --use_kp_guided_filter true
```

Outputs include:

- raw predictions
- filtered predictions
- visualizations before filtering
- visualizations after filtering
- per-image prediction JSON

## Evaluation

Example evaluation command:

```powershell
.\.venv\Scripts\python.exe evaluate_ppepose_unified.py --weights runs/ppepose/train/weights/best.pt --data ppepose_dataset.yaml --split val
```

The evaluation script reports:

- PPE detection metrics
- person bbox metrics
- pose keypoint metrics
- before/after PG-PCF filtering impact
- optional per-person compliance accuracy if compliance labels exist

## Current Validation Status

The implementation has been structurally validated with:

- successful model instantiation from `ppepose_unified.yaml`
- successful unified dataset parsing from JSON labels
- successful dual-head loss execution
- successful synthetic smoke test through loader + model + loss

This is a serious first version, but not a claim of final experimental performance. Real dataset training and ablation studies are still required.

## Notes

- Base framework: official Ultralytics repository
- This project intentionally keeps framework modifications minimal and concentrated around the unified task
- PG-PCF is implemented as a clean post-prediction module so it can be described independently in a paper
