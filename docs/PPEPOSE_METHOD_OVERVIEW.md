# PPEPose Method Overview

## Shared-Backbone Dual-Head Design

The model uses one lightweight YOLO-style backbone and neck, followed by two task-specific heads:

- A PPE detection head for `helmet` and `vest`
- A pose head for `person bbox + keypoints`

This design keeps inference efficient while allowing end-to-end optimization across both tasks in a single training run.

## Why Not Use Stock Mixed-Class Pose Format

The stock YOLO pose format assumes that every detection class shares the same keypoint target definition. That is not appropriate here because:

- PPE objects do not have meaningful human-body keypoints
- forcing `helmet`, `vest`, and `person` into one pose label space creates semantically invalid supervision
- it entangles PPE classification with keypoint targets that only belong to humans

The unified model therefore keeps PPE detection and person pose prediction in separate heads while still sharing features.

## Pose-Guided PPE Consistency Filtering

The main contribution is **Pose-Guided PPE Consistency Filtering (PG-PCF)**, a post-prediction semantic validation stage.

PG-PCF derives adaptive anatomical regions from predicted human keypoints:

- a head-centric region for helmet validation
- a torso-centric region for vest validation

Each PPE detection is scored against each detected person using:

- center-in-region consistency
- overlap with the expected anatomical region
- normalized distance to an anatomical anchor
- keypoint visibility confidence
- scale compatibility

The best PPE-to-person association is retained, and detections that are not anatomically plausible are either:

- hard-suppressed, or
- confidence-decayed and re-ranked

## Why PG-PCF Reduces False Alarms

Conventional PPE detectors can fire on isolated object-like patterns without checking whether the detection is plausible relative to a person. PG-PCF reduces these false alarms by enforcing a weak anatomical prior at inference time. A helmet-like box far from a predicted head region or a vest-like box outside the torso region receives a low association score and can be suppressed or down-weighted.

## Paper Framing

This framework is suitable for a paper contribution with the following narrative:

- a unified shared-backbone PPE+pose architecture improves deployment efficiency
- a dual-head design preserves valid supervision structure
- PG-PCF introduces an anatomically constrained semantic filtering stage
- the filtering stage improves precision by reducing implausible PPE detections with limited recall loss
- ablation switches support analysis of region construction, score fusion, and suppression mode
