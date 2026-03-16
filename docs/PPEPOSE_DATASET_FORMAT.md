# PPEPose Dataset Format

`ppepose_dataset.yaml` uses one JSON annotation file per image. This avoids forcing PPE classes into the stock YOLO pose format and keeps PPE object annotations separate from person pose annotations while still loading them together for one shared training pass.

## Directory Layout

```text
datasets/ppepose/
  images/
    train/
      sample_0001.jpg
    val/
    test/
  annotations/
    train/
      sample_0001.json
    val/
    test/
```

The image and annotation trees should mirror each other. An image at `images/train/line_a/frame_001.jpg` should have its annotation at `annotations/train/line_a/frame_001.json`.

## JSON Schema

```json
{
  "image": {
    "file_name": "sample_0001.jpg",
    "width": 1280,
    "height": 720
  },
  "detections": [
    {
      "category": "helmet",
      "bbox_xywh": [530.0, 96.0, 78.0, 64.0]
    },
    {
      "category": "vest",
      "bbox_xywh": [486.0, 178.0, 154.0, 192.0]
    }
  ],
  "persons": [
    {
      "person_id": "worker_01",
      "bbox_xywh": [452.0, 82.0, 218.0, 514.0],
      "keypoints": [
        [568.0, 118.0, 1.0],
        [555.0, 111.0, 1.0],
        [579.0, 112.0, 1.0],
        [542.0, 121.0, 1.0],
        [592.0, 122.0, 1.0],
        [520.0, 170.0, 1.0],
        [610.0, 172.0, 1.0],
        [496.0, 228.0, 1.0],
        [633.0, 230.0, 1.0],
        [478.0, 294.0, 1.0],
        [651.0, 292.0, 1.0],
        [532.0, 314.0, 1.0],
        [596.0, 316.0, 1.0],
        [520.0, 410.0, 1.0],
        [609.0, 413.0, 1.0],
        [512.0, 525.0, 1.0],
        [620.0, 527.0, 1.0]
      ],
      "compliance": {
        "helmet": true,
        "vest": true
      }
    }
  ]
}
```

## Required Fields

- `detections`: PPE objects only. Use `category` names from `names` in the dataset YAML or explicit integer `class_id`.
- `persons`: Human instances only. Each entry must include a person bounding box and a full `keypoints` array consistent with `kpt_shape`.
- `bbox_xywh`: Absolute pixel coordinates in `[x_min, y_min, width, height]` format.
- `keypoints`: Absolute pixel coordinates in `[x, y, visibility]` format. Set visibility to `0` for missing joints.

## Optional Fields

- `person_id`: Stable track or annotation id.
- `compliance`: Optional per-person PPE state used by the evaluation script for compliance metrics.
- Extra metadata fields may be added as long as the required fields above remain intact.

## Loader Behavior

- PPE and person annotations are loaded from the same JSON file.
- The loader internally merges them for shared geometric augmentation.
- After augmentation, the batch is split into:
  - PPE branch targets: `helmet`, `vest`
  - Pose branch targets: `person bbox + keypoints`
- This preserves spatial consistency without collapsing PPE classes into a mixed-class keypoint target.
