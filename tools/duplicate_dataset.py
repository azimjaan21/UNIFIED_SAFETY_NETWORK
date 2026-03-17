"""Create an overfit-style duplicated copy of the unified PPE+pose dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/ppepose"))
    parser.add_argument("--output", type=Path, default=Path("datasets/ppepose_overfit"))
    parser.add_argument("--train-copies", type=int, default=20, help="How many copies to make of each train image")
    parser.add_argument("--val-copies", type=int, default=1, help="How many copies to make of each val image")
    parser.add_argument("--test-copies", type=int, default=1, help="How many copies to make of each test image")
    return parser.parse_args()


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def duplicate_split(src_root: Path, dst_root: Path, split: str, copies: int) -> None:
    image_dir = src_root / "images" / split
    ann_dir = src_root / "annotations" / split
    out_image_dir = dst_root / "images" / split
    out_ann_dir = dst_root / "annotations" / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        return

    image_files = sorted(p for p in image_dir.iterdir() if p.is_file())
    for image_file in image_files:
        ann_file = ann_dir / f"{image_file.stem}.json"
        if not ann_file.exists():
            continue
        annotation = json.loads(ann_file.read_text(encoding="utf-8"))
        for index in range(copies):
            suffix = f"_dup{index:03d}"
            out_image = out_image_dir / f"{image_file.stem}{suffix}{image_file.suffix}"
            out_ann = out_ann_dir / f"{image_file.stem}{suffix}.json"
            shutil.copy2(image_file, out_image)
            annotation["image"]["file_name"] = out_image.name
            out_ann.write_text(json.dumps(annotation, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    clear_dir(args.output)
    duplicate_split(args.input, args.output, "train", max(args.train_copies, 1))
    duplicate_split(args.input, args.output, "val", max(args.val_copies, 1))
    duplicate_split(args.input, args.output, "test", max(args.test_copies, 1))
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "train_copies": args.train_copies,
                "val_copies": args.val_copies,
                "test_copies": args.test_copies,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
