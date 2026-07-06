#!/usr/bin/env python3
"""Inspect the competition dataset layout and write a JSON summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from var2026_bts.scene_parsers import IMAGE_EXTENSIONS  # noqa: E402


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def inspect_split(split_root: Path) -> list[dict[str, object]]:
    scenes: list[dict[str, object]] = []
    for scene_dir in sorted(item for item in split_root.iterdir() if item.is_dir()):
        test_csv = scene_dir / "test" / "test_poses.csv"
        if not test_csv.exists():
            continue
        test_df = pd.read_csv(test_csv)
        train_images = scene_dir / "train" / "images"
        test_images = scene_dir / "test" / "images"
        sparse_dir = scene_dir / "train" / "sparse" / "0"
        scenes.append(
            {
                "scene": scene_dir.name,
                "train_images": count_images(train_images),
                "test_rows": int(len(test_df)),
                "test_images": count_images(test_images),
                "widths": sorted(int(v) for v in test_df["width"].unique()),
                "heights": sorted(int(v) for v in test_df["height"].unique()),
                "has_sparse": sparse_dir.exists(),
                "columns": list(test_df.columns),
            }
        )
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect VAR 2026 round 1 dataset layout.")
    parser.add_argument("--root", type=Path, default=Path("data/round1/phase1"))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report: dict[str, object] = {}
    for split in ("public_set", "private_set1"):
        split_root = args.root / split
        if split_root.exists():
            report[split] = inspect_split(split_root)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
