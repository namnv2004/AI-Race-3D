#!/usr/bin/env python3
"""Recompress rendered JPEG submissions to a target size budget."""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

from PIL import Image

JPEG_EXTENSIONS = {".jpg", ".jpeg"}


def jpeg_size(path: Path, quality: int) -> int:
    with Image.open(path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, subsampling=0, optimize=True)
        return buffer.tell()


def save_jpeg(source: Path, dest: Path, quality: int) -> int:
    with Image.open(source) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, format="JPEG", quality=quality, subsampling=0, optimize=True)
    return dest.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompress JPEG submission images under a target MiB budget."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-mib", type=float, required=True)
    parser.add_argument("--base-quality", type=int, default=97)
    parser.add_argument("--max-quality", type=int, default=98)
    parser.add_argument("--min-quality", type=int, default=95)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        raise SystemExit(f"Source directory not found: {args.source_dir}")
    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory exists: {args.output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)

    files = sorted(
        path
        for path in args.source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in JPEG_EXTENSIONS
    )
    if not files:
        raise SystemExit(f"No JPEG files found under {args.source_dir}")

    target_bytes = int(args.target_mib * 1024 * 1024)
    qualities = list(range(args.base_quality, args.min_quality - 1, -1))
    size_by_quality: dict[int, dict[Path, int]] = {}
    base_quality = None
    base_total = None
    for quality in qualities:
        sizes = {path: jpeg_size(path, quality) for path in files}
        total = sum(sizes.values())
        size_by_quality[quality] = sizes
        if total <= target_bytes:
            base_quality = quality
            base_total = total
            break
    if base_quality is None or base_total is None:
        lowest = qualities[-1]
        lowest_mib = sum(size_by_quality[lowest].values()) / 1024 / 1024
        raise SystemExit(f"Even q{lowest} is too large: {lowest_mib:.3f} MiB > {args.target_mib:.3f} MiB")

    selected_quality: dict[Path, int] = {path: base_quality for path in files}
    selected_total = base_total
    max_sizes: dict[Path, int] | None = None
    if args.max_quality > base_quality:
        max_sizes = {path: jpeg_size(path, args.max_quality) for path in files}
        upgrades = []
        for path in files:
            extra = max_sizes[path] - size_by_quality[base_quality][path]
            if extra > 0:
                upgrades.append((extra, path))
        for extra, path in sorted(upgrades):
            if selected_total + extra > target_bytes:
                continue
            selected_quality[path] = args.max_quality
            selected_total += extra

    written_total = 0
    for path in files:
        relative = path.relative_to(args.source_dir)
        written_total += save_jpeg(path, args.output_dir / relative, selected_quality[path])

    report = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "target_mib": args.target_mib,
        "file_count": len(files),
        "base_quality": base_quality,
        "max_quality": args.max_quality,
        "base_total_mib": base_total / 1024 / 1024,
        "selected_total_mib_estimate": selected_total / 1024 / 1024,
        "written_total_mib": written_total / 1024 / 1024,
        "quality_counts": {
            str(quality): sum(1 for value in selected_quality.values() if value == quality)
            for quality in sorted(set(selected_quality.values()))
        },
        "max_quality_files": [
            str(path.relative_to(args.source_dir))
            for path, quality in selected_quality.items()
            if quality == args.max_quality
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
