#!/usr/bin/env python3
"""Create a deterministic submission ZIP from a rendered submission directory."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def make_zip(submission_dir: Path, zip_path: Path, overwrite: bool) -> None:
    if not submission_dir.is_dir():
        raise SystemExit(f"Submission directory not found: {submission_dir}")
    if zip_path.exists() and not overwrite:
        raise SystemExit(f"Zip already exists: {zip_path}. Pass --overwrite to replace it.")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(zip_path, mode="w", compression=compression, compresslevel=6) as archive:
        for path in sorted(item for item in submission_dir.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(submission_dir).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a submission ZIP from a rendered output directory.")
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    make_zip(args.submission_dir, args.zip, args.overwrite)
    print(f"wrote {args.zip}")


if __name__ == "__main__":
    main()
