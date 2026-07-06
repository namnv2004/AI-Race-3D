#!/usr/bin/env python3
"""Generate a nearest-train-view baseline submission."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from var2026_bts.scene_parsers import IMAGE_EXTENSIONS, qvec_to_rotmat  # noqa: E402


@dataclass(frozen=True)
class TrainView:
    name: str
    path: Path
    center: np.ndarray
    forward: np.ndarray
    ordinal: int | None


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def pose_center_forward(qvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = qvec_to_rotmat(qvec)
    center = -rotation.T @ tvec.astype(np.float64)
    forward = normalize(rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return center, forward


def image_ordinal(name: str) -> int | None:
    stem = Path(name).stem
    matches = re.findall(r"\d+", stem)
    if not matches:
        return None
    return int(matches[-1])


def collect_train_images(train_images_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(train_images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images[path.name] = path
    return images


def build_train_views(scene_dir: Path) -> list[TrainView]:
    train_images = collect_train_images(scene_dir / "train" / "images")
    sparse_dir = scene_dir / "train" / "sparse" / "0"
    if not train_images:
        raise RuntimeError(f"No train images found in {scene_dir / 'train' / 'images'}")
    if not sparse_dir.exists():
        raise RuntimeError(f"Missing COLMAP sparse directory: {sparse_dir}")

    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    views: list[TrainView] = []
    for image in reconstruction.images.values():
        path = train_images.get(image.name)
        if path is None:
            continue
        cam_from_world = image.cam_from_world()
        center = np.asarray(image.projection_center(), dtype=np.float64)
        rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
        forward = normalize(rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
        views.append(
            TrainView(
                name=image.name,
                path=path,
                center=center,
                forward=forward,
                ordinal=image_ordinal(image.name),
            )
        )
    if not views:
        raise RuntimeError(f"No COLMAP image poses matched train images in {scene_dir}")
    return views


def choose_nearest(target: pd.Series, train_views: list[TrainView]) -> TrainView:
    qvec = target[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)
    tvec = target[["tx", "ty", "tz"]].to_numpy(dtype=np.float64)
    target_center, target_forward = pose_center_forward(qvec, tvec)
    target_ordinal = image_ordinal(str(target["image_name"]))

    best_view = train_views[0]
    best_score = math.inf
    for view in train_views:
        center_distance = float(np.linalg.norm(view.center - target_center))
        cos_angle = float(np.clip(np.dot(view.forward, target_forward), -1.0, 1.0))
        angle = math.acos(cos_angle)
        ordinal_penalty = 0.0
        if target_ordinal is not None and view.ordinal is not None:
            ordinal_penalty = min(abs(view.ordinal - target_ordinal), 9999) * 1e-4
        score = center_distance + 0.25 * angle + ordinal_penalty
        if score < best_score:
            best_score = score
            best_view = view
    return best_view


def save_resized_rgb(src: Path, dst: Path, width: int, height: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        suffix = dst.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            image.save(dst, format="JPEG", quality=100, subsampling=0, optimize=True)
        else:
            image.save(dst, format="PNG", optimize=True)


def generate_scene(scene_dir: Path, output_root: Path, overwrite: bool) -> dict[str, int | str]:
    test_csv = scene_dir / "test" / "test_poses.csv"
    test_df = pd.read_csv(test_csv)
    scene_output = output_root / scene_dir.name
    if scene_output.exists() and overwrite:
        shutil.rmtree(scene_output)
    scene_output.mkdir(parents=True, exist_ok=True)

    train_views = build_train_views(scene_dir)
    reused = 0
    generated = 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=scene_dir.name):
        image_name = str(row["image_name"])
        width = int(row["width"])
        height = int(row["height"])
        dst = scene_output / image_name
        if dst.exists() and not overwrite:
            reused += 1
            continue
        nearest = choose_nearest(row, train_views)
        save_resized_rgb(nearest.path, dst, width, height)
        generated += 1
    return {"scene": scene_dir.name, "test_rows": int(len(test_df)), "generated": generated, "reused": reused}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a valid nearest-pose baseline submission.")
    parser.add_argument("--split-root", type=Path, default=Path("data/round1/phase1/private_set1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/submission_round1_nearest"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scenes = [item for item in sorted(args.split_root.iterdir()) if (item / "test" / "test_poses.csv").exists()]
    if not scenes:
        raise SystemExit(f"No scenes with test/test_poses.csv found in {args.split_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scene_dir in scenes:
        summaries.append(generate_scene(scene_dir, args.output_dir, args.overwrite))

    total = sum(int(item["test_rows"]) for item in summaries)
    print(f"Generated nearest-pose submission images for {len(summaries)} scenes, {total} target poses.")
    for item in summaries:
        print(item)


if __name__ == "__main__":
    main()
