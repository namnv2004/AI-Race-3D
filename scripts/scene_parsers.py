from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TEST_POSE_COLUMNS = [
    "image_name",
    "qw",
    "qx",
    "qy",
    "qz",
    "tx",
    "ty",
    "tz",
    "fx",
    "fy",
    "cx",
    "cy",
    "width",
    "height",
]


@dataclass(frozen=True)
class TrainCamera:
    name: str
    image_path: Path
    w2c: np.ndarray
    K: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class TestCamera:
    image_name: str
    w2c: np.ndarray
    K: np.ndarray
    width: int
    height: int
    render_width: int
    render_height: int


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm(qvec))
    if norm == 0.0:
        raise ValueError("Quaternion has zero norm")
    qw, qx, qy, qz = [float(v) for v in qvec / norm]
    return np.array(
        [
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw],
            [2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw],
            [2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ],
        dtype=np.float64,
    )


def w2c_from_qt(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Build a COLMAP/OpenCV world-to-camera matrix from qvec/tvec."""
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = qvec_to_rotmat(qvec)
    w2c[:3, 3] = np.asarray(tvec, dtype=np.float64)
    return w2c


def rigid3d_to_matrix(cam_from_world: object) -> np.ndarray:
    matrix_fn = getattr(cam_from_world, "matrix", None)
    if callable(matrix_fn):
        matrix = np.asarray(matrix_fn(), dtype=np.float64)
        if matrix.shape == (4, 4):
            return matrix
        if matrix.shape == (3, 4):
            out = np.eye(4, dtype=np.float64)
            out[:3, :] = matrix
            return out
    rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
    translation = np.asarray(cam_from_world.translation, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def camera_intrinsic(camera: pycolmap.Camera) -> np.ndarray:
    matrix_fn = getattr(camera, "calibration_matrix", None)
    if callable(matrix_fn):
        return np.asarray(matrix_fn(), dtype=np.float64)
    params = np.asarray(camera.params, dtype=np.float64)
    model = str(camera.model).split(".")[-1]
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
    else:
        raise RuntimeError(f"Unsupported camera model: {camera.model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def collect_image_paths(images_dir: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def parse_colmap_train_scene(scene_dir: Path, factor: int = 1) -> tuple[list[TrainCamera], np.ndarray, np.ndarray]:
    if factor < 1:
        raise ValueError("factor must be >= 1")

    train_dir = scene_dir / "train"
    sparse_dir = train_dir / "sparse" / "0"
    images_dir = train_dir / "images"
    if not sparse_dir.exists():
        raise FileNotFoundError(f"Missing COLMAP sparse directory: {sparse_dir}")
    image_paths = collect_image_paths(images_dir)
    if not image_paths:
        raise RuntimeError(f"No train images found in {images_dir}")

    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    views: list[TrainCamera] = []
    for image in reconstruction.images.values():
        image_path = image_paths.get(image.name)
        if image_path is None:
            continue
        camera = reconstruction.cameras[image.camera_id]
        K = camera_intrinsic(camera).astype(np.float64)
        width = int(camera.width)
        height = int(camera.height)
        if factor > 1:
            K[:2, :] /= factor
            width //= factor
            height //= factor
        views.append(
            TrainCamera(
                name=image.name,
                image_path=image_path,
                w2c=rigid3d_to_matrix(image.cam_from_world()),
                K=K,
                width=width,
                height=height,
            )
        )
    views.sort(key=lambda view: view.name)
    if not views:
        raise RuntimeError(f"No COLMAP images matched train images in {scene_dir}")

    points = []
    colors = []
    for point in reconstruction.points3D.values():
        points.append(np.asarray(point.xyz, dtype=np.float32))
        colors.append(np.asarray(point.color, dtype=np.float32) / 255.0)
    if not points:
        raise RuntimeError(f"No sparse points loaded from {sparse_dir}")
    return views, np.stack(points).astype(np.float32), np.stack(colors).astype(np.float32)


def test_camera_from_row(row: pd.Series, render_scale: float = 1.0) -> TestCamera:
    if render_scale <= 0.0:
        raise ValueError("render_scale must be > 0")
    qvec = row[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)
    tvec = row[["tx", "ty", "tz"]].to_numpy(dtype=np.float64)
    width = int(row["width"])
    height = int(row["height"])
    render_width = max(2, int(round(width * render_scale)))
    render_height = max(2, int(round(height * render_scale)))
    K = np.array(
        [[float(row["fx"]), 0.0, float(row["cx"])], [0.0, float(row["fy"]), float(row["cy"])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if render_scale != 1.0:
        K = K.copy()
        K[0, :] *= render_width / width
        K[1, :] *= render_height / height
    return TestCamera(
        image_name=str(row["image_name"]),
        w2c=w2c_from_qt(qvec, tvec),
        K=K,
        width=width,
        height=height,
        render_width=render_width,
        render_height=render_height,
    )


def parse_test_poses_csv(test_csv: Path, render_scale: float = 1.0) -> list[TestCamera]:
    df = pd.read_csv(test_csv)
    missing = [column for column in TEST_POSE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{test_csv} is missing columns: {missing}")
    cameras = [test_camera_from_row(row, render_scale=render_scale) for _, row in df.iterrows()]
    if not cameras:
        raise RuntimeError(f"No test poses found in {test_csv}")
    return cameras


def summarize_scene(scene_dir: Path, factor: int, render_scale: float) -> dict[str, object]:
    train_views, points, _ = parse_colmap_train_scene(scene_dir, factor=factor)
    test_cameras = parse_test_poses_csv(scene_dir / "test" / "test_poses.csv", render_scale=render_scale)
    first_test = test_cameras[0]
    return {
        "scene": scene_dir.name,
        "pose_convention": "COLMAP/OpenCV world-to-camera 4x4",
        "train_views": len(train_views),
        "sparse_points": int(len(points)),
        "train_sizes": sorted({(view.width, view.height) for view in train_views}),
        "test_views": len(test_cameras),
        "test_output_sizes": sorted({(camera.width, camera.height) for camera in test_cameras}),
        "test_render_sizes": sorted({(camera.render_width, camera.render_height) for camera in test_cameras}),
        "first_test_image": first_test.image_name,
        "first_test_w2c": first_test.w2c.tolist(),
        "first_test_K": first_test.K.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate COLMAP train cameras and test_poses.csv parsing.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize_scene(args.scene_dir, factor=args.factor, render_scale=args.render_scale)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
