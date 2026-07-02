from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycolmap
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class CameraView:
    name: str
    image_path: Path
    w2c: np.ndarray
    K: np.ndarray
    width: int
    height: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = [float(v) for v in qvec]
    return np.array(
        [
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw],
            [2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw],
            [2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ],
        dtype=np.float64,
    )


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


def load_scene(scene_dir: Path, factor: int) -> tuple[list[CameraView], np.ndarray, np.ndarray]:
    train_dir = scene_dir / "train"
    image_paths = collect_image_paths(train_dir / "images")
    reconstruction = pycolmap.Reconstruction(str(train_dir / "sparse" / "0"))
    views: list[CameraView] = []
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
            CameraView(
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
        raise RuntimeError(f"No train views loaded from {scene_dir}")

    points = []
    colors = []
    for point in reconstruction.points3D.values():
        points.append(np.asarray(point.xyz, dtype=np.float32))
        colors.append(np.asarray(point.color, dtype=np.float32) / 255.0)
    if not points:
        raise RuntimeError(f"No sparse points loaded from {scene_dir}")
    return views, np.stack(points).astype(np.float32), np.stack(colors).astype(np.float32)


def resize_rgb(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def init_splats(
    points_np: np.ndarray,
    colors_np: np.ndarray,
    max_points: int,
    init_opacity: float,
    init_scale: float,
    device: str,
) -> torch.nn.ParameterDict:
    if max_points > 0 and len(points_np) > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(points_np), size=max_points, replace=False)
        points_np = points_np[indices]
        colors_np = colors_np[indices]

    points = torch.from_numpy(points_np).float()
    colors = torch.from_numpy(np.clip(colors_np, 1e-4, 1.0 - 1e-4)).float()
    n_neighbors = min(4, len(points))
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(points_np).kneighbors(points_np)
    if n_neighbors > 1:
        dist_avg = torch.from_numpy(distances[:, 1:].mean(axis=1)).float()
    else:
        dist_avg = torch.full((len(points),), 1e-2)
    scales = torch.log(torch.clamp(dist_avg * init_scale, min=1e-4)).unsqueeze(-1).repeat(1, 3)
    quats = F.normalize(torch.randn((len(points), 4)), dim=-1)
    opacities = torch.logit(torch.full((len(points),), init_opacity))
    color_logits = torch.logit(colors)
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "colors": torch.nn.Parameter(color_logits),
        }
    ).to(device)


def make_optimizers(splats: torch.nn.ParameterDict, scene_scale: float, batch_size: int) -> dict[str, torch.optim.Optimizer]:
    lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "colors": 2.5e-3,
    }
    scale = math.sqrt(batch_size)
    return {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr * scale, "name": name}],
            eps=1e-15 / scale,
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, lr in lrs.items()
    }


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_chw = pred.permute(0, 3, 1, 2)
    target_chw = target.permute(0, 3, 1, 2)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(pred_chw, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target_chw, 11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(pred_chw * pred_chw, 11, stride=1, padding=5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target_chw * target_chw, 11, stride=1, padding=5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred_chw * target_chw, 11, stride=1, padding=5) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return 1.0 - ssim.mean()


def train(
    views: list[CameraView],
    points: np.ndarray,
    colors: np.ndarray,
    args: argparse.Namespace,
) -> torch.nn.ParameterDict:
    device = args.device
    images = [torch.from_numpy(resize_rgb(view.image_path, view.width, view.height)).float() / 255.0 for view in tqdm(views, desc="Loading train images")]
    w2cs = torch.from_numpy(np.stack([view.w2c for view in views]).astype(np.float32)).to(device)
    Ks = torch.from_numpy(np.stack([view.K for view in views]).astype(np.float32)).to(device)
    widths = [view.width for view in views]
    heights = [view.height for view in views]
    camera_centers = np.stack([np.linalg.inv(view.w2c)[:3, 3] for view in views])
    scene_scale = max(float(np.linalg.norm(camera_centers - camera_centers.mean(axis=0), axis=1).max()), 1e-3)

    splats = init_splats(points, colors, args.max_init_points, args.init_opacity, args.init_scale, device)
    optimizers = make_optimizers(splats, scene_scale, batch_size=1)
    strategy = DefaultStrategy(
        refine_start_iter=args.refine_start,
        refine_stop_iter=args.refine_stop,
        refine_every=args.refine_every,
        reset_every=args.reset_every,
        verbose=False,
    )
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / max(args.steps, 1)))

    progress = tqdm(range(args.steps), desc="Training 3DGS")
    for step in progress:
        idx = random.randrange(len(views))
        target = images[idx].to(device, non_blocking=True).unsqueeze(0)
        render, _, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.sigmoid(splats["colors"]),
            viewmats=w2cs[idx : idx + 1],
            Ks=Ks[idx : idx + 1],
            width=widths[idx],
            height=heights[idx],
            packed=False,
            absgrad=strategy.absgrad,
            rasterize_mode="classic",
        )
        pred = torch.clamp(render[..., :3], 0.0, 1.0)
        l1 = F.l1_loss(pred, target)
        loss = (1.0 - args.ssim_lambda) * l1 + args.ssim_lambda * ssim_loss(pred, target)
        strategy.step_pre_backward(splats, optimizers, strategy_state, step, info)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        strategy.step_post_backward(splats, optimizers, strategy_state, step, info, packed=False)
        if step % 50 == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}", n=len(splats["means"]))
    return splats


def target_pose(row: pd.Series) -> tuple[np.ndarray, np.ndarray, int, int]:
    qvec = row[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)
    tvec = row[["tx", "ty", "tz"]].to_numpy(dtype=np.float64)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = qvec_to_rotmat(qvec)
    w2c[:3, 3] = tvec
    K = np.array(
        [[float(row["fx"]), 0.0, float(row["cx"])], [0.0, float(row["fy"]), float(row["cy"])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return w2c, K, int(row["width"]), int(row["height"])


@torch.no_grad()
def render_targets(scene_dir: Path, splats: torch.nn.ParameterDict, output_root: Path, args: argparse.Namespace) -> None:
    device = args.device
    test_df = pd.read_csv(scene_dir / "test" / "test_poses.csv")
    if args.max_targets > 0:
        test_df = test_df.head(args.max_targets).copy()
    scene_out = output_root / scene_dir.name
    scene_out.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Rendering {scene_dir.name}"):
        w2c, K, width, height = target_pose(row)
        if args.render_scale != 1.0:
            render_width = max(2, int(round(width * args.render_scale)))
            render_height = max(2, int(round(height * args.render_scale)))
            K = K.copy()
            K[0, :] *= render_width / width
            K[1, :] *= render_height / height
        else:
            render_width = width
            render_height = height
        render, _, _ = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.sigmoid(splats["colors"]),
            viewmats=torch.from_numpy(w2c.astype(np.float32)).to(device).unsqueeze(0),
            Ks=torch.from_numpy(K.astype(np.float32)).to(device).unsqueeze(0),
            width=render_width,
            height=render_height,
            packed=False,
            rasterize_mode="classic",
        )
        image_np = (torch.clamp(render[0, ..., :3], 0.0, 1.0) * 255).byte().cpu().numpy()
        image = Image.fromarray(image_np, mode="RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        out_path = scene_out / str(row["image_name"])
        if out_path.suffix.lower() in {".jpg", ".jpeg"}:
            image.save(out_path, format="JPEG", quality=95, subsampling=1, optimize=True)
        else:
            image.save(out_path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal 3DGS scene from COLMAP and render test poses.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--max-init-points", type=int, default=200000)
    parser.add_argument("--init-opacity", type=float, default=0.1)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--ssim-lambda", type=float, default=0.2)
    parser.add_argument("--refine-start", type=int, default=500)
    parser.add_argument("--refine-stop", type=int, default=2500)
    parser.add_argument("--refine-every", type=int, default=100)
    parser.add_argument("--reset-every", type=int, default=1500)
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    if (args.output_dir / args.scene_dir.name).exists() and args.overwrite:
        shutil.rmtree(args.output_dir / args.scene_dir.name)
    views, points, colors = load_scene(args.scene_dir, args.factor)
    print(f"scene={args.scene_dir.name} train_views={len(views)} sparse_points={len(points)} factor={args.factor}")
    splats = train(views, points, colors, args)
    render_targets(args.scene_dir, splats, args.output_dir, args)
    stats = {"scene": args.scene_dir.name, "train_views": len(views), "gaussians": int(len(splats["means"]))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.scene_dir.name}_train_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
