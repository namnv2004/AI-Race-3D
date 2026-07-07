#!/usr/bin/env python3
"""Train 2D Gaussian Splatting (2DGS) on a single scene and render test poses.

Driver for the third_party/2d-gaussian-splatting codebase. 2DGS uses 2D oriented
disks (surfels) instead of 3D ellipsoids, which significantly improves
reconstruction of thin structures like tower legs and antennas.

The 2DGS repo uses the same COLMAP scene format as 3DGS, so our existing
train COLMAP sparse data is directly compatible.

Examples
--------
Basic 10k iteration training:
    python src/method_runners/run_2dgs_scene.py \\
        --scene-dir data/round1/phase1/public_set/HCM0181 \\
        --output-dir outputs/smoke/2dgs_10k_default

With custom normal/distortion weights:
    python src/method_runners/run_2dgs_scene.py ... \\
        --lambda-normal 0.05 --lambda-dist 100

Note
----
2DGS requires CUDA extensions from third_party/2d-gaussian-splatting/submodules.
If submodules are empty, run:
    git submodule update --init --recursive
    pip install third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization \\
            third_party/2d-gaussian-splatting/submodules/simple-knn
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config_defaults import (
    DEFAULT_2DGS_DENSIFY_GRAD,
    DEFAULT_2DGS_DENSIFY_UNTIL,
    DEFAULT_2DGS_DEPTH_RATIO,
    DEFAULT_2DGS_ITERATIONS,
    DEFAULT_2DGS_LAMBDA_DIST,
    DEFAULT_2DGS_LAMBDA_DSSIM,
    DEFAULT_2DGS_LAMBDA_NORMAL,
    DEFAULT_2DGS_RESOLUTION,
    DEFAULT_RENDER_SCALE,
)
from scene_parsers import TestCamera, parse_test_poses_csv


def require_2dgs_dependencies(two_dgs_root: Path) -> None:
    sys.path.insert(0, str(two_dgs_root))
    missing: list[str] = []
    try:
        import diff_surfel_rasterization  # noqa: F401
    except ImportError:
        missing.append("diff_surfel_rasterization")
    try:
        import simple_knn._C  # noqa: F401
    except ImportError:
        missing.append("simple_knn")
    if missing:
        raise RuntimeError(
            "Missing 2DGS CUDA extensions: "
            + ", ".join(missing)
            + ". Install via: pip install third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization "
            + "third_party/2d-gaussian-splatting/submodules/simple-knn"
        )


def run_train(args: argparse.Namespace, source_path: Path, model_path: Path) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "train.py",
        "-s", str(source_path.resolve()),
        "-m", str(model_path.resolve()),
        "--port", str(args.port),
        "--iterations", str(args.iterations),
        "--lambda_dssim", str(args.lambda_dssim),
        "--lambda_dist", str(args.lambda_dist),
        "--lambda_normal", str(args.lambda_normal),
        "--densify_grad_threshold", str(args.densify_grad_threshold),
        "--densify_until_iter", str(args.densify_until_iter),
    ]
    if args.resolution > 0:
        command.extend(["--resolution", str(args.resolution)])
    if args.depth_ratio is not None:
        command.extend(["--depth_ratio", str(args.depth_ratio)])
    if args.quiet:
        command.append("--quiet")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.twodgs_root.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, cwd=args.twodgs_root, env=env, check=True)


def latest_iteration(model_path: Path) -> int:
    point_cloud_dir = model_path / "point_cloud"
    iterations = []
    for path in point_cloud_dir.glob("iteration_*"):
        try:
            iterations.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if not iterations:
        raise FileNotFoundError(f"No 2DGS point cloud iterations found in {point_cloud_dir}")
    return max(iterations)


def save_render(render_rgb: torch.Tensor, output_path: Path, final_size: tuple[int, int]) -> None:
    array = (render_rgb.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(array, mode="RGB")
    if image.size != final_size:
        image = image.resize(final_size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(output_path, format="JPEG", quality=100, subsampling=0, optimize=True)
    else:
        image.save(output_path, format="PNG", optimize=True)


def make_2dgs_camera(camera: TestCamera, index: int, data_device: str):
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov

    dummy_image = torch.zeros((3, camera.render_height, camera.render_width), dtype=torch.float32)
    return Camera(
        colmap_id=index,
        R=camera.w2c[:3, :3].T,
        T=camera.w2c[:3, 3],
        FoVx=focal2fov(float(camera.K[0, 0]), camera.render_width),
        FoVy=focal2fov(float(camera.K[1, 1]), camera.render_height),
        image=dummy_image,
        gt_alpha_mask=None,
        image_name=Path(camera.image_name).stem,
        uid=index,
        data_device=data_device,
    )


def render_test_poses(args: argparse.Namespace, model_path: Path, scene_dir: Path, output_scene_dir: Path) -> None:
    sys.path.insert(0, str(args.twodgs_root.resolve()))
    from gaussian_renderer import GaussianModel, render

    iteration = latest_iteration(model_path)
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing 2DGS point cloud: {ply_path}")

    gaussians = GaussianModel(3)
    gaussians.load_ply(str(ply_path))
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, depth_ratio=0.0)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    test_cameras = parse_test_poses_csv(scene_dir / "test" / "test_poses.csv", render_scale=args.render_scale)
    if args.max_targets > 0:
        test_cameras = test_cameras[: args.max_targets]

    output_scene_dir.mkdir(parents=True, exist_ok=True)
    for idx, camera in enumerate(test_cameras):
        view = make_2dgs_camera(camera, idx, args.data_device)
        with torch.no_grad():
            rendered = render(view, gaussians, pipeline, background)["render"]
        save_render(rendered, output_scene_dir / camera.image_name, (camera.width, camera.height))
        del view, rendered
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 2DGS on one scene and render test poses.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--twodgs-root", type=Path, default=Path("third_party/2d-gaussian-splatting"))
    parser.add_argument("--model-root", type=Path, default=Path("checkpoints/2dgs"))
    parser.add_argument("--iterations", type=int, default=DEFAULT_2DGS_ITERATIONS)
    parser.add_argument("--resolution", type=int, default=DEFAULT_2DGS_RESOLUTION, help="Resolution override; -1 keeps native COLMAP resolution.")
    parser.add_argument("--render-scale", type=float, default=DEFAULT_RENDER_SCALE)
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--lambda-dssim", type=float, default=DEFAULT_2DGS_LAMBDA_DSSIM)
    parser.add_argument("--lambda-dist", type=float, default=DEFAULT_2DGS_LAMBDA_DIST)
    parser.add_argument("--lambda-normal", type=float, default=DEFAULT_2DGS_LAMBDA_NORMAL)
    parser.add_argument("--densify-grad-threshold", type=float, default=DEFAULT_2DGS_DENSIFY_GRAD)
    parser.add_argument("--densify-until-iter", type=int, default=DEFAULT_2DGS_DENSIFY_UNTIL)
    parser.add_argument("--depth-ratio", type=float, default=DEFAULT_2DGS_DEPTH_RATIO)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--data-device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not (args.twodgs_root / "train.py").exists():
        raise SystemExit(
            f"2DGS repo not found at {args.twodgs_root}. "
            "Clone https://github.com/hbb1/2d-gaussian-splatting and init submodules."
        )

    scene_name = args.scene_dir.name
    model_path = args.model_root / scene_name
    output_scene_dir = args.output_dir / scene_name

    train_source_path = (
        args.scene_dir / "train"
        if (args.scene_dir / "train").is_dir()
        else args.scene_dir
    )

    if not args.skip_train:
        require_2dgs_dependencies(args.twodgs_root)
        if model_path.exists() and args.overwrite:
            shutil.rmtree(model_path)
        run_train(args, train_source_path, model_path)
    render_test_poses(args, model_path, args.scene_dir, output_scene_dir)
    print(f"rendered test poses to {output_scene_dir}")


if __name__ == "__main__":
    main()
