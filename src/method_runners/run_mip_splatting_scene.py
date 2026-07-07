#!/usr/bin/env python3
"""Train Mip-Splatting on a single scene, render test poses, optionally with milestones.

Driver for the third_party/mip-splatting codebase. Supports:
- 3D filter + 2D mip filter (anti-aliasing)
- SAM mask-weighted loss (--mask-dir, --mask-boost)
- Milestone rendering for tuning (--milestones, --render-milestones)
- Custom densification schedule (--densify-grad-threshold, --densify-until-iter)

Examples
--------
Baseline 10k with proper anti-aliasing:
    python src/method_runners/run_mip_splatting_scene.py \\
        --scene-dir data/round1/phase1/public_set/HCM0181 \\
        --output-dir outputs/smoke/mip_10k_r4_k01 --iterations 10000 --kernel-size 0.1

Mask-weighted:
    python src/method_runners/run_mip_splatting_scene.py \\
        --scene-dir data/round1/phase1/public_set/HCM0181 \\
        --output-dir outputs/smoke/mip_mask_10k --iterations 10000 --kernel-size 0.1 \\
        --mask-dir masks --mask-boost 3 --mask-dilate 1

Tuning with milestones:
    python src/method_runners/run_mip_splatting_scene.py ... \\
        --milestones 1000 5000 10000 20000 --render-milestones
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config_defaults import (
    DEFAULT_LAMBDA_L1,
    DEFAULT_LAMBDA_LPIPS,
    DEFAULT_LAMBDA_SSIM,
    DEFAULT_MASK_BOOST,
    DEFAULT_MASK_DILATE,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_MIP_DENSIFY_GRAD,
    DEFAULT_MIP_DENSIFY_INTERVAL,
    DEFAULT_MIP_DENSIFY_UNTIL,
    DEFAULT_MIP_ITERATIONS,
    DEFAULT_MIP_KERNEL_SIZE,
    DEFAULT_MIP_OPACITY_RESET,
    DEFAULT_MIP_TRAIN_RESOLUTION,
    DEFAULT_RENDER_SCALE,
)
from scene_parsers import (
    TestCamera,
    TrainCamera,
    parse_colmap_train_scene,
    parse_test_poses_csv,
)


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    m = np.asarray(rotation, dtype=np.float64)
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = m.flat
    k = np.array(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0.0:
        qvec *= -1.0
    return qvec / np.linalg.norm(qvec)


def write_text_colmap(export_dir: Path, views: list[TrainCamera]) -> None:
    sparse_dir = export_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    camera_ids: dict[tuple[int, int, float, float, float, float], int] = {}
    image_camera_ids: list[int] = []
    for view in views:
        key = (
            int(view.width),
            int(view.height),
            round(float(view.K[0, 0]), 8),
            round(float(view.K[1, 1]), 8),
            round(float(view.K[0, 2]), 8),
            round(float(view.K[1, 2]), 8),
        )
        camera_id = camera_ids.setdefault(key, len(camera_ids) + 1)
        image_camera_ids.append(camera_id)

    cameras_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(camera_ids)}",
    ]
    for key, camera_id in sorted(camera_ids.items(), key=lambda item: item[1]):
        width, height, fx, fy, cx, cy = key
        cameras_lines.append(f"{camera_id} PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}")
    (sparse_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf-8")

    images_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(views)}, mean observations per image: 0",
    ]
    for image_id, (view, camera_id) in enumerate(zip(views, image_camera_ids), start=1):
        qvec = rotmat_to_qvec(view.w2c[:3, :3])
        tvec = view.w2c[:3, 3]
        images_lines.append(
            f"{image_id} "
            f"{qvec[0]:.17g} {qvec[1]:.17g} {qvec[2]:.17g} {qvec[3]:.17g} "
            f"{tvec[0]:.17g} {tvec[1]:.17g} {tvec[2]:.17g} "
            f"{camera_id} {view.name}"
        )
        images_lines.append("")
    (sparse_dir / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")


def write_points_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors_u8 = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property float nx\nproperty float ny\nproperty float nz\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors_u8):
            file.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"0 0 0 {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def link_or_copy_images(views: list[TrainCamera], images_dir: Path, copy_images: bool) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for view in views:
        dst = images_dir / view.name
        if dst.exists() or dst.is_symlink():
            continue
        if copy_images:
            shutil.copy2(view.image_path, dst)
        else:
            dst.symlink_to(view.image_path.resolve())


def export_mip_dataset(scene_dir: Path, export_dir: Path, overwrite: bool, copy_images: bool) -> Path:
    if export_dir.exists() and overwrite:
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    views, points, colors = parse_colmap_train_scene(scene_dir, factor=1)
    link_or_copy_images(views, export_dir / "images", copy_images=copy_images)
    write_text_colmap(export_dir, views)
    write_points_ply(export_dir / "sparse" / "0" / "points3D.ply", points, colors)
    return export_dir


def require_mip_dependencies(mip_root: Path) -> None:
    sys.path.insert(0, str(mip_root))
    missing: list[str] = []
    try:
        import diff_gaussian_rasterization  # noqa: F401
    except ImportError:
        missing.append("diff_gaussian_rasterization")
    try:
        import simple_knn._C  # noqa: F401
    except ImportError:
        missing.append("simple_knn")
    if missing:
        raise RuntimeError(
            "Missing Mip-Splatting CUDA extensions: "
            + ", ".join(missing)
            + ". Install with: pip install third_party/mip-splatting/submodules/diff-gaussian-rasterization "
            + "third_party/mip-splatting/submodules/simple-knn"
        )


def resolve_iterations(args: argparse.Namespace) -> list[int]:
    """Build the union of explicit milestones and the final iteration."""
    if args.milestones:
        iterations = [iteration for iteration in args.milestones if 0 < iteration <= args.iterations]
    else:
        iterations = [args.save_every, args.iterations]
    iterations.append(args.iterations)
    return sorted(set(iterations))


def run_train(args: argparse.Namespace, export_dir: Path, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    save_iterations = resolve_iterations(args)
    command = [
        sys.executable,
        "train.py",
        "-s",
        str(export_dir.resolve()),
        "-m",
        str(model_dir.resolve()),
        "-r",
        str(args.train_resolution),
        "--iterations",
        str(args.iterations),
        "--kernel_size",
        str(args.kernel_size),
        "--lambda_l1",
        str(args.lambda_l1),
        "--lambda_ssim",
        str(args.lambda_ssim),
        "--lambda_lpips",
        str(args.lambda_lpips),
        "--lpips_net",
        str(args.lpips_net),
        "--lpips_resize",
        str(args.lpips_resize),
        "--mask_path",
        str(args.mask_dir or ""),
        "--mask_boost",
        str(args.mask_boost),
        "--mask_threshold",
        str(args.mask_threshold),
        "--mask_dilate",
        str(args.mask_dilate),
        "--densify_grad_threshold",
        str(args.densify_grad_threshold),
        "--densify_until_iter",
        str(args.densify_until_iter),
        "--densification_interval",
        str(args.densification_interval),
        "--opacity_reset_interval",
        str(args.opacity_reset_interval),
        "--data_device",
        args.data_device,
        "--save_iterations",
        *[str(iteration) for iteration in save_iterations],
        "--checkpoint_iterations",
        *[str(iteration) for iteration in save_iterations],
    ]
    if args.quiet:
        command.append("--quiet")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.mip_root.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, cwd=args.mip_root, env=env, check=True)


def latest_iteration(model_dir: Path) -> int:
    point_cloud_dir = model_dir / "point_cloud"
    iterations = []
    for path in point_cloud_dir.glob("iteration_*"):
        try:
            iterations.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if not iterations:
        raise FileNotFoundError(f"No Mip-Splatting point cloud iterations found in {point_cloud_dir}")
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


def make_mip_camera(camera: TestCamera, index: int, data_device: str):
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


def render_test_poses(args: argparse.Namespace, model_dir: Path, scene_dir: Path, output_scene_dir: Path, iteration_override: int | None = None) -> None:
    sys.path.insert(0, str(args.mip_root.resolve()))
    from gaussian_renderer import GaussianModel, render

    iteration = iteration_override if iteration_override is not None else args.render_iteration if args.render_iteration > 0 else latest_iteration(model_dir)
    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing trained point cloud: {ply_path}")

    gaussians = GaussianModel(3)
    gaussians.load_ply(str(ply_path))
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    test_cameras = parse_test_poses_csv(scene_dir / "test" / "test_poses.csv", render_scale=args.render_scale)
    if args.max_targets > 0:
        test_cameras = test_cameras[: args.max_targets]

    output_scene_dir.mkdir(parents=True, exist_ok=True)
    for idx, camera in enumerate(test_cameras):
        view = make_mip_camera(camera, idx, args.data_device)
        with torch.no_grad():
            rendered = render(view, gaussians, pipeline, background, kernel_size=args.kernel_size)["render"]
        save_render(rendered, output_scene_dir / camera.image_name, (camera.width, camera.height))
        del view, rendered
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export, train, and render one competition scene with Mip-Splatting.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke/mip_splatting"))
    parser.add_argument("--mip-root", type=Path, default=Path("third_party/mip-splatting"))
    parser.add_argument("--workspace-dir", type=Path, default=Path("workspace/mip_splatting_scenes"))
    parser.add_argument("--model-root", type=Path, default=Path("checkpoints/mip_splatting"))
    parser.add_argument("--iterations", type=int, default=DEFAULT_MIP_ITERATIONS)
    parser.add_argument("--train-resolution", type=int, default=DEFAULT_MIP_TRAIN_RESOLUTION, help="Mip-Splatting -r value; use 1 for full-res training.")
    parser.add_argument("--render-scale", type=float, default=DEFAULT_RENDER_SCALE, help="Render lower then resize to CSV size if needed.")
    parser.add_argument("--kernel-size", type=float, default=DEFAULT_MIP_KERNEL_SIZE)
    parser.add_argument("--lambda-l1", type=float, default=DEFAULT_LAMBDA_L1, help="L1 loss weight (default 0.8)")
    parser.add_argument("--lambda-ssim", type=float, default=DEFAULT_LAMBDA_SSIM, help="SSIM loss weight (default 0.2)")
    parser.add_argument("--lambda-lpips", type=float, default=DEFAULT_LAMBDA_LPIPS, help="LPIPS loss weight (default 0 disables)")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--lpips-resize", type=int, default=256, help="Resize CHW images to this square size for LPIPS loss; <=0 keeps full size.")
    parser.add_argument("--mask-dir", type=Path, default=None, help="Per-scene or root directory containing SAM masks named <image_stem>.png.")
    parser.add_argument("--mask-boost", type=float, default=DEFAULT_MASK_BOOST, help="Additional L1 weight for foreground mask pixels; 0 disables mask weighting.")
    parser.add_argument("--mask-threshold", type=float, default=DEFAULT_MASK_THRESHOLD)
    parser.add_argument("--mask-dilate", type=int, default=DEFAULT_MASK_DILATE)
    parser.add_argument("--densify-grad-threshold", type=float, default=DEFAULT_MIP_DENSIFY_GRAD)
    parser.add_argument("--densify-until-iter", type=int, default=DEFAULT_MIP_DENSIFY_UNTIL)
    parser.add_argument("--densification-interval", type=int, default=DEFAULT_MIP_DENSIFY_INTERVAL)
    parser.add_argument("--opacity-reset-interval", type=int, default=DEFAULT_MIP_OPACITY_RESET)
    parser.add_argument("--milestones", nargs="*", type=int, default=None, help="Iterations to save/checkpoint and optionally render, e.g. 1000 5000 10000 20000.")
    parser.add_argument("--render-milestones", action="store_true", help="Render every saved milestone to output-dir/iter_<iteration>/<scene>.")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--render-iteration", type=int, default=-1)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--data-device", default="cuda")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not (args.mip_root / "train.py").exists():
        raise SystemExit(f"Mip-Splatting repo not found at {args.mip_root}. Clone https://github.com/autonomousvision/mip-splatting first.")
    if args.iterations < 15000 and not args.max_targets:
        print("[WARN] Competition runs should use 15000-30000 iterations per scene.")

    scene_name = args.scene_dir.name
    export_dir = args.workspace_dir / scene_name
    model_dir = args.model_root / scene_name
    output_scene_dir = args.output_dir / scene_name

    export_mip_dataset(args.scene_dir, export_dir, overwrite=args.overwrite, copy_images=args.copy_images)
    print(f"exported Mip-Splatting scene: {export_dir}")
    if args.export_only:
        return

    require_mip_dependencies(args.mip_root)
    if not args.skip_train:
        if model_dir.exists() and args.overwrite:
            shutil.rmtree(model_dir)
        run_train(args, export_dir, model_dir)
    if not args.skip_render:
        if output_scene_dir.exists() and args.overwrite and not args.render_milestones:
            shutil.rmtree(output_scene_dir)
        if args.render_milestones:
            for iteration in resolve_iterations(args):
                milestone_output_scene_dir = args.output_dir / f"iter_{iteration}" / scene_name
                if milestone_output_scene_dir.exists() and args.overwrite:
                    shutil.rmtree(milestone_output_scene_dir)
                render_test_poses(args, model_dir, args.scene_dir, milestone_output_scene_dir, iteration_override=iteration)
        else:
            render_test_poses(args, model_dir, args.scene_dir, output_scene_dir)
        print(f"rendered test poses to {output_scene_dir}")


if __name__ == "__main__":
    main()
