#!/usr/bin/env python3
"""Train a 3D Gaussian Splatting (gsplat) scene from COLMAP data with mask-weighted loss.

This is the primary baseline used for submission. Supports:
- Standard L1 + SSIM loss (default)
- SAM mask-weighted L1 loss (set --mask-dir and --mask-boost)
- LPIPS loss (optional, off by default for stability)

Examples
--------
Baseline (competition-stable):
    python scripts/train_gs_scene.py --scene-dir data/round1/phase1/public_set/HCM0181 \\
        --output-dir outputs/gs_10k --factor 2 --steps 10000

Mask-weighted (boost foreground BTS/tower region):
    python scripts/train_gs_scene.py --scene-dir data/round1/phase1/public_set/HCM0181 \\
        --output-dir outputs/gs_mask_10k --factor 2 --steps 10000 \\
        --mask-dir workspace/sam3_masks --mask-boost 3 --mask-dilate 1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from var2026_bts.configs.defaults import (  # noqa: E402
    DEFAULT_GS_FACTOR,
    DEFAULT_GS_INIT_OPACITY,
    DEFAULT_GS_INIT_SCALE,
    DEFAULT_GS_MAX_INIT_POINTS,
    DEFAULT_GS_REFINE_EVERY,
    DEFAULT_GS_REFINE_START,
    DEFAULT_GS_REFINE_STOP,
    DEFAULT_GS_RESET_EVERY,
    DEFAULT_GS_STEPS,
    DEFAULT_LAMBDA_L1,
    DEFAULT_LAMBDA_LPIPS,
    DEFAULT_LAMBDA_SSIM,
    DEFAULT_MASK_BOOST,
    DEFAULT_MASK_DILATE,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_RENDER_SCALE,
)
from var2026_bts.mask_utils import load_mask, resolve_mask_path, weighted_l1_loss_map  # noqa: E402
from var2026_bts.scene_parsers import (  # noqa: E402
    TrainCamera,
    parse_colmap_train_scene,
    parse_test_poses_csv,
)


def _init_lpips(net: str = "alex"):
    import lpips
    model = lpips.LPIPS(net=net).eval().cuda()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _lpips_inputs(pred_nhwc: torch.Tensor, target_nhwc: torch.Tensor, resize: int) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred_nhwc.permute(0, 3, 1, 2)
    target = target_nhwc.permute(0, 3, 1, 2)
    if resize > 0:
        pred = F.interpolate(pred, size=(resize, resize), mode="bilinear", align_corners=False)
        target = F.interpolate(target, size=(resize, resize), mode="bilinear", align_corners=False)
    return pred * 2.0 - 1.0, target * 2.0 - 1.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resize_rgb(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8).copy()


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
    views: list[TrainCamera],
    points: np.ndarray,
    colors: np.ndarray,
    scene_dir: Path,
    args: argparse.Namespace,
    lpips_fn=None,
) -> torch.nn.ParameterDict:
    device = args.device
    images = [torch.from_numpy(resize_rgb(view.image_path, view.width, view.height)).float() / 255.0 for view in tqdm(views, desc="Loading train images")]
    mask_root = args.mask_dir if args.mask_dir is not None else None
    masks = [
        torch.from_numpy(
            load_mask(
                resolve_mask_path(mask_root, scene_dir.name, view.image_path),
                view.width,
                view.height,
                args.mask_threshold,
                args.mask_dilate,
            )
        )
        for view in tqdm(views, desc="Loading train masks")
    ]
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
        foreground_mask = masks[idx].to(device, non_blocking=True).unsqueeze(0) if args.mask_boost > 0 else None
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
        l1 = weighted_l1_loss_map(pred, target, foreground_mask, args.mask_boost)
        ssim_val = ssim_loss(pred, target)
        loss = args.lambda_l1 * l1 + args.lambda_ssim * ssim_val
        if lpips_fn is not None and args.lambda_lpips > 0:
            lpips_pred, lpips_target = _lpips_inputs(pred, target, args.lpips_resize)
            lpips_val = lpips_fn(lpips_pred, lpips_target).mean()
            loss = loss + args.lambda_lpips * lpips_val
        strategy.step_pre_backward(splats, optimizers, strategy_state, step, info)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        strategy.step_post_backward(splats, optimizers, strategy_state, step, info, packed=False)
        iteration = step + 1
        if iteration in args.milestones:
            render_targets(scene_dir, splats, args.output_dir / f"iter_{iteration}", args)
            torch.cuda.empty_cache()
        if step % 50 == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}", n=len(splats["means"]))
    return splats


@torch.no_grad()
def render_targets(scene_dir: Path, splats: torch.nn.ParameterDict, output_root: Path, args: argparse.Namespace) -> None:
    device = args.device
    test_cameras = parse_test_poses_csv(scene_dir / "test" / "test_poses.csv", render_scale=args.render_scale)
    if args.max_targets > 0:
        test_cameras = test_cameras[: args.max_targets]
    scene_out = output_root / scene_dir.name
    scene_out.mkdir(parents=True, exist_ok=True)
    for camera in tqdm(test_cameras, total=len(test_cameras), desc=f"Rendering {scene_dir.name}"):
        render, _, _ = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.sigmoid(splats["colors"]),
            viewmats=torch.from_numpy(camera.w2c.astype(np.float32)).to(device).unsqueeze(0),
            Ks=torch.from_numpy(camera.K.astype(np.float32)).to(device).unsqueeze(0),
            width=camera.render_width,
            height=camera.render_height,
            packed=False,
            rasterize_mode="classic",
        )
        image_np = (torch.clamp(render[0, ..., :3], 0.0, 1.0) * 255).byte().cpu().numpy()
        image = Image.fromarray(image_np, mode="RGB")
        if image.size != (camera.width, camera.height):
            image = image.resize((camera.width, camera.height), Image.Resampling.LANCZOS)
        out_path = scene_out / camera.image_name
        if out_path.suffix.lower() in {".jpg", ".jpeg"}:
            image.save(out_path, format="JPEG", quality=100, subsampling=0, optimize=True)
        else:
            image.save(out_path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal 3DGS scene from COLMAP and render test poses.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=DEFAULT_GS_FACTOR)
    parser.add_argument("--steps", type=int, default=DEFAULT_GS_STEPS)
    parser.add_argument("--max-init-points", type=int, default=DEFAULT_GS_MAX_INIT_POINTS)
    parser.add_argument("--init-opacity", type=float, default=DEFAULT_GS_INIT_OPACITY)
    parser.add_argument("--init-scale", type=float, default=DEFAULT_GS_INIT_SCALE)
    parser.add_argument("--lambda-l1", type=float, default=DEFAULT_LAMBDA_L1)
    parser.add_argument("--lambda-ssim", type=float, default=DEFAULT_LAMBDA_SSIM)
    parser.add_argument("--lambda-lpips", type=float, default=DEFAULT_LAMBDA_LPIPS)
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--lpips-resize", type=int, default=256)
    parser.add_argument("--mask-dir", type=Path, default=None, help="Per-scene or root directory containing SAM masks named <image_stem>.png.")
    parser.add_argument("--mask-boost", type=float, default=DEFAULT_MASK_BOOST, help="Additional L1 weight for foreground mask pixels; 0 disables mask weighting.")
    parser.add_argument("--mask-threshold", type=float, default=DEFAULT_MASK_THRESHOLD)
    parser.add_argument("--mask-dilate", type=int, default=DEFAULT_MASK_DILATE)
    parser.add_argument("--milestones", nargs="*", type=int, default=[])
    parser.add_argument("--refine-start", type=int, default=DEFAULT_GS_REFINE_START)
    parser.add_argument("--refine-stop", type=int, default=DEFAULT_GS_REFINE_STOP)
    parser.add_argument("--refine-every", type=int, default=DEFAULT_GS_REFINE_EVERY)
    parser.add_argument("--reset-every", type=int, default=DEFAULT_GS_RESET_EVERY)
    parser.add_argument("--render-scale", type=float, default=DEFAULT_RENDER_SCALE)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    if (args.output_dir / args.scene_dir.name).exists() and args.overwrite:
        shutil.rmtree(args.output_dir / args.scene_dir.name)
    views, points, colors = parse_colmap_train_scene(args.scene_dir, args.factor)
    print(f"scene={args.scene_dir.name} train_views={len(views)} sparse_points={len(points)} factor={args.factor}")
    lpips_fn = _init_lpips(args.lpips_net) if args.lambda_lpips > 0 else None
    args.milestones = sorted(set(iteration for iteration in args.milestones if 0 < iteration <= args.steps))
    splats = train(views, points, colors, args.scene_dir, args, lpips_fn=lpips_fn)
    render_targets(args.scene_dir, splats, args.output_dir, args)
    stats = {"scene": args.scene_dir.name, "train_views": len(views), "gaussians": int(len(splats["means"]))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.scene_dir.name}_train_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
