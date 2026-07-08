"""Evaluate predicted test images against ground truth using PSNR, SSIM, LPIPS.

Implements the competition scoring formula:
    Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * clamp(PSNR / PSNR_MAX, 0, 1)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import lpips
except ImportError:  # pragma: no cover
    lpips = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config_defaults import (
    PSNR_MAX,
    SCORE_WEIGHT_LPIPS,
    SCORE_WEIGHT_PSNR,
    SCORE_WEIGHT_SSIM,
)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def to_lpips_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return tensor.to(device)


def evaluate_scene(
    scene_dir: Path,
    pred_scene_dir: Path,
    limit: int = 0,
    lpips_model: torch.nn.Module | None = None,
    lpips_device: str = "cuda",
    psnr_max: float = PSNR_MAX,
) -> dict[str, object]:
    test_df = pd.read_csv(scene_dir / "test" / "test_poses.csv")
    if limit > 0:
        test_df = test_df.head(limit)
    expected_count = len(test_df)
    gt_dir = scene_dir / "test" / "images"
    psnrs: list[float] = []
    ssims: list[float] = []
    lpips_values: list[float] = []
    scores: list[float] = []
    missing: list[str] = []
    exact_matches: list[str] = []
    for _, row in test_df.iterrows():
        name = str(row["image_name"])
        pred_path = pred_scene_dir / name
        gt_path = gt_dir / name
        if not pred_path.exists() or not gt_path.exists():
            missing.append(name)
            continue
        pred = read_rgb(pred_path)
        gt = read_rgb(gt_path)
        if pred.shape == gt.shape and np.array_equal(pred, gt):
            exact_matches.append(name)
        if pred.shape != gt.shape:
            pred = np.asarray(
                Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.Resampling.LANCZOS)
            )
        psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
        ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
        if lpips_model is not None:
            with torch.inference_mode():
                lpips_value = float(
                    lpips_model(to_lpips_tensor(pred, lpips_device), to_lpips_tensor(gt, lpips_device)).item()
                )
        else:
            lpips_value = float("nan")
        psnr_norm = float(np.clip(psnr / psnr_max, 0.0, 1.0))
        score = (
            SCORE_WEIGHT_LPIPS * (1.0 - lpips_value)
            + SCORE_WEIGHT_SSIM * ssim
            + SCORE_WEIGHT_PSNR * psnr_norm
            if not np.isnan(lpips_value)
            else float("nan")
        )
        psnrs.append(psnr)
        ssims.append(ssim)
        if not np.isnan(lpips_value):
            lpips_values.append(lpips_value)
            scores.append(score)
    return {
        "scene": scene_dir.name,
        "count": len(psnrs),
        "expected_count": expected_count,
        "complete": len(psnrs) == expected_count and not missing,
        "missing": missing,
        "exact_matches": exact_matches,
        "exact_match_count": len(exact_matches),
        "psnr": float(np.mean(psnrs)) if psnrs else None,
        "ssim": float(np.mean(ssims)) if ssims else None,
        "lpips": float(np.mean(lpips_values)) if lpips_values else None,
        "score": float(np.mean(scores)) if scores else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions on public_set using PSNR, SSIM, LPIPS."
    )
    parser.add_argument("--public-root", type=Path, default=Path("data/round1/phase1/public_set"))
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--limit-per-scene", type=int, default=0)
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene names to evaluate.")
    parser.add_argument(
        "--only-existing", action="store_true", help="Only evaluate scenes present in pred-dir."
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing prediction/GT files instead of failing after writing the summary.",
    )
    parser.add_argument("--with-lpips", action="store_true", help="Compute LPIPS and final Score.")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--lpips-device", default="cuda")
    parser.add_argument("--psnr-max", type=float, default=PSNR_MAX)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    lpips_model = None
    if args.with_lpips:
        if lpips is None:
            raise SystemExit("lpips package is not installed. Run: pip install lpips")
        lpips_model = lpips.LPIPS(net=args.lpips_net).to(args.lpips_device).eval()

    results = []
    for scene_dir in sorted(item for item in args.public_root.iterdir() if item.is_dir()):
        if args.scenes is not None and scene_dir.name not in args.scenes:
            continue
        if args.only_existing and not (args.pred_dir / scene_dir.name).exists():
            continue
        if (scene_dir / "test" / "test_poses.csv").exists():
            results.append(
                evaluate_scene(
                    scene_dir,
                    args.pred_dir / scene_dir.name,
                    args.limit_per_scene,
                    lpips_model=lpips_model,
                    lpips_device=args.lpips_device,
                    psnr_max=args.psnr_max,
                )
            )
    valid = [item for item in results if item["psnr"] is not None]
    valid_scores = [item for item in results if item["score"] is not None]
    incomplete = [item for item in results if not item.get("complete", False)]
    summary = {
        "scenes": results,
        "mean_psnr": float(np.mean([item["psnr"] for item in valid])) if valid else None,
        "mean_ssim": float(np.mean([item["ssim"] for item in valid])) if valid else None,
        "mean_lpips": float(np.mean([item["lpips"] for item in valid_scores])) if valid_scores else None,
        "mean_score": float(np.mean([item["score"] for item in valid_scores])) if valid_scores else None,
        "psnr_max": args.psnr_max,
        "complete": not incomplete,
        "incomplete_scenes": [str(item["scene"]) for item in incomplete],
        "exact_match_scenes": [
            str(item["scene"]) for item in results if item.get("exact_match_count", 0) > 0
        ],
        "score_formula": "0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * clamp(PSNR / psnr_max, 0, 1)",
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if incomplete and not args.allow_missing:
        raise SystemExit(
            "Evaluation incomplete for scenes: "
            + ", ".join(str(item["scene"]) for item in incomplete)
            + ". Pass --allow-missing only for intentional partial diagnostics."
        )


if __name__ == "__main__":
    main()
