"""Shared mask loading and weighted L1 loss utilities.

Used by:
- src/method_runners/run_3dgs_scene.py (gsplat + mask-weighted loss)
- src/method_runners/run_mip_splatting_scene.py (Mip-Splatting + mask-weighted loss)

The mask is binary 0/1 (after thresholding). When loaded, foreground pixels = 1,
background pixels = 0. The weighted L1 loss boosts foreground by `boost`.
Output images are NOT masked — only the loss uses the mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


def resolve_mask_path(mask_dir: Path | None, scene_name: str, image_path: Path) -> Path | None:
    """Find a mask file for a given image under the SAM mask root.

    Lookup order (per scene):
    1. <mask_dir>/<scene_name>/<image_stem>.png
    2. <mask_dir>/<scene_name>/<image_name>
    3. <mask_dir>/<image_stem>.png
    4. <mask_dir>/<image_name>

    Returns None if no mask file is found.
    """
    if mask_dir is None:
        return None
    stem = image_path.stem
    candidates = [
        mask_dir / scene_name / f"{stem}.png",
        mask_dir / scene_name / image_path.name,
        mask_dir / f"{stem}.png",
        mask_dir / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_mask(
    path: Path | None,
    width: int,
    height: int,
    threshold: float = 0.5,
    dilate: int = 0,
) -> np.ndarray:
    """Load a SAM mask as float32 array of shape (H, W, 1) in {0, 1}.

    If `path` is None, returns all-zero mask (i.e. no foreground weighting).
    Resizes with NEAREST to preserve binary values, then thresholds.
    """
    if path is None:
        return np.zeros((height, width, 1), dtype=np.float32)
    mask = Image.open(path).convert("L")
    if dilate > 0:
        mask = mask.filter(ImageFilter.MaxFilter(2 * dilate + 1))
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    array = np.asarray(mask, dtype=np.float32) / 255.0
    return (array >= threshold).astype(np.float32)[..., None]


def weighted_l1_loss_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    foreground_mask: torch.Tensor | None,
    boost: float,
) -> torch.Tensor:
    """Per-pixel L1 loss with foreground boosting.

    pred/target: shape (B, H, W, 3) in [0, 1] for gsplat, or (3, H, W) for Mip.
    foreground_mask: shape (1, H, W, 1) or (1, 1, H, W) or None.
    boost: extra weight applied to foreground pixels. 0 disables weighting.

    Returns a scalar L1 loss.
    """
    if foreground_mask is None or boost <= 0.0:
        return F.l1_loss(pred, target)
    weights = 1.0 + boost * foreground_mask
    if pred.dim() == 4 and pred.shape[-1] == 3:
        # (B, H, W, 3) layout
        return (torch.abs(pred - target) * weights).sum() / (weights.sum() * pred.shape[-1]).clamp_min(1e-6)
    # (3, H, W) layout
    weights = weights.squeeze(-1).unsqueeze(0)  # (1, H, W) -> (1, H, W)
    return (torch.abs(pred - target) * weights).sum() / (weights.sum() * 3).clamp_min(1e-6)
