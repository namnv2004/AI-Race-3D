"""JSON-backed defaults for the AI Race 3D pipeline."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = PROJECT_ROOT / "configs" / "defaults.json"


@lru_cache(maxsize=1)
def load_defaults() -> dict[str, Any]:
    with DEFAULTS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Defaults file must contain a JSON object: {DEFAULTS_PATH}")
    return data


def _get(name: str) -> Any:
    defaults = load_defaults()
    if name not in defaults:
        raise KeyError(f"Missing default config key: {name}")
    return defaults[name]


def _as_float(name: str) -> float:
    return float(_get(name))


def _as_int(name: str) -> int:
    return int(_get(name))


def _as_str(name: str) -> str:
    return str(_get(name))


SCORE_WEIGHT_LPIPS = _as_float("SCORE_WEIGHT_LPIPS")
SCORE_WEIGHT_SSIM = _as_float("SCORE_WEIGHT_SSIM")
SCORE_WEIGHT_PSNR = _as_float("SCORE_WEIGHT_PSNR")
PSNR_MAX = _as_float("PSNR_MAX")

DEFAULT_RENDER_SCALE = _as_float("DEFAULT_RENDER_SCALE")

DEFAULT_LAMBDA_L1 = _as_float("DEFAULT_LAMBDA_L1")
DEFAULT_LAMBDA_SSIM = _as_float("DEFAULT_LAMBDA_SSIM")
DEFAULT_LAMBDA_LPIPS = _as_float("DEFAULT_LAMBDA_LPIPS")
DEFAULT_MASK_BOOST = _as_float("DEFAULT_MASK_BOOST")
DEFAULT_MASK_DILATE = _as_int("DEFAULT_MASK_DILATE")
DEFAULT_MASK_THRESHOLD = _as_float("DEFAULT_MASK_THRESHOLD")

DEFAULT_GS_FACTOR = _as_int("DEFAULT_GS_FACTOR")
DEFAULT_GS_STEPS = _as_int("DEFAULT_GS_STEPS")
DEFAULT_GS_MAX_INIT_POINTS = _as_int("DEFAULT_GS_MAX_INIT_POINTS")
DEFAULT_GS_INIT_OPACITY = _as_float("DEFAULT_GS_INIT_OPACITY")
DEFAULT_GS_INIT_SCALE = _as_float("DEFAULT_GS_INIT_SCALE")
DEFAULT_GS_REFINE_START = _as_int("DEFAULT_GS_REFINE_START")
DEFAULT_GS_REFINE_STOP = _as_int("DEFAULT_GS_REFINE_STOP")
DEFAULT_GS_REFINE_EVERY = _as_int("DEFAULT_GS_REFINE_EVERY")
DEFAULT_GS_RESET_EVERY = _as_int("DEFAULT_GS_RESET_EVERY")

DEFAULT_MIP_ITERATIONS = _as_int("DEFAULT_MIP_ITERATIONS")
DEFAULT_MIP_TRAIN_RESOLUTION = _as_int("DEFAULT_MIP_TRAIN_RESOLUTION")
DEFAULT_MIP_KERNEL_SIZE = _as_float("DEFAULT_MIP_KERNEL_SIZE")
DEFAULT_MIP_DENSIFY_GRAD = _as_float("DEFAULT_MIP_DENSIFY_GRAD")
DEFAULT_MIP_DENSIFY_UNTIL = _as_int("DEFAULT_MIP_DENSIFY_UNTIL")
DEFAULT_MIP_DENSIFY_INTERVAL = _as_int("DEFAULT_MIP_DENSIFY_INTERVAL")
DEFAULT_MIP_OPACITY_RESET = _as_int("DEFAULT_MIP_OPACITY_RESET")

DEFAULT_2DGS_ITERATIONS = _as_int("DEFAULT_2DGS_ITERATIONS")
DEFAULT_2DGS_RESOLUTION = _as_int("DEFAULT_2DGS_RESOLUTION")
DEFAULT_2DGS_LAMBDA_DSSIM = _as_float("DEFAULT_2DGS_LAMBDA_DSSIM")
DEFAULT_2DGS_LAMBDA_DIST = _as_float("DEFAULT_2DGS_LAMBDA_DIST")
DEFAULT_2DGS_LAMBDA_NORMAL = _as_float("DEFAULT_2DGS_LAMBDA_NORMAL")
DEFAULT_2DGS_DENSIFY_GRAD = _as_float("DEFAULT_2DGS_DENSIFY_GRAD")
DEFAULT_2DGS_DENSIFY_UNTIL = _as_int("DEFAULT_2DGS_DENSIFY_UNTIL")
DEFAULT_2DGS_DEPTH_RATIO = _as_float("DEFAULT_2DGS_DEPTH_RATIO")

DEFAULT_SAM_MODEL = _as_str("DEFAULT_SAM_MODEL")
DEFAULT_SAM_PROMPT = _as_str("DEFAULT_SAM_PROMPT")
DEFAULT_SAM_THRESHOLD = _as_float("DEFAULT_SAM_THRESHOLD")
DEFAULT_SAM_MASK_THRESHOLD = _as_float("DEFAULT_SAM_MASK_THRESHOLD")
DEFAULT_SAM_MIN_SCORE = _as_float("DEFAULT_SAM_MIN_SCORE")

DEFAULT_MASK_ROOT = _as_str("DEFAULT_MASK_ROOT")

DEFAULT_OUTPUT_WIDTH = _as_int("DEFAULT_OUTPUT_WIDTH")
DEFAULT_OUTPUT_HEIGHT = _as_int("DEFAULT_OUTPUT_HEIGHT")
