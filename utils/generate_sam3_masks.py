#!/usr/bin/env python3
"""Generate SAM (Segment Anything Model) masks for BTS/tower foreground.

Generates binary masks for the BTS/tower region in train images. The masks
are used by training scripts to weight the L1 loss toward the foreground
object. The actual output images rendered for submission are NOT masked.

Supports both single-scene and split-wide batch generation.

Examples
--------
Single scene:
    python utils/generate_sam3_masks.py \\
        --scene-dir data/round1/phase1/public_set/HCM0181

Whole public split:
    python utils/generate_sam3_masks.py \\
        --split-root data/round1/phase1/public_set

To use facebook/sam3.1 instead, pass --model facebook/sam3.1 (gated, may
require direct checkpoint download).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config_defaults import (
    DEFAULT_MASK_ROOT,
    DEFAULT_SAM_MASK_THRESHOLD,
    DEFAULT_SAM_MIN_SCORE,
    DEFAULT_SAM_MODEL,
    DEFAULT_SAM_PROMPT,
    DEFAULT_SAM_THRESHOLD,
)
from scene_parsers import IMAGE_EXTENSIONS


def load_sam(model_name: str, device: str):
    """Load a SAM 3.x model from HuggingFace. Token via HF_TOKEN env var if gated."""
    token = os.environ.get("HF_TOKEN") or None
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:
        raise RuntimeError(
            "transformers with SAM 3 support is required. Install/upgrade transformers first."
        ) from exc
    kwargs = {"token": token} if token else {}
    processor = Sam3Processor.from_pretrained(model_name, **kwargs)
    model = Sam3Model.from_pretrained(model_name, **kwargs).to(device).eval()
    return processor, model


def combine_masks(result: dict, min_score: float, width: int, height: int) -> np.ndarray:
    """Combine all instance masks above min_score into a single binary mask."""
    masks = result.get("masks")
    scores = result.get("scores")
    if masks is None or len(masks) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    keep = torch.ones((len(masks),), dtype=torch.bool, device=masks.device)
    if scores is not None:
        keep = scores >= min_score
    selected = masks[keep]
    if len(selected) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    merged = selected.bool().any(dim=0).cpu().numpy().astype(np.uint8) * 255
    return merged


def segment_image(
    path: Path,
    out_path: Path,
    processor,
    model,
    prompt: str,
    threshold: float,
    mask_threshold: float,
    min_score: float,
    device: str,
) -> None:
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    mask = combine_masks(results, min_score=min_score, width=image.width, height=image.height)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_path)


def generate_scene_masks(
    scene_dir: Path,
    output_dir: Path,
    processor,
    model,
    args: argparse.Namespace,
) -> int:
    images_dir = scene_dir / "train" / "images"
    image_paths = [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        print(f"[WARN] no train images found in {images_dir}")
        return 0

    generated = 0
    for image_path in image_paths:
        out_path = output_dir / f"{image_path.stem}.png"
        if out_path.exists() and not args.overwrite:
            continue
        segment_image(
            image_path,
            out_path,
            processor,
            model,
            args.prompt,
            args.threshold,
            args.mask_threshold,
            args.min_score,
            args.device,
        )
        generated += 1
        print(out_path)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SAM 3.x masks for BTS/tower region in train images."
    )
    parser.add_argument("--scene-dir", type=Path, default=None)
    parser.add_argument(
        "--split-root", type=Path, default=None, help="Run all scenes under this public/private root."
    )
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Single-scene output dir, or root dir when --split-root is used.",
    )
    parser.add_argument("--model", default=DEFAULT_SAM_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_SAM_PROMPT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SAM_THRESHOLD)
    parser.add_argument("--mask-threshold", type=float, default=DEFAULT_SAM_MASK_THRESHOLD)
    parser.add_argument("--min-score", type=float, default=DEFAULT_SAM_MIN_SCORE)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.scene_dir is None and args.split_root is None:
        raise SystemExit("Pass --scene-dir or --split-root.")

    processor, model = load_sam(args.model, args.device)
    if args.split_root is not None:
        output_root = args.output_dir or Path(DEFAULT_MASK_ROOT)
        scene_dirs = [path for path in sorted(args.split_root.iterdir()) if path.is_dir()]
        if args.scenes is not None:
            scene_dirs = [path for path in scene_dirs if path.name in set(args.scenes)]
        total = 0
        for scene_dir in scene_dirs:
            total += generate_scene_masks(scene_dir, output_root / scene_dir.name, processor, model, args)
        print(f"generated {total} masks under {output_root}")
    else:
        output_dir = args.output_dir or (args.scene_dir / "train" / "masks_sam3")
        total = generate_scene_masks(args.scene_dir, output_dir, processor, model, args)
        print(f"generated {total} masks under {output_dir}")


if __name__ == "__main__":
    main()
