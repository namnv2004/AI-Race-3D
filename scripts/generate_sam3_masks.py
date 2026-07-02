from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scene_parsers import IMAGE_EXTENSIONS


def load_sam3(model_name: str, device: str):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Export it outside the repo before running SAM 3 mask generation.")
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:
        raise RuntimeError("transformers with SAM 3 support is required. Install/upgrade transformers first.") from exc
    processor = Sam3Processor.from_pretrained(model_name, token=token)
    model = Sam3Model.from_pretrained(model_name, token=token).to(device).eval()
    return processor, model


def combine_masks(result: dict[str, torch.Tensor], min_score: float, width: int, height: int) -> np.ndarray:
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


def segment_image(path: Path, out_path: Path, processor, model, prompt: str, threshold: float, mask_threshold: float, min_score: float, device: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate optional BTS object masks with SAM 3.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--prompt", default="cell tower, telecommunication tower, BTS station, antenna tower")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    images_dir = args.scene_dir / "train" / "images"
    output_dir = args.output_dir or (args.scene_dir / "train" / "masks_sam3")
    image_paths = [path for path in sorted(images_dir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise SystemExit(f"No train images found in {images_dir}")

    processor, model = load_sam3(args.model, args.device)
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
        print(out_path)


if __name__ == "__main__":
    main()
