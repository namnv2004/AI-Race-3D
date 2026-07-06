# VAR 2026 — Digital Twin for BTS

Novel view synthesis pipeline for BTS (Base Transceiver Station) tower scenes.
Trains 3D Gaussian Splatting models from COLMAP-reconstructed drone footage and
renders novel views at requested test poses.

## Layout

```
.
├── src/var2026_bts/           # Library package (importable)
│   ├── scene_parsers.py       # COLMAP + test_poses.csv parsing
│   ├── mask_utils.py          # SAM mask loading + weighted L1
│   └── configs/
│       └── defaults.py        # Hyperparameter defaults
├── scripts/                   # Executable entry points
│   ├── train_gs_scene.py      # Train gsplat (baseline)
│   ├── run_mip_splatting_scene.py
│   ├── run_2dgs_scene.py
│   ├── generate_sam3_masks.py
│   ├── evaluate_public.py
│   ├── inspect_dataset.py
│   ├── verify_submission.py
│   ├── verify_zip.py
│   ├── generate_submission_nearest.py
│   ├── setup_mip_splatting.py
│   └── setup_2dgs.py
├── third_party/               # Vendored codebases (gitignored)
│   ├── mip-splatting/
│   └── 2d-gaussian-splatting/
├── data/                      # COLMAP + test_poses.csv (gitignored)
├── workspace/                 # Intermediate exports, SAM masks (gitignored)
├── checkpoints/               # Trained model checkpoints (gitignored)
├── outputs/                   # Rendered test images (gitignored)
├── submissions/               # Final ZIP files (gitignored)
├── logs/                      # Training/eval logs (gitignored)
├── models/                    # Downloaded HF model cache (gitignored)
├── configs/                   # Optional user-supplied config presets
├── pyproject.toml
└── README.md
```

## Setup

```bash
# 1. Create venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip

# 2. Install runtime deps from pyproject.toml
pip install -e ".[dev]"

# 3. Install Mip-Splatting CUDA extensions
python scripts/setup_mip_splatting.py

# 4. (Optional) Install 2DGS CUDA extensions
python scripts/setup_2dgs.py
```

## Pipeline

### 1. Inspect dataset
```bash
python scripts/inspect_dataset.py --json-out reports/dataset.json
```

### 2. (Optional) Generate SAM masks for mask-weighted training
```bash
export HF_TOKEN=...     # only if model is gated
python scripts/generate_sam3_masks.py \\
    --split-root data/round1/phase1/public_set \\
    --output-dir workspace/sam3_masks
```

### 3. Train — baseline (3DGS / gsplat, 10k iterations)
```bash
python scripts/train_gs_scene.py \\
    --scene-dir data/round1/phase1/public_set/HCM0181 \\
    --output-dir outputs/gs_10k --factor 2 --steps 10000
```

### 4. Train — Mip-Splatting (anti-aliasing, 10k iterations)
```bash
python scripts/run_mip_splatting_scene.py \\
    --scene-dir data/round1/phase1/public_set/HCM0181 \\
    --output-dir outputs/mip_10k --iterations 10000 --kernel-size 0.1
```

### 5. Train — 2DGS (thin structures, 10k iterations)
```bash
python scripts/run_2dgs_scene.py \\
    --scene-dir data/round1/phase1/public_set/HCM0181 \\
    --output-dir outputs/2dgs_10k --iterations 10000
```

### 6. Evaluate on public set
```bash
python scripts/evaluate_public.py \\
    --public-root data/round1/phase1/public_set \\
    --pred-dir outputs/gs_10k \\
    --with-lpips
```

### 7. Verify and zip submission
```bash
python scripts/verify_submission.py \\
    --split-root data/round1/phase1/private_set1 \\
    --submission-dir outputs/gs_10k

cd outputs/gs_10k && zip -r ../../../submissions/submission_gs_10k.zip ./* && cd -

python scripts/verify_zip.py \\
    --split-root data/round1/phase1/private_set1 \\
    --zip submissions/submission_gs_10k.zip
```

### 8. Best-single-method automation (no GPU execution by default)

To compare `3dgs`, `mip`, and `2dgs` and pick one global best method, generate
the command scripts first:

```bash
python scripts/generate_experiment_commands.py \
    --config configs/best_single_method_plan.json \
    --out-dir reports/experiments/commands
```

The generator writes scripts under `reports/experiments/commands`:

- `00_preflight.sh`
- `01_generate_masks.sh`
- `10_quick_train.sh`, `11_quick_eval.sh`
- `20_public_train.sh`, `21_public_eval.sh`
- `22_select_best_single_method.sh`

After public eval JSON files exist, choose the winner and render private once:

```bash
python scripts/select_best_single_method.py \
    --config configs/best_single_method_plan.json \
    --eval-dir reports/experiments/eval/public \
    --json-out reports/experiments/final_selection.json

python scripts/generate_experiment_commands.py \
    --config configs/best_single_method_plan.json \
    --selected-experiment <experiment_id>
```

The generated `30_private_best_single_method.sh` will run:

- rendering private set with selected method/config,
- `verify_submission.py`,
- `make_submission_zip.py`,
- `verify_zip.py`.

## Metrics

```text
Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR / 30
```

- LPIPS — perceptual distance, lower is better (weight 0.4)
- SSIM — structural similarity, higher is better (weight 0.3)
- PSNR/30 — normalized PSNR, higher is better (weight 0.3)

## Configuration presets

All scripts accept CLI flags. Defaults are defined in
`src/var2026_bts/configs/defaults.py`. Override at the CLI:

```bash
python scripts/train_gs_scene.py \\
    --scene-dir <path> --output-dir <path> \\
    --steps 20000 --lambda-l1 0.8 --lambda-ssim 0.2 --lambda-lpips 0.0 \\
    --mask-dir workspace/sam3_masks --mask-boost 3 --mask-dilate 1
```

## Notes

- Outputs are full-resolution 1320x989 PNGs.
- Submission ZIP must follow `submissions/<name>.zip` with `scene/<image>.png`.
- SAM mask weighting boosts foreground (BTS/tower) L1 loss only; rendered
  outputs are NOT masked, since the leaderboard scores the whole image.
- `third_party/` is gitignored; clone submodules if needed.
