# AI Race 3D Agent Guide

This repo is primarily operated by coding agents. Prefer small, safe edits, keep paths current, and avoid running GPU workloads unless the user explicitly asks.

## Project Goal

Build and maintain a Novel View Synthesis pipeline for the AI Race 3D / Digital Twin BTS task. The pipeline consumes COLMAP scenes and test camera poses, then renders RGB images for unseen viewpoints.

## Local Skills

Project-specific skills live in `.skills/` and are written for AI Engineer / 3D Reconstruction work:

- `ai-engineer-3d-reconstruction`: end-to-end pipeline engineering, repo boundaries, no-GPU validation.
- `colmap-scene-understanding`: COLMAP scene layout, camera pose conventions, `test_poses.csv` contract.
- `gaussian-splatting-nvs`: 3DGS, Mip-Splatting, 2DGS train/render workflow rules.
- `metric-evaluation`: LPIPS, SSIM, PSNR, score formula, public evaluation behavior.
- `sota-experiment-scheduler`: automated benchmark queues, background run scripts, and SOTA comparison workflow.
- `submission-quality-control`: folder/zip validation and submission safety.

## Repository Map

- `src/`: shared Python modules and method runner entrypoints.
- `configs/`: defaults, experiment manifests, third-party registry notes.
- `src/method_runners/`: standardized method entrypoints named `run_<method>_scene.py`.
- `runs/`: generated workflow scripts live in `runs/commands/`; historical legacy `.sh` scripts live directly under `runs/` with details in `runs/README.md`.
- `eval/`: public-set scoring and metric computation.
- `verify/`: submission folder/zip validation.
- `utils/`: dataset inspection, mask generation, experiment command generation, zip/recompression helpers.
- `data/`, `outputs/`, `submissions/`, `checkpoints/`, `workspace/`, `third_party/`: local artifacts or external repos; treat as large/local by default.

## Agent Rules

- Do not reintroduce `scripts/`, `src/cli/`, or `src/configs/`; these were intentionally removed.
- Import shared modules from `src/`; defaults live in `configs/defaults.json` and are loaded via `src/config_defaults.py`.
- Keep runnable files in the directory where they are used: method runners in `src/method_runners/*.py`, generated workflows in `runs/commands/*.sh`, metrics in `eval/`, validation in `verify/`, helpers in `utils/`.
- Do not run training, rendering, CUDA extension builds, or LPIPS evaluation unless explicitly requested.
- Use lightweight checks by default: Python compile, shell syntax, and `--help` for commands that do not require heavy dependencies.
- Preserve user-generated artifacts and dirty worktree changes. Do not delete reports, submissions, data, checkpoints, or third-party repos unless explicitly requested.
- Keep documentation aligned with actual paths after every file move.
- Treat `10_smoke.sh` as a one-scene error check only. Method selection requires complete public-set runs and eval; private runs are submission-only.
- Keep render outputs in `outputs/smoke/<run_id>`, `outputs/public/<run_id>`, or `outputs/private/<run_id>`. Only verified private outputs should be zipped into `submissions/`.
- Write selection/ranking outputs under `reports/experiments/selections/`, not directly under `reports/experiments/`.

## Common Lightweight Checks

```bash
PYTHONDONTWRITEBYTECODE=1 rtk python -m py_compile src/*.py src/method_runners/*.py utils/*.py eval/*.py verify/*.py
rtk bash -n runs/commands/*.sh
PYTHONDONTWRITEBYTECODE=1 rtk python utils/generate_experiment_commands.py --help
PYTHONDONTWRITEBYTECODE=1 rtk python utils/make_submission_zip.py --help
```

If a command fails because `Pillow`, `pandas`, `torch`, `gsplat`, or CUDA-specific packages are missing, report it as an environment/dependency limitation instead of treating it as a code failure.

## Main Commands

```bash
python utils/inspect_dataset.py --json-out reports/dataset.json
python utils/generate_sam3_masks.py --split-root data/round1/phase1/public_set --output-dir masks
python src/method_runners/run_3dgs_scene.py --scene-dir data/round1/phase1/public_set/HCM0181 --output-dir outputs/smoke/3dgs_f2_10k
python eval/evaluate_public.py --public-root data/round1/phase1/public_set --pred-dir outputs/public/3dgs_f2_10k --with-lpips
python verify/verify_submission.py --split-root data/round1/phase1/private_set1 --submission-dir outputs/private/3dgs_f2_10k
python utils/make_submission_zip.py --submission-dir outputs/private/3dgs_f2_10k --zip submissions/submission_3dgs_f2_10k.zip --overwrite
```

## CodeGraph

This repo has a `.codegraph/` index. Use CodeGraph first for source exploration when available. After large moves or renames, reindex before relying on stale symbol paths.

<!-- headroom:rtk-instructions -->
## RTK (Rust Token Killer) - Token-Optimized Commands

When running shell commands, always prefix with `rtk`. This reduces context usage with no behavior change when a filter is available.

```bash
rtk git status          rtk git diff            rtk git log
rtk ls <path>           rtk read <file>         rtk grep <pattern>
rtk pytest tests/       rtk ruff check          rtk python -m py_compile
rtk bash -n runs/commands/*.sh   rtk pip list            rtk env
```

Rules:

- In command chains, prefix each segment: `rtk git status && rtk git diff`.
- For debugging a filtering issue, use the raw command once.
- `rtk proxy <cmd>` runs a command without filtering while tracking usage.
<!-- /headroom:rtk-instructions -->
