# Runs

`runs/commands/` contains the canonical generated workflows from `configs/gs_runs.json`.

The `.sh` files directly in this folder are historical high-quality runs preserved for analysis and possible resume. They came from the previous split shell workflow and should be reviewed before execution.

## Historical Run Records

| Script | Purpose | Start Time | End Time | Notes |
| --- | --- | --- | --- | --- |
| `run_hq_combined.sh` | Consolidated public HQ benchmark, winner selection, private render, verify, zip | unknown | Public 2DGS resume marker: `2026-07-05T04:34:27+07:00` | Winner marker from old run: `2dgs_30k_default_jpeg100` |
| `run_private_mip_submit.sh` | Private Mip-Splatting 30k r2 k0.1 render and zip | unknown | unknown | Original script did not persist tracked start/end markers |
| `run_private_mip_30k_r2_k01_target349.sh` | Private Mip-Splatting 30k r2 k0.1 render, recompress near 349 MiB target, verify, zip | unknown | unknown | Original script did not persist tracked start/end markers |

Known timestamp data is limited to tracked marker files. Missing start/end fields are intentionally marked `unknown` rather than inferred from untracked artifacts.

## Shared Setup

| Setting | Value |
| --- | --- |
| Python env path added to `PATH` | `/home/micace/Project/AI-3D/.venv/bin` |
| Torch library path added to `LD_LIBRARY_PATH` | `/home/micace/Project/AI-3D/.venv/lib/python3.12/site-packages/torch/lib` |
| Public root | `data/round1/phase1/public_set` |
| Private root | `data/round1/phase1/private_set1` |
| Report/log root | `reports/experiments` |
| Public scenes | `HCM0181`, `HCM0193`, `HCM0204`, `hcm0031`, `hcm0034` |
| Private scenes | `HCM0249`, `HCM0254`, `HCM0276`, `HCM1439`, `HNI0131`, `HNI0265`, `HNI0366`, `HNI0437` |
| External transfer | Omitted; no Tailscale/Ironcat step is used |

## Method Parameters

| Method | Runner | Key Parameters | Outputs |
| --- | --- | --- | --- |
| 3DGS public HQ | `src/method_runners/run_3dgs_scene.py` | `--factor 2`, `--steps 30000`, `--lambda-l1 0.8`, `--lambda-ssim 0.2`, `--lambda-lpips 0.0`, `--seed 42`, `--overwrite` | `outputs/experiments/public/3dgs_30k_f2_jpeg100` |
| 3DGS private fallback | `src/method_runners/run_3dgs_scene.py` | `--factor 2`, `--steps 10000` or `30000`, `--lambda-l1 0.8`, `--lambda-ssim 0.2`, `--lambda-lpips 0.0`, `--seed 42`, `--overwrite` | `outputs/private_auto_best_hq` |
| Mip-Splatting public HQ | `src/method_runners/run_mip_splatting_scene.py` | `--iterations 30000`, `--train-resolution 2`, `--kernel-size 0.1`, `--lambda-l1 0.8`, `--lambda-ssim 0.2`, `--lambda-lpips 0.0`, `--overwrite` | `outputs/experiments/public/mip_30k_r2_k01_jpeg100` |
| Mip-Splatting private HQ | `src/method_runners/run_mip_splatting_scene.py` | `--iterations 30000`, `--train-resolution 2`, `--kernel-size 0.1`, `--lambda-l1 0.8`, `--lambda-ssim 0.2`, `--lambda-lpips 0.0`, `--overwrite` | `outputs/private_mip_30k_r2_k01_jpeg100` |
| Mip-Splatting target349 | `src/method_runners/run_mip_splatting_scene.py` and `utils/recompress_submission_jpegs.py` | render Q100 source, recompress with `--target-mib 348.5`, `--base-quality 97`, `--max-quality 98`, `--min-quality 95`, max zip `< 350 MiB` | `outputs/private_mip_30k_r2_k01_jpeg100_target349` |
| 2DGS public HQ | `src/method_runners/run_2dgs_scene.py` | `--iterations 30000`, `--resolution -1`, `--lambda-dssim 0.2`, `--lambda-dist 0.0`, `--lambda-normal 0.05`, `--overwrite` | `outputs/experiments/public/2dgs_30k_default_jpeg100` |
| 2DGS private resume | `src/method_runners/run_2dgs_scene.py` | `--iterations 30000`, `--resolution -1`, `--lambda-dssim 0.2`, `--lambda-dist 0.0`, `--lambda-normal 0.05`, `--port 6009+` | `outputs/private_auto_best_hq` |

## Timing Markers

| Marker | Meaning |
| --- | --- |
| `reports/experiments/2dgs_public_hq_done.txt` | Public 2DGS HQ completion marker; old tracked value was `DONE 2026-07-05T04:34:27+07:00` |
| `reports/experiments/hq_batch_status.txt` | Old 3DGS public HQ completion marker; old tracked value was `DONE` without timestamp |
| `reports/experiments/final_hq_winner.txt` | Old selected HQ winner; old tracked value was `2dgs_30k_default_jpeg100` |
| `reports/experiments/auto_final_hq_done.txt` | Final private HQ done marker written by `run_hq_combined.sh` |
| `reports/experiments/private_mip_submit_done.txt` | Final marker written by `run_private_mip_submit.sh` |
| `reports/experiments/private_mip_30k_r2_k01_target349_done.txt` | Final marker written by `run_private_mip_30k_r2_k01_target349.sh` |

## Script Phases

`run_hq_combined.sh` supports:

- `all`: full historical public HQ flow, select winner, render private, verify, zip.
- `resume-public-hq`: resume interrupted 2DGS public HQ, select winner, render private, verify, zip.
- `wait-and-private`: wait for public done marker, select winner, render private, verify, zip.
- `select-and-private`: select public winner from existing eval JSON, render private, verify, zip.
- `private-2dgs-resume`: resume interrupted private 2DGS winner render, then verify and zip.
- `public-3dgs`: run only public 3DGS HQ batch and eval.
- `public-mip`: run only public Mip HQ batch and eval.
- `public-2dgs-full`: run only full public 2DGS HQ batch and eval.
- `public-2dgs-resume`: run only resumed public 2DGS HQ scenes and eval.
