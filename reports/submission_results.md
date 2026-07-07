# Submission Results

Tracking file for actual submitted zips, local public validation scores, and private packaging status.

Only rows in **Submitted Files** count as real leaderboard scores. Local public validation scores are reference only.

## ID Convention

- Run ID: `<method>_<variant>_<iterations>`, for example `3dgs_f2_10k`, `mip_r2_k01_30k`, `2dgs_default_30k`.
- Submission ID: `sub_<run_id>[_packaging]`, where `t349` means target-size packaging around 349 MiB.
- Zip Path records the actual file path, including legacy zip names when those were already submitted.

## Submitted Files

| Date | Submission ID | Zip Path | Run ID | Method | Variant | Leaderboard Score | PSNR | SSIM | LPIPS | Scenes | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-07-02 | `sub_3dgs_10k_first` | `submissions/submission_round1_3dgs10k.zip` | `3dgs_10k_legacy` | 3DGS | first generated 10k submission | 52.88530 | 19.602065 | 54.7067 | 38.2198 | 8/8 | User-reported leaderboard result for first generated 3DGS zip. |
| 2026-07-04 | `sub_3dgs_f2_10k_q95` | `submissions/submission_best_single_method.zip` | `3dgs_f2_10k` | 3DGS | factor=2, 10k, JPEG q95 | 52.29420 | 19.232366 | 53.2745 | 38.0691 | 8/8 | First valid single-method submission. |
| 2026-07-06 | `sub_2dgs_default_30k_t349` | `submissions/submission_auto_best_hq_target349.zip` | `2dgs_default_30k` | 2DGS | default, 30k, JPEG q97/q98, target 349.3 MiB | 61.98720 | 21.701095 | 66.9104 | 27.7664 | 8/8 | Current best actual submission; `num_scenes=8`, `matched_scenes=8`. |

## Local Public Validation Reference Only

These scores are local public-set validation scores from `reports/experiments/eval/public/*.json`, reweighted to the observed leaderboard scale with `PSNR_MAX=50`. They are not submitted scores.

| Rank | Run ID | Method | Variant | Local Public Score | Local PSNR | Local SSIM | Local LPIPS | Complete Public | Eval JSON | Submission Status |
|---:|---|---|---|---:|---:|---:|---:|---|---|---|
| 1 | `2dgs_default_30k` | 2DGS | default, 30k | 0.713214 | 22.767338 | 0.768822 | 0.135091 | yes | `reports/experiments/eval/public/2dgs_30k_default_jpeg100.json` | submitted as `sub_2dgs_default_30k_t349` |
| 2 | `mip_r2_k01_30k` | Mip-Splatting | resolution=2, kernel=0.1, 30k | 0.671531 | 22.880776 | 0.741951 | 0.220847 | yes | `reports/experiments/eval/public/mip_30k_r2_k01_jpeg100.json` | private trial candidate |
| 3 | `3dgs_f2_30k` | 3DGS | factor=2, 30k, JPEG q100 output | 0.608982 | 19.511557 | 0.594089 | 0.215784 | yes | `reports/experiments/eval/public/3dgs_30k_f2_jpeg100.json` | not submitted |
| 4 | `3dgs_f2_10k` | 3DGS | factor=2, 10k | 0.607701 | 20.004826 | 0.612780 | 0.240405 | yes | `reports/experiments/eval/public/3dgs_10k_f2.json` | submitted as `sub_3dgs_f2_10k_q95` |
| 5 | `nearest_pose` | Nearest baseline | nearest train-view baseline | 0.258728 | 10.337006 | 0.134873 | 0.609390 | yes | `reports/experiments/eval/public/nearest_pose.json` | not submitted |

## Public Evaluation Audit

- No training/render code path was found that reads `public_set/*/test/images`; those files are only used by `eval/evaluate_public.py` for local scoring.
- The old public scores were inflated because local eval used `PSNR_MAX=30`. The private leaderboard result matches `PSNR_MAX=50`, so local public JSON and this table were reweighted to `PSNR_MAX=50`.
- `2dgs_default_30k` changes from `0.804283` to `0.713214` on public. The remaining gap to private `0.619872` is about 9.3 points, not 18.4 points.
- Future public eval now records completeness and fails by default if predictions or GT files are missing, unless `--allow-missing` is passed intentionally.

## Current Recommendation

Current best actual submission is `sub_2dgs_default_30k_t349` using `2dgs_default_30k`, with leaderboard score `61.98720`.

Render and submit `mip_r2_k01_30k` as a private trial next because its local public score was the closest alternative to 2DGS. Do not replace the current best until an actual leaderboard score is reported.

## Current Work In Progress

| Started | Submission ID | Run ID | Method | Variant | Zip Path | Status | Notes |
|---|---|---|---|---|---|---|---|
| 2026-07-06 | `sub_mip_r2_k01_30k_t349` | `mip_r2_k01_30k` | Mip-Splatting | resolution=2, kernel=0.1, 30k, target 349 MiB | `submissions/submission_mip_r2_k01_30k_t349.zip` | running | Legacy private Mip target-size trial; canonical private workflow is now `runs/commands/30_private.sh`. No leaderboard result yet. |

## Completed Build Notes

| Started | Build ID | Run ID | Status | Zip Path | Notes |
|---|---|---|---|---|---|
| 2026-07-04 | `sub_2dgs_default_30k_t349` | `2dgs_default_30k` | complete | `submissions/submission_auto_best_hq_target349.zip` | Rendered all 8 private scenes, verified 434 images, repacked to target 349.3 MiB, then submitted to leaderboard. |
| 2026-07-04 | `sub_mip_r2_k01_30k_q100` | `mip_r2_k01_30k` | cancelled | none | Earlier Mip private build was cancelled before completion while waiting for all public families. |

## Private Scene Counts

| Scene | Expected Images |
|---|---:|
| `HCM0249` | 60 |
| `HCM0254` | 60 |
| `HCM0276` | 60 |
| `HCM1439` | 26 |
| `HNI0131` | 60 |
| `HNI0265` | 52 |
| `HNI0366` | 60 |
| `HNI0437` | 56 |
| **Total** | **434** |
