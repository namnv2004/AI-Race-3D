# Submission Results

Tracking file for actual submitted zips and their leaderboard results.

Only rows in **Submitted Files** count as real submission scores. Local public validation scores are for reference only and must not be treated as leaderboard results.

## Submitted Files

| Date | Submission Zip | Method / Config | Leaderboard Score | PSNR | SSIM | LPIPS | Scenes | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-07-02 | `submissions/submission_round1_3dgs10k.zip` | `3dgs_10k` first generated 3DGS submission | 52.88530 | 19.602065 | 54.7067 | 38.2198 | 8/8 | User-reported leaderboard result for the first 3DGS zip submitted around 2026-07-02 19:49. File mtime was 2026-07-02 18:10:04 +0700; internal zip entries around 17:49-18:09. |
| 2026-07-04 | `submissions/submission_best_single_method.zip` | `3dgs_10k_f2` (`factor=2`, `steps=10000`, JPEG q95 before HQ patch) | 52.29420 | 19.232366 | 53.2745 | 38.0691 | 8/8 | First valid best-single-method submission. |
| 2026-07-06 | `submissions/submission_auto_best_hq_target349.zip` | `2dgs_30k_default_jpeg100` final auto-best private render (`iterations=30000`, target 349.3M JPEG q97/q98 mix) | 61.98720 | 21.701095 | 66.9104 | 27.7664 | 8/8 | User-reported leaderboard result; `num_scenes=8`, `matched_scenes=8`. Current best actual submission. |

## Local Public Validation Reference Only

These scores are local public-set validation scores from `reports/experiments/eval/public/*.json`. They are not submitted scores.

| Rank | Candidate | Local Public Score | Local PSNR | Local SSIM | Local LPIPS | Complete Public | Eval JSON | Submission Status |
|---:|---|---:|---:|---:|---:|---|---|---|
| 1 | `2dgs_30k_default_jpeg100` | 0.804283 | 22.767338 | 0.768822 | 0.135091 | yes | `reports/experiments/eval/public/2dgs_30k_default_jpeg100.json` | submitted as `submission_auto_best_hq_target349.zip` |
| 2 | `mip_30k_r2_k01_jpeg100` | 0.763054 | 22.880776 | 0.741951 | 0.220847 | yes | `reports/experiments/eval/public/mip_30k_r2_k01_jpeg100.json` | private trial render in progress |
| 3 | `3dgs_10k_f2` | 0.687720 | 20.004826 | 0.612780 | 0.240405 | yes | `reports/experiments/eval/public/3dgs_10k_f2.json` | submitted |
| 4 | `3dgs_30k_f2_jpeg100` | 0.687029 | 19.511557 | 0.594089 | 0.215784 | yes | `reports/experiments/eval/public/3dgs_30k_f2_jpeg100.json` | not submitted |
| 5 | `nearest_pose` | 0.300076 | 10.337006 | 0.134873 | 0.609390 | yes | `reports/experiments/eval/public/nearest_pose.json` | not submitted |

## Quick Grid Reference Only

Quick grid scores were computed on two public scenes only (`HCM0181`, `hcm0031`) and were used only for early triage.

| Rank | Candidate | Quick Score | Quick PSNR | Quick SSIM | Quick LPIPS | Scenes | Notes |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | `3dgs_10k_f2` | 0.679883 | 19.782311 | 0.599154 | 0.244216 | 2/5 | Winner of the early quick grid. |
| 2 | `mip_10k_r4_k01` | 0.618235 | 21.350285 | 0.611724 | 0.446963 | 2/5 | Higher PSNR/SSIM than 3DGS quick, but LPIPS was much worse. |
| invalid | `2dgs_10k_default` | null | null | null | null | 0/5 | Quick run produced no valid public renders. |

## Current Recommendation

Current best actual submission is `submissions/submission_auto_best_hq_target349.zip` using `2dgs_30k_default_jpeg100`, with leaderboard score `61.98720`.

Render and submit `mip_30k_r2_k01_jpeg100` as a private trial next because its local public score was the closest alternative to 2DGS. Do not replace the current best until an actual leaderboard score is reported.

## Current Work In Progress

| Started | Candidate | Output Zip | Status | Notes |
|---|---|---|---|---|
| 2026-07-06 | `mip_30k_r2_k01_jpeg100` | `submissions/submission_mip_30k_r2_k01_jpeg100_target349.zip` | running | Private render trial launched with `run_private_mip_30k_r2_k01_target349.sh`. It trains/renders all 8 private scenes, recompresses to target size, verifies folder and zip, then sends to `ironcat:`. No leaderboard result yet. |

## Completed Build Notes

| Started | Script / Build | Status | Output Zip | Notes |
|---|---|---|---|---|
| 2026-07-04 | `run_auto_final_after_public_hq.sh` plus resume/manual target-size packaging | complete | `submissions/submission_auto_best_hq_target349.zip` | Selected `2dgs_30k_default_jpeg100`, rendered all 8 private scenes, verified 434 images, repacked to target 349.3M, sent to `ironcat:` via Tailscale, then submitted to leaderboard. |
| 2026-07-04 | `submissions/submission_mip_30k_r2_k01_jpeg100.zip` | cancelled | none | Earlier Mip private build was cancelled before completion per user request while waiting for all public families. |

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
