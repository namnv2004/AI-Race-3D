# Third-Party Gaussian Splatting Repos

`third_party/` is intentionally gitignored because external repositories are large and carry their own licenses, submodules, and CUDA build requirements.

The registry in `configs/third_party_repos.json` is kept as documentation for the repos used or evaluated by this project. Clone, patch, and install them with your existing setup process.

## Current Pipeline Methods

- `3dgs`: implemented with the `gsplat` package in `src/method_runners/run_3dgs_scene.py`.
- `mip`: uses `third_party/mip-splatting` via `src/method_runners/run_mip_splatting_scene.py`.
- `2dgs`: uses `third_party/2d-gaussian-splatting` via `src/method_runners/run_2dgs_scene.py`.

## SOTA Candidates To Port Next

- `Scaffold-GS`: quality/speed/storage tradeoffs.
- `Octree-GS`: scene scale or render LOD experiments.
- `Hierarchical 3D Gaussians`: official-group large-scene variant.
- `Gaussian Opacity Fields`: geometry/surface consistency candidate.
- `SuGaR`: surface-aligned Gaussian refinement/extraction.
- `3DGS-MCMC`: alternative densification/training strategy.
- `Gaussian Splatting Lightning`: experimentation harness for multiple 3DGS variants.

Before wiring any candidate into automation, add a wrapper that exports this competition's COLMAP/test-pose format, renders exactly the required images, and passes `verify/verify_submission.py`.
