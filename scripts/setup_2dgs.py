#!/usr/bin/env python3
"""Clone and install 2D Gaussian Splatting (2DGS) CUDA dependencies.

2DGS uses two CUDA extensions: diff-surfel-rasterization and simple-knn.
The simple-knn code is the same as Mip-Splatting's, so we can reuse the
patched version.

Examples
--------
    python scripts/setup_2dgs.py
    python scripts/setup_2dgs.py --no-install
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from setup_mip_splatting import (
    clone_repo,
    patch_simple_knn,
    run,
)


def init_submodules(twodgs_root: Path) -> None:
    submodules = twodgs_root / "submodules"
    if not (submodules / "diff-surfel-rasterization" / "diff_surfel_rasterization").exists():
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=twodgs_root)


def install_2dgs_extensions(twodgs_root: Path) -> None:
    diff_surfel = twodgs_root / "submodules" / "diff-surfel-rasterization"
    simple_knn = twodgs_root / "submodules" / "simple-knn"
    run([sys.executable, "-m", "pip", "install", "--no-build-isolation", str(diff_surfel), str(simple_knn)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone and install 2DGS CUDA dependencies.")
    parser.add_argument("--twodgs-root", type=Path, default=Path("third_party/2d-gaussian-splatting"))
    parser.add_argument("--repo-url", default="https://github.com/hbb1/2d-gaussian-splatting.git")
    parser.add_argument("--no-install", action="store_true")
    args = parser.parse_args()

    clone_repo(args.repo_url, args.twodgs_root)
    init_submodules(args.twodgs_root)
    if (args.twodgs_root / "submodules" / "simple-knn" / "simple_knn.cu").exists():
        patch_simple_knn(args.twodgs_root)
    if not args.no_install:
        install_2dgs_extensions(args.twodgs_root)


if __name__ == "__main__":
    main()
