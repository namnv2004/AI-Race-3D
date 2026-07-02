from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def clone_repo(repo_url: str, mip_root: Path) -> None:
    if (mip_root / "train.py").exists():
        print(f"Mip-Splatting already exists: {mip_root}")
        return
    mip_root.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", repo_url, str(mip_root)])


def patch_simple_knn(mip_root: Path) -> None:
    simple_knn = mip_root / "submodules" / "simple-knn" / "simple_knn.cu"
    if not simple_knn.exists():
        raise FileNotFoundError(f"Missing simple_knn.cu: {simple_knn}")
    text = simple_knn.read_text(encoding="utf-8")
    if "#include <float.h>" in text:
        print("simple-knn FLT_MAX patch already present")
        return
    marker = "#include <cub/device/device_radix_sort.cuh>\n"
    if marker not in text:
        raise RuntimeError(f"Cannot find patch marker in {simple_knn}")
    simple_knn.write_text(text.replace(marker, marker + "#include <float.h>\n", 1), encoding="utf-8")
    print(f"patched {simple_knn}")


def install_extensions(mip_root: Path) -> None:
    diff_raster = mip_root / "submodules" / "diff-gaussian-rasterization"
    simple_knn = mip_root / "submodules" / "simple-knn"
    run([sys.executable, "-m", "pip", "install", "--no-build-isolation", str(diff_raster), str(simple_knn)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone and install Mip-Splatting CUDA dependencies.")
    parser.add_argument("--mip-root", type=Path, default=Path("third_party/mip-splatting"))
    parser.add_argument("--repo-url", default="https://github.com/autonomousvision/mip-splatting.git")
    parser.add_argument("--no-install", action="store_true")
    args = parser.parse_args()

    clone_repo(args.repo_url, args.mip_root)
    patch_simple_knn(args.mip_root)
    if not args.no_install:
        install_extensions(args.mip_root)


if __name__ == "__main__":
    main()
