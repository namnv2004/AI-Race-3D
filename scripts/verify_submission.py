from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def verify_scene(scene_dir: Path, submission_scene_dir: Path) -> list[str]:
    errors: list[str] = []
    test_csv = scene_dir / "test" / "test_poses.csv"
    test_df = pd.read_csv(test_csv)
    for row_index, row in test_df.iterrows():
        image_name = str(row["image_name"])
        expected_size = (int(row["width"]), int(row["height"]))
        output_file = submission_scene_dir / image_name
        if not output_file.exists():
            errors.append(f"{scene_dir.name}: missing {image_name} at row {row_index}")
            continue
        try:
            with Image.open(output_file) as image:
                if image.size != expected_size:
                    errors.append(
                        f"{scene_dir.name}: wrong size for {image_name}: got {image.size}, expected {expected_size}"
                    )
                image.verify()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{scene_dir.name}: cannot read {image_name}: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify submission directory before zipping.")
    parser.add_argument("--split-root", type=Path, default=Path("data/round1/phase1/private_set1"))
    parser.add_argument("--submission-dir", type=Path, default=Path("outputs/submission_round1_nearest"))
    args = parser.parse_args()

    errors: list[str] = []
    total_rows = 0
    scene_count = 0
    for scene_dir in sorted(item for item in args.split_root.iterdir() if item.is_dir()):
        test_csv = scene_dir / "test" / "test_poses.csv"
        if not test_csv.exists():
            continue
        scene_count += 1
        total_rows += len(pd.read_csv(test_csv))
        errors.extend(verify_scene(scene_dir, args.submission_dir / scene_dir.name))

    if errors:
        print(f"VERIFY FAILED: {len(errors)} errors")
        for error in errors[:100]:
            print(error)
        if len(errors) > 100:
            print(f"... {len(errors) - 100} more errors")
        raise SystemExit(1)

    print(f"VERIFY PASSED: {scene_count} scenes, {total_rows} images, all files present and sizes match CSV.")


if __name__ == "__main__":
    main()
