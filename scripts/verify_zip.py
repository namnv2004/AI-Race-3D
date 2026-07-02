from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

from verify_submission import main as verify_submission_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and verify a submission zip.")
    parser.add_argument("--split-root", type=Path, default=Path("data/round1/phase1/private_set1"))
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()

    if not args.zip.exists():
        raise SystemExit(f"Zip file not found: {args.zip}")
    with tempfile.TemporaryDirectory(prefix="verify_submission_zip_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(args.zip) as archive:
            archive.extractall(tmp_path)
        import sys

        previous_argv = sys.argv
        try:
            sys.argv = [
                "verify_submission.py",
                "--split-root",
                str(args.split_root),
                "--submission-dir",
                str(tmp_path),
            ]
            verify_submission_main()
        finally:
            sys.argv = previous_argv


if __name__ == "__main__":
    main()
