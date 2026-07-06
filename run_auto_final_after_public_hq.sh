#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/micace/Project/AI-3D/.venv/bin:$PATH"
export LD_LIBRARY_PATH="/home/micace/Project/AI-3D/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PUBLIC_DONE="reports/experiments/2dgs_public_hq_done.txt"
SELECTION_JSON="reports/experiments/final_hq_selection.json"
WINNER_TXT="reports/experiments/final_hq_winner.txt"
OUTPUT_DIR="outputs/private_auto_best_hq"
ZIP_PATH="submissions/submission_auto_best_hq.zip"
PRIVATE_ROOT="data/round1/phase1/private_set1"
SCENES="HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437"

mkdir -p reports/experiments submissions "$OUTPUT_DIR"
echo "$$" > reports/experiments/auto_final_after_public_hq.pid
echo "Auto final job started at $(date -Is)"

while [ ! -f "$PUBLIC_DONE" ]; do
    if ! pgrep -f 'run_resume_2dgs_public_hq|run_2dgs_scene.py|2d-gaussian-splatting.*train.py' >/dev/null; then
        echo "ERROR: 2DGS public job is not running and $PUBLIC_DONE is missing at $(date -Is)" >&2
        exit 1
    fi
    echo "Waiting for 2DGS public HQ to finish at $(date -Is)"
    sleep 300
done

echo "2DGS public HQ finished at $(date -Is). Selecting winner."
python - <<'PY'
import json
import math
from pathlib import Path

eval_dir = Path("reports/experiments/eval/public")
public_scenes = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]
candidates = [
    "3dgs_10k_f2",
    "3dgs_30k_f2_jpeg100",
    "mip_30k_r2_k01_jpeg100",
    "2dgs_30k_default_jpeg100",
]

def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(number) or math.isinf(number))

ranking = []
for candidate in candidates:
    path = eval_dir / f"{candidate}.json"
    if not path.exists():
        ranking.append({"candidate": candidate, "eligible": False, "reason": "missing eval json", "eval_json": str(path)})
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    scored = {
        str(item.get("scene"))
        for item in data.get("scenes", [])
        if not item.get("missing") and finite(item.get("score"))
    }
    missing = [scene for scene in public_scenes if scene not in scored]
    eligible = not missing and finite(data.get("mean_score"))
    ranking.append(
        {
            "candidate": candidate,
            "eligible": eligible,
            "missing_scenes": missing,
            "mean_score": data.get("mean_score"),
            "mean_psnr": data.get("mean_psnr"),
            "mean_ssim": data.get("mean_ssim"),
            "mean_lpips": data.get("mean_lpips"),
            "eval_json": str(path),
        }
    )

def rank_key(item):
    return (
        float(item.get("mean_score") or -1),
        -float(item.get("mean_lpips") if item.get("mean_lpips") is not None else 999),
        float(item.get("mean_ssim") or -1),
        float(item.get("mean_psnr") or -1),
    )

eligible = [item for item in ranking if item.get("eligible")]
if not eligible:
    raise SystemExit("No eligible complete public candidate found")
eligible.sort(key=rank_key, reverse=True)
ranking.sort(key=lambda item: rank_key(item) if item.get("eligible") else (-1, -999, -1, -1), reverse=True)
result = {
    "selected_at": None,
    "selection_rule": "max mean_score; tie-break lower LPIPS, higher SSIM, higher PSNR",
    "best_candidate": eligible[0]["candidate"],
    "best": eligible[0],
    "ranking": ranking,
}
Path("reports/experiments/final_hq_selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
Path("reports/experiments/final_hq_winner.txt").write_text(eligible[0]["candidate"] + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY

WINNER="$(tr -d '[:space:]' < "$WINNER_TXT")"
echo "Winner: $WINNER"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

case "$WINNER" in
    3dgs_10k_f2)
        for scene in $SCENES; do
            echo "Private render $WINNER $scene at $(date -Is)"
            python scripts/train_gs_scene.py \
                --scene-dir "$PRIVATE_ROOT/$scene" \
                --output-dir "$OUTPUT_DIR" \
                --factor 2 \
                --steps 10000 \
                --lambda-l1 0.8 \
                --lambda-ssim 0.2 \
                --lambda-lpips 0.0 \
                --seed 42 \
                --overwrite > "reports/experiments/log_auto_private_${WINNER}_${scene}.txt" 2>&1
        done
        ;;
    3dgs_30k_f2_jpeg100)
        for scene in $SCENES; do
            echo "Private render $WINNER $scene at $(date -Is)"
            python scripts/train_gs_scene.py \
                --scene-dir "$PRIVATE_ROOT/$scene" \
                --output-dir "$OUTPUT_DIR" \
                --factor 2 \
                --steps 30000 \
                --lambda-l1 0.8 \
                --lambda-ssim 0.2 \
                --lambda-lpips 0.0 \
                --seed 42 \
                --overwrite > "reports/experiments/log_auto_private_${WINNER}_${scene}.txt" 2>&1
        done
        ;;
    mip_30k_r2_k01_jpeg100)
        for scene in $SCENES; do
            echo "Private render $WINNER $scene at $(date -Is)"
            python scripts/run_mip_splatting_scene.py \
                --scene-dir "$PRIVATE_ROOT/$scene" \
                --output-dir "$OUTPUT_DIR" \
                --model-root "checkpoints/experiments/private_auto_best_hq/$WINNER" \
                --workspace-dir "workspace/experiments/mip_splatting/private_auto_best_hq/$WINNER" \
                --iterations 30000 \
                --train-resolution 2 \
                --kernel-size 0.1 \
                --lambda-l1 0.8 \
                --lambda-ssim 0.2 \
                --lambda-lpips 0.0 \
                --overwrite > "reports/experiments/log_auto_private_${WINNER}_${scene}.txt" 2>&1
        done
        ;;
    2dgs_30k_default_jpeg100)
        for scene in $SCENES; do
            echo "Private render $WINNER $scene at $(date -Is)"
            python scripts/run_2dgs_scene.py \
                --scene-dir "$PRIVATE_ROOT/$scene" \
                --output-dir "$OUTPUT_DIR" \
                --model-root "checkpoints/experiments/private_auto_best_hq/$WINNER" \
                --iterations 30000 \
                --resolution -1 \
                --lambda-dssim 0.2 \
                --lambda-dist 0.0 \
                --lambda-normal 0.05 \
                --overwrite > "reports/experiments/log_auto_private_${WINNER}_${scene}.txt" 2>&1
        done
        ;;
    *)
        echo "ERROR: unsupported winner $WINNER" >&2
        exit 1
        ;;
esac

echo "Verifying final folder at $(date -Is)"
python scripts/verify_submission.py \
    --split-root "$PRIVATE_ROOT" \
    --submission-dir "$OUTPUT_DIR" > reports/experiments/log_auto_final_verify_folder.txt 2>&1

echo "Creating final zip at $(date -Is)"
python scripts/make_submission_zip.py \
    --submission-dir "$OUTPUT_DIR" \
    --zip "$ZIP_PATH" \
    --overwrite > reports/experiments/log_auto_final_zip.txt 2>&1

echo "Verifying final zip at $(date -Is)"
python scripts/verify_zip.py \
    --split-root "$PRIVATE_ROOT" \
    --zip "$ZIP_PATH" > reports/experiments/log_auto_final_verify_zip.txt 2>&1

echo "Sending final zip to ironcat at $(date -Is)"
tailscale file cp "$ZIP_PATH" ironcat: > reports/experiments/log_auto_final_taildrop.txt 2>&1

{
    echo "DONE $(date -Is)"
    echo "winner=$WINNER"
    echo "zip=$ZIP_PATH"
} > reports/experiments/auto_final_hq_done.txt
