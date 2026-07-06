#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/micace/Project/AI-3D/.venv/bin:$PATH"
export LD_LIBRARY_PATH="/home/micace/Project/AI-3D/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

SCENES="HCM0193 HCM0204 hcm0031 hcm0034"
OUTPUT_DIR="outputs/experiments/public/2dgs_30k_default_jpeg100"

mkdir -p reports/experiments/eval/public "$OUTPUT_DIR"

echo "Resuming 2DGS 30k public HQ at $(date -Is)"
for scene in $SCENES; do
    echo "Training/rendering 2DGS public $scene at $(date -Is)"
    python scripts/run_2dgs_scene.py \
        --scene-dir "data/round1/phase1/public_set/$scene" \
        --output-dir "$OUTPUT_DIR" \
        --iterations 30000 \
        --resolution -1 \
        --lambda-dssim 0.2 \
        --lambda-dist 0.0 \
        --lambda-normal 0.05 \
        --overwrite > "reports/experiments/log_2dgs_${scene}_30k_resume.txt" 2>&1
done

echo "Evaluating 2DGS 30k public HQ at $(date -Is)"
python scripts/evaluate_public.py \
    --public-root data/round1/phase1/public_set \
    --pred-dir "$OUTPUT_DIR" \
    --scenes HCM0181 HCM0193 HCM0204 hcm0031 hcm0034 \
    --json-out reports/experiments/eval/public/2dgs_30k_default_jpeg100.json \
    --with-lpips \
    --lpips-net alex \
    --lpips-device cuda > reports/experiments/log_eval_2dgs_30k_resume.txt 2>&1

echo "DONE $(date -Is)" > reports/experiments/2dgs_public_hq_done.txt
