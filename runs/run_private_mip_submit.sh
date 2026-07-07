#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/micace/Project/AI-3D/.venv/bin:$PATH"
export LD_LIBRARY_PATH="/home/micace/Project/AI-3D/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

SCENES="HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437"
OUTPUT_DIR="outputs/private_mip_30k_r2_k01_jpeg100"
ZIP_PATH="submissions/submission_mip_30k_r2_k01_jpeg100.zip"
MODEL_ROOT="checkpoints/experiments/private/mip_30k_r2_k01_jpeg100"
WORKSPACE_DIR="workspace/experiments/mip_splatting/private/mip_30k_r2_k01_jpeg100"

mkdir -p reports/experiments submissions "$OUTPUT_DIR"

echo "Starting private Mip 30k r2 k0.1 HQ render at $(date -Is)"
for scene in $SCENES; do
    echo "Training/rendering Mip private $scene at $(date -Is)"
    python src/method_runners/run_mip_splatting_scene.py \
        --scene-dir "data/round1/phase1/private_set1/$scene" \
        --output-dir "$OUTPUT_DIR" \
        --model-root "$MODEL_ROOT" \
        --workspace-dir "$WORKSPACE_DIR" \
        --iterations 30000 \
        --train-resolution 2 \
        --kernel-size 0.1 \
        --lambda-l1 0.8 \
        --lambda-ssim 0.2 \
        --lambda-lpips 0.0 \
        --overwrite > "reports/experiments/log_private_mip_${scene}_30k.txt" 2>&1
done

echo "Verifying private Mip folder at $(date -Is)"
python verify/verify_submission.py \
    --split-root data/round1/phase1/private_set1 \
    --submission-dir "$OUTPUT_DIR" > reports/experiments/log_private_mip_verify_folder.txt 2>&1

echo "Creating private Mip zip at $(date -Is)"
python utils/make_submission_zip.py \
    --submission-dir "$OUTPUT_DIR" \
    --zip "$ZIP_PATH" \
    --overwrite > reports/experiments/log_private_mip_zip.txt 2>&1

echo "Verifying private Mip zip at $(date -Is)"
python verify/verify_zip.py \
    --split-root data/round1/phase1/private_set1 \
    --zip "$ZIP_PATH" > reports/experiments/log_private_mip_verify_zip.txt 2>&1

echo "Legacy external file transfer omitted at $(date -Is)" > reports/experiments/log_private_mip_external_transfer_skipped.txt

echo "DONE $(date -Is)" > reports/experiments/private_mip_submit_done.txt
