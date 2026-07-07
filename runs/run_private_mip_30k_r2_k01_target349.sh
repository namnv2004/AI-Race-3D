#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/micace/Project/AI-3D/.venv/bin:$PATH"
export LD_LIBRARY_PATH="/home/micace/Project/AI-3D/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

SCENES="HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437"
SOURCE_OUTPUT_DIR="outputs/private_mip_30k_r2_k01_jpeg100_q100"
FINAL_OUTPUT_DIR="outputs/private_mip_30k_r2_k01_jpeg100_target349"
ZIP_PATH="submissions/submission_mip_30k_r2_k01_jpeg100_target349.zip"
MODEL_ROOT="checkpoints/experiments/private/mip_30k_r2_k01_jpeg100"
WORKSPACE_DIR="workspace/experiments/mip_splatting/private/mip_30k_r2_k01_jpeg100"
REPORT_PATH="reports/experiments/mip_30k_r2_k01_target349_mix_report.json"
MAX_ZIP_BYTES=$((350 * 1024 * 1024))

mkdir -p reports/experiments submissions "$SOURCE_OUTPUT_DIR"

echo "Starting private Mip 30k r2 k0.1 target349 run at $(date -Is)"
for scene in $SCENES; do
    checkpoint="$MODEL_ROOT/$scene/point_cloud/iteration_30000/point_cloud.ply"
    scene_verify_log="reports/experiments/log_private_mip_target349_verify_source_${scene}.txt"
    scene_log="reports/experiments/log_private_mip_target349_${scene}.txt"

    if [[ -f "$checkpoint" ]] && python verify/verify_submission.py --split-root data/round1/phase1/private_set1 --submission-dir "$SOURCE_OUTPUT_DIR" --scenes "$scene" > "$scene_verify_log" 2>&1; then
        echo "Skipping $scene; checkpoint and q100 renders already verify at $(date -Is)"
        continue
    fi

    if [[ -f "$checkpoint" ]]; then
        echo "Rendering Mip private $scene from existing 30k checkpoint at $(date -Is)"
        python src/method_runners/run_mip_splatting_scene.py \
            --scene-dir "data/round1/phase1/private_set1/$scene" \
            --output-dir "$SOURCE_OUTPUT_DIR" \
            --model-root "$MODEL_ROOT" \
            --workspace-dir "$WORKSPACE_DIR" \
            --iterations 30000 \
            --train-resolution 2 \
            --kernel-size 0.1 \
            --lambda-l1 0.8 \
            --lambda-ssim 0.2 \
            --lambda-lpips 0.0 \
            --skip-train \
            --overwrite > "$scene_log" 2>&1
    else
        echo "Training/rendering Mip private $scene at $(date -Is)"
        python src/method_runners/run_mip_splatting_scene.py \
            --scene-dir "data/round1/phase1/private_set1/$scene" \
            --output-dir "$SOURCE_OUTPUT_DIR" \
            --model-root "$MODEL_ROOT" \
            --workspace-dir "$WORKSPACE_DIR" \
            --iterations 30000 \
            --train-resolution 2 \
            --kernel-size 0.1 \
            --lambda-l1 0.8 \
            --lambda-ssim 0.2 \
            --lambda-lpips 0.0 \
            --overwrite > "$scene_log" 2>&1
    fi
done

echo "Verifying q100 Mip source folder at $(date -Is)"
python verify/verify_submission.py \
    --split-root data/round1/phase1/private_set1 \
    --submission-dir "$SOURCE_OUTPUT_DIR" > reports/experiments/log_private_mip_target349_verify_source_folder.txt 2>&1

echo "Recompressing Mip output to target size at $(date -Is)"
python utils/recompress_submission_jpegs.py \
    --source-dir "$SOURCE_OUTPUT_DIR" \
    --output-dir "$FINAL_OUTPUT_DIR" \
    --target-mib 348.5 \
    --base-quality 97 \
    --max-quality 98 \
    --min-quality 95 \
    --overwrite \
    --report "$REPORT_PATH" > reports/experiments/log_private_mip_target349_recompress.txt 2>&1

echo "Verifying final Mip folder at $(date -Is)"
python verify/verify_submission.py \
    --split-root data/round1/phase1/private_set1 \
    --submission-dir "$FINAL_OUTPUT_DIR" > reports/experiments/log_private_mip_target349_verify_folder.txt 2>&1

echo "Creating final Mip zip at $(date -Is)"
python utils/make_submission_zip.py \
    --submission-dir "$FINAL_OUTPUT_DIR" \
    --zip "$ZIP_PATH" \
    --overwrite > reports/experiments/log_private_mip_target349_zip.txt 2>&1

zip_bytes=$(stat -c '%s' "$ZIP_PATH")
echo "Mip zip bytes: $zip_bytes" > reports/experiments/log_private_mip_target349_size.txt
if (( zip_bytes >= MAX_ZIP_BYTES )); then
    echo "Zip exceeds 350 MiB limit: $zip_bytes bytes" >&2
    exit 1
fi

echo "Verifying final Mip zip at $(date -Is)"
python verify/verify_zip.py \
    --split-root data/round1/phase1/private_set1 \
    --zip "$ZIP_PATH" > reports/experiments/log_private_mip_target349_verify_zip.txt 2>&1

echo "Legacy external file transfer omitted at $(date -Is)" > reports/experiments/log_private_mip_target349_external_transfer_skipped.txt

echo "DONE $(date -Is)" > reports/experiments/private_mip_30k_r2_k01_target349_done.txt
