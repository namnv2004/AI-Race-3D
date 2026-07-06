#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/micace/Project/AI-3D/.venv/bin:$PATH"
export LD_LIBRARY_PATH="/home/micace/Project/AI-3D/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PRIVATE_ROOT="data/round1/phase1/private_set1"
OUTPUT_DIR="outputs/private_auto_best_hq"
WINNER="2dgs_30k_default_jpeg100"
ZIP_PATH="submissions/submission_auto_best_hq.zip"
SCENES="HCM1439 HNI0265 HNI0366 HNI0437"
LOG_DIR="reports/experiments"
PORT_BASE=6009

mkdir -p "$LOG_DIR"
echo "Resume private 2DGS at $(date -Is)" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"

# Scene 2: HNI0265 - fresh train (checkpoint only at 7k)
PORT=$PORT_BASE; PORT_BASE=$((PORT_BASE + 1))
echo "=== HNI0265 (fresh train, port=$PORT) at $(date -Is)" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"
rm -rf "checkpoints/experiments/private_auto_best_hq/$WINNER/HNI0265" "$OUTPUT_DIR/HNI0265"
python scripts/run_2dgs_scene.py \
    --scene-dir "$PRIVATE_ROOT/HNI0265" \
    --output-dir "$OUTPUT_DIR" \
    --model-root "checkpoints/experiments/private_auto_best_hq/$WINNER" \
    --iterations 30000 \
    --resolution -1 \
    --lambda-dssim 0.2 \
    --lambda-dist 0.0 \
    --lambda-normal 0.05 \
    --port "$PORT" \
    --overwrite \
    > "$LOG_DIR/log_resume_private_HNI0265.txt" 2>&1
echo "HNI0265 done: $(find $OUTPUT_DIR/HNI0265 -type f | wc -l) files" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"

# Scene 3: HNI0366 - fresh train
PORT=$PORT_BASE; PORT_BASE=$((PORT_BASE + 1))
echo "=== HNI0366 (fresh train, port=$PORT) at $(date -Is)" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"
python scripts/run_2dgs_scene.py \
    --scene-dir "$PRIVATE_ROOT/HNI0366" \
    --output-dir "$OUTPUT_DIR" \
    --model-root "checkpoints/experiments/private_auto_best_hq/$WINNER" \
    --iterations 30000 \
    --resolution -1 \
    --lambda-dssim 0.2 \
    --lambda-dist 0.0 \
    --lambda-normal 0.05 \
    --port "$PORT" \
    > "$LOG_DIR/log_resume_private_HNI0366.txt" 2>&1
echo "HNI0366 done: $(find $OUTPUT_DIR/HNI0366 -type f | wc -l) files" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"

# Scene 4: HNI0437 - fresh train
PORT=$PORT_BASE; PORT_BASE=$((PORT_BASE + 1))
echo "=== HNI0437 (fresh train, port=$PORT) at $(date -Is)" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"
python scripts/run_2dgs_scene.py \
    --scene-dir "$PRIVATE_ROOT/HNI0437" \
    --output-dir "$OUTPUT_DIR" \
    --model-root "checkpoints/experiments/private_auto_best_hq/$WINNER" \
    --iterations 30000 \
    --resolution -1 \
    --lambda-dssim 0.2 \
    --lambda-dist 0.0 \
    --lambda-normal 0.05 \
    --port "$PORT" \
    > "$LOG_DIR/log_resume_private_HNI0437.txt" 2>&1
echo "HNI0437 done: $(find $OUTPUT_DIR/HNI0437 -type f | wc -l) files" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"

# Verify
echo "Verifying folder at $(date -Is)" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"
python scripts/verify_submission.py \
    --split-root "$PRIVATE_ROOT" \
    --submission-dir "$OUTPUT_DIR" > "$LOG_DIR/log_resume_verify_folder.txt" 2>&1

# Zip
echo "Creating zip at $(date -Is)" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"
python scripts/make_submission_zip.py \
    --submission-dir "$OUTPUT_DIR" \
    --zip "$ZIP_PATH" \
    --overwrite > "$LOG_DIR/log_resume_zip.txt" 2>&1

# Verify zip
echo "Verifying zip at $(date -Is)" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"
python scripts/verify_zip.py \
    --split-root "$PRIVATE_ROOT" \
    --zip "$ZIP_PATH" > "$LOG_DIR/log_resume_verify_zip.txt" 2>&1

# Taildrop
echo "Sending to ironcat at $(date -Is)" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"
tailscale file cp "$ZIP_PATH" ironcat: > "$LOG_DIR/log_resume_taildrop.txt" 2>&1

# Done marker
{
    echo "DONE $(date -Is)"
    echo "winner=$WINNER"
    echo "zip=$ZIP_PATH"
} > "$LOG_DIR/auto_final_hq_done.txt"

echo "ALL DONE at $(date -Is)" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"
