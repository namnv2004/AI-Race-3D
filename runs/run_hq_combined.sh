#!/usr/bin/env bash
set -euo pipefail

# Consolidated legacy HQ workflow recovered from the split scripts:
# - run_hq_background.sh
# - run_queue_hq.sh
# - run_resume_2dgs_public_hq.sh
# - run_auto_final_after_public_hq.sh
# - run_resume_private_2dgs.sh
#
# This is a historical runner. It preserves old paths and experiment ids from
# the previous layout so prior interrupted HQ runs can be understood or resumed.

export PATH="/home/micace/Project/AI-3D/.venv/bin:$PATH"
export LD_LIBRARY_PATH="/home/micace/Project/AI-3D/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PUBLIC_ROOT="data/round1/phase1/public_set"
PRIVATE_ROOT="data/round1/phase1/private_set1"
LOG_DIR="reports/experiments"
EVAL_DIR="$LOG_DIR/eval/public"

OUTPUT_3DGS="outputs/experiments/public/3dgs_30k_f2_jpeg100"
OUTPUT_MIP="outputs/experiments/public/mip_30k_r2_k01_jpeg100"
OUTPUT_2DGS="outputs/experiments/public/2dgs_30k_default_jpeg100"
OUTPUT_PRIVATE="outputs/private_auto_best_hq"
ZIP_PATH="submissions/submission_auto_best_hq.zip"
SELECTION_JSON="$LOG_DIR/final_hq_selection.json"
WINNER_TXT="$LOG_DIR/final_hq_winner.txt"
PUBLIC_DONE="$LOG_DIR/2dgs_public_hq_done.txt"

PUBLIC_SCENES=(HCM0181 HCM0193 HCM0204 hcm0031 hcm0034)
PUBLIC_2DGS_RESUME_SCENES=(HCM0193 HCM0204 hcm0031 hcm0034)
PRIVATE_SCENES=(HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437)
PRIVATE_2DGS_RESUME_SCENES=(HNI0265 HNI0366 HNI0437)

mkdir -p "$EVAL_DIR" submissions "$OUTPUT_PRIVATE"

log() {
    echo "[$(date -Is)] $*"
}

count_files() {
    if [[ -d "$1" ]]; then
        find "$1" -type f | wc -l
    else
        echo 0
    fi
}

run_public_3dgs() {
    log "Starting 3DGS 30k factor 2 HQ public batch"
    for scene in "${PUBLIC_SCENES[@]}"; do
        log "Training 3DGS public $scene"
        python src/method_runners/run_3dgs_scene.py \
            --scene-dir "$PUBLIC_ROOT/$scene" \
            --output-dir "$OUTPUT_3DGS" \
            --factor 2 \
            --steps 30000 \
            --lambda-l1 0.8 \
            --lambda-ssim 0.2 \
            --lambda-lpips 0.0 \
            --seed 42 \
            --overwrite > "$LOG_DIR/log_${scene}_30k.txt" 2>&1
    done

    log "Evaluating 3DGS 30k factor 2 public batch"
    python eval/evaluate_public.py \
        --public-root "$PUBLIC_ROOT" \
        --pred-dir "$OUTPUT_3DGS" \
        --scenes "${PUBLIC_SCENES[@]}" \
        --json-out "$EVAL_DIR/3dgs_30k_f2_jpeg100.json" \
        --with-lpips \
        --lpips-net alex \
        --lpips-device cuda > "$LOG_DIR/log_eval_30k.txt" 2>&1

    echo "DONE" > "$LOG_DIR/hq_batch_status.txt"
}

run_public_mip() {
    log "Starting Mip-Splatting 30k r2 k0.1 HQ public batch"
    for scene in "${PUBLIC_SCENES[@]}"; do
        log "Training Mip public $scene"
        python src/method_runners/run_mip_splatting_scene.py \
            --scene-dir "$PUBLIC_ROOT/$scene" \
            --output-dir "$OUTPUT_MIP" \
            --iterations 30000 \
            --train-resolution 2 \
            --kernel-size 0.1 \
            --lambda-l1 0.8 \
            --lambda-ssim 0.2 \
            --lambda-lpips 0.0 \
            --overwrite > "$LOG_DIR/log_mip_${scene}_30k.txt" 2>&1
    done

    log "Evaluating Mip-Splatting public batch"
    python eval/evaluate_public.py \
        --public-root "$PUBLIC_ROOT" \
        --pred-dir "$OUTPUT_MIP" \
        --scenes "${PUBLIC_SCENES[@]}" \
        --json-out "$EVAL_DIR/mip_30k_r2_k01_jpeg100.json" \
        --with-lpips \
        --lpips-net alex \
        --lpips-device cuda > "$LOG_DIR/log_eval_mip_30k.txt" 2>&1
}

run_public_2dgs_full() {
    log "Starting full 2DGS 30k HQ public batch"
    for scene in "${PUBLIC_SCENES[@]}"; do
        log "Training 2DGS public $scene"
        python src/method_runners/run_2dgs_scene.py \
            --scene-dir "$PUBLIC_ROOT/$scene" \
            --output-dir "$OUTPUT_2DGS" \
            --iterations 30000 \
            --resolution -1 \
            --lambda-dssim 0.2 \
            --lambda-dist 0.0 \
            --lambda-normal 0.05 \
            --overwrite > "$LOG_DIR/log_2dgs_${scene}_30k.txt" 2>&1
    done

    evaluate_public_2dgs "$LOG_DIR/log_eval_2dgs_30k.txt"
    echo "ALL HQ BATCHES DONE" > "$LOG_DIR/all_hq_done.txt"
    echo "DONE $(date -Is)" > "$PUBLIC_DONE"
}

resume_public_2dgs() {
    log "Resuming interrupted 2DGS 30k HQ public batch"
    for scene in "${PUBLIC_2DGS_RESUME_SCENES[@]}"; do
        log "Training/rendering resumed 2DGS public $scene"
        python src/method_runners/run_2dgs_scene.py \
            --scene-dir "$PUBLIC_ROOT/$scene" \
            --output-dir "$OUTPUT_2DGS" \
            --iterations 30000 \
            --resolution -1 \
            --lambda-dssim 0.2 \
            --lambda-dist 0.0 \
            --lambda-normal 0.05 \
            --overwrite > "$LOG_DIR/log_2dgs_${scene}_30k_resume.txt" 2>&1
    done

    evaluate_public_2dgs "$LOG_DIR/log_eval_2dgs_30k_resume.txt"
    echo "DONE $(date -Is)" > "$PUBLIC_DONE"
}

evaluate_public_2dgs() {
    local log_path="$1"

    log "Evaluating 2DGS public HQ batch"
    python eval/evaluate_public.py \
        --public-root "$PUBLIC_ROOT" \
        --pred-dir "$OUTPUT_2DGS" \
        --scenes "${PUBLIC_SCENES[@]}" \
        --json-out "$EVAL_DIR/2dgs_30k_default_jpeg100.json" \
        --with-lpips \
        --lpips-net alex \
        --lpips-device cuda > "$log_path" 2>&1
}

wait_for_public_done() {
    while [[ ! -f "$PUBLIC_DONE" ]]; do
        log "Waiting for 2DGS public HQ completion marker: $PUBLIC_DONE"
        sleep 300
    done
}

select_public_winner() {
    log "Selecting best public HQ candidate"
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
}

render_private_winner() {
    local winner
    winner="$(tr -d '[:space:]' < "$WINNER_TXT")"

    log "Rendering private HQ winner: $winner"
    rm -rf "$OUTPUT_PRIVATE"
    mkdir -p "$OUTPUT_PRIVATE"

    case "$winner" in
        3dgs_10k_f2)
            for scene in "${PRIVATE_SCENES[@]}"; do
                log "Private render $winner $scene"
                python src/method_runners/run_3dgs_scene.py \
                    --scene-dir "$PRIVATE_ROOT/$scene" \
                    --output-dir "$OUTPUT_PRIVATE" \
                    --factor 2 \
                    --steps 10000 \
                    --lambda-l1 0.8 \
                    --lambda-ssim 0.2 \
                    --lambda-lpips 0.0 \
                    --seed 42 \
                    --overwrite > "$LOG_DIR/log_auto_private_${winner}_${scene}.txt" 2>&1
            done
            ;;
        3dgs_30k_f2_jpeg100)
            for scene in "${PRIVATE_SCENES[@]}"; do
                log "Private render $winner $scene"
                python src/method_runners/run_3dgs_scene.py \
                    --scene-dir "$PRIVATE_ROOT/$scene" \
                    --output-dir "$OUTPUT_PRIVATE" \
                    --factor 2 \
                    --steps 30000 \
                    --lambda-l1 0.8 \
                    --lambda-ssim 0.2 \
                    --lambda-lpips 0.0 \
                    --seed 42 \
                    --overwrite > "$LOG_DIR/log_auto_private_${winner}_${scene}.txt" 2>&1
            done
            ;;
        mip_30k_r2_k01_jpeg100)
            for scene in "${PRIVATE_SCENES[@]}"; do
                log "Private render $winner $scene"
                python src/method_runners/run_mip_splatting_scene.py \
                    --scene-dir "$PRIVATE_ROOT/$scene" \
                    --output-dir "$OUTPUT_PRIVATE" \
                    --model-root "checkpoints/experiments/private_auto_best_hq/$winner" \
                    --workspace-dir "workspace/experiments/mip_splatting/private_auto_best_hq/$winner" \
                    --iterations 30000 \
                    --train-resolution 2 \
                    --kernel-size 0.1 \
                    --lambda-l1 0.8 \
                    --lambda-ssim 0.2 \
                    --lambda-lpips 0.0 \
                    --overwrite > "$LOG_DIR/log_auto_private_${winner}_${scene}.txt" 2>&1
            done
            ;;
        2dgs_30k_default_jpeg100)
            for scene in "${PRIVATE_SCENES[@]}"; do
                log "Private render $winner $scene"
                python src/method_runners/run_2dgs_scene.py \
                    --scene-dir "$PRIVATE_ROOT/$scene" \
                    --output-dir "$OUTPUT_PRIVATE" \
                    --model-root "checkpoints/experiments/private_auto_best_hq/$winner" \
                    --iterations 30000 \
                    --resolution -1 \
                    --lambda-dssim 0.2 \
                    --lambda-dist 0.0 \
                    --lambda-normal 0.05 \
                    --overwrite > "$LOG_DIR/log_auto_private_${winner}_${scene}.txt" 2>&1
            done
            ;;
        *)
            echo "ERROR: unsupported winner $winner" >&2
            exit 1
            ;;
    esac
}

resume_private_2dgs() {
    local winner="2dgs_30k_default_jpeg100"
    local port_base="${PORT_BASE:-6009}"

    log "Resuming interrupted private 2DGS HQ render" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"
    for scene in "${PRIVATE_2DGS_RESUME_SCENES[@]}"; do
        local port="$port_base"
        local overwrite_args=()
        port_base=$((port_base + 1))

        if [[ "$scene" == "HNI0265" ]]; then
            log "Resetting partial checkpoint/output for $scene" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"
            rm -rf "checkpoints/experiments/private_auto_best_hq/$winner/$scene" "$OUTPUT_PRIVATE/$scene"
            overwrite_args=(--overwrite)
        fi

        log "Training/rendering resumed private 2DGS $scene on port $port" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"
        python src/method_runners/run_2dgs_scene.py \
            --scene-dir "$PRIVATE_ROOT/$scene" \
            --output-dir "$OUTPUT_PRIVATE" \
            --model-root "checkpoints/experiments/private_auto_best_hq/$winner" \
            --iterations 30000 \
            --resolution -1 \
            --lambda-dssim 0.2 \
            --lambda-dist 0.0 \
            --lambda-normal 0.05 \
            --port "$port" \
            "${overwrite_args[@]}" \
            > "$LOG_DIR/log_resume_private_${scene}.txt" 2>&1
        log "$scene done: $(count_files "$OUTPUT_PRIVATE/$scene") files" | tee -a "$LOG_DIR/log_resume_private_2dgs.txt"
    done

    echo "$winner" > "$WINNER_TXT"
}

verify_and_zip() {
    local prefix="$1"

    log "Verifying final HQ folder"
    python verify/verify_submission.py \
        --split-root "$PRIVATE_ROOT" \
        --submission-dir "$OUTPUT_PRIVATE" > "$LOG_DIR/log_${prefix}_verify_folder.txt" 2>&1

    log "Creating final HQ zip"
    python utils/make_submission_zip.py \
        --submission-dir "$OUTPUT_PRIVATE" \
        --zip "$ZIP_PATH" \
        --overwrite > "$LOG_DIR/log_${prefix}_zip.txt" 2>&1

    log "Verifying final HQ zip"
    python verify/verify_zip.py \
        --split-root "$PRIVATE_ROOT" \
        --zip "$ZIP_PATH" > "$LOG_DIR/log_${prefix}_verify_zip.txt" 2>&1

    echo "Legacy external file transfer omitted from consolidated archive." > "$LOG_DIR/log_${prefix}_external_transfer_skipped.txt"
}

write_done_marker() {
    local winner="unknown"
    if [[ -f "$WINNER_TXT" ]]; then
        winner="$(tr -d '[:space:]' < "$WINNER_TXT")"
    fi

    {
        echo "DONE $(date -Is)"
        echo "winner=$winner"
        echo "zip=$ZIP_PATH"
    } > "$LOG_DIR/auto_final_hq_done.txt"
}

usage() {
    cat <<'EOF'
Usage: runs/run_hq_combined.sh [phase]

Phases:
  all                 Run full historical public HQ flow, select winner, render private, zip.
  resume-public-hq    Resume interrupted 2DGS public HQ, select winner, render private, zip.
  wait-and-private    Wait for public done marker, select winner, render private, zip.
  select-and-private  Select public winner from existing eval JSON, render private, zip.
  private-2dgs-resume Resume interrupted private 2DGS winner render, then zip.
  public-3dgs         Run only public 3DGS HQ batch and eval.
  public-mip          Run only public Mip HQ batch and eval.
  public-2dgs-full    Run only full public 2DGS HQ batch and eval.
  public-2dgs-resume  Run only resumed public 2DGS HQ scenes and eval.
EOF
}

main() {
    local phase="${1:-all}"

    case "$phase" in
        all)
            run_public_3dgs
            run_public_mip
            run_public_2dgs_full
            select_public_winner
            render_private_winner
            verify_and_zip auto_final
            write_done_marker
            ;;
        resume-public-hq)
            resume_public_2dgs
            select_public_winner
            render_private_winner
            verify_and_zip auto_final
            write_done_marker
            ;;
        wait-and-private)
            wait_for_public_done
            select_public_winner
            render_private_winner
            verify_and_zip auto_final
            write_done_marker
            ;;
        select-and-private)
            select_public_winner
            render_private_winner
            verify_and_zip auto_final
            write_done_marker
            ;;
        private-2dgs-resume)
            resume_private_2dgs
            verify_and_zip resume
            write_done_marker
            ;;
        public-3dgs)
            run_public_3dgs
            ;;
        public-mip)
            run_public_mip
            ;;
        public-2dgs-full)
            run_public_2dgs_full
            ;;
        public-2dgs-resume)
            resume_public_2dgs
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            usage >&2
            echo "ERROR: unknown phase: $phase" >&2
            exit 2
            ;;
    esac
}

main "$@"
