#!/bin/bash
export PATH="/home/micace/Project/AI-3D/.venv/bin:$PATH"
export LD_LIBRARY_PATH="/home/micace/Project/AI-3D/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

echo "Starting 3DGS 30k factor 2 HQ batch..."
for scene in HCM0181 HCM0193 HCM0204 hcm0031 hcm0034; do
    echo "Training $scene..."
    python scripts/train_gs_scene.py --scene-dir "data/round1/phase1/public_set/$scene" \
        --output-dir "outputs/experiments/public/3dgs_30k_f2_jpeg100" \
        --factor 2 --steps 30000 --lambda-l1 0.8 --lambda-ssim 0.2 --lambda-lpips 0.0 \
        --seed 42 --overwrite > "reports/experiments/log_${scene}_30k.txt" 2>&1
done
echo "Batch complete. Running evaluate..."
python scripts/evaluate_public.py --public-root data/round1/phase1/public_set \
    --pred-dir outputs/experiments/public/3dgs_30k_f2_jpeg100 \
    --scenes HCM0181 HCM0193 HCM0204 hcm0031 hcm0034 \
    --json-out reports/experiments/eval/public/3dgs_30k_f2_jpeg100.json \
    --with-lpips --lpips-net alex --lpips-device cuda > reports/experiments/log_eval_30k.txt 2>&1
echo "DONE" > reports/experiments/hq_batch_status.txt
