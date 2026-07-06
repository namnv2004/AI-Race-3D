#!/bin/bash
export PATH="/home/micace/Project/AI-3D/.venv/bin:$PATH"
export LD_LIBRARY_PATH="/home/micace/Project/AI-3D/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
SCENES="HCM0181 HCM0193 HCM0204 hcm0031 hcm0034"

# Wait for 3DGS 30k to finish if it's still running
if [ -f run_hq.pid ]; then
  PID=$(cat run_hq.pid)
  echo "Waiting for PID $PID (3DGS 30k) to finish..."
  tail --pid=$PID -f /dev/null
fi

echo "Starting Mip-Splatting 30k r2 k0.1 HQ batch..."
for scene in $SCENES; do
    echo "Training Mip $scene..."
    python scripts/run_mip_splatting_scene.py --scene-dir "data/round1/phase1/public_set/$scene" \
        --output-dir "outputs/experiments/public/mip_30k_r2_k01_jpeg100" \
        --iterations 30000 --train-resolution 2 --kernel-size 0.1 \
        --lambda-l1 0.8 --lambda-ssim 0.2 --lambda-lpips 0.0 \
        --overwrite > "reports/experiments/log_mip_${scene}_30k.txt" 2>&1
done
echo "Evaluating Mip 30k..."
python scripts/evaluate_public.py --public-root data/round1/phase1/public_set \
    --pred-dir outputs/experiments/public/mip_30k_r2_k01_jpeg100 \
    --scenes $SCENES \
    --json-out reports/experiments/eval/public/mip_30k_r2_k01_jpeg100.json \
    --with-lpips --lpips-net alex --lpips-device cuda > reports/experiments/log_eval_mip_30k.txt 2>&1

echo "Starting 2DGS 30k HQ batch..."
for scene in $SCENES; do
    echo "Training 2DGS $scene..."
    python scripts/run_2dgs_scene.py --scene-dir "data/round1/phase1/public_set/$scene" \
        --output-dir "outputs/experiments/public/2dgs_30k_default_jpeg100" \
        --iterations 30000 --resolution -1 \
        --lambda-dssim 0.2 --lambda-dist 0.0 --lambda-normal 0.05 \
        --overwrite > "reports/experiments/log_2dgs_${scene}_30k.txt" 2>&1
done
echo "Evaluating 2DGS 30k..."
python scripts/evaluate_public.py --public-root data/round1/phase1/public_set \
    --pred-dir outputs/experiments/public/2dgs_30k_default_jpeg100 \
    --scenes $SCENES \
    --json-out reports/experiments/eval/public/2dgs_30k_default_jpeg100.json \
    --with-lpips --lpips-net alex --lpips-device cuda > reports/experiments/log_eval_2dgs_30k.txt 2>&1

echo "ALL HQ BATCHES DONE" > reports/experiments/all_hq_done.txt
