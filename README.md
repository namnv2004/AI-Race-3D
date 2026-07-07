# AI Race 3D - Digital Twin BTS

Pipeline Novel View Synthesis cho bài toán tái dựng Digital Twin trạm BTS từ ảnh drone/COLMAP và sinh ảnh RGB ở các góc nhìn mới.

## Cấu Trúc Repo

```text
.
├── src/                 # Module dùng chung và method runners
├── configs/             # Defaults, manifest, experiment configs
├── runs/                # Generated workflow shell commands
├── eval/                # Public evaluation
├── verify/              # Submission validation
├── utils/               # Inspect dataset, generate masks/commands, zip helpers
├── data/                # Dataset local, gitignored
├── outputs/             # Render outputs, gitignored
├── submissions/         # Submission zips, gitignored
├── checkpoints/         # Model checkpoints, gitignored
├── workspace/           # Intermediate artifacts, gitignored
└── third_party/         # External repos, gitignored
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

`third_party/` được quản lý bằng quy trình setup local hiện có của bạn.

## Lệnh Chính

```bash
# Inspect dataset
python utils/inspect_dataset.py --json-out reports/dataset.json

# Generate SAM masks
python utils/generate_sam3_masks.py \
    --split-root data/round1/phase1/public_set \
    --output-dir masks

# Train/render 3DGS
python src/method_runners/run_3dgs_scene.py \
    --scene-dir data/round1/phase1/public_set/HCM0181 \
    --output-dir outputs/smoke/3dgs_f2_10k --factor 2 --steps 10000

# Train/render Mip-Splatting
python src/method_runners/run_mip_splatting_scene.py \
    --scene-dir data/round1/phase1/public_set/HCM0181 \
    --output-dir outputs/smoke/mip_10k_r4_k01 --iterations 10000 --kernel-size 0.1

# Train/render 2DGS
python src/method_runners/run_2dgs_scene.py \
    --scene-dir data/round1/phase1/public_set/HCM0181 \
    --output-dir outputs/smoke/2dgs_10k_default --iterations 10000

# Evaluate public set
python eval/evaluate_public.py \
    --public-root data/round1/phase1/public_set \
    --pred-dir outputs/public/3dgs_f2_10k \
    --with-lpips

# Verify and zip submission
python verify/verify_submission.py \
    --split-root data/round1/phase1/private_set1 \
    --submission-dir outputs/private/3dgs_f2_10k

python utils/make_submission_zip.py \
    --submission-dir outputs/private/3dgs_f2_10k \
    --zip submissions/submission_3dgs_f2_10k.zip \
    --overwrite
```

## Experiment Workflow

- `configs/gs_runs.json`: manifest dùng chung cho smoke, public benchmark và private render.
- `runs/commands/10_smoke.sh`: chạy thử một public scene duy nhất để bắt lỗi command/dependency/runtime. Không dùng kết quả này để xếp hạng phương pháp.
- `runs/commands/20_public.sh`: benchmark chuẩn; mỗi phương pháp/config phải chạy đủ toàn bộ public scenes.
- `runs/commands/21_eval.sh`: đánh giá toàn bộ public scenes và ghi metrics vào `reports/experiments/eval/public/`.
- `runs/commands/30_private.sh`: render private bằng đúng một run đã chọn từ public benchmark.
- Render outputs dùng layout `outputs/smoke/<run_id>`, `outputs/public/<run_id>`, `outputs/private/<run_id>`.
- Chỉ zip từ `outputs/private/<run_id>` sang `submissions/` sau khi folder private đã verify pass.
- Selection/ranking outputs phải ghi vào `reports/experiments/selections/`.

## Tóm Tắt Bài Toán

Mục tiêu là tái dựng cấu trúc 3D ngầm định của một scene từ ảnh đa góc nhìn và sinh ảnh RGB tại các camera pose chưa từng xuất hiện trong dữ liệu train. Bài toán thuộc nhóm Computer Vision, 3D Vision, Neural Rendering, Novel View Synthesis và Digital Twin.

Mỗi scene gồm ảnh train, camera intrinsics/poses và sparse reconstruction từ COLMAP. Mô hình cần render ảnh đúng hình học, đúng vị trí thiết bị và có chất lượng hình ảnh chân thực.

## Cấu Trúc Dữ Liệu Scene

```text
scene/
├── train/
│   ├── images/
│   └── sparse/0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── test/
    └── test_poses.csv
```

`test_poses.csv` có format:

```csv
image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height
```

Các trường chính:

- `image_name`: tên ảnh cần sinh.
- `qw`, `qx`, `qy`, `qz`: quaternion rotation theo COLMAP.
- `tx`, `ty`, `tz`: camera translation.
- `fx`, `fy`, `cx`, `cy`: camera intrinsics.
- `width`, `height`: kích thước ảnh đầu ra.

## Submission

Submission là file ZIP chứa đầy đủ ảnh render cho toàn bộ scene/test poses:

```text
submission.zip
├── scene_001/
│   ├── 0001.png
│   └── ...
├── scene_002/
│   ├── 0001.png
│   └── ...
└── ...
```

Yêu cầu chính:

- Đúng tên scene và tên file ảnh theo `test_poses.csv`.
- Đúng kích thước `width`, `height`.
- Không thiếu hoặc thừa ảnh so với danh sách test poses.

## Metrics

Điểm được tính từ LPIPS, SSIM và PSNR chuẩn hóa:

```text
Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
PSNR_norm = clamp(PSNR / PSNR_max, 0, 1)
```

- LPIPS càng thấp càng tốt.
- SSIM càng cao càng tốt.
- PSNR càng cao càng tốt.
- `PSNR_max` mặc định là `50.0`, khớp thang leaderboard private đã quan sát.
