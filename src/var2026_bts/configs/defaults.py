"""Config defaults for VAR 2026 Digital Twin BTS competition.

These values are tuned defaults used across training and rendering scripts.
Override via CLI arguments where supported.
"""

# Competition score formula: Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR / 30
SCORE_WEIGHT_LPIPS = 0.4
SCORE_WEIGHT_SSIM = 0.3
SCORE_WEIGHT_PSNR = 0.3
PSNR_MAX = 30.0

# Default image/render dimensions (full resolution)
DEFAULT_RENDER_SCALE = 1.0

# Loss weights (competition-aligned, defaults can be overridden)
DEFAULT_LAMBDA_L1 = 0.8
DEFAULT_LAMBDA_SSIM = 0.2
DEFAULT_LAMBDA_LPIPS = 0.0
DEFAULT_MASK_BOOST = 0.0  # 0 disables mask-weighted loss
DEFAULT_MASK_DILATE = 0
DEFAULT_MASK_THRESHOLD = 0.5

# GS / gsplat training defaults
DEFAULT_GS_FACTOR = 2
DEFAULT_GS_STEPS = 10000
DEFAULT_GS_MAX_INIT_POINTS = 200000
DEFAULT_GS_INIT_OPACITY = 0.1
DEFAULT_GS_INIT_SCALE = 1.0
DEFAULT_GS_REFINE_START = 500
DEFAULT_GS_REFINE_STOP = 15000
DEFAULT_GS_REFINE_EVERY = 100
DEFAULT_GS_RESET_EVERY = 1500

# Mip-Splatting training defaults
DEFAULT_MIP_ITERATIONS = 20000
DEFAULT_MIP_TRAIN_RESOLUTION = 4
DEFAULT_MIP_KERNEL_SIZE = 0.1
DEFAULT_MIP_DENSIFY_GRAD = 0.0002
DEFAULT_MIP_DENSIFY_UNTIL = 15000
DEFAULT_MIP_DENSIFY_INTERVAL = 100
DEFAULT_MIP_OPACITY_RESET = 3000

# 2D Gaussian Splatting training defaults
DEFAULT_2DGS_ITERATIONS = 20000
DEFAULT_2DGS_RESOLUTION = -1
DEFAULT_2DGS_LAMBDA_DSSIM = 0.2
DEFAULT_2DGS_LAMBDA_DIST = 0.0
DEFAULT_2DGS_LAMBDA_NORMAL = 0.05
DEFAULT_2DGS_DENSIFY_GRAD = 0.0002
DEFAULT_2DGS_DENSIFY_UNTIL = 15000
DEFAULT_2DGS_DEPTH_RATIO = 0.0

# SAM (Segment Anything Model) defaults
DEFAULT_SAM_MODEL = "facebook/sam3"
DEFAULT_SAM_PROMPT = (
    "cell tower, telecommunication tower, BTS station, "
    "antenna tower, metal lattice tower"
)
DEFAULT_SAM_THRESHOLD = 0.5
DEFAULT_SAM_MASK_THRESHOLD = 0.5
DEFAULT_SAM_MIN_SCORE = 0.35

# Mask directory layout: workspace/sam3_masks/<scene>/<image_stem>.png
DEFAULT_MASK_ROOT = "workspace/sam3_masks"

# Render output default size for competition: 1320x989
DEFAULT_OUTPUT_WIDTH = 1320
DEFAULT_OUTPUT_HEIGHT = 989
