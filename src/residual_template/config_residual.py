# src/residual_template/config_residual.py

from pathlib import Path
import sys


# =============================================================================
# Allow importing from src/config.py
# =============================================================================

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import (
    PROJECT_ROOT,
    DATASET_ROOT,
    OUTPUT_ROOT,
    WATERMARKED_SOURCES_DIR,
    CLEAN_TARGETS_DIR,
    GROUPS,
    TARGET_MAPPING,
    COLOR_MODE,
    IMAGE_EXTENSIONS,
    EXPECTED_SUBMISSION_COUNT,
)


# =============================================================================
# Method name and paths
# =============================================================================

RESIDUAL_METHOD_NAME = "residual_template"

RESIDUAL_FORGED_ROOT = OUTPUT_ROOT / "forged" / RESIDUAL_METHOD_NAME
RESIDUAL_TEMPLATE_ROOT = OUTPUT_ROOT / "templates" / RESIDUAL_METHOD_NAME
RESIDUAL_VISUALIZATION_ROOT = OUTPUT_ROOT / "visualizations" / RESIDUAL_METHOD_NAME
RESIDUAL_QUALITY_REPORT_ROOT = OUTPUT_ROOT / "quality_reports" / RESIDUAL_METHOD_NAME

RESIDUAL_SUBMISSION_ZIP_PATH = OUTPUT_ROOT / "submission.zip"


# =============================================================================
# Core formula
# =============================================================================
# Template extraction:
#
#   r_i = x_wm_i - D(x_wm_i)
#   w'  = median_i(r_i)
#
# Forgery:
#
#   I_forged = clip(I_target + sign * alpha * mask * w')
#
# D can be gaussian, bilateral, nlm, median, or ensemble.


# =============================================================================
# Denoiser parameters
# =============================================================================

# Gaussian blur, close to the code that got your best score.
GAUSSIAN_RADIUS = 2.0

# Bilateral filtering.
BILATERAL_D = 7
BILATERAL_SIGMA_COLOR = 40
BILATERAL_SIGMA_SPACE = 40

# Non-local means.
NLM_H = 5
NLM_H_COLOR = 5
NLM_TEMPLATE_WINDOW_SIZE = 7
NLM_SEARCH_WINDOW_SIZE = 21

# Median filter.
MEDIAN_KERNEL_SIZE = 3


# =============================================================================
# Template extraction parameters
# =============================================================================

TEMPLATE_AGGREGATION = "median"  # "median" or "mean"

REMOVE_TEMPLATE_CHANNEL_MEAN = True

TEMPLATE_NORMALIZATION_QUANTILE = 0.995
RESIDUAL_SCALE = 1.0

# Options:
#   "none"
#   "original_shifted"  -> closest to your 0.32 script
#   "signed"            -> cleaner signed high-pass refinement
TEMPLATE_SHARPEN_MODE = "original_shifted"

TEMPLATE_SHARPEN_RADIUS = 1.0
TEMPLATE_SHARPEN_MIX_ORIGINAL = 0.70
TEMPLATE_SHARPEN_MIX_HIGHPASS = 0.30

CLIP_TEMPLATE_MIN = -1.0
CLIP_TEMPLATE_MAX = 1.0


# =============================================================================
# Mask parameters
# =============================================================================

USE_DETAIL_MASK = True

MASK_FLOOR = 0.25
MASK_CEILING = 1.00
MASK_DETAIL_QUANTILE = 0.95

# Denoiser used only for target detail mask.
MASK_DENOISER = "gaussian"


# =============================================================================
# Alpha parameters
# =============================================================================

ALPHA_MODE = "dynamic"  # "dynamic" or "fixed"

BASE_ALPHA = 0.10
MIN_ALPHA = 0.06
MAX_ALPHA = 0.18

# This follows the working script:
# alpha = BASE_ALPHA * (0.85 + 0.30 * tanh(strength))
ALPHA_TANH_BASE = 0.85
ALPHA_TANH_SCALE = 0.30


# =============================================================================
# Experiments
# =============================================================================
# First one reproduces the closest structured version of the code that got 0.32.
# Then we test better clean estimators.

EXPERIMENTS = [
    {
        "name": "gaussian_pos_a010",
        "denoiser": "gaussian",
        "mask_denoiser": "gaussian",
        "sign": 1.0,
        "base_alpha": 0.10,
        "min_alpha": 0.06,
        "max_alpha": 0.18,
        "mask_floor": 0.25,
        "mask_ceiling": 1.00,
        "sharpen_mode": "original_shifted",
    },
    {
        "name": "bilateral_pos_a010",
        "denoiser": "bilateral",
        "mask_denoiser": "bilateral",
        "sign": 1.0,
        "base_alpha": 0.10,
        "min_alpha": 0.06,
        "max_alpha": 0.18,
        "mask_floor": 0.25,
        "mask_ceiling": 1.00,
        "sharpen_mode": "signed",
    },
    {
        "name": "bilateral_pos_a012",
        "denoiser": "bilateral",
        "mask_denoiser": "bilateral",
        "sign": 1.0,
        "base_alpha": 0.12,
        "min_alpha": 0.08,
        "max_alpha": 0.22,
        "mask_floor": 0.20,
        "mask_ceiling": 1.00,
        "sharpen_mode": "signed",
    },
    {
        "name": "nlm_pos_a010",
        "denoiser": "nlm",
        "mask_denoiser": "gaussian",
        "sign": 1.0,
        "base_alpha": 0.10,
        "min_alpha": 0.06,
        "max_alpha": 0.18,
        "mask_floor": 0.25,
        "mask_ceiling": 1.00,
        "sharpen_mode": "signed",
    },
    {
        "name": "median_pos_a010",
        "denoiser": "median",
        "mask_denoiser": "gaussian",
        "sign": 1.0,
        "base_alpha": 0.10,
        "min_alpha": 0.06,
        "max_alpha": 0.18,
        "mask_floor": 0.25,
        "mask_ceiling": 1.00,
        "sharpen_mode": "signed",
    },
    {
        "name": "ensemble_pos_a010",
        "denoiser": "ensemble",
        "mask_denoiser": "gaussian",
        "sign": 1.0,
        "base_alpha": 0.10,
        "min_alpha": 0.06,
        "max_alpha": 0.18,
        "mask_floor": 0.25,
        "mask_ceiling": 1.00,
        "sharpen_mode": "signed",
    },
    {
        "name": "gaussian_neg_a010",
        "denoiser": "gaussian",
        "mask_denoiser": "gaussian",
        "sign": -1.0,
        "base_alpha": 0.10,
        "min_alpha": 0.06,
        "max_alpha": 0.18,
        "mask_floor": 0.25,
        "mask_ceiling": 1.00,
        "sharpen_mode": "original_shifted",
    },
{
    "name": "ensemble_pos_a012",
    "denoiser": "ensemble",
    "mask_denoiser": "gaussian",
    "sign": 1.0,
    "base_alpha": 0.12,
    "min_alpha": 0.08,
    "max_alpha": 0.22,
    "mask_floor": 0.25,
    "mask_ceiling": 1.00,
    "sharpen_mode": "signed",
},
{
    "name": "ensemble_pos_a015",
    "denoiser": "ensemble",
    "mask_denoiser": "gaussian",
    "sign": 1.0,
    "base_alpha": 0.15,
    "min_alpha": 0.10,
    "max_alpha": 0.28,
    "mask_floor": 0.25,
    "mask_ceiling": 1.00,
    "sharpen_mode": "signed",
},
{
    "name": "ensemble_texture_a012",
    "denoiser": "ensemble",
    "mask_denoiser": "gaussian",
    "sign": 1.0,
    "base_alpha": 0.12,
    "min_alpha": 0.08,
    "max_alpha": 0.22,
    "mask_floor": 0.10,
    "mask_ceiling": 1.00,
    "sharpen_mode": "signed",
},
{
    "name": "ensemble_texture_a015",
    "denoiser": "ensemble",
    "mask_denoiser": "gaussian",
    "sign": 1.0,
    "base_alpha": 0.15,
    "min_alpha": 0.10,
    "max_alpha": 0.28,
    "mask_floor": 0.10,
    "mask_ceiling": 1.00,
    "sharpen_mode": "signed",
},
{
    "name": "ensemble_neg_a010",
    "denoiser": "ensemble",
    "mask_denoiser": "gaussian",
    "sign": -1.0,
    "base_alpha": 0.10,
    "min_alpha": 0.06,
    "max_alpha": 0.18,
    "mask_floor": 0.25,
    "mask_ceiling": 1.00,
    "sharpen_mode": "signed",
},
{
    "name": "ensemble_pos_a016",
    "denoiser": "ensemble",
    "mask_denoiser": "gaussian",
    "sign": 1.0,
    "base_alpha": 0.16,
    "min_alpha": 0.10,
    "max_alpha": 0.30,
    "mask_floor": 0.25,
    "mask_ceiling": 1.00,
    "sharpen_mode": "signed",
},
{
    "name": "ensemble_pos_a017",
    "denoiser": "ensemble",
    "mask_denoiser": "gaussian",
    "sign": 1.0,
    "base_alpha": 0.17,
    "min_alpha": 0.11,
    "max_alpha": 0.32,
    "mask_floor": 0.25,
    "mask_ceiling": 1.00,
    "sharpen_mode": "signed",
},
{
    "name": "ensemble_pos_a018",
    "denoiser": "ensemble",
    "mask_denoiser": "gaussian",
    "sign": 1.0,
    "base_alpha": 0.18,
    "min_alpha": 0.12,
    "max_alpha": 0.34,
    "mask_floor": 0.25,
    "mask_ceiling": 1.00,
    "sharpen_mode": "signed",
},
{
    "name": "ensemble_pos_a014",
    "denoiser": "ensemble",
    "mask_denoiser": "gaussian",
    "sign": 1.0,
    "base_alpha": 0.14,
    "min_alpha": 0.09,
    "max_alpha": 0.26,
    "mask_floor": 0.25,
    "mask_ceiling": 1.00,
    "sharpen_mode": "signed",
},
]


# =============================================================================
# Submission choice
# =============================================================================

SUBMISSION_EXPERIMENT_NAME = "ensemble_pos_a016"

RESIDUAL_EXPECTED_SUBMISSION_COUNT = EXPECTED_SUBMISSION_COUNT


# =============================================================================
# Visualization / reporting
# =============================================================================

SAVE_TEMPLATE_NUMPY = True
SAVE_TEMPLATE_VISUALS = True
SAVE_COMPARISON_VISUALS = True

NUM_VISUALIZE_PER_GROUP = 3

DIFF_VISUAL_SCALE = 10.0

PRINT_PROGRESS = True
VERIFY_OUTPUTS = True
OVERWRITE_EXISTING_OUTPUTS = True


# =============================================================================
# Folder helpers
# =============================================================================

def get_experiment_output_dir(experiment_name: str) -> Path:
    return RESIDUAL_FORGED_ROOT / experiment_name


def get_experiment_template_dir(experiment_name: str) -> Path:
    return RESIDUAL_TEMPLATE_ROOT / experiment_name


def get_experiment_visualization_dir(experiment_name: str) -> Path:
    return RESIDUAL_VISUALIZATION_ROOT / experiment_name


def get_experiment_by_name(experiment_name: str) -> dict:
    for experiment in EXPERIMENTS:
        if experiment["name"] == experiment_name:
            return experiment

    raise KeyError(
        f"Experiment '{experiment_name}' not found. "
        f"Available: {[e['name'] for e in EXPERIMENTS]}"
    )