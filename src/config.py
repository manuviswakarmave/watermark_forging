# src/config.py

from pathlib import Path


# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "Dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

WATERMARKED_SOURCES_DIR = DATASET_ROOT / "watermarked_sources"
CLEAN_TARGETS_DIR = DATASET_ROOT / "clean_targets"

CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints" / "wm_copier"
FORGED_ROOT = OUTPUT_ROOT / "forged" / "wm_copier"
VISUALIZATION_ROOT = OUTPUT_ROOT / "visualizations" / "wm_copier"

SUBMISSION_ZIP_PATH = OUTPUT_ROOT / "submission.zip"


# =============================================================================
# Dataset structure
# =============================================================================

GROUPS = [
    "WM_1",
    "WM_2",
    "WM_3",
    "WM_4",
    "WM_5",
    "WM_6",
    "WM_7",
    "WM_8",
]

TARGET_MAPPING = {
    "WM_1": (1, 25),
    "WM_2": (26, 50),
    "WM_3": (51, 75),
    "WM_4": (76, 100),
    "WM_5": (101, 125),
    "WM_6": (126, 150),
    "WM_7": (151, 175),
    "WM_8": (176, 200),
}

COLOR_MODE = "rgb"

IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
]


# =============================================================================
# WMCopier diffusion settings
# =============================================================================

DDIM_TOTAL_STEPS = 100

# Refinement experiment:
# Use only paper default shallow inversion step T_S = 40.
DDIM_SHALLOW_STEPS = [40]


# =============================================================================
# Diffusion model architecture - 4 GB GPU safe
# =============================================================================

MODEL_BASE_CHANNELS = 16
MODEL_CHANNEL_MULTS = [1, 2, 2, 2]
MODEL_TIME_EMBED_DIM = 64
MODEL_DROPOUT = 0.10

IN_CHANNELS = 3
OUT_CHANNELS = 3


# =============================================================================
# Diffusion training hyperparameters
# =============================================================================

USE_PAPER_TRAINING_HYPERPARAMS = False

# Paper values kept only for reference.
PAPER_TRAIN_STEPS = 20_000
PAPER_BATCH_SIZE = 256
PAPER_LR = 1e-4

# If checkpoints already exist, main.py will load them and skip retraining.
# If checkpoints do not exist, it will train using these values.
TRAIN_STEPS = 1000
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

DEVICE = "cuda"


# =============================================================================
# Overfitting / memorization / memory control
# =============================================================================

USE_TRAIN_VAL_SPLIT = True
VAL_IMAGES_PER_GROUP = 5
RANDOM_SEED = 42

VALIDATE_EVERY = 100
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4

# Keep EMA off because your RTX 3050 has 4 GB VRAM.
USE_EMA = False
EMA_DECAY = 0.999

SAVE_BEST_CHECKPOINT_ONLY = True


# =============================================================================
# Watermark-preserving augmentation
# =============================================================================

USE_AUGMENTATION = True

AUGMENTED_SAMPLES_PER_GROUP = 1000

AUG_BRIGHTNESS = 0.02
AUG_CONTRAST = 0.02
AUG_GAMMA = 0.02
AUG_NOISE_STD = 0.002

AUG_JPEG_PROB = 0.10
AUG_JPEG_QUALITY_MIN = 98
AUG_JPEG_QUALITY_MAX = 100

AUG_USE_ROTATION = False
AUG_USE_CROP = False
AUG_USE_FLIP = False


# =============================================================================
# Watermark injection / forging
# =============================================================================

FORGED_METHOD_NAME = "wm_copier"

SAVE_FOR_EACH_SHALLOW_STEP = True

CLIP_OUTPUT = True


# =============================================================================
# Refinement settings
# =============================================================================

# Refinement ON for this experiment.
USE_REFINEMENT = True

# Paper default lambda.
REFINE_LAMBDA = 100.0

# Start with 30 first.
# If score improves, later try 100.
REFINE_ITERATIONS = 30

# Paper default low-noise step and step size.
REFINE_LOW_NOISE_STEP = 1
REFINE_STEP_SIZE = 1e-4

# Save only refined outputs for this run.
SAVE_UNREFINED_OUTPUTS = False
SAVE_REFINED_OUTPUTS = True


# =============================================================================
# Visualization / debugging
# =============================================================================

NUM_VISUALIZE = 1

SAVE_TRAINING_LOSS_PLOTS = True

# Keep off to reduce runtime and disk clutter.
SAVE_SAMPLE_VISUALIZATIONS = False

PRINT_EVERY = 100


# =============================================================================
# Submission settings
# =============================================================================

SUBMISSION_SHALLOW_STEP = 40
SUBMISSION_USE_REFINED = True

EXPECTED_SUBMISSION_COUNT = 200