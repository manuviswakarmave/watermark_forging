from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ZIP = PROJECT_ROOT / "Dataset.zip"
DATASET_DIR = PROJECT_ROOT / "Dataset"

CLEAN_TARGETS_DIR = DATASET_DIR / "clean_targets"
WATERMARKED_SOURCES_DIR = DATASET_DIR / "watermarked_sources"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FINAL_IMAGES_DIR = OUTPUT_DIR / "final_images"
SUBMISSION_ZIP = OUTPUT_DIR / "submission.zip"

# Mapping from watermark source folder to clean target image range
CATEGORIES = [
    ("WM_1", 1, 25),
    ("WM_2", 26, 50),
    ("WM_3", 51, 75),
    ("WM_4", 76, 100),
    ("WM_5", 101, 125),
    ("WM_6", 126, 150),
    ("WM_7", 151, 175),
    ("WM_8", 176, 200),
]

# Preference model training settings
SEED = 42
DEVICE = "cuda"   # "auto", "cuda", or "cpu"

PREFERENCE_BACKBONE = "convnextv2_tiny"
PREFERENCE_INPUT_SIZE = 256

# Training settings for ConvNeXt V2-Tiny
EPOCHS = 30
BATCH_SIZE = 4   # use 2 if you get CUDA out-of-memory

PRETRAINED = True
FREEZE_BACKBONE = False

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1

MIN_ARTIFACT_STRENGTH = 0.08
MAX_ARTIFACT_STRENGTH = 0.20

USE_ADV_NEGATIVE = False
ADV_STEPS = 1
ADV_STEP_SIZE = 0.005
ADV_WEIGHT = 0.5

VAL_FRACTION = 0.2
NUM_WORKERS = 0
PRINT_EVERY = 5

# Residual extraction settings
RESIDUAL_DIR = OUTPUT_DIR / "residuals"
PREFERENCE_CKPT_PATH = CHECKPOINT_DIR / "preference_model_best.pt"

EXTRACT_STEPS = 50
EXTRACT_LR = 0.005

# Regularization: keeps the removed residual small
EXTRACT_L2_WEIGHT = 1.0

# Maximum residual magnitude in [0, 1] image scale
EXTRACT_MAX_DELTA = 0.08

# "mean" or "median"
RESIDUAL_AGGREGATION = "mean"

# Forging settings
FORGED_IMAGES_DIR = OUTPUT_DIR / "forged_images"
SUBMISSION_ZIP = OUTPUT_DIR / "submission.zip"

# Start conservatively because your extracted residuals are quite strong.
FORGE_ALPHA_VALUES = {
    "WM_1": 0.30,
    "WM_2": 0.30,
    "WM_3": 0.30,
    "WM_4": 0.30,
    "WM_5": 0.30,
    "WM_6": 0.30,
    "WM_7": 0.30,
    "WM_8": 0.30,
}

# Extra safety clipping for applied residual after alpha scaling.
FORGE_MAX_PERTURBATION = 0.03