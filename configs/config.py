"""Shared settings for the diabetic-retinopathy mini-project."""

from pathlib import Path

# Project paths. Resolve everything from this file, not from the notebook CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "btgraham-300"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset and fixed class order.
EXPECTED_DATASET_NAME = "diabetic_retinopathy_detection"
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
NUM_CLASSES = 5
CLASS_NAMES = (
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR (PDR)",
)

# Preserve the existing TFDS splits; never make a new random split.
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"

# Reproducibility and input-pipeline defaults.
SEED = 42
SHUFFLE_BUFFER = 2048
DEFAULT_BATCH_SIZE = 4
DEFAULT_TARGET_EPOCHS = 10
CHECKPOINTS_TO_KEEP = None  # Keep every epoch checkpoint inside a run.

# Fast fixed-subset experiment. One standalone notebook creates a single manifest
# for all three official splits. Model scripts never perform random sampling.
TRAINING_SAMPLING_MODE = "undersampled"
UNDERSAMPLE_MAX_PER_CLASS = 2500
UNDERSAMPLE_SEED = SEED
TRAIN_RATIO = 70
VALIDATION_RATIO = 10
TEST_RATIO = 20
UNDERSAMPLING_MANIFEST_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "undersampling"
    / f"fixed_splits_seed{UNDERSAMPLE_SEED}_cap{UNDERSAMPLE_MAX_PER_CLASS}_70-10-20.json"
)


def normalize_training_sampling_mode(mode=None):
    """Resolve and validate the shared training-sampling mode."""
    value = TRAINING_SAMPLING_MODE if mode is None else mode
    value = str(value).strip().lower()
    if value not in {"full", "undersampled"}:
        raise ValueError(
            "Training sampling mode must be 'full' or 'undersampled', "
            f"not {value!r}."
        )
    if value == "undersampled" and UNDERSAMPLE_MAX_PER_CLASS < 1:
        raise ValueError("UNDERSAMPLE_MAX_PER_CLASS must be at least 1.")
    return value


def training_model_name(base_model_name, mode=None):
    """Return an isolated output key for the selected sampling strategy."""
    sampling_mode = normalize_training_sampling_mode(mode)
    if sampling_mode == "undersampled":
        return f"{base_model_name}_undersampled"
    return str(base_model_name)

# Unweighted EfficientNetB0 baseline.
EFFICIENTNET_VARIANT = "B0"
EFFICIENTNET_WEIGHTS = "imagenet"
EFFICIENTNET_BACKBONE_TRAINABLE = False
EFFICIENTNET_DROPOUT = 0.30
EFFICIENTNET_LEARNING_RATE = 0.001

# Class-weighted DenseNet121 experiment.
DENSENET_WEIGHTS = "imagenet"
DENSENET_BACKBONE_TRAINABLE = False
DENSENET_DROPOUT = 0.30
DENSENET_LEARNING_RATE = 0.001

# Initial, unweighted EfficientNetB0 + DenseNet121 feature-fusion experiment.
FUSION_WEIGHTS = "imagenet"
FUSION_BACKBONES_TRAINABLE = False
FUSION_CLASS_WEIGHTED = False
FUSION_DROPOUT = 0.30
FUSION_LEARNING_RATE = 0.001

# Training/evaluation control. Evaluation is always a separate explicit action.
AUTO_EVALUATE_AFTER_TRAINING = False
