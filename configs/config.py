"""Shared settings for the diabetic-retinopathy mini-project."""

from pathlib import Path

# Project paths. Resolve everything from this file, not from the notebook CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "250K"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset and fixed class order.
EXPECTED_DATASET_NAME = "diabetic_retinopathy_detection"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
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
