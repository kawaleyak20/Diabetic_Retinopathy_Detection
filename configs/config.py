"""Shared settings for the DR mini-project."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "btgraham-300"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset / labels
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

# Preserve existing TFDS splits; do not make a new random split.
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"

# Reproducibility and input pipeline
SEED = 42
SHUFFLE_BUFFER = 2048
DEFAULT_BATCH_SIZE = 4

# Initial EfficientNet baseline choice; can be revised after evaluation.
EFFICIENTNET_VARIANT = "B0"
EFFICIENTNET_WEIGHTS = "imagenet"
EFFICIENTNET_BACKBONE_TRAINABLE = False
EFFICIENTNET_DROPOUT = 0.30
EFFICIENTNET_LEARNING_RATE = 0.001
DEFAULT_TARGET_EPOCHS = 10
CHECKPOINTS_TO_KEEP = None  # Keep all epoch checkpoints within a run

# Training / evaluation control
AUTO_EVALUATE_AFTER_TRAINING = False
