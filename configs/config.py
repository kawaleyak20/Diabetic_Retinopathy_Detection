"""Single source of truth for shared project settings.

Keep the constant names in this module stable: training scripts, evaluation,
notebooks, and the inference interface import them directly. Change experiment
settings here rather than duplicating values in individual files.
"""

from pathlib import Path

# Project paths. Resolve these from the repository, never from notebook CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# All current experiments and saved manifests use the btgraham-300 TFDS format.
DATA_DIR = PROJECT_ROOT / "data" / "btgraham-300"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset identity and the fixed label order used by every model/report.
EXPECTED_DATASET_NAME = "diabetic_retinopathy_detection"
# This is the model/pipeline size; it is separate from the TFDS format name.
# Keep it aligned with completed run metadata and active-model metadata.
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

# Preserve official TFDS split names; never create a new random split here.
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"

# Reproducibility, memory, and training defaults.
SEED = 42
SHUFFLE_BUFFER = 2048
DEFAULT_BATCH_SIZE = 4
DEFAULT_TARGET_EPOCHS = 10
CHECKPOINTS_TO_KEEP = None  # Keep every epoch checkpoint inside a run.

# Sampling policy. The preparation notebook creates the manifest once; model
# scripts only consume it. Change these values only when deliberately creating
# a new experiment and regenerating the manifest.
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
    / (
        f"fixed_splits_seed{UNDERSAMPLE_SEED}_cap{UNDERSAMPLE_MAX_PER_CLASS}_"
        f"{TRAIN_RATIO}-{VALIDATION_RATIO}-{TEST_RATIO}.json"
    )
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

# EfficientNetB0 baseline. ImageNet weights are loaded explicitly; the frozen
# backbone keeps the experiment practical on the 4 GB GPU.
EFFICIENTNET_VARIANT = "B0"
EFFICIENTNET_WEIGHTS = "imagenet"
EFFICIENTNET_BACKBONE_TRAINABLE = False
EFFICIENTNET_DROPOUT = 0.30
EFFICIENTNET_LEARNING_RATE = 0.001

# Class-weighted DenseNet121 comparison experiment.
DENSENET_WEIGHTS = "imagenet"
DENSENET_BACKBONE_TRAINABLE = False
DENSENET_DROPOUT = 0.30
DENSENET_LEARNING_RATE = 0.001

# Feature-fusion comparison. Both branches are frozen and concatenated after
# global average pooling; attention is intentionally not enabled.
FUSION_WEIGHTS = "imagenet"
FUSION_BACKBONES_TRAINABLE = False
FUSION_CLASS_WEIGHTED = False
FUSION_DROPOUT = 0.30
FUSION_LEARNING_RATE = 0.001

# Evaluation remains a separate explicit notebook action; no hidden evaluation
# is triggered after training.
AUTO_EVALUATE_AFTER_TRAINING = False
