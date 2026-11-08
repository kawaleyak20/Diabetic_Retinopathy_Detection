"""Load the explicitly selected model and predict a five-grade DR label."""

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MODEL_FILE = PROJECT_ROOT / "models" / "active" / "model_metadata.json"
DEFAULT_CLASS_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR (PDR)",
]


def _load_metadata():
    if not ACTIVE_MODEL_FILE.is_file():
        raise FileNotFoundError(
            "Active-model settings are missing. Create models/active/model_metadata.json "
            "using the supplied template."
        )
    with ACTIVE_MODEL_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    model_key = metadata.get("model_key", "")
    weights_value = metadata.get("weights_path", "")
    if model_key not in {"efficientnet", "efficientnet_weighted", "densenet", "fusion"}:
        raise ValueError("Unsupported model_key in models/active/model_metadata.json.")
    if not weights_value:
        raise ValueError("Set weights_path in models/active/model_metadata.json.")

    weights_path = Path(weights_value)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Trained model weights were not found: {weights_path}. Update weights_path "
            "in models/active/model_metadata.json to a completed run."
        )

    metadata["_resolved_weights_path"] = str(weights_path)
    return metadata


def _build_model(model_key, input_size):
    # Register the project's Windows Conda DLL setup before importing TensorFlow.
    from configs import tensorflow_setup as _tensorflow_setup  # noqa: F401
    import tensorflow as tf

    _tensorflow_setup.configure_gpu_memory_growth(tf)

    # Build the same architecture with weights=None; the trained weights file
    # supplies the backbone and classifier weights without downloading ImageNet.
    input_shape = tuple(int(value) for value in input_size[:2])
    if model_key in {"efficientnet", "efficientnet_weighted"}:
        from scripts.train_efficientnet import build_efficientnet_model
        model = build_efficientnet_model(weights=None, input_size=input_shape)
    elif model_key == "densenet":
        from scripts.train_densenet import build_densenet_model
        model = build_densenet_model(weights=None, input_size=input_shape)
    else:
        from scripts.train_fusion import build_fusion_model
        model = build_fusion_model(weights=None, input_size=input_shape)

    return model


@lru_cache(maxsize=1)
def _get_model(weights_path, model_key, input_size):
    model = _build_model(model_key, input_size)
    model.load_weights(weights_path)
    return model


def _crop_black_border(image, tolerance=7):
    """Crop the near-black fundus border, preserving the retinal field."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tolerance
    if not mask.any():
        return image
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return image[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def preprocess_scan(image_bytes, image_size=300):
    """Apply a Ben Graham-style crop/contrast transform and resize to 300x300.

    Model training uses TFDS's btgraham-300 format. This inference transform is
    intended to make ordinary uploaded fundus photos closer to that input style.
    """
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise ValueError("The image is too small to be a retinal scan.")

    image = _crop_black_border(image)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=10)
    image = cv2.addWeighted(image, 4, blurred, -4, 128)
    image = np.clip(image, 0, 255).astype(np.float32)
    return np.expand_dims(image, axis=0)


def predict_scan(image_bytes):
    metadata = _load_metadata()
    input_size = metadata.get("input_size", [300, 300, 3])
    if (
        not isinstance(input_size, list)
        or len(input_size) != 3
        or any(int(value) < 1 for value in input_size)
        or int(input_size[2]) != 3
    ):
        raise ValueError("Active model metadata must define input_size as [height, width, 3].")
    image_size = (int(input_size[0]), int(input_size[1]))
    class_names = metadata.get("class_names", DEFAULT_CLASS_NAMES)
    if len(class_names) != 5:
        raise ValueError("Active model metadata must define the five DR class names.")

    image_batch = preprocess_scan(image_bytes, image_size=image_size)
    model = _get_model(
        metadata["_resolved_weights_path"], metadata["model_key"], input_size
    )
    probabilities = np.asarray(model.predict(image_batch, verbose=0))[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_stage = class_names[predicted_index]

    return {
        "dr_detected": predicted_index != 0,
        "stage": predicted_stage,
        "stage_index": predicted_index,
        "confidence_percent": round(float(probabilities[predicted_index]) * 100, 1),
    }
