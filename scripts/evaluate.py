"""Manual evaluation and explicit run-to-run comparison for the DR mini-project.

Evaluation never trains a model or changes model/checkpoint files. Each call
creates a new isolated results/evaluations/<model>/<run>/eval_NNN/ directory.
"""

from configs import tensorflow_setup as _tensorflow_setup

import tensorflow as tf

_tensorflow_setup.configure_gpu_memory_growth(tf)

import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
)

from configs.config import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    FUSION_CLASS_WEIGHTED,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RESULTS_DIR,
    SEED,
)
from scripts.data_loader import make_evaluation_dataset
from scripts.utils import load_json, run_paths, timestamp_now
from scripts.train_efficientnet import build_efficientnet_model
from scripts.train_densenet import build_densenet_model
from scripts.train_fusion import build_fusion_model


MODEL_BUILDERS = {
    "efficientnet": build_efficientnet_model,
    "efficientnet_weighted": build_efficientnet_model,
    "densenet": build_densenet_model,
    "fusion": build_fusion_model,
}
MODEL_DISPLAY_NAMES = {
    "efficientnet": "EfficientNetB0 (unweighted)",
    "efficientnet_weighted": "EfficientNetB0 (class-weighted)",
    "densenet": "DenseNet121 (class-weighted)",
    "fusion": (
        "EfficientNetB0 + DenseNet121 fusion (class-weighted)"
        if FUSION_CLASS_WEIGHTED
        else "EfficientNetB0 + DenseNet121 fusion (unweighted)"
    ),
}
MODEL_ARCHITECTURES = {
    "efficientnet": "EfficientNetB0",
    "efficientnet_weighted": "EfficientNetB0",
    "densenet": "DenseNet121",
    "fusion": "EfficientNetB0 + DenseNet121 late feature fusion",
}
MODEL_IS_WEIGHTED = {
    "efficientnet": False,
    "efficientnet_weighted": True,
    "densenet": True,
    "fusion": FUSION_CLASS_WEIGHTED,
}
RUN_ID_PATTERN = re.compile(r"^run_\d{3,}$")
EVALUATION_ID_PATTERN = re.compile(r"^eval_\d{3,}$")


def _evaluation_root(model_name, run_id):
    return RESULTS_DIR / "evaluations" / model_name / run_id


def _next_numbered_directory(parent, prefix):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{3,}})$")
    numbers = []
    for path in parent.iterdir():
        match = pattern.match(path.name)
        if path.is_dir() and match:
            numbers.append(int(match.group(1)))
    new_path = parent / f"{prefix}_{max(numbers, default=0) + 1:03d}"
    new_path.mkdir(parents=False, exist_ok=False)
    return new_path


def _check_model_run(model_name, run_id):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model_name {model_name!r}. Choose one of: {list(MODEL_BUILDERS)}"
        )
    if not RUN_ID_PATTERN.fullmatch(str(run_id)):
        raise ValueError("run_id must look like 'run_001'.")

    paths = run_paths(model_name, str(run_id))
    if not paths["metadata_path"].is_file():
        raise FileNotFoundError(f"Run metadata not found: {paths['metadata_path']}")
    metadata = load_json(paths["metadata_path"])

    if metadata.get("model_name") != model_name or metadata.get("run_id") != run_id:
        raise ValueError("The selected run metadata does not match model_name/run_id.")
    expected_architecture = MODEL_ARCHITECTURES[model_name]
    if metadata.get("architecture") != expected_architecture:
        raise ValueError(
            f"Run architecture {metadata.get('architecture')!r} does not match "
            f"the expected {expected_architecture!r} for {model_name!r}."
        )
    uses_class_weights = metadata.get("class_weights") is not None
    if uses_class_weights != MODEL_IS_WEIGHTED[model_name]:
        raise ValueError(
            f"Run class-weighting metadata does not match the experiment key {model_name!r}."
        )
    if metadata.get("dataset_format", "btgraham-300") != "btgraham-300":
        raise ValueError("This evaluator expects the project's btgraham-300 input format.")
    expected_input_size = [IMAGE_HEIGHT, IMAGE_WIDTH, 3]
    if metadata.get("input_size", expected_input_size) != expected_input_size:
        raise ValueError(
            f"Run input size {metadata.get('input_size')} does not match the current "
            f"project input size {expected_input_size}."
        )
    if metadata.get("status") != "completed":
        raise ValueError(
            f"Run {model_name}/{run_id} status is {metadata.get('status')!r}. "
            "Only successfully completed runs can be evaluated."
        )
    if not paths["best_model_path"].is_file() or paths["best_model_path"].stat().st_size == 0:
        raise FileNotFoundError(f"Best validation weights not found: {paths['best_model_path']}")
    if metadata.get("class_names") and list(metadata["class_names"]) != list(CLASS_NAMES):
        raise ValueError("Run class order differs from the current project class order.")
    return paths, metadata


def _prior_test_evaluation_exists(model_name, run_id):
    root = _evaluation_root(model_name, run_id)
    for metrics_path in root.glob("eval_*/metrics.json"):
        try:
            metrics = load_json(metrics_path)
        except (OSError, ValueError):
            continue
        if metrics.get("split") == "test":
            return True
    return False


def _write_confusion_matrix_csv(path, matrix):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label"] + list(CLASS_NAMES))
        for class_name, row in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([class_name] + row)


def _save_confusion_matrix_plot(path, matrix):
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, interpolation="nearest")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                format(int(matrix[row_index, column_index]), "d"),
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def evaluate_model(
    model_name,
    run_id,
    split="validation",
    batch_size=DEFAULT_BATCH_SIZE,
    seed=SEED,
    confirm_test=False,
    allow_repeated_test=False,
):
    """Evaluate one completed model run on the official validation or test split.

    Test evaluation requires ``confirm_test=True``. Re-evaluating the same model
    run on test requires the additional explicit ``allow_repeated_test=True``.
    Use validation metrics for model selection; do not repeatedly tune against test.
    """
    split = str(split).strip().lower()
    if split not in {"validation", "test"}:
        raise ValueError("split must be either 'validation' or 'test'; training is not evaluation.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1")
    if split == "test" and not confirm_test:
        raise ValueError(
            "Test is held out for final evaluation. To proceed deliberately, call with "
            "confirm_test=True after model selection is complete."
        )
    paths, run_metadata = _check_model_run(model_name, run_id)
    if split == "test" and _prior_test_evaluation_exists(model_name, run_id) and not allow_repeated_test:
        raise RuntimeError(
            f"{model_name}/{run_id} already has a test evaluation. Repeated test evaluation "
            "is disabled to help avoid test-set tuning. Pass allow_repeated_test=True only "
            "for a deliberate re-evaluation."
        )

    # Best artifacts contain all trained weights. Rebuild the same architecture
    # without downloading ImageNet weights, then load the selected run's artifact.
    model = MODEL_BUILDERS[model_name](weights=None)
    model.load_weights(str(paths["best_model_path"]))
    dataset = make_evaluation_dataset(split, batch_size=int(batch_size))

    true_batches = []
    probability_batches = []
    example_count = 0
    for images, labels in dataset:
        probabilities = model(images, training=False).numpy()
        labels_array = labels.numpy().astype(np.int64, copy=False)
        true_batches.append(labels_array)
        probability_batches.append(probabilities)
        example_count += int(labels_array.shape[0])
        if example_count and example_count % (int(batch_size) * 250) == 0:
            print(f"Evaluated {example_count:,} examples...")

    if not true_batches:
        raise ValueError(f"The selected '{split}' split contained no examples.")

    y_true = np.concatenate(true_batches)
    probabilities = np.concatenate(probability_batches, axis=0)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected a probability matrix with {len(CLASS_NAMES)} classes, "
            f"received shape {probabilities.shape}."
        )
    if len(y_true) != len(probabilities):
        raise ValueError("Prediction count does not match the number of true labels.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Model predictions contain NaN or infinite values; metrics were not saved.")
    y_pred = np.argmax(probabilities, axis=1).astype(np.int64)
    label_ids = list(range(len(CLASS_NAMES)))
    # Create output only after model loading, inference, and prediction checks pass.
    output_dir = _next_numbered_directory(_evaluation_root(model_name, run_id), "eval")
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )
    qwk_value = cohen_kappa_score(y_true, y_pred, labels=label_ids, weights="quadratic")
    qwk_value = float(qwk_value) if qwk_value is not None and math.isfinite(float(qwk_value)) else None

    metrics = {
        "model_name": model_name,
        "model_display_name": MODEL_DISPLAY_NAMES[model_name],
        "run_id": run_id,
        "evaluation_id": output_dir.name,
        "split": split,
        "dataset_format": run_metadata.get("dataset_format", "btgraham-300"),
        "class_names": list(CLASS_NAMES),
        "n_examples": int(len(y_true)),
        "batch_size": int(batch_size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)),
        "quadratic_weighted_kappa": qwk_value,
        "per_class_recall": {CLASS_NAMES[i]: float(recall[i]) for i in label_ids},
        "confusion_matrix": matrix.tolist(),
        "run_status_at_evaluation": run_metadata.get("status"),
        "run_best_validation_loss": run_metadata.get("best_val_loss"),
        "evaluated_at": timestamp_now(),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")

    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1_score", "support"])
        for index, class_name in enumerate(CLASS_NAMES):
            writer.writerow([index, class_name, precision[index], recall[index], f1_values[index], int(support[index])])

    _write_confusion_matrix_csv(output_dir / "confusion_matrix.csv", matrix)
    _save_confusion_matrix_plot(output_dir / "confusion_matrix.png", matrix)

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "example_index", "true_label", "true_class", "predicted_label", "predicted_class",
            "probability_no_dr", "probability_mild_npdr", "probability_moderate_npdr",
            "probability_severe_npdr", "probability_pdr",
        ])
        for index, (true_label, predicted_label, row) in enumerate(zip(y_true, y_pred, probabilities)):
            writer.writerow([
                index,
                int(true_label), CLASS_NAMES[int(true_label)],
                int(predicted_label), CLASS_NAMES[int(predicted_label)],
                *[float(value) for value in row],
            ])

    with (output_dir / "run_metadata_snapshot.json").open("w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")

    print(f"\nEvaluation saved to: {output_dir}")
    print(f"Split: {split}; examples: {len(y_true):,}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Quadratic weighted kappa: {metrics['quadratic_weighted_kappa']}")
    for name, value in metrics["per_class_recall"].items():
        print(f"Recall — {name}: {value:.4f}")

    del model
    tf.keras.backend.clear_session()
    return {"model_name": model_name, "run_id": run_id, "evaluation_id": output_dir.name,
            "split": split, "output_dir": str(output_dir), "metrics": metrics}


def list_saved_evaluations(model_name=None):
    """List evaluation reports without loading or evaluating any model."""
    if model_name is not None and model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_name: {model_name}")
    root = RESULTS_DIR / "evaluations"
    if not root.exists():
        return []
    model_names = [model_name] if model_name else sorted(MODEL_BUILDERS)
    rows = []
    for selected_model in model_names:
        if selected_model not in MODEL_BUILDERS:
            raise ValueError(f"Unknown model_name: {selected_model}")
        for metrics_path in sorted((root / selected_model).glob("run_*/eval_*/metrics.json")):
            try:
                metrics = load_json(metrics_path)
            except (OSError, ValueError):
                continue
            rows.append({
                "model_name": selected_model,
                "model_display_name": metrics.get("model_display_name", selected_model),
                "run_id": metrics.get("run_id", metrics_path.parents[1].name),
                "evaluation_id": metrics.get("evaluation_id", metrics_path.parent.name),
                "split": metrics.get("split", "unknown"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "quadratic_weighted_kappa": metrics.get("quadratic_weighted_kappa"),
                "output_dir": str(metrics_path.parent),
            })
    return rows


def compare_evaluations(selected_evaluations):
    """Compare only the exact saved evaluations supplied by the user.

    Each item must contain model_name, run_id, and evaluation_id. All selected
    reports must be from the same official split. The function never auto-selects
    or sweeps all evaluation records.
    """
    if not selected_evaluations or len(selected_evaluations) < 2:
        raise ValueError("Select at least two specific evaluation reports to compare.")

    rows = []
    seen = set()
    compared_runs = set()
    splits = set()
    dataset_formats = set()
    class_orders = set()
    example_counts = set()
    for item in selected_evaluations:
        model_name = item.get("model_name")
        run_id = item.get("run_id")
        evaluation_id = item.get("evaluation_id")
        if model_name not in MODEL_BUILDERS:
            raise ValueError(f"Unknown model_name in comparison: {model_name!r}")
        if not RUN_ID_PATTERN.fullmatch(str(run_id)):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        if not EVALUATION_ID_PATTERN.fullmatch(str(evaluation_id)):
            raise ValueError(f"Invalid evaluation_id: {evaluation_id!r}")
        identity = (model_name, run_id, evaluation_id)
        if identity in seen:
            raise ValueError(f"Duplicate evaluation selected: {identity}")
        seen.add(identity)
        run_identity = (model_name, run_id)
        if run_identity in compared_runs:
            raise ValueError(
                "Select only one evaluation per model/run. Repeated evaluations of the "
                "same run on the same split do not provide a model comparison."
            )
        compared_runs.add(run_identity)

        metrics_path = _evaluation_root(model_name, run_id) / evaluation_id / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Evaluation report not found: {metrics_path}")
        metrics = load_json(metrics_path)
        if (metrics.get("model_name"), metrics.get("run_id"), metrics.get("evaluation_id")) != identity:
            raise ValueError(f"Selected report metadata does not match its path: {metrics_path}")
        splits.add(metrics.get("split"))
        dataset_formats.add(metrics.get("dataset_format", "btgraham-300"))
        class_orders.add(tuple(metrics.get("class_names", [])))
        example_counts.add(metrics.get("n_examples"))
        row = {
            "model_name": model_name,
            "model": metrics.get("model_display_name", MODEL_DISPLAY_NAMES[model_name]),
            "run_id": run_id,
            "evaluation_id": evaluation_id,
            "split": metrics.get("split"),
            "n_examples": metrics.get("n_examples"),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "quadratic_weighted_kappa": metrics.get("quadratic_weighted_kappa"),
        }
        for class_name, recall_value in metrics.get("per_class_recall", {}).items():
            column_name = "recall_" + re.sub(r"[^a-z0-9]+", "_", class_name.lower()).strip("_")
            row[column_name] = recall_value
        rows.append(row)

    if len(splits) != 1:
        raise ValueError(f"Cannot compare reports from different splits: {sorted(str(s) for s in splits)}")
    if len(dataset_formats) != 1 or len(class_orders) != 1 or len(example_counts) != 1:
        raise ValueError(
            "Selected reports do not share the same dataset format, class order, and "
            "number of examples; a direct comparison would be misleading."
        )

    output_dir = _next_numbered_directory(RESULTS_DIR / "comparisons", "comparison")
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as file:
        json.dump({"split": next(iter(splits)), "selected_evaluations": rows,
                   "created_at": timestamp_now()}, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")

    print(f"Comparison saved to: {output_dir}")
    return {"output_dir": str(output_dir), "split": next(iter(splits)), "rows": rows}
