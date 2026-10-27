"""Train or resume a class-weighted EfficientNetB0 DR classifier.

This is a separate experiment from the unweighted EfficientNet baseline. It
reuses the same model builder and official btgraham-300 splits. Class weights
are calculated from the official training split only and applied only to the
training loss through per-example sample weights.
"""

import math
from pathlib import Path

import numpy as np

from configs import tensorflow_setup as _tensorflow_setup

import tensorflow as tf

_tensorflow_setup.configure_gpu_memory_growth(tf)

import tensorflow_datasets as tfds

from configs.config import (
    CHECKPOINTS_TO_KEEP,
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TARGET_EPOCHS,
    EFFICIENTNET_BACKBONE_TRAINABLE,
    EFFICIENTNET_DROPOUT,
    EFFICIENTNET_LEARNING_RATE,
    EFFICIENTNET_VARIANT,
    EFFICIENTNET_WEIGHTS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    SEED,
    TRAIN_SPLIT,
)
from scripts.data_loader import load_builder, make_datasets
from scripts.train_efficientnet import build_efficientnet_model, _save_weights_atomically
from scripts.utils import (
    RecoveryCheckpointCallback,
    best_model_exists,
    create_new_run,
    find_latest_resumable_run,
    save_json,
    timestamp_now,
)


MODEL_NAME = "efficientnet_weighted"


def calculate_training_class_weights():
    """Return (class_counts, class_weights) using only the official train split.

    Balanced weights are w_c = N / (K * n_c), where N is the number of
    training examples, K is the number of classes, and n_c is the training
    count for class c. Skip image decoding because this pass only needs labels.
    """
    builder = load_builder()
    label_dataset = builder.as_dataset(
        split=TRAIN_SPLIT,
        shuffle_files=False,
        read_config=tfds.ReadConfig(try_autocache=False),
        decoders={"image": tfds.decode.SkipDecoding()},
    )

    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for example in label_dataset:
        label = int(example["label"].numpy())
        if label < 0 or label >= NUM_CLASSES:
            raise ValueError(f"Found out-of-range training label: {label}")
        class_counts[label] += 1

    total_examples = int(class_counts.sum())
    if total_examples == 0:
        raise ValueError("The official training split contained no examples.")
    if np.any(class_counts == 0):
        missing_classes = np.flatnonzero(class_counts == 0).tolist()
        raise ValueError(
            "Cannot calculate balanced class weights because the training split "
            f"has no examples for class(es): {missing_classes}"
        )

    class_weights = total_examples / (NUM_CLASSES * class_counts.astype(np.float64))
    return class_counts.astype(int).tolist(), class_weights.astype(float).tolist()


def add_sample_weights(dataset, class_weights):
    """Convert (images, labels) batches to (images, labels, sample_weights)."""
    weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    return dataset.map(
        lambda images, labels: (
            images,
            labels,
            tf.gather(weights_tensor, tf.cast(labels, tf.int32)),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


class BestValidationWeightsCallback(tf.keras.callbacks.Callback):
    """Save model weights only when the unweighted validation loss improves."""

    def __init__(self, best_model_path, best_val_loss, metadata_path, metadata):
        super().__init__()
        self.best_model_path = Path(best_model_path)
        self.best_val_loss = best_val_loss
        self.metadata_path = Path(metadata_path)
        self.metadata = metadata

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get("val_loss")
        if current_loss is None:
            return

        current_loss = float(current_loss)
        if math.isfinite(current_loss) and current_loss < float(self.best_val_loss.numpy()):
            # Reuse the baseline's atomic, weight-only saver to avoid the
            # EfficientNet/Keras full-model serialization issue.
            _save_weights_atomically(self.model, self.best_model_path)
            self.best_val_loss.assign(current_loss)
            self.metadata["best_val_loss"] = current_loss
            self.metadata["best_model_format"] = "keras_weights_hdf5"
            self.metadata["updated_at"] = timestamp_now()
            save_json(self.metadata_path, self.metadata)
            print(f"Saved best validation weights: {self.best_model_path}")


def _new_metadata(paths, batch_size, target_epochs, seed, class_counts, class_weights):
    return {
        "model_name": MODEL_NAME,
        "experiment": "class-weighted EfficientNet baseline",
        "architecture": "EfficientNetB0",
        "run_id": paths["run_id"],
        "status": "initialized",
        "created_at": timestamp_now(),
        "updated_at": timestamp_now(),
        "dataset": "diabetic_retinopathy_detection",
        "dataset_format": "btgraham-300",
        "input_size": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        "class_names": list(CLASS_NAMES),
        "pretrained_weights": EFFICIENTNET_WEIGHTS,
        "backbone_trainable": EFFICIENTNET_BACKBONE_TRAINABLE,
        "dropout_rate": EFFICIENTNET_DROPOUT,
        "optimizer": "Adam",
        "learning_rate": EFFICIENTNET_LEARNING_RATE,
        "loss": "SparseCategoricalCrossentropy with per-example training sample weights",
        "training_metric": "SparseCategoricalAccuracy (unweighted metric)",
        "batch_size": batch_size,
        "batch_sizes_used": [batch_size],
        "seed": seed,
        "class_weighting": "balanced inverse-frequency: N / (K * n_c), training split only",
        "training_class_counts": class_counts,
        "class_weights": class_weights,
        "validation_loss_weighting": "none",
        "test_split_used_for_weight_calculation": False,
        "target_epochs": target_epochs,
        "last_completed_epoch": 0,
        "best_val_loss": None,
        "checkpoint_dir": str(paths["checkpoint_dir"]),
        "best_model_path": str(paths["best_model_path"]),
        "training_history_path": str(paths["history_path"]),
    }


def train_efficientnet_weighted(
    run_mode,
    batch_size=DEFAULT_BATCH_SIZE,
    epochs=DEFAULT_TARGET_EPOCHS,
    seed=SEED,
    allow_cpu=False,
    learning_rate = 0
):
    """Start a new independent weighted run or resume its latest checkpoint.

    `epochs` is the total target epoch (not the number of extra epochs). A
    resume must find the weighted experiment's own valid recovery checkpoint.
    """
    if learning_rate != 0 :
        EFFICIENTNET_LEARNING_RATE = learning_rate

    run_mode = str(run_mode).strip().lower()
    if run_mode not in {"scratch", "resume"}:
        raise ValueError("run_mode must be 'scratch' or 'resume'")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1")
    if int(epochs) < 1:
        raise ValueError("epochs must be at least 1")

    batch_size = int(batch_size)
    epochs = int(epochs)
    devices = tf.config.list_physical_devices("GPU")
    if not devices and not allow_cpu:
        raise RuntimeError(
            "TensorFlow cannot see a GPU in this environment. No training was started. "
            "Resolve GPU setup first, or deliberately pass allow_cpu=True to accept "
            "potentially very slow CPU training."
        )

    tf.keras.utils.set_random_seed(seed)
    if devices:
        print("Training device(s):", ", ".join(device.name for device in devices))
    else:
        print("WARNING: Running explicitly on CPU. Training may be very slow.")

    if run_mode == "scratch":
        # Calculate weights before creating a run folder, so a data/count error
        # cannot leave behind an empty run that appears resumable.
        class_counts, class_weights = calculate_training_class_weights()
        paths = create_new_run(MODEL_NAME)
        metadata = _new_metadata(
            paths, batch_size, epochs, seed, class_counts, class_weights
        )
        save_json(paths["metadata_path"], metadata)
        start_epoch = 0
        print(f"Starting a new, separate weighted run: {paths['run_id']}")
    else:
        paths = find_latest_resumable_run(MODEL_NAME)
        if paths is None:
            raise FileNotFoundError(
                "Resume was selected, but no valid class-weighted EfficientNet recovery "
                "checkpoint was found. Choose 'scratch' explicitly for a fresh run."
            )
        metadata = paths["metadata"]
        if metadata.get("model_name") != MODEL_NAME:
            raise ValueError(
                f"The latest run metadata is not for {MODEL_NAME}; refusing to resume it."
            )
        if "class_weights" not in metadata or "training_class_counts" not in metadata:
            raise ValueError(
                "This run has no saved training-only class weights. Refusing to resume "
                "with newly calculated weights; inspect the run or choose scratch."
            )
        class_counts = [int(value) for value in metadata["training_class_counts"]]
        class_weights = [float(value) for value in metadata["class_weights"]]
        if len(class_counts) != NUM_CLASSES or len(class_weights) != NUM_CLASSES:
            raise ValueError("Saved class counts/weights do not match NUM_CLASSES.")

        start_epoch = int(metadata.get("last_completed_epoch", 0))
        if epochs <= start_epoch:
            raise ValueError(
                f"Run {paths['run_id']} already completed epoch {start_epoch}. Set Epoch "
                "to a larger total target (not a count of extra epochs)."
            )
        if not best_model_exists(paths["best_model_path"]) and start_epoch > 0:
            raise FileNotFoundError(
                f"Recovery checkpoint exists for {paths['run_id']}, but its best validation "
                f"weights are missing: {paths['best_model_path']}. Run was not resumed."
            )

        previous_batch_size = int(metadata.get("batch_size", batch_size))
        if previous_batch_size != batch_size:
            print(
                f"Resuming with batch_size={batch_size} (previously {previous_batch_size})."
            )
        batch_sizes_used = metadata.setdefault("batch_sizes_used", [previous_batch_size])
        if batch_size not in batch_sizes_used:
            batch_sizes_used.append(batch_size)
        print(f"Resuming weighted run {paths['run_id']} from: {paths['latest_checkpoint']}")
        metadata["batch_size"] = batch_size
        metadata["target_epochs"] = epochs
        metadata["updated_at"] = timestamp_now()

    try:
        # Scratch starts from ImageNet weights. On resume, the local recovery
        # checkpoint restores all model weights; no pretrained download is needed.
        initial_weights = EFFICIENTNET_WEIGHTS if run_mode == "scratch" else None
        model = build_efficientnet_model(weights=initial_weights)

        completed_epoch = tf.Variable(
            0, dtype=tf.int64, trainable=False, name="completed_epoch"
        )
        best_val_loss = tf.Variable(
            float("inf"), dtype=tf.float32, trainable=False, name="best_val_loss"
        )
        checkpoint = tf.train.Checkpoint(
            model=model,
            optimizer=model.optimizer,
            completed_epoch=completed_epoch,
            best_val_loss=best_val_loss,
        )
        manager = tf.train.CheckpointManager(
            checkpoint,
            str(paths["checkpoint_dir"]),
            max_to_keep=CHECKPOINTS_TO_KEEP,
        )

        if run_mode == "resume":
            status = checkpoint.restore(paths["latest_checkpoint"])
            status.assert_existing_objects_matched()
            start_epoch = int(completed_epoch.numpy())
            metadata["last_completed_epoch"] = start_epoch
            # Recovery checkpoints are intentionally saved before best-weight
            # export. Metadata may therefore hold the newer best loss. When the
            # best-weight artifact exists, restore its recorded selection loss.
            if best_model_exists(paths["best_model_path"]):
                saved_best_loss = metadata.get("best_val_loss")
                if saved_best_loss is not None and math.isfinite(float(saved_best_loss)):
                    best_val_loss.assign(float(saved_best_loss))

        metadata["status"] = "running"
        metadata["updated_at"] = timestamp_now()
        metadata["batch_size"] = batch_size
        metadata["target_epochs"] = epochs
        metadata["class_weights"] = class_weights
        metadata["training_class_counts"] = class_counts
        save_json(paths["metadata_path"], metadata)

        train_ds, validation_ds, _ = make_datasets(batch_size=batch_size, seed=seed)
        weighted_train_ds = add_sample_weights(train_ds, class_weights)
        callbacks = [
            # Persist resumable model/optimizer state before best-weight export.
            RecoveryCheckpointCallback(
                manager,
                completed_epoch,
                best_val_loss,
                paths["metadata_path"],
                metadata,
            ),
            tf.keras.callbacks.CSVLogger(
                str(paths["history_path"]),
                append=(run_mode == "resume"),
            ),
            BestValidationWeightsCallback(
                paths["best_model_path"],
                best_val_loss,
                paths["metadata_path"],
                metadata,
            ),
        ]

        print(f"Input size: {IMAGE_HEIGHT}x{IMAGE_WIDTH}; classes: {NUM_CLASSES}")
        print("Training-only class counts:", dict(enumerate(class_counts)))
        print("Training-only class weights:", dict(enumerate(class_weights)))
        print(f"Validation weighting: none; best model selected by lowest validation loss")
        print(f"Starting at epoch {start_epoch}; target total epochs: {epochs}")
        history = model.fit(
            weighted_train_ds,
            validation_data=validation_ds,
            initial_epoch=start_epoch,
            epochs=epochs,
            callbacks=callbacks,
        )

        if not best_model_exists(paths["best_model_path"]):
            raise RuntimeError(
                "Training ended without best validation weights. No finite improving "
                "validation loss was recorded; inspect the history before retrying."
            )

        completed = int(completed_epoch.numpy())
        metadata["last_completed_epoch"] = completed
        metadata["best_val_loss"] = float(best_val_loss.numpy())
        metadata["status"] = "completed" if completed >= epochs else "interrupted"
        metadata["updated_at"] = timestamp_now()
        save_json(paths["metadata_path"], metadata)

        print(f"Run status: {metadata['status']} ({completed}/{epochs} epochs)")
        print(f"Recovery checkpoints: {paths['checkpoint_dir']}")
        print(f"Best validation weights: {paths['best_model_path']}")
        print(f"Training history: {paths['history_path']}")
        return {
            "run_id": paths["run_id"],
            "paths": paths,
            "metadata": metadata,
            "history": history,
        }

    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        metadata["updated_at"] = timestamp_now()
        save_json(paths["metadata_path"], metadata)
        print(
            "Training interrupted. Resume this weighted run with run_mode='resume'. "
            f"Last completed epoch: {metadata.get('last_completed_epoch', 0)}"
        )
        raise
    except Exception as error:
        metadata["status"] = "failed"
        metadata["last_error"] = f"{type(error).__name__}: {error}"
        metadata["updated_at"] = timestamp_now()
        save_json(paths["metadata_path"], metadata)
        raise
