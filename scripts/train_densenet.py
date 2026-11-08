"""Train or resume the class-weighted DenseNet121 DR experiment.

Undersampled runs consume the fixed manifest from prepare_undersampled_split.ipynb;
this script does not choose or drop random examples.
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
    DENSENET_BACKBONE_TRAINABLE,
    DENSENET_DROPOUT,
    DENSENET_LEARNING_RATE,
    DENSENET_WEIGHTS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    SEED,
    TRAIN_SPLIT,
)
from scripts.data_loader import load_builder, make_datasets, get_training_sampling_summary
from configs.config import normalize_training_sampling_mode, training_model_name
from scripts.train_efficientnet import _save_weights_atomically
from scripts.utils import (
    RecoveryCheckpointCallback,
    best_model_exists,
    create_new_run,
    find_latest_resumable_run,
    save_json,
    timestamp_now,
)

MODEL_NAME = "densenet"
ARCHITECTURE = "DenseNet121"
USE_CLASS_WEIGHTS = True


def build_densenet_model(weights=DENSENET_WEIGHTS, input_size=None):
    """Build DenseNet121 with its ImageNet preprocessing inside the model."""
    height, width = input_size or (IMAGE_HEIGHT, IMAGE_WIDTH)
    backbone = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=(int(height), int(width), 3),
    )
    backbone.trainable = DENSENET_BACKBONE_TRAINABLE

    inputs = tf.keras.Input(shape=(int(height), int(width), 3), name="retinal_image")
    processed = tf.keras.layers.Lambda(
        tf.keras.applications.densenet.preprocess_input,
        name="densenet_preprocessing",
    )(inputs)
    features = backbone(processed, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    features = tf.keras.layers.Dropout(DENSENET_DROPOUT, name="head_dropout")(features)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="dr_severity")(features)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="densenet121_dr_weighted")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=DENSENET_LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def calculate_training_class_weights(seed=SEED, training_sampling_mode=None):
    """Return counts and inverse-frequency weights for the selected train subset.

    Counts come only from the official training split. In undersampled mode they
    describe the retained examples, not the original imbalanced split.
    """
    sampling_info = get_training_sampling_summary(
        seed=seed,
        training_sampling_mode=training_sampling_mode,
    )
    class_counts = np.asarray(sampling_info["training_class_counts"], dtype=np.int64)
    total_examples = int(class_counts.sum())
    if total_examples == 0:
        raise ValueError("The selected training subset contained no examples.")
    if np.any(class_counts == 0):
        missing_classes = np.flatnonzero(class_counts == 0).tolist()
        raise ValueError(
            "Cannot calculate balanced class weights because the selected training "
            f"subset has no examples for class(es): {missing_classes}"
        )
    class_weights = total_examples / (NUM_CLASSES * class_counts.astype(np.float64))
    return class_counts.astype(int).tolist(), class_weights.astype(float).tolist()

def add_sample_weights(dataset, class_weights):
    """Attach per-example weights to batches; validation remains unweighted."""
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
    """Save best model weights when unweighted validation loss improves."""

    def __init__(self, best_model_path, best_val_loss, metadata_path, metadata):
        super().__init__()
        self.best_model_path = Path(best_model_path)
        self.best_val_loss = best_val_loss
        self.metadata_path = Path(metadata_path)
        self.metadata = metadata

    def on_epoch_end(self, epoch, logs=None):
        current_loss = (logs or {}).get("val_loss")
        if current_loss is None:
            return
        current_loss = float(current_loss)
        if math.isfinite(current_loss) and current_loss < float(self.best_val_loss.numpy()):
            _save_weights_atomically(self.model, self.best_model_path)
            self.best_val_loss.assign(current_loss)
            self.metadata["best_val_loss"] = current_loss
            self.metadata["best_model_format"] = "keras_weights_hdf5"
            self.metadata["updated_at"] = timestamp_now()
            save_json(self.metadata_path, self.metadata)
            print(f"Saved best validation weights: {self.best_model_path}")


def _new_metadata(paths, batch_size, epochs, seed, class_counts, class_weights, model_name, sampling_info):
    metadata = {
        "model_name": model_name,
        "experiment": "class-weighted DenseNet121 comparison",
        "architecture": ARCHITECTURE,
        "run_id": paths["run_id"],
        "status": "initialized",
        "created_at": timestamp_now(),
        "updated_at": timestamp_now(),
        "dataset": "diabetic_retinopathy_detection",
        "dataset_format": "btgraham-300",
        "input_size": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        "class_names": list(CLASS_NAMES),
        "pretrained_weights": DENSENET_WEIGHTS,
        "backbone_trainable": DENSENET_BACKBONE_TRAINABLE,
        "preprocessing": "tf.keras.applications.densenet.preprocess_input inside model",
        "dropout_rate": DENSENET_DROPOUT,
        "optimizer": "Adam",
        "learning_rate": DENSENET_LEARNING_RATE,
        "loss": "SparseCategoricalCrossentropy with training sample weights",
        "training_metric": "SparseCategoricalAccuracy (unweighted reporting metric)",
        "classification_head": "GlobalAveragePooling2D -> Dropout -> Dense(5, softmax)",
        "batch_size": batch_size,
        "batch_sizes_used": [batch_size],
        "seed": seed,
        "class_weighting": "balanced inverse-frequency: N / (K * n_c), selected training subset only",
        **sampling_info,
        "training_class_counts": class_counts,
        "class_weights": class_weights,
        "validation_loss_weighting": "none",
        "test_split_used_for_weight_calculation": False,
        "target_epochs": epochs,
        "last_completed_epoch": 0,
        "best_val_loss": None,
        "checkpoint_dir": str(paths["checkpoint_dir"]),
        "best_model_path": str(paths["best_model_path"]),
        "training_history_path": str(paths["history_path"]),
    }
    return metadata


def train_densenet(
    run_mode,
    batch_size=DEFAULT_BATCH_SIZE,
    epochs=DEFAULT_TARGET_EPOCHS,
    seed=SEED,
    allow_cpu=False,
    learning_rate=0,
    training_sampling_mode=None,
):
    """Train a new run or resume DenseNet's latest valid run; epochs is total target."""
    run_mode = str(run_mode).strip().lower()
    if run_mode not in {"scratch", "resume"}:
        raise ValueError("run_mode must be 'scratch' or 'resume'")
    if int(batch_size) < 1 or int(epochs) < 1:
        raise ValueError("batch_size and epochs must both be at least 1")
    batch_size, epochs = int(batch_size), int(epochs)
    sampling_mode = normalize_training_sampling_mode(training_sampling_mode)
    run_name = training_model_name("densenet", sampling_mode)
    sampling_info = get_training_sampling_summary(seed=seed, training_sampling_mode=sampling_mode)

    if learning_rate != 0 :
        EFFICIENTNET_LEARNING_RATE = learning_rate

    devices = tf.config.list_physical_devices("GPU")
    if not devices and not allow_cpu:
        raise RuntimeError("TensorFlow cannot see a GPU. No training started; explicitly opt in to CPU training if intended.")
    tf.keras.utils.set_random_seed(seed)
    print("Training device(s):" if devices else "WARNING: Explicit CPU training:", ", ".join(d.name for d in devices) if devices else "training may be very slow")

    expected_settings = {
        "architecture": ARCHITECTURE,
        "pretrained_weights": DENSENET_WEIGHTS,
        "backbone_trainable": DENSENET_BACKBONE_TRAINABLE,
        "preprocessing": "tf.keras.applications.densenet.preprocess_input inside model",
        "dropout_rate": DENSENET_DROPOUT,
        "optimizer": "Adam",
        "learning_rate": DENSENET_LEARNING_RATE,
    }

    if run_mode == "scratch":
        class_counts, class_weights = calculate_training_class_weights(seed=seed, training_sampling_mode=sampling_mode)
        paths = create_new_run(run_name)
        metadata = _new_metadata(paths, batch_size, epochs, seed, class_counts, class_weights, run_name, sampling_info)
        save_json(paths["metadata_path"], metadata)
        start_epoch = 0
        print(f"Starting a new, separate run: {run_name}/{paths['run_id']}")
    else:
        paths = find_latest_resumable_run(run_name)
        if paths is None:
            raise FileNotFoundError("No DenseNet recovery checkpoint found. Select 'scratch' for a fresh run.")
        metadata = paths["metadata"]
        saved_mode = metadata.get("training_sampling_mode", "full")
        saved_cap = metadata.get("max_training_examples_per_class")
        expected_cap = sampling_info["max_training_examples_per_class"]
        saved_sampling_seed = metadata.get("training_sampling_seed", metadata.get("seed", seed))
        expected_sampling_seed = sampling_info["training_sampling_seed"]
        saved_training_seed = metadata.get("seed", seed)
        saved_subset_sha256 = metadata.get("training_subset_sha256")
        expected_subset_sha256 = sampling_info.get("training_subset_sha256")
        saved_split_manifest_sha256 = metadata.get("split_manifest_sha256")
        expected_split_manifest_sha256 = sampling_info.get("split_manifest_sha256")
        manifest_changed = (
            sampling_mode == "undersampled"
            and (
                saved_subset_sha256 != expected_subset_sha256
                or saved_split_manifest_sha256 != expected_split_manifest_sha256
            )
        )
        if (
            saved_mode != sampling_mode
            or saved_cap != expected_cap
            or saved_sampling_seed != expected_sampling_seed
            or saved_training_seed != seed
            or manifest_changed
        ):
            raise ValueError(
                "The saved run uses a different training mode, per-class cap, seed, or "
                "fixed train/validation/test manifest. Undersampled runs must use the exact "
                "same saved split manifest to resume. Use run_mode='scratch' for a new run."
            )
        metadata.update(sampling_info)
        if metadata.get("model_name") != run_name or metadata.get("architecture") != ARCHITECTURE:
            raise ValueError("The latest run metadata does not match the DenseNet trainer.")
        for key, expected in expected_settings.items():
            if metadata.get(key) != expected:
                raise ValueError(f"Saved setting {key!r} does not match current DenseNet config. Choose scratch or restore the original config.")
        if metadata.get("class_weights") is None or metadata.get("training_class_counts") is None:
            raise ValueError("Saved run has no training-only class weights; refusing incompatible resume.")
        class_counts = metadata["training_class_counts"]
        class_weights = metadata["class_weights"]
        if len(class_counts) != NUM_CLASSES or len(class_weights) != NUM_CLASSES:
            raise ValueError("Saved class counts/weights do not match NUM_CLASSES.")
        start_epoch = int(metadata.get("last_completed_epoch", 0))
        if epochs <= start_epoch:
            raise ValueError(f"Run already completed epoch {start_epoch}; set a larger total target epoch.")
        if start_epoch > 0 and not best_model_exists(paths["best_model_path"]):
            raise FileNotFoundError(f"Best validation weights are missing: {paths['best_model_path']}")
        old_batch_size = int(metadata.get("batch_size", batch_size))
        if old_batch_size != batch_size:
            print(f"Resuming with batch_size={batch_size} (previously {old_batch_size}).")
        used = metadata.setdefault("batch_sizes_used", [old_batch_size])
        if batch_size not in used:
            used.append(batch_size)
        metadata["batch_size"] = batch_size
        metadata["target_epochs"] = epochs
        metadata["updated_at"] = timestamp_now()
        print(f"Resuming {run_name}/{paths['run_id']} from: {paths['latest_checkpoint']}")

    try:
        model = build_densenet_model(weights=DENSENET_WEIGHTS)
        completed_epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="completed_epoch")
        best_val_loss = tf.Variable(float("inf"), dtype=tf.float32, trainable=False, name="best_val_loss")
        checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer, completed_epoch=completed_epoch, best_val_loss=best_val_loss)
        manager = tf.train.CheckpointManager(checkpoint, str(paths["checkpoint_dir"]), max_to_keep=CHECKPOINTS_TO_KEEP)

        if run_mode == "resume":
            checkpoint.restore(paths["latest_checkpoint"]).assert_existing_objects_matched()
            start_epoch = int(completed_epoch.numpy())
            metadata["last_completed_epoch"] = start_epoch
            if best_val_loss.numpy() == float("inf") and metadata.get("best_val_loss") is not None:
                best_val_loss.assign(float(metadata["best_val_loss"]))

        metadata["status"] = "running"
        metadata["updated_at"] = timestamp_now()
        metadata["batch_size"] = batch_size
        metadata["target_epochs"] = epochs
        save_json(paths["metadata_path"], metadata)

        train_ds, validation_ds, _ = make_datasets(batch_size=batch_size, seed=seed, training_sampling_mode=sampling_mode)
        weighted_train_ds = add_sample_weights(train_ds, class_weights)
        callbacks = [
            BestValidationWeightsCallback(paths["best_model_path"], best_val_loss, paths["metadata_path"], metadata),
            tf.keras.callbacks.CSVLogger(str(paths["history_path"]), append=(run_mode == "resume")),
            RecoveryCheckpointCallback(manager, completed_epoch, best_val_loss, paths["metadata_path"], metadata),
        ]
        print(f"Training sampling mode: {sampling_mode}; cap: {sampling_info['max_training_examples_per_class']}")
        print(f"Input size: {IMAGE_HEIGHT}x{IMAGE_WIDTH}; classes: {NUM_CLASSES}")
        print("Training-only class counts:", dict(enumerate(class_counts)))
        print("Training-only class weights:", dict(enumerate(class_weights)))
        print("Validation remains unweighted; best model selected by lowest validation loss.")
        print(f"Starting at epoch {start_epoch}; total target epochs: {epochs}")
        history = model.fit(weighted_train_ds, validation_data=validation_ds, initial_epoch=start_epoch, epochs=epochs, callbacks=callbacks)

        if not best_model_exists(paths["best_model_path"]):
            raise RuntimeError("Training ended without best validation weights; inspect training history.")
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
        return {"run_id": paths["run_id"], "paths": paths, "metadata": metadata, "history": history}
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        metadata["updated_at"] = timestamp_now()
        save_json(paths["metadata_path"], metadata)
        print(f"Interrupted; resume {run_name}/{paths['run_id']} with run_mode='resume'.")
        raise
    except Exception as error:
        metadata["status"] = "failed"
        metadata["last_error"] = f"{type(error).__name__}: {error}"
        metadata["updated_at"] = timestamp_now()
        save_json(paths["metadata_path"], metadata)
        raise
