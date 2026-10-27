"""Train or resume the class-weighted DenseNet121 DR experiment."""

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
from scripts.data_loader import load_builder, make_datasets
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


def build_densenet_model(weights=DENSENET_WEIGHTS):
    """Build DenseNet121 with its ImageNet preprocessing inside the model."""
    backbone = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    )
    backbone.trainable = DENSENET_BACKBONE_TRAINABLE

    inputs = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), name="retinal_image")
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


def calculate_training_class_weights():
    """Calculate balanced inverse-frequency weights from train labels only."""
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
        missing = np.flatnonzero(class_counts == 0).tolist()
        raise ValueError(f"Cannot calculate class weights; training split lacks class(es): {missing}")

    weights = total_examples / (NUM_CLASSES * class_counts.astype(np.float64))
    return class_counts.astype(int).tolist(), weights.astype(float).tolist()


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


def _new_metadata(paths, batch_size, epochs, seed, class_counts, class_weights):
    metadata = {
        "model_name": MODEL_NAME,
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
        "class_weighting": "balanced inverse-frequency: N / (K * n_c), training split only",
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


def train_densenet(run_mode, batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_TARGET_EPOCHS, seed=SEED, allow_cpu=False,learning_rate=0):
    """Train a new run or resume DenseNet's latest valid run; epochs is total target."""
    run_mode = str(run_mode).strip().lower()
    if run_mode not in {"scratch", "resume"}:
        raise ValueError("run_mode must be 'scratch' or 'resume'")
    if int(batch_size) < 1 or int(epochs) < 1:
        raise ValueError("batch_size and epochs must both be at least 1")
    batch_size, epochs = int(batch_size), int(epochs)

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
        class_counts, class_weights = calculate_training_class_weights()
        paths = create_new_run(MODEL_NAME)
        metadata = _new_metadata(paths, batch_size, epochs, seed, class_counts, class_weights)
        save_json(paths["metadata_path"], metadata)
        start_epoch = 0
        print(f"Starting a new, separate run: {MODEL_NAME}/{paths['run_id']}")
    else:
        paths = find_latest_resumable_run(MODEL_NAME)
        if paths is None:
            raise FileNotFoundError("No DenseNet recovery checkpoint found. Select 'scratch' for a fresh run.")
        metadata = paths["metadata"]
        if metadata.get("model_name") != MODEL_NAME or metadata.get("architecture") != ARCHITECTURE:
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
        print(f"Resuming {MODEL_NAME}/{paths['run_id']} from: {paths['latest_checkpoint']}")

    try:
        # Scratch starts from ImageNet weights. Resume restores the complete local
        # model state from the checkpoint and must not require a pretrained download.
        initial_weights = DENSENET_WEIGHTS if run_mode == "scratch" else None
        model = build_densenet_model(weights=initial_weights)
        completed_epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="completed_epoch")
        best_val_loss = tf.Variable(float("inf"), dtype=tf.float32, trainable=False, name="best_val_loss")
        checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer, completed_epoch=completed_epoch, best_val_loss=best_val_loss)
        manager = tf.train.CheckpointManager(checkpoint, str(paths["checkpoint_dir"]), max_to_keep=CHECKPOINTS_TO_KEEP)

        if run_mode == "resume":
            checkpoint.restore(paths["latest_checkpoint"]).assert_existing_objects_matched()
            start_epoch = int(completed_epoch.numpy())
            metadata["last_completed_epoch"] = start_epoch
            # Recovery is saved before best-weight export, so metadata can contain
            # a newer best loss than the checkpoint variable. Prefer metadata when
            # the corresponding best-weight artifact exists.
            if best_model_exists(paths["best_model_path"]):
                saved_best_loss = metadata.get("best_val_loss")
                if saved_best_loss is not None and math.isfinite(float(saved_best_loss)):
                    best_val_loss.assign(float(saved_best_loss))

        metadata["status"] = "running"
        metadata["updated_at"] = timestamp_now()
        metadata["batch_size"] = batch_size
        metadata["target_epochs"] = epochs
        save_json(paths["metadata_path"], metadata)

        train_ds, validation_ds, _ = make_datasets(batch_size=batch_size, seed=seed)
        weighted_train_ds = add_sample_weights(train_ds, class_weights)
        callbacks = [
            # Save the resumable state first so a later best-weight export failure
            # cannot discard the completed epoch's model/optimizer state.
            RecoveryCheckpointCallback(manager, completed_epoch, best_val_loss, paths["metadata_path"], metadata),
            tf.keras.callbacks.CSVLogger(str(paths["history_path"]), append=(run_mode == "resume")),
            BestValidationWeightsCallback(paths["best_model_path"], best_val_loss, paths["metadata_path"], metadata),
        ]
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
        print(f"Interrupted; resume {MODEL_NAME}/{paths['run_id']} with run_mode='resume'.")
        raise
    except Exception as error:
        metadata["status"] = "failed"
        metadata["last_error"] = f"{type(error).__name__}: {error}"
        metadata["updated_at"] = timestamp_now()
        save_json(paths["metadata_path"], metadata)
        raise
