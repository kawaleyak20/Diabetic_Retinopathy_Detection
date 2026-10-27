"""Train or resume the initial EfficientNetB0 diabetic-retinopathy baseline."""

from pathlib import Path

import csv
import math

from configs import tensorflow_setup as _tensorflow_setup

import tensorflow as tf

_tensorflow_setup.configure_gpu_memory_growth(tf)

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
)
from scripts.data_loader import make_datasets
from scripts.utils import (
    RecoveryCheckpointCallback,
    best_model_exists,
    create_new_run,
    find_latest_resumable_run,
    save_json,
    timestamp_now,
)

def build_efficientnet_model(weights=EFFICIENTNET_WEIGHTS):
    """Create a frozen EfficientNetB0 feature extractor and five-class head.

    Keras EfficientNet includes its own input rescaling, so pass the loader's
    float32 image tensors in the 0-255 range without applying another rescale.
    """
    if EFFICIENTNET_VARIANT != "B0":
        raise ValueError(
            "This starter script implements EfficientNetB0 only. Update the model "
            "builder deliberately before changing EFFICIENTNET_VARIANT."
        )

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    )
    backbone.trainable = EFFICIENTNET_BACKBONE_TRAINABLE

    inputs = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), name="retinal_image")
    features = backbone(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    features = tf.keras.layers.Dropout(EFFICIENTNET_DROPOUT, name="head_dropout")(features)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="dr_severity",
    )(features)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_b0_dr")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=EFFICIENTNET_LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def _save_weights_atomically(model, weights_path):
    """Write weights to a temporary HDF5 file before replacing the best copy."""
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = weights_path.with_name(weights_path.stem + ".tmp.h5")
    try:
        # Weight-only saving skips model-config JSON serialization, which fails
        # for this EfficientNet model under the project's TF/Keras 2.10 setup.
        model.save_weights(str(temporary_path), save_format="h5")
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"Best-model weights were not written: {temporary_path}")
        temporary_path.replace(weights_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    if not best_model_exists(weights_path):
        raise RuntimeError(f"Best-model weights are missing or empty: {weights_path}")


def _read_validation_loss_for_epoch(history_path, completed_epoch):
    """Read the validation loss logged for a completed one-based epoch."""
    history_path = Path(history_path)
    if not history_path.is_file():
        return None

    with history_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            try:
                row_epoch = int(float(row.get("epoch", ""))) + 1
                val_loss = float(row.get("val_loss", "nan"))
            except (TypeError, ValueError):
                continue
            if row_epoch == int(completed_epoch) and math.isfinite(val_loss):
                return val_loss
    return None


class BestValidationModelCallback(tf.keras.callbacks.Callback):
    """Save only improved validation-loss models, without replacing a better one."""

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
            _save_weights_atomically(self.model, self.best_model_path)
            self.best_val_loss.assign(current_loss)
            self.metadata["best_val_loss"] = current_loss
            self.metadata["best_model_format"] = "keras_weights_hdf5"
            self.metadata["updated_at"] = timestamp_now()
            save_json(self.metadata_path, self.metadata)
            print(f"Saved best validation model: {self.best_model_path}")


def _new_metadata(paths, batch_size, target_epochs, seed):
    return {
        "model_name": "efficientnet",
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
        "batch_size": batch_size,
        "batch_sizes_used": [batch_size],
        "seed": seed,
        "class_weighting": "none (initial baseline)",
        "target_epochs": target_epochs,
        "last_completed_epoch": 0,
        "best_val_loss": None,
        "checkpoint_dir": str(paths["checkpoint_dir"]),
        "best_model_path": str(paths["best_model_path"]),
        "training_history_path": str(paths["history_path"]),
    }


def train_efficientnet(
    run_mode,
    batch_size=DEFAULT_BATCH_SIZE,
    epochs=DEFAULT_TARGET_EPOCHS,
    seed=SEED,
    allow_cpu=False,
    learning_rate = 0
):
    """Start a new run or resume the newest checkpointed EfficientNet run.

    `epochs` is the total target epoch number, not the number of extra epochs.
    CPU training is blocked by default while TensorFlow cannot see a GPU; pass
    allow_cpu=True only when deliberately accepting potentially very slow training.
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
        paths = create_new_run("efficientnet")
        metadata = _new_metadata(paths, batch_size, epochs, seed)
        save_json(paths["metadata_path"], metadata)
        start_epoch = 0
        print(f"Starting a new, separate run: {paths['run_id']}")
    else:
        paths = find_latest_resumable_run("efficientnet")
        if paths is None:
            raise FileNotFoundError(
                "Resume was selected, but no valid EfficientNet recovery checkpoint "
                "was found. Choose 'scratch' explicitly to create a fresh run."
            )
        metadata = paths["metadata"]
        start_epoch = int(metadata.get("last_completed_epoch", 0))
        if epochs <= start_epoch:
            raise ValueError(
                f"Run {paths['run_id']} already completed epoch {start_epoch}. Set epochs "
                "to a larger total target (not a count of extra epochs)."
            )
        previous_batch_size = int(metadata.get("batch_size", batch_size))
        if previous_batch_size != batch_size:
            print(
                f"Resuming with batch_size={batch_size} (previously {previous_batch_size})."
            )
        batch_sizes_used = metadata.setdefault("batch_sizes_used", [previous_batch_size])
        if batch_size not in batch_sizes_used:
            batch_sizes_used.append(batch_size)
        print(f"Resuming {paths['run_id']} from checkpoint: {paths['latest_checkpoint']}")
        metadata["batch_size"] = batch_size
        metadata["target_epochs"] = epochs
        metadata["updated_at"] = timestamp_now()

    try:
        # Scratch starts from ImageNet weights. Resume restores every model weight
        # from the local recovery checkpoint, so it must not depend on a download.
        initial_weights = EFFICIENTNET_WEIGHTS if run_mode == "scratch" else None
        model = build_efficientnet_model(weights=initial_weights)

        completed_epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="completed_epoch")
        best_val_loss = tf.Variable(float("inf"), dtype=tf.float32, trainable=False, name="best_val_loss")
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
            # The recovery checkpoint is written before best-model serialization.
            # Metadata may therefore contain a newer best loss than the checkpoint.
            # Trust the metadata when a best model exists; if saving failed before
            # the first best model was created, start best-loss tracking at infinity.
            if best_model_exists(paths["best_model_path"]):
                if metadata.get("best_val_loss") is not None:
                    best_val_loss.assign(float(metadata["best_val_loss"]))
            else:
                # Migration for a run that completed epoch 1 with the previous
                # full-model saver: its recovery checkpoint exists, but export
                # failed. If CSVLogger recorded epoch 1 validation loss, the
                # restored model is exactly that epoch's model, so preserve it
                # as the initial best-weight candidate before resuming.
                epoch_one_loss = None
                if start_epoch == 1:
                    epoch_one_loss = _read_validation_loss_for_epoch(
                        paths["history_path"], completed_epoch=1
                    )
                if epoch_one_loss is not None:
                    _save_weights_atomically(model, paths["best_model_path"])
                    best_val_loss.assign(epoch_one_loss)
                    metadata["best_val_loss"] = epoch_one_loss
                    metadata["best_model_format"] = "keras_weights_hdf5"
                    print(
                        "Recovered epoch-1 best weights from the recovery checkpoint; "
                        f"validation loss: {epoch_one_loss:.6f}"
                    )
                else:
                    best_val_loss.assign(float("inf"))
                    metadata["best_val_loss"] = None
                    if start_epoch > 0:
                        print(
                            "WARNING: this resumed run has no best-weight file, and its "
                            "previous best validation loss could not be recovered from "
                            "training_history.csv. Best-loss tracking will restart from "
                            "the next completed epoch."
                        )

        metadata["status"] = "running"
        metadata["updated_at"] = timestamp_now()
        metadata["batch_size"] = batch_size
        metadata["target_epochs"] = epochs
        metadata["best_model_path"] = str(paths["best_model_path"])
        metadata["best_model_format"] = "keras_weights_hdf5"
        save_json(paths["metadata_path"], metadata)

        train_ds, validation_ds, _ = make_datasets(batch_size=batch_size, seed=seed)
        callbacks = [
            # Save resumable model+optimizer state before best-model serialization.
            # If best-model export fails, this epoch can still be resumed safely.
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
            BestValidationModelCallback(
                paths["best_model_path"],
                best_val_loss,
                paths["metadata_path"],
                metadata,
            ),
        ]

        print(f"Input size: {IMAGE_HEIGHT}x{IMAGE_WIDTH}; classes: {NUM_CLASSES}")
        print(f"Starting at epoch {start_epoch}; target total epochs: {epochs}")
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            initial_epoch=start_epoch,
            epochs=epochs,
            callbacks=callbacks,
        )

        if not best_model_exists(paths["best_model_path"]):
            raise RuntimeError(
                "Training ended without valid best-validation weights. No finite "
                "improving validation loss was saved; inspect the history before retrying."
            )

        completed = int(completed_epoch.numpy())
        metadata["last_completed_epoch"] = completed
        metadata["best_val_loss"] = float(best_val_loss.numpy())
        metadata["status"] = "completed" if completed >= epochs else "interrupted"
        metadata["updated_at"] = timestamp_now()
        save_json(paths["metadata_path"], metadata)
        print(f"Run status: {metadata['status']} ({completed}/{epochs} epochs)")
        print(f"Recovery checkpoints: {paths['checkpoint_dir']}")
        print(f"Best validation model: {paths['best_model_path']}")
        print(f"Training history: {paths['history_path']}")
        return {"run_id": paths["run_id"], "paths": paths, "metadata": metadata, "history": history}

    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        metadata["updated_at"] = timestamp_now()
        save_json(paths["metadata_path"], metadata)
        print(
            f"Training interrupted. Resume this run with run_mode='resume'. "
            f"Last completed epoch: {metadata.get('last_completed_epoch', 0)}"
        )
        raise
    except Exception as error:
        metadata["status"] = "failed"
        metadata["last_error"] = f"{type(error).__name__}: {error}"
        metadata["updated_at"] = timestamp_now()
        save_json(paths["metadata_path"], metadata)
        raise
