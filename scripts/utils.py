"""Small shared helpers for model runs, checkpoints, and status files."""

import json
import math
import re
from datetime import datetime
from pathlib import Path

from configs import tensorflow_setup as _tensorflow_setup

import tensorflow as tf

_tensorflow_setup.configure_gpu_memory_growth(tf)

from configs.config import MODELS_DIR


RUN_PATTERN = re.compile(r"^run_(\d{3,})$")


def timestamp_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def model_directories(model_name):
    """Return the model, checkpoint, and per-run output directories."""
    model_root = MODELS_DIR / model_name
    return {
        "model_root": model_root,
        "checkpoints_root": model_root / "checkpoints",
        "runs_root": model_root / "runs",
    }


def run_paths(model_name, run_id):
    """Build paths for a particular run without creating or overwriting it."""
    roots = model_directories(model_name)
    return {
        **roots,
        "run_id": run_id,
        "checkpoint_dir": roots["checkpoints_root"] / run_id,
        "output_dir": roots["runs_root"] / run_id,
        "metadata_path": roots["runs_root"] / run_id / "run_metadata.json",
        "history_path": roots["runs_root"] / run_id / "training_history.csv",
        # Save only the model weights for best-validation selection. This avoids
        # serializing EfficientNet's model config, which raises an EagerTensor
        # JSON serialization error in this TensorFlow/Keras 2.10 environment.
        "best_model_path": roots["runs_root"] / run_id / "best_model.weights.h5",
    }


def best_model_exists(path):
    """Return True only when non-empty best-validation weights are present."""
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def _existing_run_numbers(model_name):
    roots = model_directories(model_name)
    numbers = set()
    for parent in (roots["checkpoints_root"], roots["runs_root"]):
        if parent.is_dir():
            for path in parent.iterdir():
                match = RUN_PATTERN.match(path.name)
                if path.is_dir() and match:
                    numbers.add(int(match.group(1)))
    return numbers


def create_new_run(model_name):
    """Create a never-before-used run directory pair."""
    roots = model_directories(model_name)
    next_number = max(_existing_run_numbers(model_name), default=0) + 1
    run_id = f"run_{next_number:03d}"
    paths = run_paths(model_name, run_id)

    # Refuse to reuse a directory, even if it was left behind by a failed run.
    if paths["checkpoint_dir"].exists() or paths["output_dir"].exists():
        raise FileExistsError(f"Run directory already exists: {run_id}")

    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=False)
    paths["output_dir"].mkdir(parents=True, exist_ok=False)
    return paths


def save_json(path, data):
    """Write JSON metadata as UTF-8 with readable indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def find_latest_resumable_run(model_name):
    """Find the newest run, refusing to fall back to an older run silently."""
    numbers = sorted(_existing_run_numbers(model_name), reverse=True)
    if not numbers:
        return None

    # If the newest run is incomplete or has no checkpoint, stop instead of
    # silently resuming an older experiment and confusing its outputs.
    run_id = f"run_{numbers[0]:03d}"
    paths = run_paths(model_name, run_id)
    if not paths["metadata_path"].is_file():
        raise FileNotFoundError(
            f"The newest run ({run_id}) has no run_metadata.json. "
            "Inspect the run folders before choosing resume."
        )

    latest_checkpoint = tf.train.latest_checkpoint(str(paths["checkpoint_dir"]))
    if not latest_checkpoint:
        raise FileNotFoundError(
            f"The newest run ({run_id}) has no valid recovery checkpoint. "
            "Resume stopped rather than selecting an older run. Choose 'scratch' "
            "for a fresh run, or inspect this run before continuing."
        )

    metadata = load_json(paths["metadata_path"])
    if metadata.get("run_id") != run_id:
        raise ValueError(
            f"The newest run folder is {run_id}, but its metadata identifies "
            f"{metadata.get('run_id')!r}. Inspect the run before resuming."
        )
    if metadata.get("model_name") != model_name:
        raise ValueError(
            f"The newest run folder is under model {model_name!r}, but its metadata "
            f"identifies {metadata.get('model_name')!r}. Inspect the run before resuming."
        )

    paths["latest_checkpoint"] = latest_checkpoint
    paths["metadata"] = metadata
    return paths

def list_model_runs(model_name):
    """Return lightweight status rows for all discovered runs of one model."""
    numbers = sorted(_existing_run_numbers(model_name))
    rows = []
    for number in numbers:
        run_id = f"run_{number:03d}"
        paths = run_paths(model_name, run_id)
        metadata = load_json(paths["metadata_path"]) if paths["metadata_path"].is_file() else {}
        latest_checkpoint = tf.train.latest_checkpoint(str(paths["checkpoint_dir"]))
        rows.append(
            {
                "run_id": run_id,
                "status": metadata.get("status", "metadata missing"),
                "last_completed_epoch": metadata.get("last_completed_epoch", 0),
                "target_epochs": metadata.get("target_epochs", ""),
                "checkpoint": "yes" if latest_checkpoint else "no",
                "best_model": "yes" if best_model_exists(paths["best_model_path"]) else "no",
                "output_dir": str(paths["output_dir"]),
            }
        )
    return rows


def print_run_status(model_name):
    """Print all run status rows for one model without loading model weights."""
    rows = list_model_runs(model_name)
    print(f"Model: {model_name}")
    if not rows:
        print("No runs found yet.")
        return rows

    for row in rows:
        print(
            f"{row['run_id']}: status={row['status']}, "
            f"epoch={row['last_completed_epoch']}/{row['target_epochs']}, "
            f"checkpoint={row['checkpoint']}, best_model={row['best_model']}"
        )
        print(f"  Output: {row['output_dir']}")
    return rows


class RecoveryCheckpointCallback(tf.keras.callbacks.Callback):
    """Save full model + optimizer state after every completed epoch."""

    def __init__(self, checkpoint_manager, completed_epoch, best_val_loss, metadata_path, metadata):
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.completed_epoch = completed_epoch
        self.best_val_loss = best_val_loss
        self.metadata_path = Path(metadata_path)
        self.metadata = metadata

    def on_epoch_end(self, epoch, logs=None):
        completed = int(epoch) + 1
        self.completed_epoch.assign(completed)
        self.checkpoint_manager.save(checkpoint_number=completed)

        self.metadata["last_completed_epoch"] = completed
        best_loss = float(self.best_val_loss.numpy())
        self.metadata["best_val_loss"] = best_loss if math.isfinite(best_loss) else None
        self.metadata["last_checkpoint"] = self.checkpoint_manager.latest_checkpoint
        self.metadata["status"] = "running"
        self.metadata["updated_at"] = timestamp_now()
        save_json(self.metadata_path, self.metadata)
