"""Load btgraham-300 data using one fixed train/validation/test manifest.

Sampling is performed only by prepare_undersampled_split.ipynb. In undersampled
mode, all trainers and the evaluator consume identical saved indices. Image bytes
are filtered BEFORE decoding so the reduced evaluation splits are faster to read.
"""

from configs import tensorflow_setup as _tensorflow_setup

import hashlib
import json
from functools import lru_cache

import tensorflow as tf

_tensorflow_setup.configure_gpu_memory_growth(tf)

import tensorflow_datasets as tfds

from configs.config import (
    DATA_DIR,
    EXPECTED_DATASET_NAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    CLASS_NAMES,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    SEED,
    SHUFFLE_BUFFER,
    DEFAULT_BATCH_SIZE,
    UNDERSAMPLE_MAX_PER_CLASS,
    UNDERSAMPLE_SEED,
    UNDERSAMPLING_MANIFEST_PATH,
    PROJECT_ROOT,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    normalize_training_sampling_mode,
)

SPLIT_NAMES = {
    "train": TRAIN_SPLIT,
    "validation": VALIDATION_SPLIT,
    "test": TEST_SPLIT,
}


def load_builder():
    """Open the local prepared TFDS dataset only; never downloads or rebuilds it."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory was not found: {DATA_DIR}")
    required_files = ("features.json", "dataset_info.json")
    missing = [name for name in required_files if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {DATA_DIR}; check DATA_DIR.")
    builder = tfds.builder_from_directory(str(DATA_DIR))
    if builder.name != EXPECTED_DATASET_NAME:
        raise ValueError(
            f"Expected TFDS dataset {EXPECTED_DATASET_NAME!r}, found {builder.name!r}."
        )
    return builder


def inspect_dataset():
    """Print local TFDS metadata and official split sizes."""
    builder = load_builder()
    print(f"Dataset directory: {DATA_DIR}")
    print(f"Dataset name:      {builder.name}")
    print(f"Dataset version:   {builder.info.version}")
    for name, split_info in builder.info.splits.items():
        print(f"  {name}: {split_info.num_examples:,} examples")
    return builder.info


def _prepare_decoded_example(example):
    """Resize an already-decoded image and validate the five-class label."""
    image = tf.cast(example["image"], tf.float32)
    image = tf.image.resize(
        image, [IMAGE_HEIGHT, IMAGE_WIDTH], method="bilinear", antialias=True
    )
    image = tf.ensure_shape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    label = tf.cast(example["label"], tf.int32)
    label = tf.ensure_shape(label, [])
    checks = [
        tf.debugging.assert_greater_equal(label, 0, message="DR label is below 0"),
        tf.debugging.assert_less(label, NUM_CLASSES, message="DR label exceeds 4"),
    ]
    with tf.control_dependencies(checks):
        label = tf.identity(label)
    return image, label


def _decode_and_prepare_example(example):
    """Decode an encoded image only after its index has passed manifest filtering."""
    image = tf.io.decode_image(
        example["image"], channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    decoded = {"image": image, "label": example["label"]}
    return _prepare_decoded_example(decoded)


def _indices_sha256(indices):
    payload = json.dumps([int(x) for x in indices], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_sha256(manifest):
    unsigned = dict(manifest)
    unsigned.pop("split_manifest_sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_undersampling_manifest(builder=None):
    """Load and validate the fixed manifest for all three splits; never sample here."""
    if not UNDERSAMPLING_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "Fixed split manifest not found:\n"
            f"  {UNDERSAMPLING_MANIFEST_PATH}\n"
            "Run prepare_undersampled_split.ipynb once before training/evaluation."
        )
    with UNDERSAMPLING_MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if builder is None:
        builder = load_builder()
    info = builder.info

    if manifest.get("manifest_version") != 2:
        raise ValueError("Unsupported split manifest. Rerun prepare_undersampled_split.ipynb.")
    if manifest.get("dataset_name") != builder.name:
        raise ValueError("Manifest dataset name does not match the local dataset.")
    if str(manifest.get("dataset_version")) != str(info.version):
        raise ValueError("Manifest dataset version differs from local TFDS dataset.")
    if int(manifest.get("sampling_seed", -1)) != int(UNDERSAMPLE_SEED):
        raise ValueError("Manifest seed differs from config.py. Regenerate the manifest.")
    if int(manifest.get("max_per_class", -1)) != int(UNDERSAMPLE_MAX_PER_CLASS):
        raise ValueError("Manifest per-class cap differs from config.py. Regenerate it.")
    expected_ratios = {"train": TRAIN_RATIO, "validation": VALIDATION_RATIO, "test": TEST_RATIO}
    if manifest.get("split_ratios") != expected_ratios:
        raise ValueError("Manifest split ratios differ from config.py. Regenerate it.")

    all_splits = manifest.get("splits")
    if not isinstance(all_splits, dict):
        raise ValueError("Manifest has no split definitions.")
    for role, split_name in SPLIT_NAMES.items():
        if split_name not in info.splits or role not in all_splits:
            raise ValueError(f"Required manifest or TFDS split missing: {role}/{split_name}.")
        split_item = all_splits[role]
        official_count = int(info.splits[split_name].num_examples)
        if int(split_item.get("official_count", -1)) != official_count:
            raise ValueError(f"Official {role} count differs from saved manifest.")
        indices = split_item.get("indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"Manifest has no retained indices for {role}.")
        if any(not isinstance(x, int) or isinstance(x, bool) for x in indices):
            raise ValueError(f"Manifest indices for {role} must be integers.")
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError(f"Manifest indices for {role} must be sorted and unique.")
        if indices[0] < 0 or indices[-1] >= official_count:
            raise ValueError(f"Manifest indices for {role} fall outside the official split.")
        if len(indices) != int(split_item.get("retained_count", -1)):
            raise ValueError(f"Saved retained count for {role} differs from its indices.")
        if _indices_sha256(indices) != split_item.get("indices_sha256"):
            raise ValueError(f"Saved index checksum for {role} is invalid.")
        class_counts = split_item.get("retained_class_counts")
        if not isinstance(class_counts, list) or len(class_counts) != NUM_CLASSES:
            raise ValueError(f"Saved class counts for {role} must contain five values.")
        if sum(int(x) for x in class_counts) != len(indices):
            raise ValueError(f"Saved class counts for {role} do not sum to retained total.")

    expected_train_counts = [
        min(int(x), int(UNDERSAMPLE_MAX_PER_CLASS))
        for x in all_splits["train"]["original_class_counts"]
    ]
    if [int(x) for x in all_splits["train"]["retained_class_counts"]] != expected_train_counts:
        raise ValueError("Training class counts do not match the configured per-class cap.")
    if _manifest_sha256(manifest) != manifest.get("split_manifest_sha256"):
        raise ValueError("Whole-manifest checksum failed. Regenerate the fixed splits.")

    manifest["manifest_path"] = str(UNDERSAMPLING_MANIFEST_PATH.relative_to(PROJECT_ROOT))
    # Keep compatibility aliases for metadata/evaluator code that expects train fields.
    train_item = all_splits["train"]
    manifest["original_training_count"] = train_item["official_count"]
    manifest["retained_training_count"] = train_item["retained_count"]
    manifest["original_class_counts"] = train_item["original_class_counts"]
    manifest["retained_class_counts"] = train_item["retained_class_counts"]
    manifest["retained_indices"] = train_item["indices"]
    manifest["subset_sha256"] = train_item["indices_sha256"]
    return manifest


def get_training_sampling_summary(seed=SEED, training_sampling_mode=None):
    """Return train and evaluation subset metadata for model-run records."""
    mode = normalize_training_sampling_mode(training_sampling_mode)
    builder = load_builder()
    official_train_count = int(builder.info.splits[TRAIN_SPLIT].num_examples)
    if mode == "undersampled":
        manifest = load_undersampling_manifest(builder)
        splits = manifest["splits"]
        tr = splits["train"]
        va = splits["validation"]
        te = splits["test"]
        return {
            "training_sampling_mode": "undersampled",
            "training_sampling": "fixed 70:10:20 train/validation/test subset manifest",
            "max_training_examples_per_class": int(manifest["max_per_class"]),
            "training_sampling_seed": int(manifest["sampling_seed"]),
            "original_training_examples": int(tr["official_count"]),
            "retained_training_examples": int(tr["retained_count"]),
            "original_training_class_counts": [int(x) for x in tr["original_class_counts"]],
            "training_class_counts": [int(x) for x in tr["retained_class_counts"]],
            "training_subset_manifest_path": manifest["manifest_path"],
            "training_subset_sha256": tr["indices_sha256"],
            "validation_examples_after_sampling": int(va["retained_count"]),
            "validation_subset_sha256": va["indices_sha256"],
            "test_examples_after_sampling": int(te["retained_count"]),
            "test_subset_sha256": te["indices_sha256"],
            "split_manifest_sha256": manifest["split_manifest_sha256"],
            "split_manifest_path": manifest["manifest_path"],
            "split_ratios": dict(manifest["split_ratios"]),
        }

    counts = _read_class_counts(TRAIN_SPLIT)
    val_count = int(builder.info.splits[VALIDATION_SPLIT].num_examples)
    test_count = int(builder.info.splits[TEST_SPLIT].num_examples)
    return {
        "training_sampling_mode": "full",
        "training_sampling": "all official examples; complete official validation/test splits",
        "max_training_examples_per_class": None,
        "training_sampling_seed": int(seed),
        "original_training_examples": official_train_count,
        "retained_training_examples": official_train_count,
        "original_training_class_counts": counts,
        "training_class_counts": list(counts),
        "training_subset_manifest_path": None,
        "training_subset_sha256": None,
        "validation_examples_after_sampling": val_count,
        "validation_subset_sha256": None,
        "test_examples_after_sampling": test_count,
        "test_subset_sha256": None,
        "split_manifest_sha256": None,
        "split_manifest_path": None,
        "split_ratios": None,
    }


@lru_cache(maxsize=3)
def _read_class_counts(split_name):
    """Count class labels without decoding images."""
    builder = load_builder()
    ds = builder.as_dataset(
        split=split_name,
        shuffle_files=False,
        read_config=tfds.ReadConfig(try_autocache=False),
        decoders={"image": tfds.decode.SkipDecoding()},
    )
    counts = [0] * NUM_CLASSES
    for example in ds:
        label = int(example["label"].numpy())
        if not 0 <= label < NUM_CLASSES:
            raise ValueError(f"Out-of-range label {label} in split {split_name}.")
        counts[label] += 1
    return counts


def _filter_to_manifest_indices(dataset, indices, expected_count):
    """Filter by source position before image decoding."""
    wanted = tf.constant(indices, dtype=tf.int64)
    lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            wanted,
            tf.ones_like(wanted, dtype=tf.int32),
            key_dtype=tf.int64,
            value_dtype=tf.int32,
        ),
        default_value=0,
    )
    dataset = dataset.enumerate()
    dataset = dataset.filter(lambda index, example: lookup.lookup(index) > 0)
    dataset = dataset.map(lambda index, example: example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(int(expected_count)))
    return dataset


def _build_split_dataset(builder, role, batch_size, sampling_mode, seed=None):
    split_name = SPLIT_NAMES[role]
    mode = normalize_training_sampling_mode(sampling_mode)
    split_count = int(builder.info.splits[split_name].num_examples)
    manifest = load_undersampling_manifest(builder) if mode == "undersampled" else None
    if manifest is not None:
        split_manifest = manifest["splits"][role]
        dataset = builder.as_dataset(
            split=split_name,
            shuffle_files=False,
            read_config=tfds.ReadConfig(try_autocache=False),
            decoders={"image": tfds.decode.SkipDecoding()},
        )
        dataset = _filter_to_manifest_indices(
            dataset, split_manifest["indices"], split_manifest["retained_count"]
        )
        prepare_function = _decode_and_prepare_example
        print(
            f"{role.capitalize():10s}: {split_manifest['retained_count']:,} selected / "
            f"{split_count:,} official; class counts={split_manifest['retained_class_counts']}"
        )
    else:
        dataset = builder.as_dataset(
            split=split_name,
            shuffle_files=False,
            read_config=tfds.ReadConfig(try_autocache=False),
        )
        prepare_function = _prepare_decoded_example
        print(f"{role.capitalize():10s}: complete official split ({split_count:,} examples)")

    if role == "train":
        effective_count = (
            int(manifest["splits"][role]["retained_count"]) if manifest else split_count
        )
        # Shuffle encoded bytes/uint8 tensors before casting and resizing to float32,
        # keeping memory usage manageable on the 4 GB VRAM/limited-RAM laptop.
        dataset = dataset.shuffle(
            buffer_size=min(effective_count, SHUFFLE_BUFFER),
            seed=SEED if seed is None else int(seed),
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(prepare_function, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(int(batch_size)).prefetch(tf.data.AUTOTUNE)


def make_datasets(batch_size=DEFAULT_BATCH_SIZE, seed=SEED, training_sampling_mode=None):
    """Return train, validation, test streams under one common fixed split policy."""
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1")
    mode = normalize_training_sampling_mode(training_sampling_mode)
    builder = load_builder()
    for split_name in SPLIT_NAMES.values():
        if split_name not in builder.info.splits:
            raise ValueError(f"Required official split {split_name!r} was not found.")
    if "image" not in builder.info.features or "label" not in builder.info.features:
        raise ValueError("Expected TFDS features named 'image' and 'label'.")
    if mode == "undersampled":
        manifest = load_undersampling_manifest(builder)
        print(
            "Using fixed 70:10:20 subset manifest: "
            f"train={manifest['splits']['train']['retained_count']:,}, "
            f"validation={manifest['splits']['validation']['retained_count']:,}, "
            f"test={manifest['splits']['test']['retained_count']:,}; "
            f"SHA-256={manifest['split_manifest_sha256'][:12]}…"
        )
    train_ds = _build_split_dataset(builder, "train", batch_size, mode, seed=seed)
    validation_ds = _build_split_dataset(builder, "validation", batch_size, mode)
    test_ds = _build_split_dataset(builder, "test", batch_size, mode)
    return train_ds, validation_ds, test_ds


def get_split_sampling_summary(split_name, sampling_mode="full"):
    """Return selected example count/checksum for one validation or test split."""
    if split_name not in (VALIDATION_SPLIT, TEST_SPLIT):
        raise ValueError("Only validation and test can be evaluated.")
    mode = normalize_training_sampling_mode(sampling_mode)
    builder = load_builder()
    if mode == "full":
        return {
            "split": split_name,
            "n_examples": int(builder.info.splits[split_name].num_examples),
            "evaluation_subset_sha256": None,
            "split_manifest_sha256": None,
        }
    manifest = load_undersampling_manifest(builder)
    role = "validation" if split_name == VALIDATION_SPLIT else "test"
    selected = manifest["splits"][role]
    return {
        "split": split_name,
        "n_examples": int(selected["retained_count"]),
        "evaluation_subset_sha256": selected["indices_sha256"],
        "split_manifest_sha256": manifest["split_manifest_sha256"],
    }


def make_evaluation_dataset(
    split_name,
    batch_size=DEFAULT_BATCH_SIZE,
    sampling_mode="full",
):
    """Build the full or fixed-manifest validation/test dataset without shuffling."""
    if split_name not in (VALIDATION_SPLIT, TEST_SPLIT):
        raise ValueError(f"Evaluation split must be {VALIDATION_SPLIT!r} or {TEST_SPLIT!r}.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1")
    mode = normalize_training_sampling_mode(sampling_mode)
    builder = load_builder()
    if split_name not in builder.info.splits:
        raise ValueError(f"Required evaluation split {split_name!r} was not found.")
    role = "validation" if split_name == VALIDATION_SPLIT else "test"
    return _build_split_dataset(builder, role, int(batch_size), mode)
