"""Load the existing btgraham-300 TFDS dataset without modifying it."""

from configs import tensorflow_setup as _tensorflow_setup

import tensorflow as tf

_tensorflow_setup.configure_gpu_memory_growth(tf)

import tensorflow_datasets as tfds

from configs.config import (
    DATA_DIR,
    EXPECTED_DATASET_NAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    SEED,
    SHUFFLE_BUFFER,
    DEFAULT_BATCH_SIZE,
)


def load_builder():
    """Open an existing local TFDS dataset; never download or rebuild it."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory was not found: {DATA_DIR}")

    required_files = ("features.json", "dataset_info.json")
    missing = [name for name in required_files if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {DATA_DIR}. Check that DATA_DIR points to the "
            "btgraham-300 TFDS directory containing its metadata files."
        )

    # This opens the prepared local dataset; it does not call download_and_prepare().
    builder = tfds.builder_from_directory(str(DATA_DIR))
    if builder.name != EXPECTED_DATASET_NAME:
        raise ValueError(
            f"Expected TFDS dataset '{EXPECTED_DATASET_NAME}', found '{builder.name}'. "
            f"Check DATA_DIR: {DATA_DIR}"
        )
    return builder


def inspect_dataset():
    """Print the local dataset's actual features and available split sizes."""
    builder = load_builder()
    info = builder.info

    print(f"Dataset directory: {DATA_DIR}")
    print(f"Dataset name:      {builder.name}")
    print(f"Dataset version:   {info.version}")
    print(f"Features:          {list(info.features.keys())}")
    print("Available splits:")

    for split_name, split_info in info.splits.items():
        print(f"  {split_name}: {split_info.num_examples:,} examples")

    if "image" in info.features:
        print(f"Image feature:     {info.features['image']}")
    if "label" in info.features:
        print(f"Label feature:     {info.features['label']}")

    return info


def _prepare_example(example):
    """Resize a decoded retinal image to model input size and return its label.

    TFDS's btgraham-300 preprocessing sets the eyeball radius to 300 pixels
    and crops to 90% of that radius. It therefore commonly stores a 540x540
    image, not a 300x300 image. Resize here to the shared model input size.
    Pixel values remain in the 0-255 range; model-specific preprocessing is
    handled by each model implementation.
    """
    image = tf.ensure_shape(example["image"], [None, None, 3])
    image = tf.image.resize(
        tf.cast(image, tf.float32),
        size=[IMAGE_HEIGHT, IMAGE_WIDTH],
        method="bilinear",
        antialias=True,
    )
    image = tf.ensure_shape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    label = tf.cast(example["label"], tf.int32)
    label = tf.ensure_shape(label, [])

    checks = [
        tf.debugging.assert_greater_equal(label, 0, message="DR label is below 0"),
        tf.debugging.assert_less(
            label, NUM_CLASSES, message="DR label is outside the expected 0-4 range"
        ),
    ]
    with tf.control_dependencies(checks):
        label = tf.identity(label)

    return image, label


def make_datasets(batch_size=DEFAULT_BATCH_SIZE, seed=SEED):
    """Return train, validation, and test datasets from the existing official splits.

    Images are decoded by TFDS, cast to float32, and kept at 0-255. Model-specific
    preprocessing should be applied in the corresponding model implementation.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    builder = load_builder()
    info = builder.info
    required_splits = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
    missing_splits = [name for name in required_splits if name not in info.splits]
    if missing_splits:
        raise ValueError(
            f"Required official split(s) not found: {missing_splits}. "
            f"Available splits: {list(info.splits.keys())}"
        )

    feature_names = list(info.features.keys())
    if "image" not in info.features or "label" not in info.features:
        raise ValueError(
            "Expected TFDS features named 'image' and 'label', but found: "
            f"{feature_names}"
        )

    read_config = tfds.ReadConfig(try_autocache=False)
    output_datasets = {}

    for role, split_name in (
        ("train", TRAIN_SPLIT),
        ("validation", VALIDATION_SPLIT),
        ("test", TEST_SPLIT),
    ):
        split_count = info.splits[split_name].num_examples
        if split_count is None or split_count < 1:
            raise ValueError(
                f"Split '{split_name}' has an invalid example count: {split_count}"
            )

        print(f"{role.capitalize():10s}: {split_name} ({split_count:,} examples)")
        dataset = builder.as_dataset(
            split=split_name,
            shuffle_files=False,
            read_config=read_config,
        )

        # Shuffle only the training order. Split membership remains unchanged.
        if role == "train":
            dataset = dataset.shuffle(
                buffer_size=min(split_count, SHUFFLE_BUFFER),
                seed=seed,
                reshuffle_each_iteration=True,
            )

        dataset = dataset.map(
            _prepare_example,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        output_datasets[role] = dataset

    return (
        output_datasets["train"],
        output_datasets["validation"],
        output_datasets["test"],
    )
