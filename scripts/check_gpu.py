"""Verify TensorFlow GPU detection and a real GPU matrix-multiplication op."""

import sys

from configs import tensorflow_setup as _tensorflow_setup

import tensorflow as tf

# Set memory growth before the first TensorFlow operation allocates GPU memory.
gpus = _tensorflow_setup.configure_gpu_memory_growth(tf)

print("Python:", sys.executable)
print("TensorFlow:", tf.__version__)
print("CUDA DLL directory:", _tensorflow_setup.CUDA_DLL_DIRECTORY)
print("Detected GPUs:", gpus)

if not gpus:
    raise RuntimeError(
        "TensorFlow cannot detect a GPU. Confirm this script uses the dr_tf "
        "interpreter and that the environment has the expected CUDA/cuDNN DLLs."
    )

# Force placement on the GPU so CPU fallback cannot hide a broken setup.
tf.config.set_soft_device_placement(False)
with tf.device("/GPU:0"):
    a = tf.random.normal((1024, 1024))
    b = tf.random.normal((1024, 1024))
    c = tf.matmul(a, b)

print("Operation device:", c.device)
print("Test result:", float(tf.reduce_sum(c).numpy()))
print("GPU test passed.")
