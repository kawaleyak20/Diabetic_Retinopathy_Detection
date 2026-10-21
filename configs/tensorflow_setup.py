"""Windows/Conda TensorFlow runtime setup.

Import this module BEFORE importing TensorFlow in every Python entry point and
notebook. In this project's native-Windows TensorFlow 2.10 environment, CUDA
and cuDNN DLLs live in ``<conda-env>\\Library\\bin``. PyCharm may launch Python
without that directory on PATH even when the DLLs are installed.

This module intentionally does not import TensorFlow itself.
"""

import os
import sys
from pathlib import Path

# Keep DLL-directory handles alive for the life of the Python process.
_DLL_DIRECTORY_HANDLES = []
_CONFIGURED_DLL_DIRECTORIES = set()
_GPU_MEMORY_GROWTH_CONFIGURED = False


def configure_tensorflow_dlls():
    """Expose this Conda environment's DLL directory to Windows DLL loading."""
    if os.name != "nt":
        return None

    dll_directory = Path(sys.prefix) / "Library" / "bin"
    if not dll_directory.is_dir():
        return None

    dll_directory = dll_directory.resolve()
    dll_directory_text = str(dll_directory)
    normalized_target = os.path.normcase(os.path.normpath(dll_directory_text))

    # Python/Conda DLL discovery can depend on PATH when launched from an IDE.
    existing_path = os.environ.get("PATH", "")
    path_entries = [entry.strip().strip('"') for entry in existing_path.split(os.pathsep)]
    normalized_entries = {
        os.path.normcase(os.path.normpath(os.path.abspath(entry)))
        for entry in path_entries
        if entry
    }
    if normalized_target not in normalized_entries:
        os.environ["PATH"] = (
            dll_directory_text + os.pathsep + existing_path
            if existing_path
            else dll_directory_text
        )

    # PATH alone was insufficient in this PyCharm launch environment. Register
    # the directory explicitly as well, and retain the handle globally.
    if normalized_target not in _CONFIGURED_DLL_DIRECTORIES:
        if hasattr(os, "add_dll_directory"):
            handle = os.add_dll_directory(dll_directory_text)
            _DLL_DIRECTORY_HANDLES.append(handle)
        _CONFIGURED_DLL_DIRECTORIES.add(normalized_target)

    return dll_directory_text


def configure_gpu_memory_growth(tf_module):
    """Enable gradual GPU memory allocation after TensorFlow is imported.

    Call immediately after ``import tensorflow as tf`` and before creating any
    tensors/models. This is useful for the project's 4 GB RTX 3050. Calling the
    helper repeatedly in the same process is safe after successful setup.
    """
    global _GPU_MEMORY_GROWTH_CONFIGURED

    devices = tf_module.config.list_physical_devices("GPU")
    if not devices or _GPU_MEMORY_GROWTH_CONFIGURED:
        return devices

    try:
        for device in devices:
            tf_module.config.experimental.set_memory_growth(device, True)
    except RuntimeError as error:
        raise RuntimeError(
            "GPU memory growth could not be configured because TensorFlow may have "
            "already initialized the GPU. Restart the PyCharm run process or notebook "
            "kernel, then ensure configs.tensorflow_setup is imported before TensorFlow "
            "and configure_gpu_memory_growth(tf) is called before any GPU operations."
        ) from error

    _GPU_MEMORY_GROWTH_CONFIGURED = True
    return devices


# Apply DLL search-path setup at import time, before a caller imports TensorFlow.
CUDA_DLL_DIRECTORY = configure_tensorflow_dlls()
