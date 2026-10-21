# Diabetic Retinopathy Detection Mini-Project

This is a small, local M.Tech mini-project for five-class diabetic-retinopathy severity classification. Read `PROJECT.md` before changing the modeling plan.

## Environment

The confirmed environment is Python 3.10.20, TensorFlow 2.10.1, Keras 2.10.0, TensorFlow Datasets 4.8.3, and NumPy 1.23.5 in the Conda environment `dr_tf`. The NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) was detected and tested successfully from PyCharm.

Keep your existing dataset at `data/btgraham-300/`. The scripts only read it; they do not download, rebuild, or modify the dataset.

## Initial training implementation

The first baseline is a working implementation choice of **EfficientNetB0 with ImageNet weights, a frozen backbone, global average pooling, dropout, and a five-class softmax head**. It is a baseline, not a final claim that B0 is the best architecture. The script expects `btgraham-300` images to be resized in the shared loader to `300 x 300`; EfficientNet's built-in input rescaling is retained, so do not add another `1/255` rescale.

If ImageNet weights are not already cached locally, the first model initialization needs internet access to retrieve them. The script does not silently fall back to random initialization.

## PyCharm CUDA setup

`configs/tensorflow_setup.py` is the shared setup for native-Windows TensorFlow 2.10 in the `dr_tf` Conda environment. It exposes `dr_tf\Library\bin` to Windows DLL loading before TensorFlow imports, then offers a shared GPU memory-growth setup for the 4 GB RTX 3050. All project Python files and notebooks that import TensorFlow must import the setup module first. Do not import TensorFlow earlier in the same process.

After replacing project files, restart the PyCharm notebook kernel so the CUDA DLL setup runs before TensorFlow is loaded. To verify GPU compute, right-click `scripts/check_gpu.py` in the Project panel and choose **Run 'check_gpu'**. A successful check reports `GPU test passed.` and an operation device ending in `/device:GPU:0`. If the DLL warning returns, confirm the selected PyCharm interpreter is `C:\Users\kawal\miniconda3\envs\dr_tf\python.exe` and that the setup import comes before TensorFlow. Do not reinstall packages unless the diagnostic indicates they are missing.

## Running the notebook

1. Keep this folder as the project root and place your existing `data/btgraham-300/` directory under it.
2. Open `training.ipynb` with the `dr_tf` Jupyter kernel.
3. Run the environment and optional model-preview cells first. Opening the notebook never starts training.
4. Restart the kernel after project updates, then run the environment check. The training pipeline applies the shared CUDA DLL setup and enables GPU memory growth before TensorFlow operations. If no GPU is visible, the training cell requires you to type `RUN-ON-CPU` to deliberately accept a CPU run.
5. In the EfficientNet training cell, choose `scratch` or `resume`, set batch size, and enter the **total target epoch** count.

For a first run choose `scratch`. Resume targets the newest run. If that run has no valid TensorFlow recovery checkpoint, it stops rather than falling back to an older run or silently starting from scratch. Set the target epoch count higher than the run's last completed epoch to continue training.

## Output locations

- Recovery checkpoints: `models/efficientnet/checkpoints/run_001/`
- Best model, history, and metadata: `models/efficientnet/runs/run_001/`
- Each scratch run gets a new run ID. Prior runs are preserved.

This first baseline does not apply class weights; the imbalance strategy remains an open decision. Training checkpoints save model weights, optimizer state, and the completed epoch after each epoch. `best_model/` is a TensorFlow SavedModel directory, separate from recovery checkpoints, and is updated only when validation loss improves. It can be loaded with `tf.keras.models.load_model()` in the project's TensorFlow 2.10 environment. The model status cell reads metadata and does not evaluate a model.

The DenseNet, fusion, formal evaluation, and interface/activation code are not implemented in this initial bundle. Do not treat the EfficientNet baseline as trained until a run completes successfully.
