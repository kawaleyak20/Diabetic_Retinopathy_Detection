# Diabetic Retinopathy Detection Mini-Project

**Purpose:** Persistent project context and implementation contract. Read this file before continuing project work after a context refresh. Update it when a project decision changes.

**Last updated:** 2026-10-22

## 1. Project paths

- **Project root:** `D:\M L\Diabetic_Retinopathy_Detection_mini_project`
- **Input dataset:** `D:\M L\Diabetic_Retinopathy_Detection_mini_project\data\btgraham-300\`
- **Completed dataset analysis notebook:** `dataset_analysis.ipynb` (kept where it currently resides; do not rebuild unnecessarily).
- **Environment:** Windows 11, Conda environment `dr_tf`, Python 3.10.20, TensorFlow 2.10.1, Keras 2.10.0, TensorFlow Datasets 4.8.3, NumPy 1.23.5. These versions were confirmed by a diagnostic cell in the active `dr_tf` kernel.
- **Hardware constraint:** NVIDIA RTX 3050 laptop GPU with 4 GB VRAM. GPU detection was fixed and verified when launching from PyCharm: TensorFlow 2.10.1 sees `/physical_device:GPU:0`, and a forced matrix multiplication executed on `/device:GPU:0`. The common setup module `configs/tensorflow_setup.py` must be imported before TensorFlow in every project Python entry point and notebook. It prepends the active Conda environment’s `Library\bin` to `PATH` and registers it with `os.add_dll_directory()`. After importing TensorFlow, call `_tensorflow_setup.configure_gpu_memory_growth(tf)` before creating tensors/models.

The project root is a Windows-local path and was not accessible from the assistant's current container. This `PROJECT.md` is therefore provided as a file to place in the project root.

## 2. Project objective and scope

- Individual M.Tech mini-project; **not** production software.
- Detect/classify diabetic retinopathy severity from retinal fundus images.
- Dataset labels and class order:
  - `0`: No DR
  - `1`: Mild NPDR
  - `2`: Moderate NPDR
  - `3`: Severe NPDR
  - `4`: Proliferative DR (PDR)
- Dataset: Kaggle Diabetic Retinopathy Detection.
- The label distribution is imbalanced, with many more grade-0 cases. Address imbalance deliberately and report per-class performance; do not distort the held-out evaluation distribution merely to make it balanced.

## 3. Dataset and completed work

- Dataset acquisition and TFDS preprocessing have already been completed.
- Available formats include `1M`, `250K`, and `btgraham-300`.
- **Final input choice for all planned model experiments: `btgraham-300`.** Do not switch planned model experiments to `250K` or `1M` without an explicit decision.
- Treat `data\btgraham-300\` as existing input data and do not modify it in training code.
- `dataset_analysis.ipynb` was run successfully by the user: every operation completed without errors. It covers dataset checks/EDA, class imbalance, sample images, and format comparison. Treat data analysis as completed; do not repeat it without a concrete need.
- Note: the saved EDA notebook was configured with `CONFIG = "250K"`. Its split counts and overlap checks therefore describe that folder; confirm the corresponding metadata in `btgraham-300` by running the updated `inspect_dataset()` before the one-batch smoke test. The loader is designed to use the split names actually present and will stop clearly if an expected official split is missing.
- Native extracted images were deleted earlier; use the retained TFDS-formatted dataset.

## 4. Planned project structure

```text
Diabetic_Retinopathy_Detection_mini_project/
├── PROJECT.md                    # Persistent project decisions/context
├── training.ipynb                 # Central, manually controlled training notebook
├── dataset_analysis.ipynb         # Completed EDA notebook; only its startup setup was updated
├── data/
│   └── btgraham-300/              # Existing input dataset; do not modify
├── scripts/
│   ├── check_gpu.py
│   ├── data_loader.py
│   ├── utils.py
│   ├── train_efficientnet.py
│   ├── train_densenet.py
│   ├── train_fusion.py
│   └── evaluate.py
├── models/
│   ├── active/                    # Selected model and metadata for interface
│   │   └── model_metadata.json
│   ├── efficientnet/
│   │   ├── checkpoints/run_001/   # TensorFlow recovery checkpoints
│   │   └── runs/run_001/          # Best model, history, run metadata
│   ├── densenet/
│   │   ├── checkpoints/run_001/
│   │   └── runs/run_001/
│   └── fusion/
│       ├── checkpoints/run_001/
│       └── runs/run_001/
├── interface/
│   ├── app.py
│   └── predict.py
├── results/
│   ├── metrics/
│   ├── plots/
│   └── predictions/
├── configs/
│   ├── config.py
│   └── tensorflow_setup.py
├── requirements.txt
├── README.md
└── .gitignore
```

Create folders/files only as needed. Model files, history files, checkpoints, reports, and plots are outputs generated during implementation/training; empty placeholder model artifacts are not required. Keep the completed EDA notebook where it currently resides.

## 5. Model experiments

Planned experiments, all using `btgraham-300` and a consistent evaluation protocol:

1. EfficientNet baseline.
2. DenseNet baseline.
3. Fusion model combining the two architectures/features, if feasible on the available GPU.

**Initial implementation choice for the first baseline:** EfficientNetB0 with ImageNet weights, a frozen backbone, global average pooling, dropout, and a five-class softmax head. This first baseline does not apply class weights; the imbalance strategy remains an open decision. This is a resource-conscious starter choice for the 4 GB GPU, not a final conclusion that B0 is the best variant. ImageNet weights must load successfully; the code must not silently fall back to random initialization if downloading or cache access fails. The fusion architecture (including whether to use attention fusion) is still undecided.

## 6. Central training notebook

`training.ipynb` is the control panel. Keep a separate, clearly labelled cell/block for each model (EfficientNet, DenseNet, fusion). Scripts hold reusable implementation; the notebook controls when operations run.

Each model's training block should explicitly ask the user for:

1. **Run mode:** resume latest checkpoint or start a new run from scratch.
2. **Batch size:** ask independently per model; default is `4`.
3. **Epoch count:** request/allow the user to set it.

Opening/importing the notebook must not automatically start training. Execute training only when the relevant cell is run. Keep model experiments independent.

## 7. Input and GPU defaults

- **Model input size:** `300 x 300` for all planned experiments. This is separate from the dataset format name: TFDS `btgraham-300` sets the eyeball radius to 300 pixels and crops to 90% of that radius, so decoded source images are commonly `540 x 540`. The loader resizes these decoded images to `300 x 300` using bilinear interpolation with antialiasing.
- **Windows CUDA initialization:** import `configs.tensorflow_setup` before any `tensorflow` import, then call `configure_gpu_memory_growth(tf)` immediately after importing TensorFlow and before tensor/model operations. This is required for PyCharm and notebook kernels in the current Conda environment. Restart the run process or notebook kernel after changing the import order. Do not remove CUDA/cuDNN or reinstall the working environment just to fix the PyCharm DLL lookup issue.
- **Default batch size:** `4`, configurable per model/run.
- If VRAM runs out, first reduce the batch size rather than silently changing input resolution.
- Use mixed precision only if supported by the installed TensorFlow/GPU combination; verify compatibility during setup.
- Actual VRAM use depends on the architecture and fusion implementation, so batch size 4 is a starting point, not a guarantee.

## 8. Checkpointing and resumable training (mandatory)

Apply the same straightforward rules to every model:

- Save recovery checkpoints automatically after each completed epoch, not only at the end. The current config keeps every epoch checkpoint within a run; older runs are also never overwritten.
- A recovery checkpoint must preserve model weights and optimizer state where supported, plus the epoch/progress needed to continue correctly.
- Resume mode must restore the latest checkpoint for the newest run of that model and continue from saved progress; do not silently restart if no valid checkpoint exists—stop and ask the user to select scratch mode or inspect the checkpoint folders. Do not silently fall back to an older run if the newest run has no valid checkpoint.
- Scratch mode starts a separate new run with fresh initialization and no previous optimizer state. It must not overwrite prior runs.
- Store checkpoints in run-specific subdirectories, e.g. `models\efficientnet\checkpoints\run_001\`. Store that run's best model, training history, and metadata under `models\efficientnet\runs\run_001\`.
- Keep recovery checkpoints separate from the best validation model. Recovery files are for resuming; the best model is the candidate for evaluation/deployment. The current TF 2.10-compatible baseline writes the best model as a TensorFlow SavedModel directory at `models\efficientnet\runs\run_001\best_model\`. This avoids the known EfficientNetB0 HDF5 JSON-serialization error in TF 2.10.
- Preserve prior checkpoints and results.

## 9. Evaluation controls and safeguards (mandatory)

- Evaluation must be explicitly controlled by the user.
- Make post-training automatic evaluation opt-in; default `AUTO_EVALUATE_AFTER_TRAINING = False`.
- If the user enables automatic evaluation, evaluate **only the model/run whose training just completed successfully**. Never sweep over or automatically evaluate all saved models.
- Keep a separate manual evaluation cell that requires explicit model/run selection.
- Do not evaluate a run marked failed/interrupted or mistake an old model file for the latest training result.
- Evaluation must not retrain a model or overwrite recovery checkpoints/best weights.
- Preserve the existing TFDS `train`, `validation`, and `test` splits for every experiment; do not create a new random 80/10/10 split. The completed EDA verified the split sizes and found no image-name overlap across these splits. The training loader should shuffle only the order of the training examples, never split membership.
- Planned metrics: accuracy, macro-F1, per-class recall, confusion matrix, and quadratic weighted kappa (QWK). Include other metrics only if they meaningfully support the project.

## 10. Active model and inference interface

- `interface\app.py` provides the user interface.
- `interface\predict.py` handles inference-time image preprocessing and prediction.
- The interface should load only the model selected under `models\active\` and use metadata to identify model file, architecture, input preprocessing, and class order.
- The active model's inference preprocessing and class order must match its training setup.
- Model selection/activation should be explicit; training another model must not silently change the active interface model.

## 11. Minimal-coding rule (project-wide)

This is a simple academic mini-project, not production software.

- Write only code needed for the agreed pipeline and operations.
- Prefer simple, readable functions over nested functions, elaborate class hierarchies, or excessive abstraction.
- Avoid unnecessary frameworks, production-grade logging systems, complex configuration layers, and boilerplate.
- Reuse shared functions where useful; do not duplicate substantial common logic.
- Keep outputs understandable: training progress and relevant evaluation metrics.
- Retain essential safeguards: checkpoint/resume, run separation, controlled evaluation, and consistent preprocessing.
- Guiding rule: **the simplest code that reliably performs the agreed task.**

## 12. Current implementation status and next steps

Completed:
- Dataset acquisition/preprocessing and EDA; the user's `dataset_analysis.ipynb` has run successfully.
- Selection of `btgraham-300` as the input format for all planned experiments.
- Local environment diagnosis confirmed Python 3.10.20, TensorFlow 2.10.1, Keras 2.10.0, TFDS 4.8.3, and NumPy 1.23.5.
- The user reran the loader smoke test after restarting the Jupyter kernel and confirmed the corrected loader now works.

Implemented in the current code bundle (Python syntax checked):
- Shared configuration and data loading in `configs/config.py` and `scripts/data_loader.py`.
- Loader reads the existing local TFDS output, preserves official train/validation/test splits (35,126 / 10,906 / 42,670 examples), shuffles only training order, and resizes decoded images to `300 x 300` at pipeline time without altering saved dataset files.
- `scripts/utils.py`: run-specific directories, readable JSON metadata, run status listing, checkpoint discovery, and a recovery callback.
- `scripts/train_efficientnet.py`: initial EfficientNetB0 baseline with frozen ImageNet backbone and five-class head; explicit scratch/resume mode; per-epoch model+optimizer recovery checkpoints; separate best-validation TensorFlow SavedModel directory (to avoid TF 2.10 EfficientNetB0 HDF5 serialization failure); CSV history; and run metadata.
- `training.ipynb`: manual environment check, optional loader smoke test, architecture preview, controlled EfficientNet training block, and run status section. Opening the notebook does not train automatically.
- `configs/tensorflow_setup.py`: shared Windows/Conda CUDA DLL discovery before TensorFlow import, plus one-time GPU memory-growth configuration.
- `scripts/check_gpu.py`: repeatable PyCharm-friendly GPU detection and real GPU operation check.
- `README.md`, `requirements.txt`, and `.gitignore`.

Not yet implemented/confirmed:
- A fully successful end-to-end training run and local runtime verification of checkpoint restore/resume. The first run reached the end of Epoch 1's training/validation work but failed while exporting the best model because TF 2.10 EfficientNetB0 HDF5 saving attempts to JSON-serialize an EagerTensor. The current fix writes the best model as a TensorFlow SavedModel directory and saves the recovery checkpoint before best-model export. Because the old callback raised before its recovery callback ran, inspect `models\efficientnet\checkpoints\run_001\`; if it has no valid checkpoint, choose `scratch` for the next attempt rather than `resume` (which should stop safely).
- DenseNet and fusion implementations, formal manual evaluation (`scripts/evaluate.py`), and interface/model activation.
- The final imbalance strategy; do not add class weights or undersample classes until the approach is explicitly selected. Never rebalance validation/test.

**Next step:** restart the PyCharm notebook kernel, run the environment check and optional EfficientNet model-preview cell, and confirm that the shared setup reports the RTX 3050. Then test the first training run with a small epoch target before committing to a longer run. The training cell requires an explicit user action and will not run CPU training unless the user types `RUN-ON-CPU`.
