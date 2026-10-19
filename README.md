# Running the MVP on Windows and Linux

## 1. Standalone Requirements

`research_mvp` can run independently of the historical research directory. It requires only the source code in this directory, the dataset, Python dependencies, and initialization weights. It does not depend on historical Python scripts, logs, papers, or previous experiment weights in the parent directory.

The only supported pipeline is:

```text
Pretrained initialization with yolov8s.pt
→ Training on the full training set with validation → best.pt from the current run
→ Chronological inference on the test set
→ Clean → Interpolate → Consistent evaluation of all three stages
```

The smoke test used only 8 training images, 4 validation images, and 24 test images, with an input size of 320 and just 1 training epoch. Its results do not represent full-training performance. Using the complete training and validation sets, an input size of 1280, and 50 or 100 epochs constitutes a subsequent new experiment. More data and additional training epochs do not guarantee improvement; performance must be assessed using validation and independent test results.

**The full-training commands in this guide have not been executed as part of this setup.** Validation so far is limited to a small-scale, 1-epoch pipeline on a Windows CPU. The Linux instructions reflect the current entry points and path-handling mechanisms but have not been tested.

## 2. Download and Arrange the Dataset

Dataset:

[Badminton360-Shuttlecock — Hugging Face](https://huggingface.co/datasets/aissur320/Badminton360-Shuttlecock)

Download the dataset from the link above, extract it, and name the dataset directory **`project`**. Place it **alongside `research_mvp` in the same parent directory**:

```text
your_workspace/
├── project/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── research_mvp/
    ├── mvp.py
    ├── fetch_weights.py
    ├── configs/
    ├── src/
    ├── tests/
    └── ...
```

Ensure that `images/` and `labels/` are directly inside `project/`, without an additional nested directory of the same name. Every training, validation, and test image must have a corresponding TXT label file with the same base filename. Background images must have empty label files.

The external `project/` directory contains the source data. The `prepare-data` command below copies the complete dataset into `research_mvp/data/project/` and maps the confirmed source classes 0 and 3 to the single shuttlecock class 0 **in the copy only**. The original dataset remains unchanged. Ensure that sufficient disk space is available for the copy.

## 3. Windows: Set Up the Environment

Use PowerShell and install a working Python 3.12 environment first. Open a terminal in the directory containing both `project/` and `research_mvp/`, then enter the MVP directory:

```powershell
cd research_mvp
python -m venv .venv
```

Create a separate `.venv` on each device. Do not reuse a virtual environment copied from another device. The commands below invoke the virtual environment's Python directly, so activation is unnecessary.

### CPU Installation

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

Full training at an input size of 1280 may take considerable time on a CPU. If a compatible NVIDIA GPU is available, use the GPU installation below instead.

### NVIDIA GPU Installation

In a fresh virtual environment, choose a compatible PyTorch CUDA installation command for your driver and platform:

[Official PyTorch Installation Instructions for Previous Versions](https://pytorch.org/get-started/previous-versions/)

Install the compatible pair `torch==2.7.1` and `torchvision==0.22.1`, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The final output must be `True` before setting `device: "0"` in Section 5. Choose either the CPU or GPU installation method. Do not install `requirements-cpu.txt` first in an environment intended for GPU use.

### Check Dependencies and Download Initialization Weights

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe fetch_weights.py
```

After downloading, `weights/yolov8s.pt` should exist. Skip the download if this file already exists and its source is trusted. The downloader does not overwrite existing files.

## 4. Linux: Set Up the Environment

Use Bash and prepare Python 3.12 with `venv` support. Open a terminal in the directory containing both `project/` and `research_mvp/`. The example assumes that `python3` points to the required version:

```bash
cd research_mvp
python3 --version
python3 -m venv .venv
```

### CPU Installation

```bash
./.venv/bin/python -m pip install -r requirements-cpu.txt
```

### NVIDIA GPU Installation

Follow the [official PyTorch installation instructions for previous versions](https://pytorch.org/get-started/previous-versions/) to install compatible CUDA builds of `torch==2.7.1` and `torchvision==0.22.1` in a fresh virtual environment, then run:

```bash
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The CUDA availability check must return `True` for GPU use. Choose either the CPU or GPU installation method.

### Check Dependencies and Download Initialization Weights

```bash
./.venv/bin/python -m pip check
./.venv/bin/python fetch_weights.py
```

Skip the download if a trusted `weights/yolov8s.pt` already exists. Do not rename weights from another model to `yolov8s.pt`.

## 5. Create a Full-Training Configuration

Copy the complete example configuration to create your own local configuration.

Windows:

```powershell
Copy-Item configs/experiment.example.json configs/local.full.json
```

Linux:

```bash
cp configs/experiment.example.json configs/local.full.json
```

Open `configs/local.full.json` in a text editor and save it as **UTF-8 without BOM**. Preserve the other fields and configure the following settings for your device:

| Field | Setting |
|---|---|
| `run_name` | Use a new, unique name, such as `yolov8s_full_50e_cpu` or `yolov8s_full_50e_gpu`. |
| `paths.data_root` | Keep `data/project` for the imported dataset copy. |
| `paths.pretrained_weights` | Keep `weights/yolov8s.pt`. |
| `paths.output_root` | Keep `outputs`, or specify a new output directory inside the MVP directory, separate from the data. |
| `train.epochs` | Set to `50`, or `100` for a 100-epoch run. The example configuration defaults to 100. |
| `train.imgsz` | Keep `1280`. |
| `train.batch` | Start with `2` on a CPU. For GPU training, choose a value appropriate for the available VRAM. The example defaults to 16. |
| `train.device` | Use `"cpu"` for CPU training or `"0"` for the first available NVIDIA GPU. |
| `predict.imgsz` | Keep `1280`. |
| `predict.device` | Match the selected device: `"cpu"` or `"0"`. |

Values such as `epochs` and `batch` are numbers; `device` is a string. Configuration path fields are resolved relative to `research_mvp/`. Relative configuration filenames passed to `--config` are resolved against the current working directory, so all commands in this guide are executed from the MVP directory.

At runtime, the program resolves absolute paths based on the current project location rather than using fixed paths from the developer's computer. File paths saved in `config.resolved.json` and the training `args.yaml` are relative to `research_mvp/` and use `/` separators. The train/val paths in `dataset.yaml` are relative to the directory containing that YAML file; no machine-specific `path` field is set. When moving the entire project, preserve the relative directory structure of `data/`, `weights/`, and `outputs/`.

New users should create their own runs from the example configuration. `outputs/` contains local run artifacts and is ignored by Git. `.venv/` and `.cache/` are also excluded from publication and should not be copied between devices. Reinstall dependencies on the target device using the instructions above.

Keep `source_class_map` set to `{"0": 0, "3": 0}`. If the downloaded dataset contains other class IDs, verify their meaning before adding mappings.

**Do not modify the configuration after training starts**, including the prediction device, thresholds, or `run_name`. Subsequent stages check that the configuration matches the one saved during training.

## 6. Windows: Run the Full Pipeline

The following commands use the same configuration created in Section 5. Run each step only after the previous step succeeds.

```powershell
# Inspect the external source dataset without modifying it and save an audit report in the MVP
.\.venv\Scripts\python.exe -X utf8 mvp.py audit --data-root ../project

# Copy the complete dataset and map labels; this does not start training
.\.venv\Scripts\python.exe -X utf8 mvp.py prepare-data --source ../project --config configs/local.full.json

# Train on the full training set with validation; 50/100 epochs require this explicit flag
.\.venv\Scripts\python.exe -X utf8 mvp.py train --config configs/local.full.json --allow-full-train

# Load best.pt from the current run and perform chronological inference on the complete test set
.\.venv\Scripts\python.exe -X utf8 mvp.py predict --config configs/local.full.json

# Save Clean and Interpolate outputs separately
.\.venv\Scripts\python.exe -X utf8 mvp.py postprocess --config configs/local.full.json

# Evaluate raw, Clean, and Interpolate outputs with the same evaluator
.\.venv\Scripts\python.exe -X utf8 mvp.py evaluate --config configs/local.full.json
```

## 7. Linux: Run the Full Pipeline

```bash
./.venv/bin/python -X utf8 mvp.py audit --data-root ../project
./.venv/bin/python -X utf8 mvp.py prepare-data --source ../project --config configs/local.full.json
./.venv/bin/python -X utf8 mvp.py train --config configs/local.full.json --allow-full-train
./.venv/bin/python -X utf8 mvp.py predict --config configs/local.full.json
./.venv/bin/python -X utf8 mvp.py postprocess --config configs/local.full.json
./.venv/bin/python -X utf8 mvp.py evaluate --config configs/local.full.json
```

Run `prepare-data` only once for a given destination dataset directory. If `data/project/` already contains a complete, successfully imported and mapped copy, reuse it for subsequent experiments without importing it again. If a previous import was incomplete, specify a new `paths.data_root` and import into that directory to avoid overwriting or mixing data.

Do not use `run-smoke` for full training. That command forces 1 epoch and limits the number of samples. Use the staged commands above for full training.

## 8. Inspect Results and Run New Experiments

For example, with `run_name: "yolov8s_full_50e_gpu"`:

```text
outputs/yolov8s_full_50e_gpu/
├── config.resolved.json
├── dataset_audit.json
├── manifest.json
├── training.json
├── train/
│   ├── args.yaml
│   ├── results.csv
│   └── weights/
│       ├── best.pt
│       └── last.pt
├── raw.json
├── clean.json
├── interpolated.json
└── metrics.json
```

- `best.pt` is selected based on validation performance during training. Inference accepts only the `best.pt` recorded for the current run and verifies its SHA-256 hash; it does not use weights from previous smoke tests.
- `metrics.json` contains TP/FP/FN, P/R@0.5, AP50, and mAP50–95 for all three stages. The definitions used by the training framework's internal validation metrics differ from those of this unified evaluator and should not be treated as identical.
- `manifest.json` explicitly records sequence IDs, numeric frame indices, and frame order. Missing frames, duplicate frame indices, incorrect ordering, or changes in image dimensions cause an error. Post-processing does not operate across sequence boundaries.
- Clean and Interpolate use only predictions and permitted frame information. Test ground truth is used only for final evaluation. Training and validation labels are used for supervised training and validation.
- This is a new experiment using actual image widths and a new AP101 evaluation protocol. Its results must not be described as a direct reproduction of the paper.

Use a new `run_name` for every new training run. Existing outputs cannot be overwritten. After changing the configuration, you cannot reuse an old run for prediction or evaluation. The current entry point fixes `resume=False` and provides no command for resuming interrupted training. Use a new run name when restarting an experiment after an interruption.

If GPU or system memory is insufficient, stop the failed run and reduce `train.batch` in a new configuration with a new run name. Keeping the input size at 1280 avoids silently changing a full experiment to the smoke test's input size of 320. If a different input size is necessary, record it as a new experimental setting.

Optionally, run the existing unit tests:

Windows:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

The MVP's `.gitignore` excludes `data/`, `weights/`, `outputs/`, `.venv/`, caches, and `configs/local*.json`. Do not add the external sibling `project/` directory to the public repository either. Publish only the MVP source code.

This guide provides instructions only. It does not automatically download files, install dependencies, import data, run tests, or start training.
