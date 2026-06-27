# 📡 Satellite Communication Modulation Classification

A deep learning system for automatic modulation classification of satellite communication signals. The model distinguishes between **16APSK**, **8PSK**, and **QPSK** modulation schemes using raw IQ (In-phase / Quadrature) signal data, and supports deployment to AMD Xilinx **Versal ACAP** hardware via Vitis-AI / DPU.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Environment Variables (Data Paths)](#environment-variables-data-paths)
- [How to Run](#how-to-run)
  - [1. Training](#1-training)
  - [2. Testing / Evaluation](#2-testing--evaluation)
  - [3. SNR Estimation](#3-snr-estimation)
  - [4. Versal Hardware Deployment](#4-versal-hardware-deployment)
- [Key Files Reference](#key-files-reference)
- [Results](#results)

---

## Project Overview

This project was developed as a final project for an Electrical Engineering degree. The goal is to classify the modulation scheme of incoming satellite signals in real time with high accuracy and low latency, ultimately running on an AMD Versal ACAP neural processing unit (DPU).

**Supported Modulation Schemes:**
| Label | Modulation |
|-------|------------|
| 0     | 16APSK     |
| 1     | 8PSK       |
| 2     | QPSK       |

**Input Signal Representation:**
Each IQ sample is processed into a **4-channel feature tensor** of shape `(4, 512)`:
- Channel 0 – I (In-phase)
- Channel 1 – Q (Quadrature)
- Channel 2 – Amplitude
- Channel 3 – Phase

---

## Architecture

Three CNN/ResNet variants are implemented and can be selected for training:

| Model File | Class | Description |
|---|---|---|
| `modulation_model_cnn.py` | `CNNClassifier` | Custom ResNet-18-like 1D CNN with residual blocks |
| `modulation_model_resnet18.py` | `resnet18` | ResNet-18 adapted for 1D IQ signals (Vitis-AI DPU optimized) |
| `modulation_model_resnet34.py` | `CNNClassifier` | ResNet-34 adapted for 1D IQ signals |

All models take input shape `(batch, 4, 512)` and output class logits.

**Training configuration (defaults in `modulation_main.py`):**
- Segment length: 512 samples
- Batch size: 64
- Optimizer: AdamW (lr=1e-3, weight decay=1e-3)
- Scheduler: CosineAnnealingLR
- Loss: CrossEntropyLoss with label smoothing (0.05)
- Max epochs: 400 with early stopping (patience=100)

---

## Project Structure

```
codesfortrainEEfinalproj/
│
├── modulation_main.py            # Main training script (ResNet-34 variant)
├── modulation_main_resnet18.py   # Training script for ResNet-18 (DPU-optimized)
├── modulation_main_resnet34.py   # Training script for ResNet-34
│
├── modulation_test_cnn.py        # Model testing & confusion matrix generation
│
├── data_manager.py               # SatComDataManager: data loading & saving
├── host_preprocessor.py          # Offline preprocessor: IQ -> binary tensors for Versal
├── hardware_validation.py        # Zero-preprocessing validation script on Versal DPU
│
├── create_versal_bin_third.py    # Utility: create equal class-balanced bin splits
├── calculate_snr.py              # SNR estimation using Welch's PSD method
├── signal_plotter.py             # IQ signal visualization utility
├── matrics_plot.py               # Metrics plotting helpers
│
├── modulation_models/            # Model definitions
│   ├── modulation_model_cnn.py
│   ├── modulation_model_resnet18.py
│   ├── modulation_model_resnet34.py
│  
│
├── data_ready_SR.json            # Signal metadata database (modcod, symbol rate, etc.)
├── class_mapping_SR.json         # Mappings for symbol rate & rolloff indices
│
├── best_resnet18_dpu.pth         # Pre-trained ResNet-18 weights (DPU-optimized)
├── best_resnet_cnn.pth           # Pre-trained ResNet-CNN weights
│
└── README.md
```

---

## Requirements

### Python Version
Python 3.8+

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib seaborn tqdm scikit-learn psutil
```

> **For Versal / DPU hardware validation only:**
> `xir` and `vart` packages are provided by AMD Vitis-AI and must be installed on the Versal target board running PetaLinux. They are **not** available on a standard PC.

---

## Environment Variables (Data Paths)

The `SatComDataManager` uses environment variables for portability across machines. Set these before running any script:

| Variable | Description | Default |
|---|---|---|
| `DATA_DIR` | Path to directory containing `.iq` signal files | `/data/tempo/iq_augmented_cut` |
| `OUTPUT_DIR` | Path for saving results / predictions | `/app/results` |
| `METADATA_FILE` | Path to `data_ready_SR.json` metadata file | `/app/data_ready_SR.json` |

**Windows example:**
```powershell
$env:DATA_DIR      = "D:\iq_augmented_cut"
$env:OUTPUT_DIR    = "D:\results"
$env:METADATA_FILE = "C:\path\to\data_ready_SR.json"
```

**Linux / macOS example:**
```bash
export DATA_DIR=/data/iq_augmented_cut
export OUTPUT_DIR=/app/results
export METADATA_FILE=/app/data_ready_SR.json
```

---

## How to Run

### 1. Training

**Train with ResNet-34 (default):**
```bash
python modulation_main.py
```

**Train with ResNet-18 (recommended for Versal/DPU export):**
```bash
python modulation_main_resnet18.py
```

**Train with ResNet-34 (dedicated script):**
```bash
python modulation_main_resnet34.py
```

The training script will:
1. Load `.iq` files from `DATA_DIR` and match them to the metadata in `data_ready_SR.json`.
2. Split data 80/20 for training/validation.
3. Apply data augmentation (random phase rotation) during training.
4. Save the best model weights to `models_results/best_<name>.pth`.
5. Save training metrics plots (`Accuracy` and `Loss` curves) to `models_results/`.

> **GPU Support:** Training automatically uses CUDA if a compatible GPU is available.

---

### 2. Testing / Evaluation

Edit the paths in `modulation_test_cnn.py` (`TestConfig` dataclass) to match your local setup:

```python
@dataclass
class TestConfig:
    model_name: str = "best_resnet18_dpu.pth"           # Model file to load
    metadata_path: str = r"C:\path\to\data_ready_SR.json"
    test_data_dir: str = r"D:\iq_augmented_cut"
    test_split: float = 0.10  # Use 10% of files for testing
```

Then run:
```bash
python modulation_test_cnn.py
```

**Output:**
- Overall accuracy, average confidence, average inference latency
- A confusion matrix saved as `poster_results_enlarged.png`

---

### 3. SNR Estimation

Estimate the Signal-to-Noise Ratio of your IQ data using Welch's PSD method:

```bash
python calculate_snr.py <path_to_iq_directory>
```

**Example:**
```bash
python calculate_snr.py D:\iq_augmented_cut
```

Requires `data_ready_SR.json` and `class_mapping_SR.json` in the script's directory (or set `DATA_DIR`).

---

### 4. Versal Hardware Deployment

This section describes the full pipeline for deploying and validating the model on the AMD Versal ACAP DPU.

#### Step 1 — Preprocess IQ files into DPU-ready binaries (run on PC)

Edit the paths at the top of `host_preprocessor.py`:
```python
INPUT_DIR     = r"D:\iq_augmented_cut"      # Raw .iq files
OUTPUT_DIR    = r"D:\versal_ready_bins"      # Output directory for .bin files
METADATA_PATH = r"C:\path\to\data_ready_SR.json"
```

Then run:
```bash
python host_preprocessor.py
```

This generates:
- `<signal_name>.bin` — float32 tensor files with shape `(1, 4, 1, 512)` ready for DPU ingestion
- `versal_manifest.json` — label manifest for all exported bins

#### Step 2 — (Optional) Create class-balanced bin subset

```bash
python create_versal_bin_third.py \
    --src_dir D:\versal_ready_bins \
    --dest_dir D:\versal_ready_equal_bins \
    --samples_per_class 6000
```

#### Step 3 — Validate on Versal DPU (run **on the Versal board**)

Copy the `.bin` files, `versal_manifest.json`, and the compiled `.xmodel` to the Versal board, then run:

```bash
python hardware_validation.py --model resnet18_16bf.xmodel --data_dir /path/to/versal_ready_bins
```

**Output:**
- Final hardware accuracy (%)
- Average DPU inference latency (ms)
- Sustained throughput (FPS)
- Per-class accuracy breakdown

---

## Key Files Reference

| File | Purpose |
|---|---|
| `modulation_main.py` | Main training entry point |
| `modulation_test_cnn.py` | Model evaluation & confusion matrix |
| `data_manager.py` | Data loading utilities for `.iq` files |
| `host_preprocessor.py` | Converts IQ files to DPU-ready `.bin` tensors |
| `hardware_validation.py` | On-board DPU inference benchmarking |
| `calculate_snr.py` | Estimates SNR per signal file via PSD |
| `data_ready_SR.json` | Signal metadata (modulation, symbol rate, etc.) |
| `best_resnet18_dpu.pth` | Pre-trained ResNet-18 weights |
| `best_resnet_cnn.pth` | Pre-trained ResNet-CNN weights |

---

## Results

| Metric | Value |
|---|---|
| Modulation classes | 16APSK, 8PSK, QPSK |
| Input tensor shape | (4, 512) — [I, Q, Amp, Phase] |
| Hardware platform | AMD Versal ACAP (Vitis-AI DPU) |

Sample results are included as images:
- `poster_results_enlarged.png` — Confusion matrix from software evaluation
- `binary_validation_results.png` — Results from on-hardware binary validation
