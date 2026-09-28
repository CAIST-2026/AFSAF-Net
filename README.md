# AFSAF-Net: Adaptive Frequency-Spatial Attention Fusion Network with IMU Guidance

> Robust indoor localization under low-light and texture-degraded environments.

[![paper](https://img.shields.io/badge/paper-2026-2ea44f)](.)
[![pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?\&style=flat\&logo=PyTorch\&logoColor=white)](https://pytorch.org)
[![params](https://img.shields.io/badge/params-3.08M-blue)](#architecture)
[![opensource](https://img.shields.io/badge/Open%20source-%E2%9C%93-brightgreen)](#license)

***

## Overview

AFSAF-Net is an **Adaptive Frequency-Spatial Attention Fusion Network** that jointly learns frequency-domain and spatial-domain visual representations, and fuses them adaptively with synchronized inertial (IMU) measurements.

Unlike previous LBP-FFT-CNN or fixed visual-IMU fusion methods, AFSAF-Net explicitly models:

| Module                      | What it learns                                               | Paper Section |
| --------------------------- | ------------------------------------------------------------ | ------------- |
| 🧱 **Spatial Branch**       | Local texture, edges, and structural cues from RGB images    | §3.2          |
| 🌊 **Frequency Branch**     | Adaptive spectral representations via learnable FFT encoding | §3.3          |
| 🔀 **Attention Fusion**     | Per-channel weight map `A` balancing spatial vs. frequency   | §3.4          |
| 🧭 **IMU Reliability Gate** | Dynamic coefficient `G` balancing visual vs. inertial cues   | §3.5          |
| 🎯 **Localization Head**    | Regression to position `(x, y, z)` and orientation `θ`       | §3.6          |

### Key Equations

```
F_s = f_s(I; θ_s)                                    # spatial features
F_f = f_f(FFT(I); θ_f)                               # frequency features
A   = σ(W_a · [F_s, F_f] + b_a)                      # attention map
F_v = A ⊙ F_s + (1 − A) ⊙ F_f                        # fused visual
F_i = f_i(M; θ_i),   G = σ(W_g · F_i + b_g)          # IMU + reliability gate
F   = G ⊙ F_i + (1 − G) ⊙ F_v                        # multimodal representation
P̂   = Head(F)                                        # [x, y, z, θ]
```

***

#  Architecture

```
RGB Image ──► Spatial Branch ──┐
                                ├──► Attention Fusion ──┐
RGB Image ──► Frequency Branch ─┘                       ├──► IMU Reliability Gate ──► Localization Head ──► (x, y, z, θ)
                                                         ▲
IMU (ax,ay,az,ωx,ωy,ωz) ───────────────────► IMU Encoder ┘
```

| Component            | Output Channels | Params    |
| -------------------- | --------------- | --------- |
| Spatial Branch       | 128             | 0.72M     |
| Frequency Branch     | 128             | 0.72M     |
| Attention Fusion     | 128             | 0.03M     |
| IMU Reliability Gate | 128             | 0.02M     |
| Localization Head    | —               | 1.77M     |
| **Total**            | <br />          | **3.27M** |

***

## 📁 Repository Structure

```
AFSAF-Net/
├── afsaf_net/
│   ├── __init__.py        # Package entry
│   ├── model.py           # 5 modules: Spatial/Freq/Attn/IMU/Head
│   ├── data.py            # RGB+IMU dataset pipeline
│   ├── train.py           # Training & evaluation (CLI)
│   └── metrics.py         # MLE / RMSE / R² / PSI
├── data/
│   ├── train.csv          # Sample: sample_id,x,y,z,theta
│   ├── val.csv
│   ├── images/            # <sample_id>.jpg  (640×480 RGB)
│   └── imus/              # <sample_id>.npy  (6,) → [ax,ay,az,ωx,ωy,ωz]
├── requirements.txt
└── README.md
```

***

## Quick Start

### 1. Environment

```bash
pip install -r requirements.txt
```

Requirements: Python ≥ 3.8, PyTorch ≥ 2.0.

### 2. Run with Dummy Data (no real data needed)

```bash
python -m afsaf_net.train --epochs 5 --batch_size 4
```

A synthetic dataset is generated on-the-fly. This verifies the full pipeline end-to-end.

### 3. Run with Your MC4L-IMU Data

```bash
python -m afsaf_net.train \
    --train_csv data/train.csv \
    --val_csv   data/val.csv \
    --image_dir data/images \
    --imu_dir   data/imus \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-3 \
    --weight_decay 1e-4
```

Expected output:

```
Epoch 001 | 19.0s | train_loss=1.4869 mle=0.968 rmse_xyz=0.568 r2=-0.4869 | val_mle=0.856 val_r2=-0.5331
Epoch 002 | 18.4s | train_loss=1.2033 mle=0.799 rmse_xyz=0.512 r2=-0.2033 | val_mle=0.868 val_r2=-0.6841
...
```

### 4. Model Inference

```python
from afsaf_net import build_afsaf_net
import torch

model = build_afsaf_net()
model.eval()

img = torch.randn(1, 3, 480, 640)   # RGB (B, C, H, W)
imu = torch.randn(1, 6)              # IMU [ax, ay, az, ωx, ωy, ωz]

with torch.no_grad():
    pred = model(img, imu)            # (1, 4) → [x, y, z, θ]
```

### 5. PSI Stability Metric

```python
from afsaf_net import compute_psi

# Compare test error distribution against baseline
psi_value = compute_psi(expected=baseline_errors, actual=afsafnet_errors, bins=10)
print(f"PSI = {psi_value:.4f}")   # 0 = identical; larger = more shift
```

***

## Results (from Paper)

### Quantitative Comparison

| Method            | MLE (m) | RMSE\_xyz (m) | R² |
| ----------------- | ------- | ------------- | -- |
| LRA               | —       | —             | —  |
| LBP-CNN           | —       | —             | —  |
| LBP-FFT-CNN       | —       | —             | —  |
| Visual-IMU Fusion | —       | —             | —  |
| **AFSAF-Net**     | —       | —             | —  |

### Ablation Study

| Configuration           | Freq Branch | Attn Fusion | IMU Gate | MLE (m) |
| ----------------------- | :---------: | :---------: | :------: | ------- |
| Baseline (Spatial only) |      —      |      —      |     —    | —       |
| + Frequency Branch      |      ✅      |      —      |     —    | —       |
| + Attention Fusion      |      ✅      |      ✅      |     —    | —       |
| + IMU Reliability Gate  |      ✅      |      ✅      |     ✅    | —       |

### Complexity

| Method            | Params     | FLOPs      | Inference   |
| ----------------- | ---------- | ---------- | ----------- |
| LBP-CNN           | 1.82 M     | 1.35 G     | 18.6 ms     |
| LBP-FFT-CNN       | 2.14 M     | 1.72 G     | 23.4 ms     |
| Visual-IMU Fusion | 2.56 M     | 2.05 G     | 27.8 ms     |
| **AFSAF-Net**     | **3.08 M** | **2.41 G** | **31.6 ms** |

***

## About the Dataset

**The** **`data/`** **folder contains only a tiny synthetic demo subset (16 samples).**

### Why not the full MC4L-IMU dataset?

The complete MC4L-IMU dataset used in the paper contains **2,880 synchronized RGB+IMU samples** collected across **4 indoor scenes** with **48 measurement positions** (60 observations per position). At full resolution (640×480 RGB + 6-channel IMU per sample), the dataset totals approximately **2 GB** — well beyond GitHub's 100 MB single-file limit and 1 GB soft repository cap.

**What is included here:**

- 16 synthetic samples (4 positions × 4 observations) with procedurally generated "indoor scene" RGB images and plausible 6-axis IMU readings

- Matching `train.csv` / `val.csv` with ground-truth `(x, y, z, θ)`

- A full `AFSAFDataset` pipeline that loads identically-formatted real data with zero code changes

**To use real data:**

1. Clone this repository structure and place real MC4L-IMU images under `data/images/` and `.npy` IMU arrays under `data/imus/`
2. Generate `train.csv` and `val.csv` using the paper's 80:20 position-wise split (§4.1, Table 1)
3. Point `--train_csv` and `--val_csv` at your real CSV files

***

## Citation

If you find this work useful, please cite:

```bibtex
@article{yin2026afsaf,
  title   = {Adaptive Frequency-Spatial Attention Fusion Network with IMU
             Guidance for Robust Indoor Localization under Low-Light and
             Texture-Degraded Environments},
  author  = {Yin, N. and Sun, Y. and Zou, Z. and Zhang, H. and Yang, W.
             and Yang, L. and Kim, J. S.},
  year    = {2026}
}
```

***

## &#x20;License

The core AFSAF-Net code is released under the **MIT License**. The complete dataset and trained weights may be made available upon request to the corresponding author.

***

##  Acknowledgements

This research was funded by the **2026 Guangdong Provincial Colleges and Universities Characteristic Innovation Project** and supported by the Canada Advanced Institute of Science and Technology (CAIST) research platform.
