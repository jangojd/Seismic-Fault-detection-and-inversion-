# Seismic Interpretation & Inversion with Deep Learning

Reproduction of Zheng et al. (2019) paper implementing CNNs for seismic data analysis.

<div align="center">
  <img src="https://github.com/jangojd/Seismic-Fault-detection-and-inversion-/blob/43ddbf62101e7957dcd14e70573730e39ef8b281/Seismic_inversion_Fault_detection.ipynb" alt="Project Results" width="800">
  <p><em>Training results for fault detection and seismic inversion models</em></p>
</div>

## Two Case Studies

**1. Automatic Fault Detection**
- Input: 2D seismic sections (100×100)
- Output: Fault probability, dip angle, azimuth
- Uses: U-Net encoder-decoder architecture

**2. Seismic Inversion**
- Input: Prestack seismic traces (200 samples)
- Output: Velocity & density profiles (100 samples)
- Uses: 1D CNN with dual regression heads

## Installation

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib
```

## Quick Start

```bash
python seismic_ml.py
```

Or in Jupyter:
```python
from seismic_ml import *

# Generate data
gen = SeismicDataGenerator(nx=100, nz=100, nt=200)
v_model = gen.generate_velocity_model(faulted=True)

# Train fault detection
model_fault = FaultDetectionCNN()
train_fault_detection(model_fault, train_loader, val_loader, epochs=20)

# Train inversion
model_inv = SeismicInversionCNN()
train_seismic_inversion(model_inv, train_loader, val_loader, epochs=20)
```

## Features

- ✅ Synthetic seismic data generation (Ricker wavelet convolution)
- ✅ Multi-task learning for fault detection
- ✅ Regression networks for velocity/density inversion
- ✅ PyTorch implementation with GPU support
- ✅ Training curves and visualization

## Parameters

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Batch Size | 16 |
| Learning Rate | 0.001 |
| Optimizer | Adam |

## Results

- **Fault Detection Loss**: ~0.20-0.30
- **Inversion RMSE**: ~250-450 m/s

## Citation

```bibtex
@article{zheng2019applications,
  title={Applications of supervised deep learning for seismic interpretation and inversion},
  author={Zheng, York and Zhang, Qie and Yusifov, Anar and Shi, Yunzhi},
  journal={The Leading Edge},
  volume={38},
  number={7},
  pages={526--533},
  year={2019}
}
```

## References

- Original Paper: Zheng et al., The Leading Edge, 2019
- U-Net: Ronneberger et al., MICCAI 2015
