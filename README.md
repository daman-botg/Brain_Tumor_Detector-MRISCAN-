# Brain Tumor Detector — MRISCAN

A Jupyter-Notebook-based project for detecting brain tumors from MRI scans using the "Brain Tumor MRI Dataset" (Kaggle). The experiments in this repo use PyTorch and achieve approximately 94% recall and 94% accuracy on the evaluated split.

- Repo: daman-botg/Brain_Tumor_Detector-MRISCAN-
- Primary language: Jupyter Notebook (100%)
- Dataset used: Brain Tumor MRI Dataset — https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data

## Table of contents

- [About](#about)
- [Dataset](#dataset)
- [Results (summary)](#results-summary)
- [Notebooks / What’s included](#notebooks--whats-included)
- [Requirements (PyTorch)](#requirements-pytorch)
- [Reproduce the experiments](#reproduce-the-experiments)
- [Quick inference (PyTorch)](#quick-inference-pytorch)
- [Notes, limitations & next steps](#notes-limitations--next-steps)
- [License & contact](#license--contact)

## About

This repository contains Jupyter notebooks implementing a PyTorch-based pipeline for brain tumor detection from MRI images. The notebooks cover data loading, preprocessing, model definition (custom CNNs and/or transfer learning with torchvision models), training loops, evaluation, and inference examples.

## Dataset

Download the Kaggle dataset:
- Brain Tumor MRI Dataset — https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data

Using the Kaggle CLI:
```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/
```
Place the extracted dataset under `data/` or update notebook paths accordingly.

## Results (summary)

- Recall: ~94%
- Accuracy: ~94%

See the evaluation cells in the main notebook for confusion matrix, per-class metrics, and experiment details (train/val/test split, augmentation, seed).

## Notebooks / What's included

- Data exploration & visualization
- Dataset & DataLoader implementation (PyTorch)
- Preprocessing & augmentations (torchvision / albumentations)
- Model architectures (CNNs, transfer learning)
- Training loop with metrics tracking (accuracy, recall)
- Evaluation (confusion matrix, classification report)
- Inference examples and visualization (Grad-CAM optional)

## Requirements (PyTorch)

Suggested environment:

- Python 3.8+
- JupyterLab or Jupyter Notebook
- PyTorch (CPU or with CUDA if you have a GPU)
- torchvision
- numpy
- pandas
- scikit-learn
- matplotlib / seaborn
- opencv-python or pillow
- albumentations (optional, for advanced augmentations)
- kaggle (optional, for dataset download)

Install with pip (CPU example):
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install jupyterlab torch torchvision numpy pandas scikit-learn matplotlib opencv-python albumentations kaggle
```

If you want GPU support, install the appropriate torch + CUDA build from https://pytorch.org.

## Reproduce the experiments

1. Clone the repo:
```bash
git clone https://github.com/daman-botg/Brain_Tumor_Detector-MRISCAN-.git
cd Brain_Tumor_Detector-MRISCAN-
```

2. Download and place the dataset in `data/`.

3. Create and activate the environment and install dependencies (see Requirements).

4. Start Jupyter and open the main notebook (e.g., `Brain_Tumor_Detection.ipynb`):
```bash
jupyter lab
# or
jupyter notebook
```

5. Run cells in order. Update the configuration cell (paths, hyperparameters, device) at the top.

6. Training: run the training cells. Example training-device selection in a notebook:
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Adjust preprocessing to match training transforms (grayscale vs RGB, normalization values, resize).

## Notes, limitations & next steps

- Metrics depend on split, augmentation, and preprocessing — validate with an independent hold-out set or cross-validation.
- Consider stronger augmentations, transfer learning (pretrained backbones), ensembling, and model explainability (Grad-CAM).
- Address class imbalance with weighted loss, oversampling, or focal loss if necessary.
- Save training logs (e.g., TensorBoard, Weights & Biases) for reproducibility.

## Contact

- Author: Guhan Venkat — https://github.com/daman-botg
