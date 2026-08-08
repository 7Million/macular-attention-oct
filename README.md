# Macular Attention OCT

Attention-enhanced Convolutional Neural Networks for macular disease classification using Optical Coherence Tomography (OCT) images (_not the actual topic_).

This project was developed as part of an **MSc Computer Science dissertation at Keele University**.

#### Project Overview

The goal of this research is to improve the accuracy and explainability of deep learning models for classifying macular diseases from OCT scans. The models classify images into four categories:

- **CNV** (Choroidal Neovascularization)
- **DME** (Diabetic Macular Edema)
- **DRUSEN**
- **NORMAL**

#### Models Implemented

- ResNet50 (Baseline)
- ConvNeXt-Tiny (Baseline)
- Attention-enhanced versions using CBAM and SE blocks

#### Repository Structure

- `src/models/` → Model architectures
- `src/train.py` → Training script
- `app/` → Gradio demo application
- `results/` → Performance metrics, plots, and heatmaps

#### Dataset

This project uses the publicly available **Kermany 2018 OCT** [dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018?select=OCT2017+).

#### Author

Developed by Tadiwanashe Mataruse.

#### License

This project is intended for educational and research purposes.