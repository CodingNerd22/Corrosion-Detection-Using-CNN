# Corrosion Detection in Ship Hulls using CNN-Based Image Segmentation

This repository provides a PyTorch-based implementation for detecting and segmenting corrosion in ship hull images using a convolutional neural network (CNN).

## 📌 Project Overview

This project involves training a simple CNN model to perform binary segmentation of corrosion areas in ship hulls from high-resolution images. It includes:

- Custom dataset loader for handling images and segmentation masks.
- CNN-based encoder-decoder model for segmentation.
- Training script with PyTorch DataLoader.
- Interactive interface for testing the trained model on new images.

The dataset used is available from [Mendeley Data](https://data.mendeley.com/datasets/ry392rp8cj/1).

---

## 🗃️ Dataset

**Dataset source:** [ry392rp8cj/1 – Mendeley Data](https://data.mendeley.com/datasets/ry392rp8cj/1)

Use the `HiRes/raw/` folder for input images and `HiRes/labeled/` for the corresponding segmentation masks. Masks should be named in the format `filename_label.jpg`.

---

## 🧠 Model Architecture

The model is a basic encoder-decoder CNN for binary segmentation, consisting of:

- 3-layer convolutional encoder
- 3-layer transposed convolutional decoder
- Sigmoid activation for output mask

---

## 🛠️ Getting Started

### Prerequisites

Make sure you have Python 3.7+ and install the required dependencies:

```bash
pip install -r requirements.txt
