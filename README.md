# 🎬 Action Recognition in Videos

A deep learning system that classifies human actions in short video clips using a pretrained **3D CNN (R3D-18)** with transfer learning on a **5-class subset of UCF101**.

**Domain:** Computer Vision · **Framework:** PyTorch · **Dataset:** UCF101 (5 classes)

---

## 🎯 Action Classes

| Class | Description |
|-------|-------------|
| JumpingJack | Full-body jumping exercise |
| PushUps | Upper-body strength exercise |
| HorseRiding | Equestrian activity |
| Swimming | Aquatic locomotion |
| WalkingWithDog | Human-animal interaction |

---

## 📁 Project Structure

```
action-recognition-project/
│
├── Action_Recognition_5Class.ipynb   # Colab notebook (full pipeline)
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── class_mapping.json                # Class-to-index mapping
├── kaggle.json                       # (you add this — not committed)
│
├── src/
│   ├── config.py                     # All hyperparameters & paths
│   ├── dataset.py                    # VideoDataset + dataloaders
│   ├── model.py                      # R3D-18 architecture
│   ├── train.py                      # Training pipeline
│   ├── evaluate.py                   # Metrics, confusion matrix, benchmark
│   ├── predict.py                    # Single video inference
│   └── prepare_dataset.py            # Dataset download & preparation
│
├── data/                             # Dataset (not committed)
│   ├── UCF-101/                      # Raw UCF101 (after extraction)
│   └── filtered_dataset/             # 5-class subset (auto-created)
│
├── model/
│   ├── best_action_model.pth         # Best checkpoint (after training)
│   └── action_recognition_final.pth  # Final weights (after training)
│
├── sample_results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── prediction_visualization.png
│
└── demo/
    └── https://youtu.be/7CvP-XhDTnk?si=kERmA91vj7pjCVrwdemo_video_link          # YouTube demo link
```

---

## 🚀 Quick Start (3 steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download & prepare dataset

**Option A — Kaggle (recommended):**
1. Get your `kaggle.json` from [kaggle.com](https://www.kaggle.com) → Settings → API → Create New Token
2. Place `kaggle.json` in the project root folder
3. Run:
```bash
python src/prepare_dataset.py --method kaggle
```

**Option B — Manual:**
1. Download `UCF101.rar` from https://www.crcv.ucf.edu/data/UCF101.php
2. Extract it into `data/` so you have `data/UCF-101/`
3. Run:
```bash
python src/prepare_dataset.py --method manual
```

### Step 3 — Train, evaluate, predict
```bash
# Train the model (saves best checkpoint to model/)
python src/train.py

# Evaluate on test set (metrics + confusion matrix + visualization)
python src/evaluate.py

# Run inference on any video
python src/predict.py --video path/to/your_video.avi
```

---

## 💻 Google Colab Usage

1. Open `Action_Recognition_5Class.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells top to bottom

---

## 🧠 Model Architecture

- **Backbone:** R3D-18 (3D ResNet-18) pretrained on Kinetics-400
- **Reference:** Tran et al., 2018. *A Closer Look at Spatiotemporal Convolutions for Action Recognition.* CVPR.
- **Input:** 16-frame clips at 112×112
- **Head:** Dropout(0.3) + Linear(512 → 5)

---

## 🔄 Training Details

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| LR scheduler | StepLR (step=5, γ=0.1) |
| Epochs | 10 |
| Batch size | 4 |
| Split | 70% train / 15% val / 15% test (stratified) |
| Seed | 42 |

**Augmentation (train only):**
- RandomResizedCrop (scale 0.7–1.0)
- RandomHorizontalFlip (p=0.5)
- Temporal jitter (±4 frames)

**Normalization:** Kinetics-400 mean/std
- Mean: `[0.43216, 0.394666, 0.37645]`
- Std: `[0.22803, 0.221459, 0.216321]`

---

## ♻️ Reproducibility

Fixed seed (42) across all libraries. Run with:
```bash
python src/train.py  # seed is set automatically
```

**Environment:** Python 3.10 · PyTorch 2.x · CUDA 11.8 · Colab T4 GPU

---

## 📦 Dataset

**UCF101** — 101 human action classes from videos in the wild.
- **Citation:** Soomro, K., Zamir, A. R., & Shah, M. (2012). UCF101: A dataset of 101 human actions classes from videos in the wild. arXiv:1212.0402.
- **Used:** 5-class subset
- **Download:** https://www.crcv.ucf.edu/data/UCF101.php

---

## 🎥 Demo

[Watch the demo video](demo/https://youtu.be/7CvP-XhDTnk?si=kERmA91vj7pjCVrw)
