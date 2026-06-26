# Satellite Image Classification with CNN and PyTorch

A deep learning pipeline that classifies satellite images into 10 terrain categories using a custom Convolutional Neural Network built with PyTorch. Includes a Shiny web application for interactive inference.

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | **92.60%** |
| Training Epochs | 50 |
| Best Epoch | 47 |
| Model Parameters | 1,241,578 |
| Dataset Size | 10,000 images |

## Dataset

[EuroSAT RGB](https://github.com/phelber/EuroSAT) — 10,000 satellite images (64×64 px, RGB) across 10 terrain classes:

`AnnualCrop` · `Forest` · `HerbaceousVegetation` · `Highway` · `Industrial` · `Pasture` · `PermanentCrop` · `Residential` · `River` · `SeaLake`

Split: **80% training (8,000)** / **20% validation (2,000)**, stratified so each class is equally represented in both sets.

## Model Architecture

Custom CNN — no pretrained weights or transfer learning.

```
Input (3 × 64 × 64)
  → ConvBlock(3 → 32)    # 64 → 32
  → ConvBlock(32 → 64)   # 32 → 16
  → ConvBlock(64 → 128)  # 16 → 8
  → ConvBlock(128 → 256) #  8 → 4
  → GlobalAveragePool    #  4 → 1
  → Linear(256 → 256) + ReLU + Dropout
  → Linear(256 → 10)
```

Each ConvBlock: `Conv2d → BN → ReLU → Conv2d → BN → ReLU → MaxPool2d → Dropout2d`

**Training setup:**
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Loss: CrossEntropyLoss
- Early stopping: patience=10
- Data augmentation: random horizontal/vertical flip, rotation ±15°, color jitter

## Training Progress

| Epoch | Train Loss | Val Loss | Val Acc |
|---|---|---|---|
| 10 | 0.8585 | 0.6195 | 78.65% |
| 20 | 0.6657 | 0.4617 | 84.35% |
| 30 | 0.5175 | 0.3216 | 89.55% |
| 40 | 0.4219 | 0.2387 | 91.95% |
| 47 | 0.3846 | **0.2196** | **92.60%** ✓ best |

## Project Structure

```
├── k10.ipynb               # Main notebook (all 5 sections)
├── app/
│   ├── app.py              # Shiny web application
│   └── best_model.pth      # Trained model weights
├── assets/
│   ├── plots/
│   │   ├── random_samples.png
│   │   ├── average_pixel_distribution.png
│   │   ├── average_brightness.png
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   └── misclassified.png
│   └── weights/
│       └── best_model.pth
└── submission.csv          # Test set predictions for challenge server
```

## How to Run

### Notebook (Google Colab)
1. Upload `k10.ipynb` to Google Colab
2. Place dataset on Google Drive at `MyDrive/Satellite_Dataset/data`
3. Run all cells top to bottom

### Shiny Web App (locally)
```bash
pip install shiny torch torchvision pillow pandas numpy
cd app
python -m shiny run app.py
```
Then open `http://127.0.0.1:8000` in your browser.

The app lets you upload any satellite image and shows the predicted terrain class, confidence score, and a full probability table for all 10 classes.

## Requirements

```
torch
torchvision
pandas
numpy
matplotlib
seaborn
Pillow
scikit-learn
shiny
```

## Author

Ehtesham — Student ID: k10
