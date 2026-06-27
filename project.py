# -*- coding: utf-8 -*-


# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset from Drive to fast local storage
import os

if not os.path.exists('/content/data'):
    print('Copying dataset to local storage...')
    !cp -r '/content/drive/MyDrive/Satellite_Dataset/data' /content/data
    print('Done copying data.')
else:
    print('Local data already exists, skipping copy.')

if not os.path.exists('/content/test_data'):
    print('Copying test set...')
    !cp -r '/content/drive/MyDrive/Satellite_Dataset/public_test_data' /content/test_data
    print('Done copying test data.')
else:
    print('Local test data already exists, skipping copy.')

DATA_FOLDER = '/content/data'
TEST_FOLDER = '/content/test_data'
print(f'DATA_FOLDER = {DATA_FOLDER}')
print(f'TEST_FOLDER = {TEST_FOLDER}')

# Install any missing packages
!pip install -q shiny

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

#  Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#  Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

#  Paths
DATA_FOLDER = '/content/drive/MyDrive/Satellite_Dataset/data'
TEST_FOLDER = '/content/drive/MyDrive/Satellite_Dataset/public_test_data'

os.makedirs('assets/plots', exist_ok=True)
os.makedirs('assets/weights', exist_ok=True)

"""---
## 1. Data Handling and Pre-processing
"""

def preprocess(data_folder: str) -> tuple[pd.DataFrame, dict]:

    # Collect all class directories, sorted for reproducibility
    classes = sorted(
        d for d in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, d))
    )
    label_dict = {cls: idx for idx, cls in enumerate(classes)}

    records = []
    for cls in classes:
        cls_path = os.path.join(data_folder, cls)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                records.append({
                    'folder':    cls,
                    'file_name': fname,
                    'label':     label_dict[cls]
                })

    df = pd.DataFrame(records)
    return df, label_dict


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / validation split.

    Returns
    -------
    train_df, val_df : pd.DataFrame
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

# Run preprocessing
df, label_dict = preprocess(DATA_FOLDER)
train_df, val_df = split_data(df, test_size=0.2)

# Inverse mapping: numeric label -> class name (useful later)
idx_to_class = {v: k for k, v in label_dict.items()}

print(f'Total images : {len(df)}')
print(f'Training set : {len(train_df)}')
print(f'Validation set: {len(val_df)}')
print()
print('Label dictionary:')
print(label_dict)
print()
print('Sample rows:')
df.head(10)

# Verify stratification: class distribution should be ~equal in train and val
train_dist = train_df['folder'].value_counts(normalize=True).sort_index()
val_dist   = val_df['folder'].value_counts(normalize=True).sort_index()

dist_check = pd.DataFrame({'train_%': train_dist, 'val_%': val_dist})
print('Class distribution (proportions):')
print(dist_check.round(4))

"""---
## 2. Exploratory Data Analysis (EDA)
"""

from concurrent.futures import ThreadPoolExecutor

def _load_image_array(data_folder: str, row: pd.Series) -> np.ndarray:
    """Load one image as a uint8 numpy array (H, W, 3)."""
    path = os.path.join(data_folder, row['folder'], row['file_name'])
    return np.array(Image.open(path).convert('RGB'))


def _load_images_parallel(data_folder: str, df_subset: pd.DataFrame, max_workers: int = 8) -> np.ndarray:
    """
    Load a batch of images in parallel using threads.
    Returns array of shape (N, H, W, 3) as float32.
    """
    rows = [row for _, row in df_subset.iterrows()]

    def _load(row):
        path = os.path.join(data_folder, row['folder'], row['file_name'])
        return np.array(Image.open(path).convert('RGB'), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        imgs = list(ex.map(_load, rows))
    return np.stack(imgs)   # (N, 64, 64, 3)


def show_samples(df: pd.DataFrame, num_samples: int = 5) -> None:

    sample = df.sample(n=num_samples, random_state=SEED).reset_index(drop=True)

    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3.5))
    if num_samples == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, sample.iterrows()):
        img = _load_image_array(DATA_FOLDER, row)
        ax.imshow(img)
        ax.set_title(row['folder'], fontsize=9, wrap=True)
        ax.axis('off')

    fig.suptitle('Random Sample Images', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('assets/plots/random_samples.png', bbox_inches='tight', dpi=150)
    plt.show()
    print('Saved → assets/plots/random_samples.png')

show_samples(train_df, num_samples=5)

def average_pixel_plot(df: pd.DataFrame, n_sample: int = 1000) -> None:

    sample = df.sample(n=min(n_sample, len(df)), random_state=SEED)

    print(f'Loading {len(sample)} images in parallel...')
    arr = _load_images_parallel(DATA_FOLDER, sample)  # (N, 64, 64, 3)

    # Vectorized means: average over H and W axes for each channel
    r_means = arr[:, :, :, 0].mean(axis=(1, 2))  # (N,)
    g_means = arr[:, :, :, 1].mean(axis=(1, 2))
    b_means = arr[:, :, :, 2].mean(axis=(1, 2))

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, data, color in [('Red', r_means, 'red'), ('Green', g_means, 'green'), ('Blue', b_means, 'royalblue')]:
        ax.hist(data, bins=60, alpha=0.55, color=color, label=label, edgecolor='none')

    ax.set_xlabel('Average Pixel Value (0–255)', fontsize=11)
    ax.set_ylabel('Number of Images', fontsize=11)
    ax.set_title(f'Distribution of Average Pixel Values per Channel (n={len(sample)})', fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('assets/plots/average_pixel_distribution.png', dpi=150)
    plt.show()
    print('Saved → assets/plots/average_pixel_distribution.png')

average_pixel_plot(train_df)

def average_brightness_per_class(df: pd.DataFrame, n_per_class: int = 100) -> None:

    sampled = (
        df.groupby('folder', group_keys=False)
          .apply(lambda g: g.sample(n=min(n_per_class, len(g)), random_state=SEED))
          .reset_index(drop=True)
    )

    print(f'Loading {len(sampled)} images in parallel...')
    arr = _load_images_parallel(DATA_FOLDER, sampled)  # (N, 64, 64, 3)

    # Vectorized brightness: mean over H, W, and all 3 channels
    brightness = arr.mean(axis=(1, 2, 3))  # (N,)

    brightness_df = pd.DataFrame({
        'class':      sampled['folder'].values,
        'brightness': brightness
    })

    order = (
        brightness_df.groupby('class')['brightness']
        .median().sort_values().index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=brightness_df, x='class', y='brightness',
                order=order, palette='Set2', ax=ax)
    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('Average Brightness (0–255)', fontsize=11)
    ax.set_title(f'Average Brightness per Class (n={n_per_class} per class)', fontsize=12)
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig('assets/plots/average_brightness.png', dpi=150)
    plt.show()
    print('Saved → assets/plots/average_brightness.png')

average_brightness_per_class(train_df)

"""---
## 3. CNN Implementation and Training
"""

# ── Normalization constants (ImageNet stats work well for RGB satellite imagery)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# Same as val_transform — used for Shiny app too
test_transform = val_transform

class SatelliteDataset(Dataset):
    """PyTorch Dataset for labelled satellite images."""

    def __init__(self, df: pd.DataFrame, data_folder: str, transform=None):
        self.df          = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.transform   = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = os.path.join(self.data_folder, row['folder'], row['file_name'])
        img  = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(row['label'])


# ── DataLoaders ──────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
NUM_WORKERS = 2

train_dataset = SatelliteDataset(train_df, DATA_FOLDER, transform=train_transform)
val_dataset   = SatelliteDataset(val_df,   DATA_FOLDER, transform=val_transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

print(f'Train batches : {len(train_loader)}')
print(f'Val batches   : {len(val_loader)}')

class SatelliteCNN(nn.Module):


    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()

        def conv_block(in_ch, out_ch, drop=dropout):
            return nn.Sequential(
                nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(drop),
            )

        self.features = nn.Sequential(
            conv_block(3,    32),   # 64 → 32
            conv_block(32,   64),   # 32 → 16
            conv_block(64,  128),   # 16 →  8
            conv_block(128, 256),   #  8 →  4
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # 4 → 1  (robust to size changes)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 2),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.classifier(x)


model = SatelliteCNN(num_classes=10).to(DEVICE)

# Quick sanity check
dummy = torch.zeros(1, 3, 64, 64).to(DEVICE)
out   = model(dummy)
print(f'Output shape: {out.shape}')   # should be (1, 10)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {total_params:,}')

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    save_path: str = 'assets/weights/best_model.pth'
):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc  = 0.0
    epochs_no_imp = 0   # counter for early stopping

    for epoch in range(1, num_epochs + 1):

        # ── Training phase ──────────────────────────────────────────────────
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ── Validation phase ────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        correct     = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item() * imgs.size(0)
                correct  += (logits.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc   = correct / len(val_loader.dataset)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # ── Save best model ─────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            epochs_no_imp = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_imp += 1

        print(
            f'Epoch {epoch:3d}/{num_epochs} | '
            f'Train loss: {train_loss:.4f} | '
            f'Val loss: {val_loss:.4f} | '
            f'Val acc: {val_acc:.4f}'
            + (' ✓ best' if epochs_no_imp == 0 else '')
        )

        # ── Early stopping ───────────────────────────────────────────────────
        if epochs_no_imp >= patience:
            print(f'\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).')
            break

    print(f'\nBest val accuracy: {best_val_acc:.4f}')
    print(f'Model saved to: {save_path}')
    return history

MODEL_PATH = 'assets/weights/best_model.pth'

history = train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    lr=1e-3,
    patience=10,
    save_path=MODEL_PATH
)

# ── Auto-save model to Drive immediately after training ───────────────────────
import shutil
shutil.copy(MODEL_PATH, '/content/drive/MyDrive/best_model.pth')
print('Model saved to Google Drive ✓')
print('Safe to disconnect now — reload from Drive next session.')

# Load best checkpoint for evaluation
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
_ = model.load_state_dict(state_dict)
_ = model.eval()
print('Best model loaded.')

"""---
## 4. Model Evaluation and Analysis
"""

def plot_training_curves(history: dict) -> None:
    """
    Plot training & validation loss (left) and validation accuracy (right)
    side-by-side. Saves to assets/plots/training_curves.png.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, history['train_loss'], label='Train loss')
    ax1.plot(epochs, history['val_loss'],   label='Val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], color='green', label='Val accuracy')
    ax2.axhline(max(history['val_acc']) * 100, color='green', linestyle='--',
                alpha=0.4, label=f'Best: {max(history["val_acc"]) * 100:.2f}%')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('assets/plots/training_curves.png', dpi=150)
    plt.show()
    print('Saved → assets/plots/training_curves.png')


plot_training_curves(history)

def get_val_predictions(model, val_loader):
    """
    Run the best model over the full validation set.
    Returns (all_preds, all_labels) as numpy arrays.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


val_preds, val_labels = get_val_predictions(model, val_loader)
val_acc = (val_preds == val_labels).mean()
print(f'Validation accuracy: {val_acc * 100:.2f}%')

from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, class_names: list) -> None:
    """
    Plot normalised confusion matrix.
    Saves to assets/plots/confusion_matrix.png.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9)

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_title('Confusion Matrix (normalised)', fontsize=12)
    plt.tight_layout()
    plt.savefig('assets/plots/confusion_matrix.png', dpi=150)
    plt.show()
    print('Saved → assets/plots/confusion_matrix.png')


class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
plot_confusion_matrix(val_labels, val_preds, class_names)

print('\nPer-class report:')
print(classification_report(val_labels, val_preds, target_names=class_names))

def plot_misclassified(model, val_df, num_samples: int = 5) -> None:
    """
    Show `num_samples` misclassified images with true and predicted labels.
    Saves to assets/plots/misclassified.png.
    """
    model.eval()
    misclassified = []

    with torch.no_grad():
        for idx, row in val_df.iterrows():
            if len(misclassified) >= num_samples:
                break
            path = os.path.join(DATA_FOLDER, row['folder'], row['file_name'])
            img_pil = Image.open(path).convert('RGB')
            tensor  = val_transform(img_pil).unsqueeze(0).to(DEVICE)
            pred    = model(tensor).argmax(1).item()
            true    = int(row['label'])
            if pred != true:
                misclassified.append({
                    'img':  img_pil,
                    'true': idx_to_class[true],
                    'pred': idx_to_class[pred],
                })

    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3.8))
    for ax, sample in zip(axes, misclassified):
        ax.imshow(sample['img'])
        ax.set_title(
            f"True: {sample['true']}\nPred: {sample['pred']}",
            fontsize=8, color='red'
        )
        ax.axis('off')

    fig.suptitle('Misclassified Samples', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('assets/plots/misclassified.png', bbox_inches='tight', dpi=150)
    plt.show()
    print('Saved → assets/plots/misclassified.png')


plot_misclassified(model, val_df)

def preprocess_test(data_folder: str) -> pd.DataFrame:
    """
    Build a DataFrame for the test set (flat folder, no class subfolders).

    Returns
    -------
    df : pd.DataFrame with columns [folder, file_name]
    """
    files = [
        f for f in os.listdir(data_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    df = pd.DataFrame({
        'folder':    os.path.basename(data_folder),
        'file_name': files
    })
    return df


class TestDataset(Dataset):
    """Dataset for unlabelled test images (flat folder)."""

    def __init__(self, df: pd.DataFrame, data_folder: str, transform=None):
        self.df          = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.transform   = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = os.path.join(self.data_folder, row['file_name'])
        img  = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, row['file_name']   # return filename so we can build submission


test_df = preprocess_test(TEST_FOLDER)
print(f'Test images: {len(test_df)}')
test_df.head()

test_dataset = TestDataset(test_df, TEST_FOLDER, transform=test_transform)
test_loader  = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

# Run inference
model.eval()
filenames, predictions = [], []

with torch.no_grad():
    for imgs, fnames in test_loader:
        imgs  = imgs.to(DEVICE)
        preds = model(imgs).argmax(1).cpu().numpy()
        filenames.extend(fnames)
        predictions.extend([idx_to_class[p] for p in preds])

# Build and save submission (no header, as required)
submission = pd.DataFrame({'file_name': filenames, 'label': predictions})
submission.to_csv('submission.csv', index=False, header=False)

print(f'submission.csv saved — {len(submission)} rows')
print('Preview:')
print(submission.head(7).to_string(index=False))

# Save submission.csv to Drive so you don't lose it
import shutil
shutil.copy('submission.csv', '/content/drive/MyDrive/Satellite_Dataset/submission.csv')
print('submission.csv saved to Drive ✓')

"""---
## 5. Shiny Web Application
The app lives in the `app/` directory. Run the cell below to copy the model weights there, then launch with `shiny run app/app.py`.
"""

!mkdir -p app

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app/app.py
# 
# 
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# import pandas as pd
# import numpy as np
# from pathlib import Path
# 
# from shiny.express import input, render, ui
# from shiny import reactive
# 
# # ── Constants ─────────────────────────────────────────────────────────────────
# DEVICE = torch.device("cpu")
# 
# # Class names in sorted order (must match training label_dict)
# CLASS_NAMES = [
#     "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
#     "Industrial", "Pasture", "PermanentCrop", "Residential",
#     "River", "SeaLake",
# ]
# 
# MODEL_PATH = Path(__file__).parent / "best_model.pth"
# 
# 
# # ── Model (must match training architecture exactly) ──────────────────────────
# class SatelliteCNN(nn.Module):
#     def __init__(self, num_classes: int = 10, dropout: float = 0.25):
#         super().__init__()
# 
#         def conv_block(in_ch, out_ch, drop=dropout):
#             return nn.Sequential(
#                 nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
#                 nn.BatchNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#                 nn.BatchNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(2),
#                 nn.Dropout2d(drop),
#             )
# 
#         self.features = nn.Sequential(
#             conv_block(3,    32),
#             conv_block(32,   64),
#             conv_block(64,  128),
#             conv_block(128, 256),
#         )
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout * 2),
#             nn.Linear(256, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(256, num_classes),
#         )
# 
#     def forward(self, x):
#         x = self.features(x)
#         x = self.gap(x)
#         x = x.flatten(1)
#         return self.classifier(x)
# 
# 
# # ── Load model ────────────────────────────────────────────────────────────────
# model = SatelliteCNN(num_classes=len(CLASS_NAMES))
# state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
# _ = model.load_state_dict(state_dict)
# _ = model.eval()
# 
# # ── Transforms (identical to val_transform used during training) ──────────────
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),   # safety resize
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])
# 
# 
# def predict(img_pil: Image.Image):
#     """Return (class_name, confidence_float, probs_array)."""
#     tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         logits = model(tensor)
#     probs   = torch.softmax(logits, dim=1).squeeze().numpy()
#     pred_idx = int(probs.argmax())
#     return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs
# 
# 
# # ── Page layout ───────────────────────────────────────────────────────────────
# ui.page_opts(title="Satellite Image Classifier", fillable=True)
# 
# with ui.sidebar(width=270):
#     ui.h4("Upload Image")
#     ui.input_file(
#         "image_file",
#         "Choose a satellite image (.jpg / .png):",
#         accept=[".jpg", ".jpeg", ".png"],
#         multiple=False,
#     )
#     ui.hr()
#     ui.markdown("""
# **How to use:**
# 1. Upload a 64 × 64 satellite image
# 2. The CNN model predicts the terrain type
# 3. Confidence score and per-class probabilities are shown on the right
# """)
# 
# # ── Top row: image + prediction ───────────────────────────────────────────────
# with ui.layout_columns(col_widths=[4, 8]):
# 
#     with ui.card(full_screen=False):
#         ui.card_header("Uploaded Image")
# 
#         @render.image
#         def uploaded_image():
#             f = input.image_file()
#             if not f:
#                 return None
#             return {"src": f[0]["datapath"], "width": "100%", "alt": "Uploaded satellite image"}
# 
#     with ui.card():
#         ui.card_header("Prediction Result")
# 
#         @render.ui
#         def prediction_result():
#             f = input.image_file()
#             if not f:
#                 return ui.p("Upload an image to see the prediction.", style="color: #888;")
# 
#             img          = Image.open(f[0]["datapath"]).convert("RGB")
#             cls, conf, _ = predict(img)
#             conf_pct     = conf * 100
#             color        = "#28a745" if conf_pct >= 75 else "#fd7e14" if conf_pct >= 50 else "#dc3545"
# 
#             return ui.div(
#                 ui.div(
#                     ui.h2(cls, style=f"margin: 0; color: {color};"),
#                     style="padding: 12px 0 4px 0;"
#                 ),
#                 ui.div(
#                     ui.h5(
#                         f"Confidence: {conf_pct:.1f}%",
#                         style=f"margin: 0; color: {color};"
#                     ),
#                     style="padding-bottom: 8px;"
#                 ),
#                 ui.p(
#                     "The model classified this satellite image as ",
#                     ui.strong(cls),
#                     f" with {conf_pct:.1f}% confidence.",
#                     style="color: #555; margin-top: 8px;"
#                 ),
#             )
# 
# # ── Bottom row: probability table ─────────────────────────────────────────────
# with ui.card():
#     ui.card_header("Class Probabilities")
# 
#     @render.data_frame
#     def prob_table():
#         f = input.image_file()
# 
#         if not f:
#             df = pd.DataFrame({
#                 "Rank":             range(1, len(CLASS_NAMES) + 1),
#                 "Class":            CLASS_NAMES,
#                 "Probability (%)":  ["—"] * len(CLASS_NAMES),
#             })
#             return df
# 
#         img           = Image.open(f[0]["datapath"]).convert("RGB")
#         _, _, probs   = predict(img)
# 
#         df = pd.DataFrame({
#             "Class":           CLASS_NAMES,
#             "Probability (%)": [round(float(p) * 100, 2) for p in probs],
#         }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
#         df.insert(0, "Rank", range(1, len(df) + 1))
# 
#         return df
#

import os, shutil

os.makedirs('app', exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print('ERROR: Model weights not found.')
else:
    shutil.copy(MODEL_PATH, 'app/best_model.pth')
    print('app/app.py         ✓')
    print('app/best_model.pth ✓')
    print('Files in app/:', os.listdir('app'))