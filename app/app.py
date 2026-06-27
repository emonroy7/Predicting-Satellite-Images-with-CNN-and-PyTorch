# Name: Ehtesham, Student ID: k10

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path

from shiny.express import input, render, ui
from shiny import reactive

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")

# Class names in sorted order (must match training label_dict)
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

MODEL_PATH = Path(__file__).parent / "best_model.pth"


# ── Model (must match training architecture exactly) ──────────────────────────
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
            conv_block(3,    32),
            conv_block(32,   64),
            conv_block(64,  128),
            conv_block(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
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


# ── Load model ────────────────────────────────────────────────────────────────
model = SatelliteCNN(num_classes=len(CLASS_NAMES))
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
_ = model.load_state_dict(state_dict)
_ = model.eval()

# ── Transforms (identical to val_transform used during training) ──────────────
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # safety resize
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(img_pil: Image.Image):
    """Return (class_name, confidence_float, probs_array)."""
    tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
    probs   = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx = int(probs.argmax())
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs


# ── Page layout ───────────────────────────────────────────────────────────────
ui.page_opts(title="Satellite Image Classifier", fillable=True)

with ui.sidebar(width=270):
    ui.h4("Upload Image")
    ui.input_file(
        "image_file",
        "Choose a satellite image (.jpg / .png):",
        accept=[".jpg", ".jpeg", ".png"],
        multiple=False,
    )
    ui.hr()
    ui.markdown("""
**How to use:**
1. Upload a 64 × 64 satellite image
2. The CNN model predicts the terrain type
3. Confidence score and per-class probabilities are shown on the right
""")

# ── Top row: image + prediction ───────────────────────────────────────────────
with ui.layout_columns(col_widths=[4, 8]):

    with ui.card(full_screen=False):
        ui.card_header("Uploaded Image")

        @render.image
        def uploaded_image():
            f = input.image_file()
            if not f:
                return None
            return {"src": f[0]["datapath"], "width": "100%", "alt": "Uploaded satellite image"}

    with ui.card():
        ui.card_header("Prediction Result")

        @render.ui
        def prediction_result():
            f = input.image_file()
            if not f:
                return ui.p("Upload an image to see the prediction.", style="color: #888;")

            img          = Image.open(f[0]["datapath"]).convert("RGB")
            cls, conf, _ = predict(img)
            conf_pct     = conf * 100
            color        = "#28a745" if conf_pct >= 75 else "#fd7e14" if conf_pct >= 50 else "#dc3545"

            return ui.div(
                ui.div(
                    ui.h2(cls, style=f"margin: 0; color: {color};"),
                    style="padding: 12px 0 4px 0;"
                ),
                ui.div(
                    ui.h5(
                        f"Confidence: {conf_pct:.1f}%",
                        style=f"margin: 0; color: {color};"
                    ),
                    style="padding-bottom: 8px;"
                ),
                ui.p(
                    "The model classified this satellite image as ",
                    ui.strong(cls),
                    f" with {conf_pct:.1f}% confidence.",
                    style="color: #555; margin-top: 8px;"
                ),
            )

# ── Bottom row: probability table ─────────────────────────────────────────────
with ui.card():
    ui.card_header("Class Probabilities")

    @render.data_frame
    def prob_table():
        f = input.image_file()

        if not f:
            df = pd.DataFrame({
                "Rank":             range(1, len(CLASS_NAMES) + 1),
                "Class":            CLASS_NAMES,
                "Probability (%)":  ["—"] * len(CLASS_NAMES),
            })
            return df

        img           = Image.open(f[0]["datapath"]).convert("RGB")
        _, _, probs   = predict(img)

        df = pd.DataFrame({
            "Class":           CLASS_NAMES,
            "Probability (%)": [round(float(p) * 100, 2) for p in probs],
        }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        return df
