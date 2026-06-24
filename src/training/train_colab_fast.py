
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset_loader import GeoSynthDataset
from src.models.geosynth_unet import GeoSynthUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PATH = "/content/drive/MyDrive/GeoSynth-CloudX/data/dataset/train"
VAL_PATH = "/content/drive/MyDrive/GeoSynth-CloudX/data/dataset/val"

BATCH_SIZE = 16
EPOCHS = 10

def combined_loss(pred, target):
    l1 = nn.functional.l1_loss(pred, target)
    mse = nn.functional.mse_loss(pred, target)
    return 0.7*l1 + 0.3*mse

train_dataset = GeoSynthDataset(TRAIN_PATH)
val_dataset = GeoSynthDataset(VAL_PATH)

train_subset = train_dataset
val_subset = val_dataset

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

model = GeoSynthUNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Device:", DEVICE)
print("Train Samples:", len(train_subset))
print("Val Samples:", len(val_subset))

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for x,y,_ in tqdm(train_loader):

        x=x.to(DEVICE)
        y=y.to(DEVICE)

        optimizer.zero_grad()

        pred=model(x)

        loss=combined_loss(pred,y)

        loss.backward()
        optimizer.step()

        train_loss+=loss.item()

    model.eval()

    val_loss=0

    with torch.no_grad():

        for x,y,_ in val_loader:

            x=x.to(DEVICE)
            y=y.to(DEVICE)

            pred=model(x)

            loss=combined_loss(pred,y)

            val_loss+=loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss {train_loss/len(train_loader):.4f} | "
        f"Val Loss {val_loss/len(val_loader):.4f}"
    )

torch.save(
    model.state_dict(),
    "/content/drive/MyDrive/GeoSynth-CloudX/outputs/checkpoints/geosynth_colab_fast.pth"
)

print("Model Saved")