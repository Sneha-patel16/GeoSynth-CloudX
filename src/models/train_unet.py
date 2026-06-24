import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import GeoSynthDataset
from src.models.geosynth_unet import GeoSynthUNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PATH = "data/dataset/train"
VAL_PATH = "data/dataset/val"

CHECKPOINT_DIR = "outputs/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train():
    train_dataset = GeoSynthDataset(TRAIN_PATH)
    val_dataset = GeoSynthDataset(VAL_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    model = GeoSynthUNet(in_channels=16, out_channels=13).to(DEVICE)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 3

    print("Device:", DEVICE)
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                pred = model(x)
                loss = criterion(pred, y)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"geosynth_unet_epoch_{epoch+1}.pth"
        )

        torch.save(model.state_dict(), checkpoint_path)
        print("Saved:", checkpoint_path)


if __name__ == "__main__":
    train()