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

CHECKPOINT_DIR = "outputs/checkpoints/v2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def combined_loss(pred, target):
    l1 = nn.functional.l1_loss(pred, target)
    mse = nn.functional.mse_loss(pred, target)
    return 0.7 * l1 + 0.3 * mse


def train():
    train_dataset = GeoSynthDataset(TRAIN_PATH)
    val_dataset = GeoSynthDataset(VAL_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )

    model = GeoSynthUNet(in_channels=16, out_channels=13).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_epochs = 2
    best_val_loss = float("inf")

    print("Device:", DEVICE)
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    for epoch in range(1, total_epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            pred = model(x)
            loss = combined_loss(pred, y)

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
                loss = combined_loss(pred, y)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch}/{total_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        epoch_path = os.path.join(
            CHECKPOINT_DIR,
            f"geosynth_unet_v2_epoch_{epoch}.pth"
        )
        torch.save(model.state_dict(), epoch_path)
        print("Saved:", epoch_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "geosynth_unet_v2_best.pth")
            torch.save(model.state_dict(), best_path)
            print("Best model saved:", best_path)


if __name__ == "__main__":
    train()