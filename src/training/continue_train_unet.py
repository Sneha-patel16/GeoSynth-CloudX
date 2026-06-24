import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_loader import GeoSynthDataset
from src.models.geosynth_unet import GeoSynthUNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PATH = "data/dataset/train"
VAL_PATH = "data/dataset/val"

CHECKPOINT_DIR = "outputs/checkpoints"
START_CHECKPOINT = "outputs/checkpoints/geosynth_unet_epoch_2.pth"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train():
    train_dataset = GeoSynthDataset(TRAIN_PATH)
    val_dataset = GeoSynthDataset(VAL_PATH)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

    model = GeoSynthUNet(in_channels=16, out_channels=13).to(DEVICE)

    model.load_state_dict(torch.load(START_CHECKPOINT, map_location=DEVICE))
    print("Loaded checkpoint:", START_CHECKPOINT)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    start_epoch = 3
    total_epochs = 20

    print("Device:", DEVICE)
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    best_val_loss = 999

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}"):
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
            f"Epoch [{epoch}/{total_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"geosynth_unet_epoch_{epoch}.pth"
        )

        torch.save(model.state_dict(), checkpoint_path)
        print("Saved:", checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "geosynth_unet_best.pth")
            torch.save(model.state_dict(), best_path)
            print("Best model saved:", best_path)


if __name__ == "__main__":
    train()