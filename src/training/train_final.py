import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.geosynth_dataset import GeoSynthDataset
from src.models.geosynth_unet_final import GeoSynthUNetFinal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PATH = "/content/drive/MyDrive/GeoSynth-CloudX/data/dataset/train"
VAL_PATH = "/content/drive/MyDrive/GeoSynth-CloudX/data/dataset/val"
SAVE_PATH = "/content/drive/MyDrive/GeoSynth-CloudX/outputs/checkpoints/geosynth_final_best.pth"

BATCH_SIZE = 16
EPOCHS = 8

train_ds = GeoSynthDataset(TRAIN_PATH)
val_ds = GeoSynthDataset(VAL_PATH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

model = GeoSynthUNetFinal().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val = 999

print("Device:", DEVICE)
print("Train:", len(train_ds))
print("Val:", len(val_ds))

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0

    for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        pred = model(x)

        loss = 0.8 * F.l1_loss(pred, y) + 0.2 * F.mse_loss(pred, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            pred = model(x)
            loss = 0.8 * F.l1_loss(pred, y) + 0.2 * F.mse_loss(pred, y)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print("Best saved:", SAVE_PATH)
