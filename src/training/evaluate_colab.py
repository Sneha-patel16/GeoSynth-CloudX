import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset_loader import GeoSynthDataset
from src.models.geosynth_unet import GeoSynthUNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PATH = "/content/drive/MyDrive/GeoSynth-CloudX/data/dataset/test"

CHECKPOINT = "/content/drive/MyDrive/GeoSynth-CloudX/outputs/checkpoints/geosynth_colab_fast.pth"


dataset = GeoSynthDataset(TEST_PATH)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2
)

model = GeoSynthUNet().to(DEVICE)

model.load_state_dict(
    torch.load(CHECKPOINT, map_location=DEVICE)
)

model.eval()

mae_list = []
rmse_list = []
psnr_list = []

with torch.no_grad():

    for x, y, _ in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)

        mae = torch.mean(torch.abs(pred - y))

        rmse = torch.sqrt(
            torch.mean((pred - y) ** 2)
        )

        mse = torch.mean(
            (pred - y) ** 2
        )

        psnr = 20 * torch.log10(
            torch.tensor(2.0, device=DEVICE)
        ) - 10 * torch.log10(mse)

        mae_list.append(mae.item())
        rmse_list.append(rmse.item())
        psnr_list.append(psnr.item())

print("MAE :", np.mean(mae_list))
print("RMSE:", np.mean(rmse_list))
print("PSNR:", np.mean(psnr_list))