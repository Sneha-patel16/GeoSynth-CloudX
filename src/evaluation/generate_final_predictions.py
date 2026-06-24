import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data.geosynth_dataset import GeoSynthDataset
from src.models.geosynth_unet_final import GeoSynthUNetFinal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PATH = "/content/drive/MyDrive/GeoSynth-CloudX/data/dataset/test"
CHECKPOINT = "/content/drive/MyDrive/GeoSynth-CloudX/outputs/checkpoints/geosynth_final_best.pth"
SAVE_DIR = "/content/drive/MyDrive/GeoSynth-CloudX/outputs/final_predictions"

os.makedirs(SAVE_DIR, exist_ok=True)

def rgb_raw(arr13):
    rgb = arr13[[3, 2, 1], :, :]
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = (rgb + 1) / 2
    return np.clip(rgb, 0, 1)

def apply_same_stretch(img, lo, hi):
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)

dataset = GeoSynthDataset(TEST_PATH)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = GeoSynthUNetFinal().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

with torch.no_grad():
    for idx, (x, y, name) in enumerate(loader):
        if idx >= 30:
            break

        x = x.to(DEVICE)
        pred = model(x)

        cloudy = x[0, 3:, :, :].cpu().numpy()
        clear = y[0].cpu().numpy()
        pred = pred[0].cpu().numpy()

        cloudy_rgb = rgb_raw(cloudy)
        clear_rgb = rgb_raw(clear)
        pred_rgb = rgb_raw(pred)

        # SAME contrast based on GT only
        lo, hi = np.percentile(clear_rgb, (2, 98))

        cloudy_rgb = apply_same_stretch(cloudy_rgb, lo, hi)
        clear_rgb = apply_same_stretch(clear_rgb, lo, hi)
        pred_rgb = apply_same_stretch(pred_rgb, lo, hi)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(cloudy_rgb); plt.title("Cloudy Input"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(pred_rgb); plt.title("GeoSynth Output"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(clear_rgb); plt.title("Ground Truth Clear"); plt.axis("off")
        plt.suptitle(name[0])
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"final_comparison_{idx+1}.png"), dpi=200)
        plt.close()

print("Saved to:", SAVE_DIR)
