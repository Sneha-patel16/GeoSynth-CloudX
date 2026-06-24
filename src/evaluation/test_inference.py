import os
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import GeoSynthDataset
from src.models.geosynth_unet import GeoSynthUNet
from src.evaluation.metrics import mae, rmse, psnr
from src.utils.save_visuals import save_comparison


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PATH = "data/dataset/test"
CHECKPOINT_PATH = "C:/Users/HP/OneDrive/GeoSynth-CloudX/outputs/checkpoints/geosynth_unet_epoch_2.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"


def run_test():
    dataset = GeoSynthDataset(TEST_PATH)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = GeoSynthUNet(in_channels=16, out_channels=13).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    total_mae = 0
    total_rmse = 0
    total_psnr = 0

    print(f"Running evaluation on {len(dataset)} samples using {DEVICE}...")
    start_time = perf_counter()

    with torch.inference_mode():
        for i, (x, y, names) in enumerate(loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            total_mae += mae(pred, y)
            total_rmse += rmse(pred, y)
            total_psnr += psnr(pred, y)

            if i < 10:
                x_np = x.cpu().numpy()[0]
                y_np = y.cpu().numpy()[0]
                pred_np = pred.cpu().numpy()[0]

                sar = x_np[:3]
                cloudy = x_np[3:]

                save_path = os.path.join(
                    OUTPUT_DIR,
                    f"comparison_{i+1}.png"
                )

                save_comparison(
                    sar=sar,
                    cloudy=cloudy,
                    clear=y_np,
                    pred=pred_np,
                    save_path=str(save_path)
                )

            if (i + 1) % 25 == 0 or (i + 1) == len(loader):
                elapsed = perf_counter() - start_time
                print(f"Processed {i + 1}/{len(loader)} samples in {elapsed:.1f}s")

    n = len(loader)

    print("Test MAE:", total_mae / n)
    print("Test RMSE:", total_rmse / n)
    print("Test PSNR:", total_psnr / n)
    print("Saved predictions in:", OUTPUT_DIR)


if __name__ == "__main__":
    run_test()