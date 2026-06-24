import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.models.geosynth_unet_final import GeoSynthUNetFinal
from src.detection.cloud_detector import detect_cloud_mask, s2_to_rgb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path):
    model = GeoSynthUNetFinal().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def sar_to_rgb(sar):
    sar = np.transpose(sar, (1, 2, 0))
    sar = (sar + 1) / 2
    return np.clip(sar, 0, 1)

def enhance(img):
    p2, p98 = np.percentile(img, (2, 98))
    return np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)

def compute_metrics(pred, gt):
    pred = np.clip(pred, 0, 1)
    gt = np.clip(gt, 0, 1)

    mae = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=1.0))
    ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0))

    return mae, rmse, psnr, ssim

def reconstruct(npz_path, model, brightness=92, whiteness=70):
    data = np.load(npz_path)

    sar = data["sar"].astype(np.float32)
    cloudy = data["s2_cloudy"].astype(np.float32)
    clear = data["s2_clear"].astype(np.float32)

    x = np.concatenate([sar, cloudy], axis=0)
    x = torch.tensor(x).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = model(x)[0].cpu().numpy()

    sar_rgb = enhance(sar_to_rgb(sar))
    cloudy_rgb = enhance(s2_to_rgb(cloudy))
    pred_rgb = enhance(s2_to_rgb(pred))
    clear_rgb = enhance(s2_to_rgb(clear))

    mask = detect_cloud_mask(cloudy, brightness, whiteness)
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)

    final_output = cloudy_rgb * (1 - mask3) + pred_rgb * mask3
    final_output = np.clip(final_output, 0, 1)

    mae, rmse, psnr, ssim = compute_metrics(final_output, clear_rgb)
    cloud_percent = float(mask.mean() * 100)

    metric_text = f"""
Cloud Coverage: {cloud_percent:.2f}%
MAE: {mae:.4f}
RMSE: {rmse:.4f}
PSNR: {psnr:.2f} dB
SSIM: {ssim:.4f}
"""

    return sar_rgb, cloudy_rgb, mask, pred_rgb, final_output, clear_rgb, metric_text
