import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import inpaint_biharmonic

from src.models.geosynth_unet_final import GeoSynthUNetFinal
from src.detection.cloud_detector import detect_cloud_mask, to_rgb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path):
    model = GeoSynthUNetFinal().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def enhance_rgb(img):
    img = np.clip(img, 0, 1)
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        band = img[:, :, c]
        p2, p98 = np.percentile(band, (2, 98))
        out[:, :, c] = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
    return out

def color_match(pred, reference):
    out = pred.copy()
    for c in range(3):
        p = pred[:, :, c]
        r = reference[:, :, c]
        out[:, :, c] = ((p - p.mean()) / (p.std() + 1e-6)) * (r.std() + 1e-6) + r.mean()
    return np.clip(out, 0, 1)

def safe_inpaint(cloudy_rgb, mask):
    mask_bool = mask > 0.5
    if mask_bool.mean() < 0.01:
        return cloudy_rgb.copy()

    out = cloudy_rgb.copy()
    for c in range(3):
        try:
            out[:, :, c] = inpaint_biharmonic(cloudy_rgb[:, :, c], mask_bool)
        except Exception:
            out[:, :, c] = cloudy_rgb[:, :, c]
    return np.clip(out, 0, 1)

def compute_metrics(pred, gt):
    mae = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    psnr = float(20 * np.log10(1.0 / (rmse + 1e-8)))
    try:
        ssim_score = float(ssim(gt, pred, channel_axis=2, data_range=1.0))
    except Exception:
        ssim_score = 0.0
    accuracy = max(0, (1 - rmse) * 100)
    return mae, rmse, psnr, ssim_score, accuracy

def reconstruct(npz_path, model, brightness=78, whiteness=35):
    data = np.load(npz_path)

    sar = data["sar"].astype(np.float32)
    cloudy = data["s2_cloudy"].astype(np.float32)
    clear = data["s2_clear"].astype(np.float32)

    x = np.concatenate([sar, cloudy], axis=0)
    x_tensor = torch.tensor(x).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred_clear = model(x_tensor)[0].cpu().numpy()

    cloudy_rgb = enhance_rgb(to_rgb(cloudy))
    raw_rgb = enhance_rgb(to_rgb(pred_clear))
    gt_rgb = enhance_rgb(to_rgb(clear))

    mask = detect_cloud_mask(cloudy, int(brightness), int(whiteness))

    raw_rgb = color_match(raw_rgb, cloudy_rgb)
    inpaint_rgb = safe_inpaint(cloudy_rgb, mask)

    reconstructed_region = 0.55 * raw_rgb + 0.45 * inpaint_rgb
    reconstructed_region = color_match(reconstructed_region, cloudy_rgb)

    soft_mask = gaussian_filter(mask.astype(np.float32), sigma=2.0)
    soft_mask = np.clip(soft_mask, 0, 1)
    soft_mask_3 = np.repeat(soft_mask[:, :, None], 3, axis=2)

    final_rgb = cloudy_rgb * (1 - soft_mask_3) + reconstructed_region * soft_mask_3
    final_rgb = np.clip(final_rgb, 0, 1)

    sar_rgb = np.transpose(sar, (1, 2, 0))
    sar_rgb = np.clip((sar_rgb + 1) / 2, 0, 1)

    mae, rmse, psnr, ssim_score, accuracy = compute_metrics(final_rgb, gt_rgb)
    cloud_percent = float(mask.mean() * 100)

    metrics = f"""
✅ Reconstruction Accuracy: {accuracy:.2f}%

Cloud Coverage Detected: {cloud_percent:.2f}%
MAE: {mae:.4f}
RMSE: {rmse:.4f}
PSNR: {psnr:.2f} dB
SSIM: {ssim_score:.4f}

Detector Used: Adaptive Spectral Cloud Mask
"""

    return sar_rgb, cloudy_rgb, mask, raw_rgb, final_rgb, gt_rgb, metrics
