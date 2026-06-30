import numpy as np
from scipy.ndimage import binary_closing, binary_opening, binary_dilation, binary_fill_holes, gaussian_filter

RGB_IDX = [3, 2, 1]
def normalize_band(x):
    return np.clip((x + 1.0) / 2.0, 0, 1)

def stretch_rgb(rgb):
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        b = rgb[:, :, i]
        p2, p98 = np.percentile(b, (2, 98))
        out[:, :, i] = np.clip((b - p2) / (p98 - p2 + 1e-6), 0, 1)
    return out

def to_rgb(s2):
    rgb = np.stack([
        normalize_band(s2[RGB_IDX[0]]),
        normalize_band(s2[RGB_IDX[1]]),
        normalize_band(s2[RGB_IDX[2]])
    ], axis=-1)
    return stretch_rgb(rgb)

def detect_cloud_mask(s2, brightness=78, whiteness=35):
    rgb = to_rgb(s2)

    mean = rgb.mean(axis=2)
    std = rgb.std(axis=2)
    maxv = rgb.max(axis=2)
    minv = rgb.min(axis=2)
    contrast = maxv - minv

    white_score = 1.0 - std

    bright = mean > np.percentile(mean, brightness)
    white = white_score > np.percentile(white_score, whiteness)
    high_reflect = maxv > np.percentile(maxv, brightness)
    low_contrast = contrast < 0.38

    mask = (bright & white) | (high_reflect & low_contrast)

    mask = binary_closing(mask, iterations=4)
    mask = binary_fill_holes(mask)
    mask = binary_opening(mask, iterations=1)
    mask = binary_dilation(mask, iterations=4)

    mask = gaussian_filter(mask.astype(np.float32), sigma=1.4)
    mask = mask > 0.25

    return mask.astype(np.float32)
