import numpy as np
from scipy.ndimage import binary_closing, binary_opening, binary_dilation, binary_fill_holes, gaussian_filter

RGB_IDX = [3, 2, 1]

def normalize_band(x):
    return np.clip((x + 1.0) / 2.0, 0, 1)

def stretch_rgb(rgb):
    out = np.zeros_like(rgb)
    for i in range(3):
        b = rgb[:, :, i]
        p1, p99 = np.percentile(b, (1, 99))
        out[:, :, i] = np.clip((b - p1) / (p99 - p1 + 1e-6), 0, 1)
    return out

def to_rgb(s2):
    rgb = np.stack([
        normalize_band(s2[RGB_IDX[0]]),
        normalize_band(s2[RGB_IDX[1]]),
        normalize_band(s2[RGB_IDX[2]])
    ], axis=-1)
    return stretch_rgb(rgb)

def detect_cloud_mask(s2, brightness=78, whiteness=35, full_coverage=True):
    rgb = to_rgb(s2)

    mean_rgb = rgb.mean(axis=2)
    std_rgb = rgb.std(axis=2)
    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)

    whiteness_score = 1 - std_rgb
    brightness_thr = np.percentile(mean_rgb, brightness)
    white_thr = np.percentile(whiteness_score, whiteness)
    max_thr = np.percentile(max_rgb, brightness)

    bright = mean_rgb >= brightness_thr
    white = whiteness_score >= white_thr
    high_reflectance = max_rgb >= max_thr
    low_contrast = (max_rgb - min_rgb) < 0.35

    mask = (bright & white) | (high_reflectance & low_contrast) | (bright & high_reflectance)

    if full_coverage:
        possible_cloud = mean_rgb > np.percentile(mean_rgb, 60)
        mask = mask | (possible_cloud & low_contrast)

    mask = binary_closing(mask, iterations=6)
    mask = binary_fill_holes(mask)
    mask = binary_opening(mask, iterations=1)
    mask = binary_dilation(mask, iterations=8)

    mask = gaussian_filter(mask.astype(np.float32), sigma=1.8)
    mask = mask > 0.18

    return mask.astype(np.float32)
