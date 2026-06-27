import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    binary_dilation,
    binary_fill_holes,
    gaussian_filter
)

RGB_IDX = [3, 2, 1]  # Sentinel-2 RGB = B4, B3, B2


def normalize_band(band):
    return np.clip((band + 1.0) / 2.0, 0, 1)


def contrast_stretch(img):
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        p1, p99 = np.percentile(img[:, :, c], (1, 99))
        out[:, :, c] = np.clip((img[:, :, c] - p1) / (p99 - p1 + 1e-6), 0, 1)
    return out


def to_rgb(s2):
    rgb = np.stack(
        [
            normalize_band(s2[RGB_IDX[0]]),
            normalize_band(s2[RGB_IDX[1]]),
            normalize_band(s2[RGB_IDX[2]])
        ],
        axis=-1
    )
    return contrast_stretch(rgb)


def detect_cloud_mask(s2_cloudy, brightness=76, whiteness=32, coverage_boost=True):
    rgb = to_rgb(s2_cloudy)

    mean_rgb = rgb.mean(axis=2)
    std_rgb = rgb.std(axis=2)
    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)

    whiteness_score = 1.0 - std_rgb
    contrast_score = max_rgb - min_rgb

    bright_thr = np.percentile(mean_rgb, brightness)
    white_thr = np.percentile(whiteness_score, whiteness)
    high_thr = np.percentile(max_rgb, brightness)

    bright_cloud = mean_rgb >= bright_thr
    white_cloud = whiteness_score >= white_thr
    high_reflectance = max_rgb >= high_thr
    low_contrast = contrast_score <= 0.42

    mask = (
        (bright_cloud & white_cloud) |
        (high_reflectance & low_contrast) |
        (bright_cloud & high_reflectance)
    )

    if coverage_boost:
        weak_cloud = (mean_rgb >= np.percentile(mean_rgb, 55)) & low_contrast
        mask = mask | weak_cloud

    mask = binary_closing(mask, iterations=7)
    mask = binary_fill_holes(mask)
    mask = binary_opening(mask, iterations=1)
    mask = binary_dilation(mask, iterations=9)

    mask = gaussian_filter(mask.astype(np.float32), sigma=2.0)
    mask = mask > 0.16

    return mask.astype(np.float32)
