import numpy as np
from scipy.ndimage import binary_closing, binary_opening, binary_dilation

def s2_to_rgb(s2):
    rgb = s2[[3, 2, 1], :, :]  # B4, B3, B2
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = (rgb + 1) / 2
    return np.clip(rgb, 0, 1)

def detect_cloud_mask(s2_cloudy, brightness_percentile=92, whiteness_percentile=70):
    rgb = s2_to_rgb(s2_cloudy)

    brightness = rgb.mean(axis=2)
    whiteness = 1.0 - rgb.std(axis=2)

    b_thr = np.percentile(brightness, brightness_percentile)
    w_thr = np.percentile(whiteness, whiteness_percentile)

    mask = (brightness > b_thr) & (whiteness > w_thr)

    mask = binary_opening(mask, iterations=1)
    mask = binary_closing(mask, iterations=1)
    mask = binary_dilation(mask, iterations=1)

    return mask.astype(np.float32)
