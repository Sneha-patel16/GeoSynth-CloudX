import numpy as np
import matplotlib.pyplot as plt


def normalize_for_display(img):
    img = img.astype("float32")
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img


def sentinel_rgb(s2_array):
    # Sentinel-2 bands: B4=Red index 3, B3=Green index 2, B2=Blue index 1
    rgb = np.stack([
        s2_array[3],
        s2_array[2],
        s2_array[1]
    ], axis=-1)

    return normalize_for_display(rgb)


def sar_rgb(sar_array):
    sar = np.transpose(sar_array, (1, 2, 0))
    return normalize_for_display(sar)


def show_sample(npz_path):
    data = np.load(npz_path)

    sar = data["sar"]
    cloudy = data["s2_cloudy"]
    clear = data["s2_clear"]

    sar_img = sar_rgb(sar)
    cloudy_img = sentinel_rgb(cloudy)
    clear_img = sentinel_rgb(clear)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(sar_img)
    plt.title("Sentinel-1 SAR")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cloudy_img)
    plt.title("Sentinel-2 Cloudy")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(clear_img)
    plt.title("Sentinel-2 Clear")
    plt.axis("off")

    plt.tight_layout()
    plt.show()