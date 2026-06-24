import cv2
import numpy as np


def generate_basic_cloud_mask(image_array, threshold=200):
    """
    Generates a basic cloud mask using brightness thresholding.
    White/bright pixels are treated as cloud regions.
    """

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return mask


def calculate_cloud_percentage(mask):
    """
    Calculates percentage of cloud-covered pixels.
    """

    cloud_pixels = np.sum(mask == 255)
    total_pixels = mask.shape[0] * mask.shape[1]

    cloud_percentage = (cloud_pixels / total_pixels) * 100

    return round(cloud_percentage, 2)


def apply_cloud_overlay(image_array, mask):
    """
    Applies red overlay on detected cloud regions.
    """

    overlay = image_array.copy()
    overlay[mask == 255] = [255, 0, 0]

    blended = cv2.addWeighted(image_array, 0.7, overlay, 0.3, 0)

    return blended