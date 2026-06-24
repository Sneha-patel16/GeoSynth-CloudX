from PIL import Image
import numpy as np


def load_image(image_file):
    """
    Loads an uploaded image and converts it into RGB numpy array.
    """
    image = Image.open(image_file).convert("RGB")
    return image, np.array(image)


def resize_image(image, size=(256, 256)):
    """
    Resizes satellite image for model input.
    """
    return image.resize(size)


def normalize_image(image_array):
    """
    Normalizes image pixels from 0-255 to 0-1.
    """
    return image_array / 255.0