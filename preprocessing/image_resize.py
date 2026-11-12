"""
Image resizing utilities for Diabetic Retinopathy Detection.
"""

from PIL import Image
from model.preprocess import IMG_SIZE, load_and_validate_image


def resize_retinal_image(image_input, size=IMG_SIZE):
    """
    Resize input retinal image to standard dimensions (224, 224) using high-quality Lanczos filter.
    """
    pil_img = load_and_validate_image(image_input)
    return pil_img.resize(size, Image.Resampling.LANCZOS)
