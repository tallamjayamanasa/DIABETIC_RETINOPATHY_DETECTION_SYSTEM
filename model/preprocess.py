"""
Retinal Fundus Image Preprocessing Module.
Standardized for both Training and Inference.
"""

import os
import numpy as np
from PIL import Image
import cv2

# Disease Classes (International Clinical Diabetic Retinopathy Disease Severity Scale)
CLASS_NAMES = [
    'No_DR',
    'Mild',
    'Moderate',
    'Severe',
    'Proliferative_DR'
]

CLASS_DISPLAY_NAMES = {
    'No_DR': 'No Diabetic Retinopathy (Grade 0)',
    'Mild': 'Mild NPDR (Grade 1)',
    'Moderate': 'Moderate NPDR (Grade 2)',
    'Severe': 'Severe NPDR (Grade 3)',
    'Proliferative_DR': 'Proliferative DR (Grade 4)'
}

SEVERITY_LEVELS = {
    'No_DR': 'Normal',
    'Mild': 'Low',
    'Moderate': 'Moderate',
    'Severe': 'High',
    'Proliferative_DR': 'Critical'
}

IMG_SIZE = (224, 224)

# Normalization parameters (standard ImageNet / Medical imaging statistics)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_and_validate_image(image_input):
    """
    Safely load an image from a path, file-like object, or PIL Image.
    Returns RGB PIL Image.
    """
    if isinstance(image_input, Image.Image):
        pil_img = image_input.copy()
    elif hasattr(image_input, 'read'):
        # File-like object (e.g. Django UploadedFile)
        image_input.seek(0)
        pil_img = Image.open(image_input)
    elif isinstance(image_input, (str, os.PathLike)):
        pil_img = Image.open(image_input)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Convert to RGB mode (removes Alpha, Palette, Grayscale issues)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    return pil_img


def apply_retinal_clahe(cv_img_rgb):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance microaneurysms, hemorrhages, and exudates in retinal fundus images.
    """
    # Convert RGB to LAB color space
    lab = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge and convert back to RGB
    limg = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def preprocess_image(image_input, target_size=IMG_SIZE, use_clahe=False, return_tensor_format=False):
    """
    Preprocess image for deep learning model inference.
    
    Args:
        image_input: File path, file-like object, or PIL Image.
        target_size: Tuple (width, height), default (224, 224).
        use_clahe: If True, apply CLAHE contrast enhancement.
        return_tensor_format: If True, returns (1, C, H, W) for PyTorch or (1, H, W, C) for TensorFlow.
        
    Returns:
        normalized_array: float32 NumPy array scaled to [0, 1] or normalized.
    """
    pil_img = load_and_validate_image(image_input)
    
    # High-quality Lanczos resampling
    resized_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(resized_img, dtype=np.uint8)

    if use_clahe:
        img_array = apply_retinal_clahe(img_array)

    # Normalize to [0.0, 1.0]
    normalized_array = img_array.astype(np.float32) / 255.0

    return normalized_array


def prepare_torch_tensor(img_array):
    """
    Convert (H, W, C) float array [0, 1] to PyTorch tensor format (1, C, H, W).
    """
    import torch
    # Standard normalization
    norm = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    # Transpose from (H, W, C) to (C, H, W)
    chw = np.transpose(norm, (2, 0, 1))
    tensor = torch.tensor(chw, dtype=torch.float32).unsqueeze(0)
    return tensor


def prepare_keras_tensor(img_array):
    """
    Convert (H, W, C) float array [0, 1] to Keras/TF tensor format (1, H, W, C).
    """
    # Standard normalization or [0, 1]
    norm = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    batch = np.expand_dims(norm, axis=0)
    return batch


def generate_fundus_previews(image_input, target_size=IMG_SIZE):
    """
    Generates diagnostic preprocessed versions of the fundus photograph:
    1. Resized standard RGB
    2. Green channel extraction (highest contrast for retinal microaneurysms and vessels)
    3. CLAHE (Contrast-Limited Adaptive Histogram Equalization) enhanced RGB
    
    Returns:
        dict: Containing base64 data URLs for UI preview and PIL images.
    """
    import base64
    import io

    pil_img = load_and_validate_image(image_input)
    resized_pil = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    rgb_array = np.array(resized_pil, dtype=np.uint8)

    # 1. CLAHE enhanced
    clahe_rgb = apply_retinal_clahe(rgb_array)
    clahe_pil = Image.fromarray(clahe_rgb)

    # 2. Green channel enhancement (pseudocolor / grayscale)
    green_ch = rgb_array[:, :, 1]
    clahe_green = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(green_ch)
    # Convert to high-contrast cyan-green pseudocolor or grayscale
    green_rgb = cv2.applyColorMap(clahe_green, cv2.COLORMAP_VIRIDIS)
    green_rgb = cv2.cvtColor(green_rgb, cv2.COLOR_BGR2RGB)
    green_pil = Image.fromarray(green_rgb)

    def pil_to_base64(img):
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=90)
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')

    return {
        'original_b64': pil_to_base64(resized_pil),
        'clahe_b64': pil_to_base64(clahe_pil),
        'green_b64': pil_to_base64(green_pil),
        'clahe_pil': clahe_pil,
        'resized_pil': resized_pil,
    }

