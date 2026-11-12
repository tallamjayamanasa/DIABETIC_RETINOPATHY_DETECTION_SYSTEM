"""
Fundus Camera Image Validation Module.
Verifies whether an uploaded or camera-captured image is a genuine retinal fundus photograph
and rejects selfies, faces, normal photographs, documents, screenshots, and non-retinal images.
"""

import os
import io
import cv2
import numpy as np
from PIL import Image


class FundusValidationError(Exception):
    """Raised when an image fails retinal fundus validation."""
    pass


class FundusValidator:
    """
    Validates retinal fundus photographs using multi-stage bio-optical,
    spectral, circular aperture, and vascular tree analysis.
    """

    INVALID_MESSAGE = "Invalid Image. Please upload a valid retinal fundus image."

    @classmethod
    def load_image(cls, image_input):
        """
        Loads image input (filepath, file-like object, PIL Image, or numpy array)
        into RGB PIL Image and RGB numpy array.
        """
        if isinstance(image_input, Image.Image):
            pil_img = image_input.copy()
        elif hasattr(image_input, 'read'):
            image_input.seek(0)
            data = image_input.read()
            pil_img = Image.open(io.BytesIO(data))
            if hasattr(image_input, 'seek'):
                image_input.seek(0)
        elif isinstance(image_input, (str, os.PathLike)):
            pil_img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = np.clip(image_input * 255.0, 0, 255).astype(np.uint8)
            if len(image_input.shape) == 2:
                pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB))
            else:
                pil_img = Image.fromarray(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        rgb_array = np.array(pil_img, dtype=np.uint8)
        return pil_img, rgb_array

    @classmethod
    def evaluate_retinal_characteristics(cls, rgb_array):
        """
        Bio-optical validation rules for authentic retinal fundus images:
        1. Spectral Distribution (Red dominant from choroid/RPE, low Blue): R/B >= 2.2
        2. Retinal Hue percentage (HSV orange-red-amber spectrum): >= 60%
        3. Circular Optical Aperture (Dark corners vs illuminated center): Corners < 60
        4. Retinal Blood Vessels (Green-channel morphological black-hat response): >= 1.8
        """
        h, w, c = rgb_array.shape
        reasons = []
        is_valid = True

        r_mean = float(np.mean(rgb_array[:, :, 0]))
        g_mean = float(np.mean(rgb_array[:, :, 1]))
        b_mean = float(np.mean(rgb_array[:, :, 2]))

        # 1. Color Dominance Check
        rb_ratio = r_mean / max(b_mean, 1e-4)
        rg_ratio = r_mean / max(g_mean, 1e-4)

        if b_mean >= r_mean or rb_ratio < 2.2:
            is_valid = False
            reasons.append(f"Non-retinal color spectrum (Red-to-Blue ratio {rb_ratio:.2f} < 2.2). Normal photos, selfies, and daylight scenes lack retinal choroidal reflectance.")

        # 2. Retinal Hue Percentage Check in HSV
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        non_dark_mask = (rgb_array[:, :, 0] > 20) | (rgb_array[:, :, 1] > 20)
        total_non_dark = int(np.sum(non_dark_mask))

        if total_non_dark < (h * w * 0.15):
            is_valid = False
            reasons.append("Image is completely dark or underexposed with insufficient illuminated retinal area.")
            retinal_hue_pct = 0.0
        else:
            h_channel = hsv[:, :, 0]
            retinal_hue_pixels = np.sum(non_dark_mask & ((h_channel <= 25) | (h_channel >= 160)))
            retinal_hue_pct = float(retinal_hue_pixels / total_non_dark) * 100.0

            if retinal_hue_pct < 60.0:
                is_valid = False
                reasons.append(f"Only {retinal_hue_pct:.1f}% of pixels match retinal fundus chromaticity. Non-retinal objects, landscapes, clothing, or documents detected.")

        # 3. Circular Field of View (FOV) / Optical Aperture Check
        cw, ch = max(4, w // 10), max(4, h // 10)
        c1 = rgb_array[:ch, :cw]
        c2 = rgb_array[:ch, -cw:]
        c3 = rgb_array[-ch:, :cw]
        c4 = rgb_array[-ch:, -cw:]
        corners_mean = float(np.mean([c1.mean(), c2.mean(), c3.mean(), c4.mean()]))
        center_mean = float(np.mean(rgb_array[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]))

        has_circular_aperture = corners_mean < 60.0 or corners_mean < (center_mean * 0.65)

        if not has_circular_aperture:
            is_valid = False
            reasons.append(f"Missing circular optical aperture or dark margins (Corner brightness: {corners_mean:.1f}). Selfies, full-frame photos, and screenshots lack fundus camera aperture boundaries.")

        # 4. Retinal Vascular Tree & Texture Detection via Green Channel
        green = rgb_array[:, :, 1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
        illuminated = rgb_array[:, :, 0] > 30
        vessel_score = float(blackhat[illuminated].mean()) if np.any(illuminated) else 0.0

        if vessel_score < 1.8:
            is_valid = False
            reasons.append(f"Insufficient retinal blood vessel structures detected (Vessel score: {vessel_score:.2f} < 1.8). The image appears flat, artificial, or non-retinal.")

        # Compute Quality Score (0 to 100%)
        quality_score = 0.0
        if is_valid:
            quality_score = min(100.0, 50.0 + (rb_ratio * 5.0) + (vessel_score * 5.0) + (10.0 if corners_mean < 30.0 else 0.0))
            quality_score = round(quality_score, 1)
            clinical_message = "Genuine Retinal Fundus Image verified. Spectral profile, optical aperture, and vascular tree match clinical standards."
        else:
            quality_score = 0.0
            clinical_message = cls.INVALID_MESSAGE

        details = {
            'r_mean': round(r_mean, 1),
            'g_mean': round(g_mean, 1),
            'b_mean': round(b_mean, 1),
            'rb_ratio': round(rb_ratio, 2),
            'rg_ratio': round(rg_ratio, 2),
            'retinal_hue_pct': round(retinal_hue_pct, 1),
            'vessel_score': round(vessel_score, 2),
            'corners_mean': round(corners_mean, 1),
            'center_mean': round(center_mean, 1),
            'has_circular_aperture': bool(has_circular_aperture),
            'quality_score': quality_score,
        }

        return {
            'is_valid': is_valid,
            'reasons': reasons,
            'quality_score': quality_score,
            'details': details,
            'clinical_message': clinical_message,
        }

    @classmethod
    def validate_image(cls, image_input):
        """
        Primary entry point. Validates an image and returns validation report.
        """
        try:
            pil_img, rgb_array = cls.load_image(image_input)
        except Exception as e:
            return {
                'is_valid': False,
                'reasons': [f"Corrupted or unreadable image file: {str(e)}"],
                'quality_score': 0.0,
                'details': {},
                'clinical_message': cls.INVALID_MESSAGE
            }

        h, w = rgb_array.shape[:2]
        if w < 64 or h < 64:
            return {
                'is_valid': False,
                'reasons': ["Image dimensions are too small (minimum 64x64 pixels required)."],
                'quality_score': 0.0,
                'details': {'width': w, 'height': h},
                'clinical_message': cls.INVALID_MESSAGE
            }

        return cls.evaluate_retinal_characteristics(rgb_array)
