"""
Diabetic Retinopathy CNN Inference & Prediction Pipeline.
Loads trained weights and predicts DR stage with confidence scores.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from model.preprocess import (
    CLASS_NAMES,
    CLASS_DISPLAY_NAMES,
    SEVERITY_LEVELS,
    IMG_SIZE,
    preprocess_image,
    prepare_torch_tensor
)

MODEL_DIR = BASE_DIR / 'model'
PYTORCH_MODEL_PATH = MODEL_DIR / 'diabetic_retinopathy_model.pth'
KERAS_MODEL_PATH = MODEL_DIR / 'diabetic_retinopathy_model.keras'
INFO_PATH = MODEL_DIR / 'model_info.json'

# Global cached model reference
_CACHED_MODEL = None
_MODEL_TYPE = None


def is_model_available():
    """
    Check if a trained model weight file is available.
    """
    return PYTORCH_MODEL_PATH.exists() or KERAS_MODEL_PATH.exists()


def get_model():
    """
    Load and return the trained CNN model (cached in memory).
    """
    global _CACHED_MODEL, _MODEL_TYPE
    
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, _MODEL_TYPE

    # Check PyTorch model first
    if PYTORCH_MODEL_PATH.exists():
        from model.train_model import RetinalCNN
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RetinalCNN(num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        _CACHED_MODEL = model
        _MODEL_TYPE = 'pytorch'
        print(f"[OK] Loaded PyTorch RetinalCNN model from {PYTORCH_MODEL_PATH}")
        return _CACHED_MODEL, _MODEL_TYPE

    # Check Keras model
    if KERAS_MODEL_PATH.exists():
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(KERAS_MODEL_PATH))
            _CACHED_MODEL = model
            _MODEL_TYPE = 'keras'
            print(f"[OK] Loaded Keras model from {KERAS_MODEL_PATH}")
            return _CACHED_MODEL, _MODEL_TYPE
        except Exception as e:
            print(f"[!] Could not load Keras model: {e}")

    return None, None


def predict_retinopathy(image_input):
    """
    Run prediction on a single retinal fundus image.
    
    Args:
        image_input: File path, file-like object, or PIL Image.
        
    Returns:
        dict: Prediction results containing predicted_class, confidence, probabilities, etc.
    """
    if not is_model_available():
        return {
            'status': 'ERROR',
            'error': (
                "Deep learning model weights not found. "
                "Please run `python model/train_model.py` to train the CNN model on the dataset, "
                "or place `diabetic_retinopathy_model.pth` / `diabetic_retinopathy_model.keras` "
                "in the `model/` folder."
            ),
            'predicted_class': None,
            'confidence': 0.0,
            'probabilities': {cls: 0.0 for cls in CLASS_NAMES}
        }

    model, model_type = get_model()
    if model is None:
        return {
            'status': 'ERROR',
            'error': "Failed to load model weights. Ensure weights match the architecture.",
            'predicted_class': None,
            'confidence': 0.0,
            'probabilities': {cls: 0.0 for cls in CLASS_NAMES}
        }

    try:
        # Preprocess input image to normalized float32 array
        norm_array = preprocess_image(image_input, target_size=IMG_SIZE)

        if model_type == 'pytorch':
            tensor = prepare_torch_tensor(norm_array)
            device = next(model.parameters()).device
            tensor = tensor.to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        elif model_type == 'keras':
            from model.preprocess import prepare_keras_tensor
            batch = prepare_keras_tensor(norm_array)
            preds = model.predict(batch)
            probs = preds[0]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compute predictions
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        with open(INFO_PATH, "r") as f:
            model_info = json.load(f)

        model_classes = model_info["classes"]
        predicted_class = model_classes[pred_idx]

        # Map probabilities explicitly to the canonical 5 medical classes in Grade 0 to Grade 4 order
        class_to_prob = {
            cls: float(probs[i])
            for i, cls in enumerate(model_classes)
        }
        probabilities_dict = {
            cls: round(class_to_prob.get(cls, 0.0) * 100, 2)
            for cls in CLASS_NAMES
        }

        return {
            'status': 'SUCCESS',
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'confidence_percentage': f"{confidence * 100:.1f}%",
            'probabilities': probabilities_dict,
            'severity_level': SEVERITY_LEVELS.get(predicted_class, 'Normal'),
            'display_name': CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class),
            'error': None
        }

    except Exception as e:
        return {
            'status': 'ERROR',
            'error': f"Prediction error: {str(e)}",
            'predicted_class': None,
            'confidence': 0.0,
            'probabilities': {cls: 0.0 for cls in CLASS_NAMES}
        }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python model/predict.py <path_to_retinal_image.jpg>")
    else:
        img_path = sys.argv[1]
        print(f"Analyzing {img_path}...")
        result = predict_retinopathy(img_path)
        print(json.dumps(result, indent=2))
