"""
Detection App Prediction Interface.
Connects Django views with the CNN model inference pipeline.
"""

from model.predict import predict_retinopathy, is_model_available

__all__ = ['predict_retinopathy', 'is_model_available']
