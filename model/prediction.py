"""
Prediction module wrapper for model/predict.py.
"""

from model.predict import predict_retinopathy, is_model_available

if __name__ == '__main__':
    import sys
    from model.predict import __name__ as predict_main
    if len(sys.argv) > 1:
        import json
        print(json.dumps(predict_retinopathy(sys.argv[1]), indent=2))
