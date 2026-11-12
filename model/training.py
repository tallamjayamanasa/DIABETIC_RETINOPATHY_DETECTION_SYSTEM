"""
Diabetic Retinopathy CNN Training Script (wrapper).
"""

from model.train_model import (
    RetinalCNN,
    build_keras_model,
    train_pytorch,
    CLASS_NAMES,
    IMG_SIZE
)

if __name__ == '__main__':
    from model.train_model import train_path, test_path, args, has_samples
    if not has_samples:
        print("Please populate dataset/train and dataset/test with retinal fundus images.")
    else:
        train_pytorch(str(train_path), str(test_path))
