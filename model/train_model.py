"""
CNN Deep Learning Training Pipeline for Diabetic Retinopathy Classification.
Supports 5 Clinical Stages: No_DR, Mild, Moderate, Severe, Proliferative_DR.
Provides training for both PyTorch and Keras backends.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from model.preprocess import CLASS_NAMES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

MODEL_DIR = BASE_DIR / 'model'
KERAS_MODEL_PATH = MODEL_DIR / 'diabetic_retinopathy_model.keras'
PYTORCH_MODEL_PATH = MODEL_DIR / 'diabetic_retinopathy_model.pth'
ONNX_MODEL_PATH = MODEL_DIR / 'diabetic_retinopathy_model.onnx'
INFO_PATH = MODEL_DIR / 'model_info.json'

NUM_CLASSES = len(CLASS_NAMES)


# =====================================================================
# PYTORCH CNN ARCHITECTURE (Production & Windows-compatible)
# =====================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class RetinalCNN(nn.Module):
    """
    Convolutional Neural Network tailored for Retinal Fundus Feature Extraction.
    Consists of 4 Convolutional Feature Extraction blocks,
    Batch Normalization, Max Pooling, and a Dense Classification Head.
    """
    def __init__(self, num_classes=5):
        super(RetinalCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)  # 224 -> 112

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 -> 56

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 -> 28

        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2)  # 28 -> 14

        # Adaptive Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


# =====================================================================
# KERAS / TENSORFLOW CNN ARCHITECTURE
# =====================================================================
def build_keras_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Build equivalent CNN using TensorFlow Keras Sequential API.
    """
    try:
        from tensorflow.keras import layers, models
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        print(f"[!] Keras model builder info: {e}")
        return None


def get_data_transforms():
    """
    Training and Testing Data Transforms with Clinical Data Augmentation.
    """
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    ])

    test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    ])

    return train_transform, test_transform


def train_pytorch(train_dir, test_dir, epochs=10, batch_size=16, lr=0.001):
    """
    Train PyTorch Retinal CNN model with validation and test evaluation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training RetinalCNN on device: {device}")

    train_tf, test_tf = get_data_transforms()
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_tf)

    

    print(f"[*] Found {len(train_dataset)} training samples, {len(test_dataset)} testing samples across classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = RetinalCNN(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)

        # Evaluation on test/val set
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                test_correct += torch.sum(preds == labels.data).item()
                test_total += labels.size(0)

        val_loss = test_loss / max(test_total, 1)
        val_acc = test_correct / max(test_total, 1)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), PYTORCH_MODEL_PATH)
            print(f"  [+] Saved best model weights to: {PYTORCH_MODEL_PATH}")

    # Save metadata info
    info = {
        'architecture': 'RetinalCNN',
        'classes': train_dataset.classes,
        'input_size': list(IMG_SIZE),
        'best_val_accuracy': float(best_acc),
        'device': str(device)
    }
    with open(INFO_PATH, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"[OK] Training complete. Model metadata saved to {INFO_PATH}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Diabetic Retinopathy CNN Classifier")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_path = BASE_DIR / 'dataset' / 'train'
    test_path = BASE_DIR / 'dataset' / 'test'

    print("Checking dataset directories...")
    # Check if samples exist in train folders
    has_samples = any(len(list(p.glob('*.*'))) > 0 for p in train_path.iterdir() if p.is_dir())

    if not has_samples:
        print("[!] No images found in dataset/train/. Please add images to train the model.")
        print(f"    Expected classes: {', '.join(CLASS_NAMES)}")
    else:
        train_pytorch(str(train_path), str(test_path), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
