"""
Helper utility to generate sample retinal fundus training images for academic setup,
demonstrating the 5 stages of Diabetic Retinopathy.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent.parent

CLASSES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']


def generate_synthetic_fundus(stage_name, size=(224, 224), seed=42):
    """
    Generate a synthetic retinal fundus image representing clinical features of each stage.
    """
    np.random.seed(seed)
    w, h = size
    
    # Background: black
    img = Image.new('RGB', (w, h), (10, 10, 12))
    draw = ImageDraw.Draw(img)

    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 2 - 8

    # Create retinal disc gradient (orange/reddish tone)
    retina = Image.new('RGB', (w, h), (0, 0, 0))
    retina_draw = ImageDraw.Draw(retina)

    # Base retina color
    retina_color = (190, 75, 30)
    retina_draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        fill=retina_color
    )

    # Optic Disc (yellowish-bright circle on nasal side)
    od_x = center_x + int(radius * 0.45)
    od_y = center_y + int(np.random.uniform(-10, 10))
    od_r = int(radius * 0.22)
    retina_draw.ellipse(
        [od_x - od_r, od_y - od_r, od_x + od_r, od_y + od_r],
        fill=(255, 220, 130)
    )

    # Fovea / Macula (dark spot on temporal side)
    mac_x = center_x - int(radius * 0.3)
    mac_y = center_y
    mac_r = int(radius * 0.15)
    retina_draw.ellipse(
        [mac_x - mac_r, mac_y - mac_r, mac_x + mac_r, mac_y + mac_r],
        fill=(130, 45, 15)
    )

    # Retinal Blood Vessels (branching curves)
    vessel_color = (120, 20, 10)
    for _ in range(12):
        start_x, start_y = od_x, od_y
        for step in range(5):
            angle = np.random.uniform(0.7 * np.pi, 1.8 * np.pi)
            length = np.random.uniform(15, 35)
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            # clamp within retina
            dist = np.hypot(end_x - center_x, end_y - center_y)
            if dist < radius - 5:
                retina_draw.line([(start_x, start_y), (end_x, end_y)], fill=vessel_color, width=max(1, 4 - step))
                start_x, start_y = end_x, end_y

    # Pathology lesions according to grade:
    if stage_name == 'Mild':
        # Microaneurysms: small pinhead red dots
        for _ in range(6):
            rx = int(center_x + np.random.uniform(-radius*0.7, radius*0.7))
            ry = int(center_y + np.random.uniform(-radius*0.7, radius*0.7))
            if np.hypot(rx - center_x, ry - center_y) < radius - 15:
                retina_draw.ellipse([rx-2, ry-2, rx+2, ry+2], fill=(180, 10, 10))

    elif stage_name == 'Moderate':
        # Microaneurysms + blot hemorrhages + hard exudates
        for _ in range(15):
            rx = int(center_x + np.random.uniform(-radius*0.7, radius*0.7))
            ry = int(center_y + np.random.uniform(-radius*0.7, radius*0.7))
            if np.hypot(rx - center_x, ry - center_y) < radius - 15:
                retina_draw.ellipse([rx-3, ry-3, rx+3, ry+3], fill=(160, 15, 10))
        # Exudates (yellow speckles)
        for _ in range(8):
            ex = int(center_x + np.random.uniform(-radius*0.5, radius*0.5))
            ey = int(center_y + np.random.uniform(-radius*0.5, radius*0.5))
            retina_draw.ellipse([ex-2, ey-2, ex+2, ey+2], fill=(245, 235, 160))

    elif stage_name == 'Severe':
        # Extensive blot hemorrhages in multiple quadrants & cotton wool spots
        for _ in range(35):
            rx = int(center_x + np.random.uniform(-radius*0.8, radius*0.8))
            ry = int(center_y + np.random.uniform(-radius*0.8, radius*0.8))
            r_sz = np.random.randint(3, 7)
            if np.hypot(rx - center_x, ry - center_y) < radius - 15:
                retina_draw.ellipse([rx-r_sz, ry-r_sz, rx+r_sz, ry+r_sz], fill=(140, 5, 5))
        # Cotton wool spots (fluffy grey/white lesions)
        for _ in range(6):
            cx = int(center_x + np.random.uniform(-radius*0.6, radius*0.6))
            cy = int(center_y + np.random.uniform(-radius*0.6, radius*0.6))
            retina_draw.ellipse([cx-5, cy-4, cx+5, cy+4], fill=(210, 210, 200))

    elif stage_name == 'Proliferative_DR':
        # Neovascularization (tangle of fine abnormal new vessels) + large vitreous hemorrhage
        for _ in range(8):
            nx = int(od_x + np.random.uniform(-25, 25))
            ny = int(od_y + np.random.uniform(-25, 25))
            retina_draw.ellipse([nx-3, ny-3, nx+3, ny+3], fill=(220, 30, 20))
        # Vitreous hemorrhage (large dark red patch)
        vx = center_x + int(np.random.uniform(-40, 40))
        vy = center_y + int(np.random.uniform(-40, 40))
        retina_draw.ellipse([vx-22, vy-18, vx+22, vy+18], fill=(100, 5, 5))

    # Mask image with smooth circular border
    mask = Image.new('L', (w, h), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill=255)

    final_img = Image.composite(retina, img, mask)
    return final_img


def populate_dataset(train_count=12, test_count=4):
    """
    Populate dataset/train and dataset/test with sample images for initial training.
    """
    for cls in CLASSES:
        train_cls_dir = BASE_DIR / 'dataset' / 'train' / cls
        test_cls_dir = BASE_DIR / 'dataset' / 'test' / cls
        train_cls_dir.mkdir(parents=True, exist_ok=True)
        test_cls_dir.mkdir(parents=True, exist_ok=True)

        for i in range(train_count):
            img = generate_synthetic_fundus(cls, seed=100 + i * 7 + CLASSES.index(cls) * 31)
            img.save(train_cls_dir / f"{cls}_train_{i+1:03d}.jpg", quality=95)

        for i in range(test_count):
            img = generate_synthetic_fundus(cls, seed=900 + i * 13 + CLASSES.index(cls) * 19)
            img.save(test_cls_dir / f"{cls}_test_{i+1:03d}.jpg", quality=95)

    print(f"[OK] Successfully populated dataset with {train_count * len(CLASSES)} training and {test_count * len(CLASSES)} testing images.")


if __name__ == '__main__':
    populate_dataset()
