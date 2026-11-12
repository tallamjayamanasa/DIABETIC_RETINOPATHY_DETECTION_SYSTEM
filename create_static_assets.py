"""
Create high-quality medical imagery for Diabetic Retinopathy Detection System static assets.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / 'static' / 'images'
IMG_DIR.mkdir(parents=True, exist_ok=True)


def create_logo():
    """Create a sleek, modern medical AI eye logo (256x256 PNG with transparency)."""
    size = (300, 300)
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Stylized eye shape
    # Outer circle / gradient teal glow
    draw.ellipse([20, 20, 280, 280], fill=(14, 165, 233, 25), outline=(13, 148, 136, 180), width=3)
    
    # Stylized Iris (Deep blue/teal gradient)
    draw.ellipse([65, 65, 235, 235], fill=(15, 43, 72, 240), outline=(14, 165, 233, 255), width=4)
    
    # Retinal vessel lines
    vessel_color = (244, 63, 94, 220)
    draw.arc([80, 80, 220, 220], start=45, end=210, fill=vessel_color, width=3)
    draw.line([(150, 150), (195, 105)], fill=vessel_color, width=3)
    draw.line([(195, 105), (220, 95)], fill=vessel_color, width=2)
    draw.line([(195, 105), (215, 125)], fill=vessel_color, width=2)
    draw.line([(150, 150), (110, 195)], fill=vessel_color, width=3)
    draw.line([(110, 195), (85, 205)], fill=vessel_color, width=2)

    # Pupil with AI aperture crosshairs
    draw.ellipse([115, 115, 185, 185], fill=(6, 182, 212, 255))
    draw.ellipse([132, 132, 168, 168], fill=(10, 25, 47, 255))

    # Optic disc highlight
    draw.ellipse([185, 140, 210, 165], fill=(255, 230, 130, 240))

    # AI node points
    points = [(70, 150), (230, 150), (150, 70), (150, 230), (95, 95), (205, 205)]
    for pt in points:
        draw.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill=(20, 184, 166, 255))

    img.save(IMG_DIR / 'logo.png')
    print("[OK] Created logo.png")


def create_hero():
    """Create a rich, professional retinal fundus medical hero image (1200x800)."""
    w, h = 1200, 800
    # Deep navy clinical background
    img = Image.new('RGB', (w, h), (11, 25, 44))
    draw = ImageDraw.Draw(img)

    # Ambient teal/cyan glow in the center-right
    cx, cy = 720, 400
    r = 340

    # Draw retinal fundus globe
    fundus = Image.new('RGB', (w, h), (11, 25, 44))
    fdraw = ImageDraw.Draw(fundus)

    # Orange-red retinal field
    fdraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(195, 65, 25))

    # Optic disc
    od_x, od_y = cx + 150, cy - 20
    od_r = 75
    fdraw.ellipse([od_x - od_r, od_y - od_r, od_x + od_r, od_y + od_r], fill=(255, 225, 140))

    # Macula
    m_x, m_y = cx - 110, cy + 10
    m_r = 50
    fdraw.ellipse([m_x - m_r, m_y - m_r, m_x + m_r, m_y + m_r], fill=(125, 35, 15))

    # Blood vessels
    np.random.seed(101)
    vessel_color = (130, 15, 10)
    for _ in range(16):
        sx, sy = od_x, od_y
        for step in range(6):
            ang = np.random.uniform(0.7 * np.pi, 1.85 * np.pi)
            step_len = np.random.uniform(35, 65)
            ex = int(sx + step_len * np.cos(ang))
            ey = int(sy + step_len * np.sin(ang))
            if np.hypot(ex - cx, ey - cy) < r - 10:
                fdraw.line([(sx, sy), (ex, ey)], fill=vessel_color, width=max(2, 6 - step))
                sx, sy = ex, ey

    # Add lesions representing DR clinical patterns
    # Microaneurysms
    for _ in range(25):
        rx = int(cx + np.random.uniform(-r*0.7, r*0.7))
        ry = int(cy + np.random.uniform(-r*0.7, r*0.7))
        if np.hypot(rx - cx, ry - cy) < r - 25:
            fdraw.ellipse([rx-4, ry-4, rx+4, ry+4], fill=(190, 10, 10))

    # Hard exudates
    for _ in range(15):
        ex = int(cx + np.random.uniform(-r*0.5, r*0.5))
        ey = int(cy + np.random.uniform(-r*0.5, r*0.5))
        fdraw.ellipse([ex-3, ey-3, ex+3, ey+3], fill=(250, 240, 170))

    # Soft exudates / Cotton wool spots
    for _ in range(5):
        cx_spot = int(cx + np.random.uniform(-r*0.6, r*0.6))
        cy_spot = int(cy + np.random.uniform(-r*0.6, r*0.6))
        fdraw.ellipse([cx_spot-10, cy_spot-7, cx_spot+10, cy_spot+7], fill=(220, 220, 205))

    # Circular mask
    mask = Image.new('L', (w, h), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    composite = Image.composite(fundus, img, mask)
    cdraw = ImageDraw.Draw(composite)

    # Scanner overlay / AI bounding boxes
    cdraw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(14, 165, 233), width=3)
    
    # AI detection bounding box highlights
    bbox1 = [cx - 70, cy - 90, cx + 20, cy + 10]
    cdraw.rectangle(bbox1, outline=(6, 182, 212), width=2)
    bbox2 = [cx + 80, cy + 40, cx + 180, cy + 130]
    cdraw.rectangle(bbox2, outline=(244, 63, 94), width=2)

    # Soft gradient overlay on left for text readability
    gradient = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(gradient)
    for x in range(w):
        alpha = int(max(0, 1 - (x / 700)) * 230)
        gdraw.line([(x, 0), (x, h)], fill=(11, 25, 44, alpha))

    final_img = Image.alpha_composite(composite.convert('RGBA'), gradient).convert('RGB')
    final_img.save(IMG_DIR / 'hero.jpg', quality=92)
    print("[OK] Created hero.jpg")


def create_background():
    """Create a subtle dark-teal geometric medical background pattern (1920x1080)."""
    w, h = 1920, 1080
    img = Image.new('RGB', (w, h), (10, 23, 40))
    draw = ImageDraw.Draw(img)

    # Subtle grid lines
    grid_color = (18, 38, 64)
    for x in range(0, w, 60):
        draw.line([(x, 0), (x, h)], fill=grid_color, width=1)
    for y in range(0, h, 60):
        draw.line([(0, y), (w, y)], fill=grid_color, width=1)

    # Soft glowing radial gradients
    glow = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    gdraw.ellipse([w - 600, -200, w + 400, 600], fill=(14, 165, 233, 20))
    gdraw.ellipse([-200, h - 500, 600, h + 300], fill=(13, 148, 136, 18))

    final = Image.alpha_composite(img.convert('RGBA'), glow).convert('RGB')
    final.save(IMG_DIR / 'background.jpg', quality=90)
    print("[OK] Created background.jpg")


if __name__ == '__main__':
    create_logo()
    create_hero()
    create_background()
