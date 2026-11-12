# Diabetic Retinopathy Detection System

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Django Version](https://img.shields.io/badge/Django-6.1%20%2F%205.0+-green.svg)](https://www.djangoproject.com/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN%20%2F%20PyTorch%20%2F%20Keras-orange.svg)](https://pytorch.org/)
[![Bootstrap](https://img.shields.io/badge/UI-Bootstrap%205%20%2B%20Chart.js-purple.svg)](https://getbootstrap.com/)
[![License](https://img.shields.io/badge/Academic-Diploma%20Final%20Year-teal.svg)](#)

An intelligent, full-stack medical AI decision-support web application for the automated screening, classification, and grading of **Diabetic Retinopathy (DR)** from digital retinal fundus photographs using deep Convolutional Neural Networks (CNNs).

---

## 1. Project Overview

Diabetic Retinopathy (DR) is a severe microvascular complication of diabetes mellitus and a leading cause of preventable blindness worldwide. Early identification of microaneurysms, hemorrhages, hard exudates, and neovascularization is critical to halting vision loss.

This platform provides:
- Automated clinical triage into the **5 International Clinical Diabetic Retinopathy Disease Severity Scale** stages.
- Real-time deep learning inference with confidence probability scores and class probability distributions.
- Secure clinician portal with historical records, clinical follow-up recommendations, and printable reports.
- Dedicated Staff Administration Portal for system-wide health and epidemiological monitoring.

---

## 2. Key Features

- **Deep CNN Image Classifier**: 4-block deep convolutional neural network with batch normalization, max-pooling, adaptive pooling, and dropout.
- **5-Stage Severity Classification**:
  - **Grade 0**: No Diabetic Retinopathy
  - **Grade 1**: Mild Non-Proliferative DR (NPDR)
  - **Grade 2**: Moderate Non-Proliferative DR (NPDR)
  - **Grade 3**: Severe Non-Proliferative DR (NPDR)
  - **Grade 4**: Proliferative Diabetic Retinopathy (PDR)
- **Interactive Visual Analytics**: Embedded **Chart.js** probability distribution charts and severity meters.
- **Clinician Dashboard**: Track total uploads, past analyses, recent activities, and individual patient scans.
- **Drag-and-Drop Retinal Upload**: Instant file pre-validation, client-side preview, animated scanning laser indicator, and 15 MB limit.
- **Strict Data Security & Privacy**: User ownership isolation ensuring clinicians can only view their own patient scans.
- **Staff Administration Portal**: System analytics, user directory, global inferences log, and direct Django admin bridge.
- **Responsive Healthcare Theme**: Navy Blue, Teal, Cyan, and White aesthetic with complete mobile, tablet, and desktop support.

---

## 3. Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, Bootstrap Icons, Chart.js 4.4 |
| **Backend** | Python 3.10+, Django 6.1 / 5.0+, Django ORM, Sessions, Messages |
| **Deep Learning** | PyTorch 2.x, TorchVision, Keras 3 / TensorFlow, NumPy, OpenCV, Pillow |
| **Database** | SQLite3 (`db.sqlite3`) |
| **Fonts & Icons** | Google Fonts (Inter & Poppins), Bootstrap Icons |

---

## 4. Project Structure

```text
Diabetic_Retinopathy_Detection/
│
├── manage.py                               # Django CLI management entry point
├── requirements.txt                        # Python dependencies
├── README.md                               # Academic project documentation
├── create_static_assets.py                 # Generator for medical images & logos
│
├── dataset/                                # Training and testing dataset
│   ├── train/
│   │   ├── No_DR/
│   │   ├── Mild/
│   │   ├── Moderate/
│   │   ├── Severe/
│   │   └── Proliferative_DR/
│   └── test/
│       ├── No_DR/
│       ├── Mild/
│       ├── Moderate/
│       ├── Severe/
│       └── Proliferative_DR/
│
├── model/                                  # Deep learning CNN pipeline
│   ├── preprocess.py                       # Retinal resizing, normalization & CLAHE
│   ├── train_model.py                      # RetinalCNN model architecture & training
│   ├── predict.py                          # CNN inference pipeline & softmax probabilities
│   ├── generate_sample_data.py             # Generates sample fundus images for training
│   ├── model_info.json                     # Model metadata & validation accuracy
│   └── diabetic_retinopathy_model.pth      # Trained model weights file
│
├── diabetic_retinopathy/                   # Core Django project configuration
│   ├── settings.py                         # Settings (apps, templates, static, media)
│   ├── urls.py                             # Root routing and media handlers
│   ├── asgi.py                             # ASGI configuration
│   └── wsgi.py                             # WSGI configuration
│
├── detection/                              # Main application module
│   ├── models.py                           # PredictionRecord, UserProfile, ContactMessage
│   ├── views.py                            # Auth, dashboard, upload, results, admin views
│   ├── forms.py                            # Upload, registration, login, profile forms
│   ├── urls.py                             # App URL routing
│   ├── admin.py                            # Customized Django admin interface
│   ├── prediction.py                       # Interface to model inference
│   ├── apps.py                             # App configuration
│   ├── tests.py                            # 13 unit & integration test cases
│   └── migrations/                         # Database migrations
│
├── templates/                              # Django HTML templates
│   ├── base.html                           # Base layout with navbar, footer & disclaimer
│   ├── home.html                           # Landing page with hero & disease categories
│   ├── about.html                          # Clinical statement, CNN theory & workflow
│   ├── contact.html                        # Contact form & laboratory details
│   ├── accounts/
│   │   ├── login.html                      # Sign in with password toggle
│   │   ├── register.html                   # Registration with validation
│   │   └── profile.html                    # Profile management & password change
│   ├── dashboard/
│   │   └── dashboard.html                  # Personalized clinician dashboard
│   ├── prediction/
│   │   ├── upload.html                     # Drag-and-drop retinal upload
│   │   ├── result.html                     # Diagnostic report with Chart.js
│   │   └── history.html                    # Scan history with delete modal & pagination
│   └── admin_panel/
│       ├── dashboard.html                  # Staff telemetry & doughnut chart
│       ├── users.html                      # User management table
│       └── predictions.html                # Global scan registry
│
├── static/                                 # Static assets
│   ├── css/
│   │   ├── style.css                       # Medical theme styles
│   │   ├── responsive.css                  # Mobile & tablet media queries
│   │   └── animation.css                   # Transitions & laser scan keyframes
│   ├── js/
│   │   ├── script.js                       # Upload dropzone, preview, Chart.js renderers
│   │   └── validation.js                   # Client-side form validation
│   └── images/
│       ├── logo.png                        # Retinal eye aperture logo
│       ├── hero.jpg                        # Retinal fundus medical hero image
│       └── background.jpg                  # Ambient clinical background pattern
│
├── media/                                  # Uploaded patient fundus photographs
│   └── retinal_images/
│
└── db.sqlite3                              # SQLite3 database
```

---

## 5. Setup & Installation Guide (Windows)

### Step 1: Open Terminal in Project Directory
Open PowerShell or Command Prompt in the project folder:
```powershell
cd C:\Users\Admin\Desktop\DIABETIC_RETINOPATHY_DETECTION_SYSTEM
```

### Step 2: Virtual Environment
Use the existing virtual environment or create a new one:
```powershell
# Activate the existing virtual environment
.\myvenv\Scripts\Activate.ps1

# Or in Command Prompt (cmd.exe):
# myvenv\Scripts\activate.bat
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Apply Database Migrations
```powershell
python manage.py makemigrations detection
python manage.py migrate
```

### Step 5: (Optional) Retrain or Update the CNN Model
The project already comes with pre-trained weights in `model/diabetic_retinopathy_model.pth`.
To generate fresh sample data or train on custom images:
```powershell
# Generate sample images (if needed):
python model/generate_sample_data.py

# Train the CNN model:
python model/train_model.py --epochs 5 --batch-size 16
```

### Step 6: Run Automated Tests
Verify all system components:
```powershell
python manage.py test detection
```
*(All 13 unit tests pass with `OK`.)*

### Step 7: Launch the Django Development Server
```powershell
python manage.py runserver
```
Visit `http://127.0.0.1:8000/` in your browser.

---

## 6. Pre-Configured Test Accounts

| Role | Username | Password | Access Level |
|---|---|---|---|
| **Staff Administrator** | `admin` | `admin123` | Full access, Staff Admin Panel (`/admin-panel/`), Django Admin (`/admin/`) |
| **Clinician Doctor** | `doctor` | `doctor123` | Personal Clinician Dashboard, Upload, Result Reports, History |

*You can also register any new account at `/accounts/register/`.*

---

## 7. Dataset Preparation Guide

To train on real-world clinical datasets (e.g., **APTOS 2019 Blindness Detection** or **Kaggle Diabetic Retinopathy Challenge**):

1. Download the dataset images.
2. Organize the images into the corresponding class subdirectories:
   ```text
   dataset/train/No_DR/
   dataset/train/Mild/
   dataset/train/Moderate/
   dataset/train/Severe/
   dataset/train/Proliferative_DR/
   ```
3. Run the training script:
   ```powershell
   python model/train_model.py --epochs 20 --batch-size 32 --lr 0.0005
   ```
4. The script automatically applies data augmentation (rotations, flips, color jitter), saves the best weights to `model/diabetic_retinopathy_model.pth`, and updates `model/model_info.json`.

---

## 8. Clinical Disease Severity Scale Reference

| Grade | Clinical Label | Pathological Features | Recommended Action |
|:---:|---|---|---|
| **0** | **No DR** | No observable vascular abnormalities. Capillaries intact. | Annual dilated fundus screening. |
| **1** | **Mild NPDR** | Microaneurysms only (isolated focal capillary dilations). | Follow-up dilated exam in 9–12 months. |
| **2** | **Moderate NPDR** | Microaneurysms, dot-and-blot hemorrhages, venous beading, cotton wool spots. | Referral to ophthalmologist in 3–6 months. |
| **3** | **Severe NPDR** | &gt;20 intraretinal hemorrhages in all 4 quadrants, venous beading in 2+ quadrants, or prominent IRMA in 1+ quadrant (4:2:1 rule). | Prompt retinal specialist consultation in 2–4 weeks. |
| **4** | **Proliferative DR (PDR)** | Neovascularization (fragile new vessels) or preretinal/vitreous hemorrhage. | Urgent ophthalmological referral within 24–48 hours. |

---

## 9. Academic Project Limitations

1. **Academic Decision-Support Prototype**: Designed for triaging support and educational demonstration. It is not an FDA/CE approved Software as a Medical Device (SaMD).
2. **Single-Modality Input**: Processes 2D fundus images and does not include Optical Coherence Tomography (OCT) cross-sections or systemic blood glucose telemetry (HbA1c).
3. **Artifact Vulnerability**: Scan blurriness, severe cataract opacity, or illumination glare may influence feature detection.

---

## 10. Medical Disclaimer

> **IMPORTANT MEDICAL NOTICE**:
> This software is intended strictly for academic research, education, and decision-support demonstration. It **does NOT** produce a definitive medical diagnosis and must **NEVER** replace clinical evaluation, biomicroscopic examination, or treatment planning by a licensed ophthalmologist or healthcare practitioner.
