"""
Comprehensive Test Suite for Diabetic Retinopathy Detection System.
Tests Authentication, Dashboard, Image Upload Validation, Ownership Isolation,
Inference Flow, and Admin Protection.
"""

import io
from PIL import Image
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile

from detection.models import PredictionRecord, UserProfile, ContactMessage
from detection.prediction import is_model_available, predict_retinopathy


class RetinopathySystemTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Normal Test User 1
        self.user1 = User.objects.create_user(
            username='dr_smith',
            email='smith@eyecare.org',
            password='Password123!',
            first_name='John',
            last_name='Smith'
        )

        # Normal Test User 2 (for ownership isolation tests)
        self.user2 = User.objects.create_user(
            username='dr_jones',
            email='jones@eyecare.org',
            password='Password123!',
            first_name='Sarah',
            last_name='Jones'
        )

        # Staff / Superuser
        self.staff_user = User.objects.create_superuser(
            username='admin_doc',
            email='admin@hospital.org',
            password='AdminPassword123!'
        )

        # Generate a valid test retinal fundus image in memory using real dataset sample
        self.valid_image_bytes = self._create_test_image()

    def _create_test_image(self, size=(224, 224), color=(180, 60, 20)):
        import os
        sample_path = 'dataset/train/No_DR/No_DR_train_001.jpg'
        if os.path.exists(sample_path):
            with open(sample_path, 'rb') as f:
                return f.read()
        # Fallback: draw circular fundus with dark corners
        img = Image.new('RGB', size, color=(10, 10, 10))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.ellipse([20, 20, 204, 204], fill=color, outline=(220, 100, 30))
        draw.ellipse([60, 90, 90, 120], fill=(240, 220, 140))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return buf.getvalue()

    # 1. Public Pages Tests
    def test_home_page_loads(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "AI-Powered Diabetic Retinopathy Detection")
        self.assertContains(response, "Analyze Retinal Image")

    def test_about_page_loads(self):
        response = self.client.get(reverse('about'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "About Diabetic Retinopathy Detection")
        self.assertContains(response, "Convolutional Neural Network")

    def test_contact_submission(self):
        data = {
            'name': 'Test Clinician',
            'email': 'tester@clinic.com',
            'subject': 'Inquiry about CNN accuracy',
            'message': 'Hello, could you explain the dataset preprocessing?'
        }
        response = self.client.post(reverse('contact'), data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(ContactMessage.objects.filter(email='tester@clinic.com').exists())

    # 2. Authentication Tests
    def test_user_registration(self):
        data = {
            'first_name': 'Alice',
            'last_name': 'Brown',
            'username': 'alice_dr',
            'email': 'alice@hospital.org',
            'password': 'SecurePassword123!',
            'confirm_password': 'SecurePassword123!'
        }
        response = self.client.post(reverse('register'), data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(User.objects.filter(username='alice_dr').exists())

    def test_user_login_and_logout(self):
        response = self.client.post(reverse('login'), {
            'login_id': 'dr_smith',
            'password': 'Password123!'
        }, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['user'].is_authenticated)

        # Logout
        logout_resp = self.client.get(reverse('logout'), follow=True)
        self.assertEqual(logout_resp.status_code, 200)
        self.assertFalse(logout_resp.context['user'].is_authenticated)

    # 3. Protected Dashboard Tests
    def test_dashboard_requires_login(self):
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('login'), response.url)

    def test_dashboard_authenticated(self):
        self.client.login(username='dr_smith', password='Password123!')
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Welcome, John!")

    # 4. Retinal Image Upload & Validation Tests
    def test_upload_valid_retinal_image(self):
        self.client.login(username='dr_smith', password='Password123!')
        uploaded = SimpleUploadedFile(
            name='test_retina.jpg',
            content=self.valid_image_bytes,
            content_type='image/jpeg'
        )
        response = self.client.post(reverse('upload'), {
            'image': uploaded,
            'notes': 'Test patient scan OD'
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(PredictionRecord.objects.filter(user=self.user1).exists())
        record = PredictionRecord.objects.filter(user=self.user1).first()
        self.assertIsNotNone(record.predicted_class)
        self.assertGreaterEqual(record.confidence, 0.0)

    def test_upload_rejects_corrupted_or_non_image(self):
        self.client.login(username='dr_smith', password='Password123!')
        fake_file = SimpleUploadedFile(
            name='fake_scan.jpg',
            content=b'This is clearly not a real image file content.',
            content_type='image/jpeg'
        )
        response = self.client.post(reverse('upload'), {
            'image': fake_file
        }, follow=True)
        self.assertContains(response, "Upload a valid image")

    def test_upload_rejects_invalid_extension(self):
        self.client.login(username='dr_smith', password='Password123!')
        script_file = SimpleUploadedFile(
            name='malicious.exe',
            content=b'binary contents',
            content_type='application/octet-stream'
        )
        response = self.client.post(reverse('upload'), {
            'image': script_file
        }, follow=True)
        self.assertContains(response, "Upload a valid image")

    # 5. Prediction History & Ownership Isolation Tests
    def test_user_cannot_view_another_users_prediction(self):
        # Create record for User 1
        uploaded = SimpleUploadedFile('retina_u1.jpg', self.valid_image_bytes, 'image/jpeg')
        rec = PredictionRecord.objects.create(
            user=self.user1,
            image=uploaded,
            predicted_class='No_DR',
            confidence=0.92,
            status='COMPLETED'
        )

        # Login as User 2 and attempt to view User 1's prediction
        self.client.login(username='dr_jones', password='Password123!')
        response = self.client.get(reverse('result', kwargs={'pk': rec.pk}))
        self.assertEqual(response.status_code, 404)

        # Login as User 1 and view own prediction
        self.client.login(username='dr_smith', password='Password123!')
        response = self.client.get(reverse('result', kwargs={'pk': rec.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No Diabetic Retinopathy")

    # 6. Admin Panel Access Control Tests
    def test_regular_user_cannot_access_admin_panel(self):
        self.client.login(username='dr_smith', password='Password123!')
        response = self.client.get(reverse('admin_dashboard'))
        # Staff check redirects non-staff users
        self.assertEqual(response.status_code, 302)

    def test_staff_user_can_access_admin_panel(self):
        self.client.login(username='admin_doc', password='AdminPassword123!')
        response = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "System Analytics &amp; Control Panel")

    # 7. Retinal Image Upload & Five-Class Prediction Tests
    def test_upload_rejects_non_retinal_image(self):
        self.client.login(username='dr_smith', password='Password123!')
        # Blue non-retinal image
        blue_img = Image.new('RGB', (224, 224), color=(30, 90, 220))
        buf = io.BytesIO()
        blue_img.save(buf, format='JPEG')
        uploaded = SimpleUploadedFile('scenery_photo.jpg', buf.getvalue(), 'image/jpeg')

        response = self.client.post(reverse('upload'), {
            'image': uploaded,
            'notes': 'Test non-fundus rejection'
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Invalid Image. Please upload a valid retinal fundus image.")
        # Ensure invalid image is NEVER saved to database as a prediction record
        self.assertFalse(PredictionRecord.objects.filter(notes='Test non-fundus rejection').exists())

    def test_upload_five_class_prediction_pipeline(self):
        self.client.login(username='dr_smith', password='Password123!')
        uploaded = SimpleUploadedFile('retina_scan.jpg', self.valid_image_bytes, 'image/jpeg')

        response = self.client.post(reverse('upload'), {
            'image': uploaded,
            'notes': 'OD Retinal Screening'
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        record = PredictionRecord.objects.filter(user=self.user1, notes='OD Retinal Screening').first()
        self.assertIsNotNone(record)
        self.assertEqual(record.status, 'COMPLETED')
        self.assertIn(record.predicted_class, ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR'])
        self.assertGreater(record.confidence, 0.0)

        # Verify all five classes exist in probabilities dict
        expected_classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']
        for cls_name in expected_classes:
            self.assertIn(cls_name, record.probabilities)
            self.assertGreaterEqual(record.probabilities[cls_name], 0.0)
