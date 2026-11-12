"""
Database Models for Diabetic Retinopathy Detection System.
"""

from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    """
    Profile extension for Django User model.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    phone_number = models.CharField(max_length=20, blank=True)
    organization_or_hospital = models.CharField(max_length=150, blank=True)
    role_title = models.CharField(max_length=100, default='Healthcare Researcher', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} Profile"


@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)
    else:
        if hasattr(instance, 'profile'):
            instance.profile.save()


class PredictionRecord(models.Model):
    """
    Stores retinal fundus scan uploads and CNN prediction results.
    """
    CLASS_CHOICES = [
        ('No_DR', 'No Diabetic Retinopathy (Grade 0)'),
        ('Mild', 'Mild NPDR (Grade 1)'),
        ('Moderate', 'Moderate NPDR (Grade 2)'),
        ('Severe', 'Severe NPDR (Grade 3)'),
        ('Proliferative_DR', 'Proliferative DR (Grade 4)'),
    ]

    STATUS_CHOICES = [
        ('COMPLETED', 'Completed'),
        ('PENDING', 'Pending'),
        ('FAILED', 'Failed'),
    ]

    SEVERITY_LEVELS = [
        ('Normal', 'Normal'),
        ('Low', 'Low Risk'),
        ('Moderate', 'Moderate Risk'),
        ('High', 'High Risk'),
        ('Critical', 'Critical / Urgent'),
    ]

    SOURCE_CHOICES = [
        ('UPLOAD', 'Direct Retinal Upload'),
        ('CAMERA', 'Fundus Camera Capture'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    image = models.ImageField(upload_to='retinal_images/')
    capture_source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default='UPLOAD')
    is_fundus_validated = models.BooleanField(default=True, help_text="Verified as genuine retinal fundus image")
    validation_details = models.JSONField(default=dict, blank=True, help_text="Bio-optical validation metrics")
    preprocessed_image = models.ImageField(upload_to='preprocessed_images/', blank=True, null=True, help_text="Enhanced preprocessed scan (CLAHE/Green channel)")
    predicted_class = models.CharField(max_length=50, choices=CLASS_CHOICES, default='No_DR', db_index=True)
    confidence = models.FloatField(default=0.0, help_text="Model confidence probability between 0.0 and 1.0")
    probabilities = models.JSONField(default=dict, blank=True, help_text="Probabilities for each of the 5 classes")
    severity_level = models.CharField(max_length=20, choices=SEVERITY_LEVELS, default='Normal')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='COMPLETED')
    notes = models.TextField(blank=True, null=True, help_text="Clinical notes or user observations")
    error_message = models.TextField(blank=True, null=True, help_text="Diagnostic error details if failed")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Prediction Record"
        verbose_name_plural = "Prediction Records"
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['predicted_class']),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.get_predicted_class_display()} ({self.confidence_percentage})"

    @property
    def confidence_percentage(self):
        """Returns confidence formatted as a percentage string."""
        return f"{self.confidence * 100:.1f}%"

    @property
    def badge_class(self):
        """Returns a Bootstrap color badge class corresponding to severity."""
        mapping = {
            'No_DR': 'bg-success',
            'Mild': 'bg-info text-dark',
            'Moderate': 'bg-warning text-dark',
            'Severe': 'bg-orange text-white',
            'Proliferative_DR': 'bg-danger text-white',
        }
        return mapping.get(self.predicted_class, 'bg-secondary')

    @property
    def human_description(self):
        """Clinical stage description for patient/clinician decision support."""
        descriptions = {
            'No_DR': 'No apparent signs of diabetic retinopathy. Retinal blood vessels, macula, and optic disc appear normal with no microaneurysms detected.',
            'Mild': 'Microaneurysms only. Earliest clinically observable stage of non-proliferative diabetic retinopathy (NPDR). Small balloon-like swellings in fine retinal capillaries.',
            'Moderate': 'More than microaneurysms but less than severe NPDR. Retinal microaneurysms, dot-and-blot hemorrhages, venous beading, and soft exudates (cotton wool spots) may be present.',
            'Severe': 'Extensive intraretinal hemorrhages in all four quadrants, definite venous beading in 2+ quadrants, or prominent IRMA in 1+ quadrant (4:2:1 rule). High risk of progression.',
            'Proliferative_DR': 'Neovascularization (growth of abnormal, fragile new blood vessels) or preretinal/vitreous hemorrhage. Advanced stage requiring immediate ophthalmological intervention.',
        }
        return descriptions.get(self.predicted_class, 'Clinical evaluation pending.')

    @property
    def clinical_recommendation(self):
        """Clinical recommendation guidance (decision-support prototype)."""
        recommendations = {
            'No_DR': 'Continue routine annual dilated eye examination and maintain optimal glycemic, blood pressure, and lipid control.',
            'Mild': 'Repeat dilated fundus examination in 9 to 12 months. Reinforce diabetes self-management and blood sugar monitoring.',
            'Moderate': 'Referral to an ophthalmologist / retinal specialist for formal assessment within 3 to 6 months. Closer monitoring warranted.',
            'Severe': 'Prompt referral to a vitreoretinal specialist within 2 to 4 weeks. Evaluation for panretinal photocoagulation or anti-VEGF therapy.',
            'Proliferative_DR': 'Urgent ophthalmological referral within 24 to 48 hours for immediate retinal assessment, imaging, and intervention.',
        }
        return recommendations.get(self.predicted_class, 'Consult a qualified ophthalmologist for in-person examination.')


class ContactMessage(models.Model):
    """
    Stores contact messages submitted through the contact page.
    """
    name = models.CharField(max_length=100)
    email = models.EmailField()
    subject = models.CharField(max_length=200)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_resolved = models.BooleanField(default=False)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Contact Message"
        verbose_name_plural = "Contact Messages"

    def __str__(self):
        return f"Message from {self.name} - {self.subject}"
