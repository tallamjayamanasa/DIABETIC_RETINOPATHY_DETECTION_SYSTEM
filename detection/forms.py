"""
Forms for Diabetic Retinopathy Detection System.
"""

from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm, PasswordChangeForm
from django.core.exceptions import ValidationError
from PIL import Image
import io

from .models import PredictionRecord, UserProfile, ContactMessage


class UserRegistrationForm(forms.ModelForm):
    """
    User registration form with full name, username, email, and password confirmation.
    """
    first_name = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. John',
            'autocomplete': 'given-name'
        })
    )
    last_name = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. Doe',
            'autocomplete': 'family-name'
        })
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'name@example.com',
            'autocomplete': 'email'
        })
    )
    username = forms.CharField(
        max_length=150,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Choose a username',
            'autocomplete': 'username'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter a secure password (min. 8 characters)',
            'autocomplete': 'new-password',
            'id': 'register-password'
        }),
        min_length=8
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm your password',
            'autocomplete': 'new-password',
            'id': 'register-confirm-password'
        })
    )

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'username', 'email']

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username__iexact=username).exists():
            raise ValidationError("This username is already taken. Please choose a different one.")
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email__iexact=email).exists():
            raise ValidationError("An account with this email address already exists.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')

        if password and confirm_password and password != confirm_password:
            self.add_error('confirm_password', "Passwords do not match. Please re-enter.")
        return cleaned_data


class UserLoginForm(forms.Form):
    """
    Login form supporting username or email.
    """
    login_id = forms.CharField(
        label="Username or Email",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your username or email',
            'autocomplete': 'username',
            'autofocus': True
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password',
            'autocomplete': 'current-password',
            'id': 'login-password'
        })
    )


class UserProfileUpdateForm(forms.ModelForm):
    """
    Form for updating basic user information.
    """
    first_name = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    last_name = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

    def __init__(self, *args, **kwargs):
        self.user_instance = kwargs.get('instance')
        super().__init__(*args, **kwargs)

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if self.user_instance:
            if User.objects.filter(email__iexact=email).exclude(pk=self.user_instance.pk).exists():
                raise ValidationError("This email is already in use by another account.")
        return email


class ProfileDetailsForm(forms.ModelForm):
    """
    Form for updating extended profile details.
    """
    phone_number = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '+1 (555) 000-0000'})
    )
    organization_or_hospital = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Apex Eye Hospital / University'})
    )
    role_title = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Clinical Ophthalmologist / Researcher'})
    )

    class Meta:
        model = UserProfile
        fields = ['phone_number', 'organization_or_hospital', 'role_title']


class RetinalImageUploadForm(forms.ModelForm):
    """
    Upload form with strict validation for retinal fundus photographs.
    """
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Optional: Enter patient ID, eye side (OD / OS), or clinical context...'
        })
    )

    class Meta:
        model = PredictionRecord
        fields = ['image', 'notes']
        widgets = {
            'image': forms.ClearableFileInput(attrs={
                'class': 'form-control visually-hidden',
                'id': 'retinalImageInput',
                'accept': '.jpg,.jpeg,.png,image/jpeg,image/png'
            })
        }

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if not image:
            raise ValidationError("Please select or upload a retinal fundus image.")

        # 1. Size Validation: Max 15 MB
        max_size = 15 * 1024 * 1024
        if image.size > max_size:
            raise ValidationError(f"File size exceeds 15 MB limit ({image.size / (1024 * 1024):.1f} MB uploaded).")

        # 2. Extension check
        allowed_extensions = ('.jpg', '.jpeg', '.png')
        if not image.name.lower().endswith(allowed_extensions):
            raise ValidationError("Invalid file format. Only JPG, JPEG, and PNG images are supported.")

        # 3. Readability & Integrity verification via Pillow
        try:
            image_content = image.read()
            image_stream = io.BytesIO(image_content)
            pil_image = Image.open(image_stream)
            pil_image.verify()  # Verifies file integrity

            # Re-open after verify() closes/invalidates the stream
            image_stream.seek(0)
            pil_image = Image.open(image_stream)

            # Check dimensions
            width, height = pil_image.size
            if width < 64 or height < 64:
                raise ValidationError("Image dimensions are too small (minimum 64x64 pixels required for analysis).")

            # Reset pointer for Django storage
            image.seek(0)
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError("The uploaded file is not a valid, readable image file or is corrupted.")

        return image


class ContactForm(forms.ModelForm):
    """
    Contact us form.
    """
    class Meta:
        model = ContactMessage
        fields = ['name', 'email', 'subject', 'message']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Your Full Name'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Your Email Address'}),
            'subject': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Subject of your inquiry'}),
            'message': forms.Textarea(attrs={'class': 'form-control', 'rows': 5, 'placeholder': 'Write your message or inquiry here...'}),
        }
