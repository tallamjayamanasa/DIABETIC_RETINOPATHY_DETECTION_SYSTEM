"""
Views for Diabetic Retinopathy Detection System.
Full-stack views handling authentication, dashboard, upload, inference, and reporting.
"""

import json
import base64
import io
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.models import User
from django.contrib.auth.forms import PasswordChangeForm
from django.core.paginator import Paginator
from django.core.files.base import ContentFile
from django.db.models import Count, Q
from django.http import JsonResponse, Http404

from .models import PredictionRecord, UserProfile, ContactMessage
from .forms import (
    UserRegistrationForm,
    UserLoginForm,
    UserProfileUpdateForm,
    ProfileDetailsForm,
    RetinalImageUploadForm,
    ContactForm,
)
from .prediction import predict_retinopathy, is_model_available
from .fundus_validator import FundusValidator
from model.preprocess import generate_fundus_previews


# =====================================================================
# PUBLIC VIEWS
# =====================================================================

def home(request):
    """
    Public landing page with healthcare AI theme, hero section,
    clinical stages breakdown, and tech stack details.
    """
    total_predictions = PredictionRecord.objects.filter(status='COMPLETED').count()
    total_users = User.objects.count()
    model_ready = is_model_available()

    stages_info = [
        {
            'grade': 'Grade 0',
            'name': 'No Diabetic Retinopathy',
            'badge': 'bg-success',
            'desc': 'No clinical abnormalities. Blood vessels are intact with no microaneurysms or hemorrhages.',
            'recommendation': 'Annual dilated retinal screening.'
        },
        {
            'grade': 'Grade 1',
            'name': 'Mild NPDR',
            'badge': 'bg-info text-dark',
            'desc': 'Presence of microaneurysms only. Earliest stage of capillary dilation.',
            'recommendation': 'Follow-up exam within 9-12 months.'
        },
        {
            'grade': 'Grade 2',
            'name': 'Moderate NPDR',
            'badge': 'bg-warning text-dark',
            'desc': 'Microaneurysms, blot hemorrhages, venous beading, or cotton wool spots.',
            'recommendation': 'Ophthalmological referral within 3-6 months.'
        },
        {
            'grade': 'Grade 3',
            'name': 'Severe NPDR',
            'badge': 'bg-orange text-white',
            'desc': 'Extensive hemorrhages in 4 quadrants or prominent IRMA (4:2:1 rule). High progression risk.',
            'recommendation': 'Retinal specialist consultation in 2-4 weeks.'
        },
        {
            'grade': 'Grade 4',
            'name': 'Proliferative DR',
            'badge': 'bg-danger text-white',
            'desc': 'Neovascularization or vitreous/preretinal hemorrhage. High blindness risk without intervention.',
            'recommendation': 'Urgent ophthalmology referral within 24-48 hours.'
        },
    ]

    context = {
        'total_predictions': total_predictions,
        'total_users': total_users,
        'model_ready': model_ready,
        'stages_info': stages_info,
    }
    return render(request, 'home.html', context)


def about(request):
    """
    Detailed about page explaining problem statement, CNN architecture,
    dataset methodology, and clinical decision support objectives.
    """
    return render(request, 'about.html')


def contact(request):
    """
    Contact page with validation and persistent message recording.
    """
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(
                request,
                "Thank you for contacting us! Your inquiry has been received. Our clinical/technical team will review it shortly."
            )
            return redirect('contact')
        else:
            messages.error(request, "Please correct the errors in the contact form below.")
    else:
        form = ContactForm()

    return render(request, 'contact.html', {'form': form})


# =====================================================================
# AUTHENTICATION VIEWS
# =====================================================================

def register_view(request):
    """
    User registration view.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            login(request, user)
            messages.success(request, f"Welcome to the platform, {user.first_name or user.username}! Your account has been created.")
            return redirect('dashboard')
        else:
            messages.error(request, "Please resolve the errors highlighted below.")
    else:
        form = UserRegistrationForm()

    return render(request, 'accounts/register.html', {'form': form})


def login_view(request):
    """
    User login view supporting either username or email identifier.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')

    next_url = request.GET.get('next', 'dashboard')

    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            login_id = form.cleaned_data['login_id'].strip()
            password = form.cleaned_data['password']

            # Check if login_id matches email or username
            user = authenticate(request, username=login_id, password=password)
            if user is None:
                try:
                    user_obj = User.objects.get(email__iexact=login_id)
                    user = authenticate(request, username=user_obj.username, password=password)
                except User.DoesNotExist:
                    user = None

            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {user.first_name or user.username}!")
                return redirect(next_url if next_url else 'dashboard')
            else:
                messages.error(request, "Invalid username/email or password. Please try again.")
    else:
        form = UserLoginForm()

    return render(request, 'accounts/login.html', {'form': form})


def logout_view(request):
    """
    Secure logout view.
    """
    logout(request)
    messages.info(request, "You have been securely signed out.")
    return redirect('home')


@login_required
def profile_view(request):
    """
    User profile management (details, hospital affiliation, password change).
    """
    user = request.user
    profile, _ = UserProfile.objects.get_or_create(user=user)

    user_form = UserProfileUpdateForm(instance=user)
    profile_form = ProfileDetailsForm(instance=profile)
    password_form = PasswordChangeForm(user=user)

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'update_profile':
            user_form = UserProfileUpdateForm(request.POST, instance=user)
            profile_form = ProfileDetailsForm(request.POST, instance=profile)
            if user_form.is_valid() and profile_form.is_valid():
                user_form.save()
                profile_form.save()
                messages.success(request, "Your profile details have been successfully updated.")
                return redirect('profile')
            else:
                messages.error(request, "Error updating profile. Please check form fields.")

        elif action == 'change_password':
            password_form = PasswordChangeForm(user=user, data=request.POST)
            if password_form.is_valid():
                password_form.save()
                messages.success(request, "Your password has been changed successfully. Please log in with your new password.")
                return redirect('login')
            else:
                messages.error(request, "Password change failed. Please review the criteria below.")

    context = {
        'user_form': user_form,
        'profile_form': profile_form,
        'password_form': password_form,
        'profile': profile,
    }
    return render(request, 'accounts/profile.html', context)


# =====================================================================
# DASHBOARD & PREDICTION VIEWS
# =====================================================================

@login_required
def dashboard_view(request):
    """
    Personalized clinical dashboard for the authenticated user.
    Displays completed, valid retinal fundus analysis records.
    """
    user_records = PredictionRecord.objects.filter(user=request.user, status='COMPLETED')
    total_uploads = user_records.count()
    completed_predictions = total_uploads

    # Stage counts for user
    normal_count = user_records.filter(predicted_class='No_DR').count()
    dr_detected_count = user_records.exclude(predicted_class='No_DR').count()

    # Recent predictions (latest 5)
    recent_predictions = user_records[:5]
    latest_prediction = user_records.first()

    context = {
        'total_uploads': total_uploads,
        'completed_predictions': completed_predictions,
        'normal_count': normal_count,
        'dr_detected_count': dr_detected_count,
        'recent_predictions': recent_predictions,
        'latest_prediction': latest_prediction,
        'model_ready': is_model_available(),
    }
    return render(request, 'dashboard/dashboard.html', context)


@login_required
def upload_view(request):
    """
    Retinal fundus image upload and CNN prediction execution.
    Validates genuine retinal fundus image before CNN inference.
    """
    if request.method == 'POST':
        form = RetinalImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.cleaned_data['image']

            # Step 1: Validate genuine retinal fundus image
            validation = FundusValidator.validate_image(uploaded_image)
            if not validation['is_valid']:
                messages.error(
                    request,
                    "Invalid Image. Please upload a valid retinal fundus image."
                )
                return render(request, 'prediction/upload.html', {
                    'form': form,
                    'model_ready': is_model_available(),
                    'validation_error': "Invalid Image. Please upload a valid retinal fundus image."
                })

            # Step 2: Create prediction record associated with current user
            record = form.save(commit=False)
            record.user = request.user
            record.capture_source = 'UPLOAD'
            record.is_fundus_validated = True
            record.validation_details = validation['details']
            record.status = 'PENDING'
            record.confidence = 0.0
            record.predicted_class = 'No_DR'
            record.save()

            # Step 3: Archive CLAHE preprocessed scan
            try:
                preview_data = generate_fundus_previews(record.image.path)
                clahe_pil = preview_data['clahe_pil']
                clahe_buf = io.BytesIO()
                clahe_pil.save(clahe_buf, format='JPEG', quality=95)
                clahe_buf.seek(0)
                record.preprocessed_image.save(
                    f"preprocessed_{record.pk}.jpg",
                    ContentFile(clahe_buf.getvalue()),
                    save=False
                )
            except Exception as e:
                print(f"[!] Preprocessing archiving notice: {e}")

            # Step 4: Execute CNN prediction on uploaded image
            prediction_result = predict_retinopathy(record.image.path)

            if prediction_result.get('status') == 'SUCCESS':
                record.predicted_class = prediction_result['predicted_class']
                record.confidence = prediction_result['confidence']
                record.probabilities = prediction_result['probabilities']
                record.severity_level = prediction_result['severity_level']
                record.status = 'COMPLETED'
                record.save()
                messages.success(request, f"Retinal analysis complete: {record.get_predicted_class_display()}")
                return redirect('result', pk=record.pk)
            else:
                record.status = 'FAILED'
                record.error_message = prediction_result.get('error', 'Unknown inference error')
                record.save()
                messages.warning(
                    request,
                    f"Scan uploaded, but model inference encountered an issue: {record.error_message}"
                )
                return redirect('result', pk=record.pk)
        else:
            messages.error(request, "Upload failed. Please ensure the file is a valid image under 15 MB.")
    else:
        form = RetinalImageUploadForm()

    context = {
        'form': form,
        'model_ready': is_model_available(),
    }
    return render(request, 'prediction/upload.html', context)


@login_required
def result_view(request, pk):
    """
    Detailed prediction result view for a specific scan.
    Guarantees user ownership isolation.
    """
    record = get_object_or_404(PredictionRecord, pk=pk)

    # Restrict viewing to the owner or staff members
    if record.user != request.user and not request.user.is_staff:
        raise Http404("Prediction record not found.")

    probabilities_json = json.dumps(record.probabilities or {})

    context = {
        'record': record,
        'probabilities_json': probabilities_json,
    }
    return render(request, 'prediction/result.html', context)


@login_required
def history_view(request):
    """
    List past valid prediction records for the current user with pagination and filtering.
    """
    records_qs = PredictionRecord.objects.filter(user=request.user, status='COMPLETED')

    # Filter by classification if requested
    stage_filter = request.GET.get('stage')
    if stage_filter and stage_filter in [c[0] for c in PredictionRecord.CLASS_CHOICES]:
        records_qs = records_qs.filter(predicted_class=stage_filter)

    # Search notes or date
    search_query = request.GET.get('q')
    if search_query:
        records_qs = records_qs.filter(
            Q(notes__icontains=search_query) |
            Q(predicted_class__icontains=search_query)
        )

    paginator = Paginator(records_qs, 9)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'stage_filter': stage_filter,
        'search_query': search_query,
        'class_choices': PredictionRecord.CLASS_CHOICES,
    }
    return render(request, 'prediction/history.html', context)


@login_required
def delete_prediction_view(request, pk):
    """
    Secure deletion of a prediction record by its owner.
    """
    record = get_object_or_404(PredictionRecord, pk=pk, user=request.user)
    
    if request.method == 'POST':
        # Safely delete image file from storage if it exists
        try:
            if record.image and hasattr(record.image, 'path'):
                import os
                if os.path.isfile(record.image.path):
                    os.remove(record.image.path)
        except Exception:
            pass

        record.delete()
        messages.success(request, "Prediction record and associated scan deleted successfully.")
    
    return redirect('history')


# =====================================================================
# CUSTOM ADMIN PANEL VIEWS (Staff/Superuser Only)
# =====================================================================

@staff_member_required
def admin_dashboard_view(request):
    """
    Administrative analytics dashboard for doctors / project evaluators.
    """
    total_users = User.objects.count()
    total_predictions = PredictionRecord.objects.count()
    completed_predictions = PredictionRecord.objects.filter(status='COMPLETED').count()

    # Breakdown by DR Class
    class_distribution = {}
    for code, label in PredictionRecord.CLASS_CHOICES:
        cnt = PredictionRecord.objects.filter(predicted_class=code, status='COMPLETED').count()
        class_distribution[code] = {
            'label': label,
            'count': cnt
        }

    # Chart data
    chart_labels = [c[1].split(' (')[0] for c in PredictionRecord.CLASS_CHOICES]
    chart_data = [PredictionRecord.objects.filter(predicted_class=c[0], status='COMPLETED').count() for c in PredictionRecord.CLASS_CHOICES]

    recent_predictions = PredictionRecord.objects.select_related('user').all()[:10]
    recent_users = User.objects.order_by('-date_joined')[:5]

    context = {
        'total_users': total_users,
        'total_predictions': total_predictions,
        'completed_predictions': completed_predictions,
        'class_distribution': class_distribution,
        'chart_labels_json': json.dumps(chart_labels),
        'chart_data_json': json.dumps(chart_data),
        'recent_predictions': recent_predictions,
        'recent_users': recent_users,
        'model_ready': is_model_available(),
    }
    return render(request, 'admin_panel/dashboard.html', context)


@staff_member_required
def admin_users_view(request):
    """
    User management overview in staff admin panel.
    """
    users_qs = User.objects.annotate(
        prediction_count=Count('predictions')
    ).order_by('-date_joined')

    search_query = request.GET.get('q')
    if search_query:
        users_qs = users_qs.filter(
            Q(username__icontains=search_query) |
            Q(email__icontains=search_query) |
            Q(first_name__icontains=search_query) |
            Q(last_name__icontains=search_query)
        )

    paginator = Paginator(users_qs, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'search_query': search_query,
    }
    return render(request, 'admin_panel/users.html', context)


@staff_member_required
def admin_predictions_view(request):
    """
    Global predictions log in staff admin panel.
    """
    records_qs = PredictionRecord.objects.select_related('user').all().order_by('-created_at')

    stage_filter = request.GET.get('stage')
    if stage_filter:
        records_qs = records_qs.filter(predicted_class=stage_filter)

    search_query = request.GET.get('q')
    if search_query:
        records_qs = records_qs.filter(
            Q(user__username__icontains=search_query) |
            Q(notes__icontains=search_query)
        )

    paginator = Paginator(records_qs, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'stage_filter': stage_filter,
        'search_query': search_query,
        'class_choices': PredictionRecord.CLASS_CHOICES,
    }
    return render(request, 'admin_panel/predictions.html', context)
