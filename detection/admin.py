
"""
Django Admin Configuration for Diabetic Retinopathy Detection System.
"""

from django.contrib import admin
from django.utils.html import format_html
from .models import PredictionRecord, UserProfile, ContactMessage


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = (
        'user',
        'organization_or_hospital',
        'role_title',
        'phone_number',
        'created_at'
    )
    search_fields = (
        'user__username',
        'user__email',
        'organization_or_hospital',
        'role_title'
    )
    list_filter = ('role_title', 'created_at')


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):

    list_display = (
        'id',
        'user',
        'image_thumbnail',
        'predicted_class',
        'confidence_badge',
        'severity_level',
        'status',
        'created_at'
    )

    list_filter = (
        'predicted_class',
        'status',
        'severity_level',
        'created_at'
    )

    search_fields = ('user__username', 'user__email', 'notes')

    readonly_fields = (
        'image_preview',
        'confidence',
        'probabilities',
        'created_at',
        'updated_at'
    )

    def image_thumbnail(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="width:50px;height:50px;object-fit:cover;border-radius:6px;" />',
                obj.image.url
            )
        return "No Image"

    image_thumbnail.short_description = "Scan Preview"

    def image_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-width:320px;border-radius:8px;" />',
                obj.image.url
            )
        return "No Image"

    image_preview.short_description = "Retinal Scan"

    def confidence_badge(self, obj):
        try:
            confidence = float(obj.confidence or 0)
            pct = confidence * 100
            pct_text = f"{pct:.1f}"

            return format_html(
                '<span style="font-weight:bold;color:{};">{}%</span>',
                '#0d6efd' if pct >= 70 else '#6c757d',
                pct_text
            )
        except (ValueError, TypeError):
            return "0.0%"

    confidence_badge.short_description = "Confidence"


@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):

    list_display = (
        'name',
        'email',
        'subject',
        'created_at',
        'is_resolved'
    )

    list_filter = ('is_resolved', 'created_at')

    search_fields = (
        'name',
        'email',
        'subject',
        'message'
    )
