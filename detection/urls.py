"""
URL Configuration for detection app.
"""

from django.urls import path
from . import views

urlpatterns = [
    # Public pages
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),

    # Authentication
    path('accounts/register/', views.register_view, name='register'),
    path('accounts/login/', views.login_view, name='login'),
    path('accounts/logout/', views.logout_view, name='logout'),
    path('accounts/profile/', views.profile_view, name='profile'),

    # Dashboard & Prediction
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('prediction/upload/', views.upload_view, name='upload'),
    path('prediction/result/<int:pk>/', views.result_view, name='result'),
    path('prediction/history/', views.history_view, name='history'),
    path('prediction/delete/<int:pk>/', views.delete_prediction_view, name='delete_prediction'),

    # Custom Staff Admin Panel
    path('admin-panel/', views.admin_dashboard_view, name='admin_dashboard'),
    path('admin-panel/users/', views.admin_users_view, name='admin_users'),
    path('admin-panel/predictions/', views.admin_predictions_view, name='admin_predictions'),
]
