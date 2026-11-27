"""Routing URL untuk aplikasi klasifikasi."""

from django.urls import path

from . import views

urlpatterns = [
    path("", views.beranda, name="beranda"),
    path("health/", views.health_check, name="health_check"),
]
