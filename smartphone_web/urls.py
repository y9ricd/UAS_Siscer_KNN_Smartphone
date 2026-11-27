"""Konfigurasi URL utama proyek Django."""

from django.urls import include, path

urlpatterns = [
    path("", include("klasifikasi.urls")),
]
