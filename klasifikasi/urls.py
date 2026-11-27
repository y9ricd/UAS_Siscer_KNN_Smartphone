"""
Routing URL untuk aplikasi klasifikasi harga smartphone.
Menghubungkan halaman utama, endpoint prediksi JSON, evaluasi JSON, dan health check.
"""

from django.urls import path

from . import views

urlpatterns = [
    # Halaman utama (Tab Simulasi Harga & Tab Evaluasi Model)
    path("", views.beranda, name="beranda"),
    # Endpoint POST JSON untuk prediksi real-time dari slider via fetch()
    path("api/predict/", views.api_predict, name="api_predict"),
    # Endpoint GET JSON untuk membaca ringkasan evaluasi aktual model
    path("api/evaluasi/", views.evaluasi_json, name="api_evaluasi"),
    # Endpoint pemeriksaan kesehatan aplikasi saat deploy
    path("health/", views.health_check, name="health_check"),
]
