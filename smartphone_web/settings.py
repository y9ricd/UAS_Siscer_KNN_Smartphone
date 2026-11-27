"""
Pengaturan proyek Django untuk Sistem Klasifikasi Rentang Harga Smartphone (KNN).
"""

from pathlib import Path

# Direktori root proyek (tempat mobile_price.csv dan model_knn_smartphone.joblib berada)
BASE_DIR = Path(__file__).resolve().parent.parent

# Kunci rahasia untuk lingkungan pengembangan lokal / demo UAS
SECRET_KEY = "uas-sistem-cerdas-knn-smartphone-local-dev-key"

DEBUG = True

ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]

# Daftar aplikasi minimal agar tidak membutuhkan migrasi basis data saat demo
INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "klasifikasi",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "smartphone_web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "smartphone_web.wsgi.application"

# Basis data SQLite standar (dipakai oleh test runner Django bila diperlukan)
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

LANGUAGE_CODE = "id"
TIME_ZONE = "Asia/Jakarta"
USE_I18N = True
USE_TZ = True

# Konfigurasi berkas statis (CSS dan aset antarmuka)
STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
