"""
Pengaturan proyek Django untuk Sistem Klasifikasi Rentang Harga Smartphone (KNN).
Siap untuk pengembangan lokal maupun deployment produksi dengan environment variables.
"""

import os
from pathlib import Path

# Direktori root proyek (tempat mobile_price.csv, model, dan manage.py berada)
BASE_DIR = Path(__file__).resolve().parent.parent

# Baca SECRET_KEY dari environment variable; fallback hanya untuk pengembangan lokal
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-local-dev-only-uas-siscer-knn-smartphone",
)

# Kontrol mode DEBUG lewat environment variable (default True di lokal)
DEBUG = os.environ.get("DJANGO_DEBUG", "True").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Gabungkan host lokal dengan host produksi dari environment variable DJANGO_ALLOWED_HOSTS
_env_hosts = os.environ.get("DJANGO_ALLOWED_HOSTS", "")
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"] + [
    host.strip() for host in _env_hosts.split(",") if host.strip()
]

# Daftar aplikasi minimal agar tidak memerlukan migrasi basis data saat dijalankan
INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "klasifikasi",
]

# WhiteNoise ditempatkan tepat setelah SecurityMiddleware untuk menyajikan file statis saat deploy
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
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

# Konfigurasi file statis untuk lokal dan produksi (collectstatic + WhiteNoise)
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedStaticFilesStorage",
    },
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
