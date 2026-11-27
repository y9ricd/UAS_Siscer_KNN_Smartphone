#!/usr/bin/env python
"""Utilitas baris perintah Django untuk proyek Klasifikasi Harga Smartphone (KNN)."""

import os
import sys


def main():
    """Menjalankan perintah administratif Django."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "smartphone_web.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django tidak ditemukan. Jalankan `pip install -r requirements.txt` terlebih dahulu."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
