"""Konfigurasi WSGI untuk deployment standar Django."""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "smartphone_web.settings")
application = get_wsgi_application()
