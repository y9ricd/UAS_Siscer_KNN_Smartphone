"""
View Django untuk antarmuka simulasi harga smartphone, evaluasi model KNN,
serta endpoint JSON `/api/predict/` untuk prediksi real-time.
"""

import json

from django.http import JsonResponse'
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from .services import (
    FITUR_UTAMA,
    LABEL_FITUR,
    PRESET_SPESIFIKASI,
    SATUAN_FITUR,
    ambil_rentang_fitur,
    muat_metrik,
    muat_model,
    prediksi_kategori_harga,
    siapkan_data_evaluasi,
    validasi_input_spesifikasi,
)

# Muat model & metrik sekali pada level modul saat aplikasi mulai berjalan
# agar tidak terjadi pembacaan disk berulang di setiap request HTTP
try:
    MODEL_KNN = muat_model()
    METRIK_RINGKAS = muat_metrik()
except FileNotFoundError:
    MODEL_KNN = None
    METRIK_RINGKAS = None


def _susun_daftar_slider(rentang: dict, nilai_awal: dict) -> list[dict]:
    """Menyusun metadata 6 slider spesifikasi hardware untuk dirender di template."""
    konfigurasi_step = {
        "ram": 16,
        "battery_power": 10,
        "int_memory": 1,
        "px_width": 10,
        "px_height": 10,
        "pc": 1,
    }
    daftar = []
    for fitur in FITUR_UTAMA:
        batas = rentang[fitur]
        daftar.append(
            {
                "name": fitur,
                "label": f"{LABEL_FITUR[fitur]} ({SATUAN_FITUR[fitur]})",
                "unit": SATUAN_FITUR[fitur],
                "min": int(batas["min"]),
                "max": int(batas["max"]),
                "step": konfigurasi_step[fitur],
                "value": int(nilai_awal[fitur]),
            }
        )
    return daftar


@ensure_csrf_cookie
@require_GET
def beranda(request):
    """Menampilkan halaman utama dengan Tab Simulasi Harga dan Tab Evaluasi Model."""
    error_sistem = None
    hasil_awal = None
    data_evaluasi = None

    rentang = ambil_rentang_fitur()
    nilai_awal = dict(PRESET_SPESIFIKASI["Mid-Range"])

    try:
        hasil_awal = prediksi_kategori_harga(nilai_awal)
        data_evaluasi = siapkan_data_evaluasi()
    except FileNotFoundError as exc:
        error_sistem = str(exc)

    tab_aktif = request.GET.get("tab", "simulasi")
    if tab_aktif not in ("simulasi", "evaluasi"):
        tab_aktif = "simulasi"

    konteks = {
        "daftar_slider": _susun_daftar_slider(rentang, nilai_awal),
        "preset_json": json.dumps(PRESET_SPESIFIKASI),
        "nilai_reset_json": json.dumps({k: int(v["median"]) for k, v in rentang.items()}),
        "hasil_awal": hasil_awal,
        "hasil_awal_json": json.dumps(hasil_awal) if hasil_awal else "null",
        "data_evaluasi": data_evaluasi,
        "tab_aktif": tab_aktif,
        "error_sistem": error_sistem,
    }
    return render(request, "klasifikasi/index.html", konteks)


@require_POST
def api_predict(request):
    """
    Endpoint JSON `POST /api/predict/` untuk prediksi real-time tanpa reload halaman.
    Melakukan validasi tipe numerik serta batas min-max dan mengembalikan HTTP 400 jika tidak valid.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse(
            {
                "error": "Body request harus berupa JSON yang valid.",
                "detail": {"_general": "JSON tidak dapat diparsing."},
            },
            status=400,
        )

    data_bersih, errors = validasi_input_spesifikasi(payload)
    if errors:
        return JsonResponse(
            {
                "error": "Input spesifikasi tidak valid atau berada di luar rentang dataset.",
                "detail": errors,
            },
            status=400,
        )

    try:
        hasil = prediksi_kategori_harga(data_bersih)
    except FileNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=503)

    return JsonResponse(hasil, status=200)


@require_GET
def evaluasi_json(request):
    """Menyediakan metrik evaluasi aktual dari metrics_summary.json dalam format JSON."""
    try:
        data = siapkan_data_evaluasi()
        return JsonResponse(data, status=200)
    except FileNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=503)


@require_GET
def health_check(request):
    """Endpoint ringan untuk mengecek kesiapan server Django dan artefak model KNN."""
    try:
        evaluasi = siapkan_data_evaluasi()
        return JsonResponse(
            {
                "status": "ok",
                "best_k": evaluasi["best_k"],
                "test_accuracy": evaluasi["test_accuracy"],
            },
            status=200,
        )
    except FileNotFoundError as exc:
        return JsonResponse({"status": "error", "detail": str(exc)}, status=503)
