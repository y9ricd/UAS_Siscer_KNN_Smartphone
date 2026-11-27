"""
View Django untuk menampilkan antarmuka prediksi harga smartphone dan evaluasi model KNN.
"""

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_http_methods

from .forms import PrediksiSmartphoneForm
from .services import (
    ambil_rentang_fitur,
    prediksi_kategori_harga,
    siapkan_data_evaluasi,
)


@require_http_methods(["GET", "POST"])
def beranda(request):
    """Menampilkan halaman utama dengan Tab Prediksi Harga dan Tab Evaluasi Model."""
    hasil_prediksi = None
    error_sistem = None
    tab_aktif = request.GET.get("tab", "prediksi")
    if tab_aktif not in ("prediksi", "evaluasi"):
        tab_aktif = "prediksi"

    rentang = ambil_rentang_fitur()
    nilai_awal = {k: int(v["median"]) for k, v in rentang.items()}

    if request.method == "POST":
        tab_aktif = "prediksi"
        form = PrediksiSmartphoneForm(request.POST)
        if form.is_valid():
            try:
                hasil_prediksi = prediksi_kategori_harga(form.cleaned_data)
            except FileNotFoundError as exc:
                error_sistem = str(exc)
    else:
        form = PrediksiSmartphoneForm(initial=nilai_awal)

    try:
        data_evaluasi = siapkan_data_evaluasi()
    except FileNotFoundError as exc:
        data_evaluasi = None
        error_sistem = str(exc)

    # Susun metadata slider agar mudah dirender rapi dalam dua kolom
    daftar_slider = []
    for field in form:
        nilai_saat_ini = field.value()
        if nilai_saat_ini is None or nilai_saat_ini == "":
            nilai_saat_ini = field.field.initial
        daftar_slider.append(
            {
                "name": field.name,
                "id_for_label": field.id_for_label,
                "label": field.label,
                "help_text": field.help_text,
                "errors": field.errors,
                "value": nilai_saat_ini,
                "min": field.field.min_value,
                "max": field.field.max_value,
                "step": field.field.widget.attrs.get("step", 1),
                "unit": field.field.widget.attrs.get("data-unit", ""),
                "group": field.field.widget.attrs.get("data-group", ""),
            }
        )

    konteks = {
        "form": form,
        "slider_performa": [
            s for s in daftar_slider if s["group"] == "Performa & Penyimpanan"
        ],
        "slider_layar": [
            s for s in daftar_slider if s["group"] == "Layar & Kamera"
        ],
        "hasil_prediksi": hasil_prediksi,
        "data_evaluasi": data_evaluasi,
        "tab_aktif": tab_aktif,
        "error_sistem": error_sistem,
    }
    return render(request, "klasifikasi/index.html", konteks)


@require_GET
def health_check(request):
    """Endpoint ringan untuk memeriksa kesiapan server Django dan artefak model."""
    try:
        evaluasi = siapkan_data_evaluasi()
        return JsonResponse(
            {
                "status": "ok",
                "best_k": evaluasi["best_k"],
                "test_accuracy": evaluasi["test_accuracy"],
            }
        )
    except FileNotFoundError as exc:
        return JsonResponse({"status": "error", "detail": str(exc)}, status=503)
