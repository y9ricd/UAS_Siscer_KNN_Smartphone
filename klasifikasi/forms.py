"""
Formulir Django untuk memvalidasi 6 input spesifikasi hardware smartphone.
"""

from django import forms
from django.core.validators import MaxValueValidator, MinValueValidator

from .services import ambil_rentang_fitur


class PrediksiSmartphoneForm(forms.Form):
    """Validasi batas min-max dataset untuk 6 fitur hardware utama."""

    ram = forms.IntegerField(label="RAM (MB)")
    battery_power = forms.IntegerField(label="Kapasitas Baterai (mAh)")
    int_memory = forms.IntegerField(label="Memori Internal (GB)")
    px_width = forms.IntegerField(label="Lebar Resolusi Layar (px)")
    px_height = forms.IntegerField(label="Tinggi Resolusi Layar (px)")
    pc = forms.IntegerField(label="Kamera Utama (MP)")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rentang = ambil_rentang_fitur()

        # Terapkan min, max, dan nilai awal (median) sesuai dataset mobile_price.csv
        konfigurasi = {
            "ram": ("MB", 16, "Performa & Penyimpanan"),
            "battery_power": ("mAh", 10, "Performa & Penyimpanan"),
            "int_memory": ("GB", 1, "Performa & Penyimpanan"),
            "px_width": ("px", 10, "Layar & Kamera"),
            "px_height": ("px", 10, "Layar & Kamera"),
            "pc": ("MP", 1, "Layar & Kamera"),
        }

        for nama_field, (satuan, step, grup) in konfigurasi.items():
            batas = rentang[nama_field]
            field = self.fields[nama_field]
            field.min_value = int(batas["min"])
            field.max_value = int(batas["max"])
            field.initial = int(batas["median"])
            field.help_text = (
                f"Rentang dataset: {batas['min']} – {batas['max']} {satuan}"
            )
            field.widget = forms.NumberInput(
                attrs={
                    "type": "range",
                    "min": int(batas["min"]),
                    "max": int(batas["max"]),
                    "step": step,
                    "data-unit": satuan,
                    "data-group": grup,
                }
            )
            # Sinkronkan validator batas min dan max pada Django Form
            field.validators.clear()
            field.validators.append(MinValueValidator(int(batas["min"])))
            field.validators.append(MaxValueValidator(int(batas["max"])))
