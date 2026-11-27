"""
Layanan Machine Learning untuk aplikasi Django.
Memuat model KNN terlatih dan ringkasan metrik dari berkas hasil train_model.py.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings

MODEL_PATH = Path(settings.BASE_DIR) / "model_knn_smartphone.joblib"
METRICS_PATH = Path(settings.BASE_DIR) / "metrics_summary.json"

# Urutan fitur wajib identik dengan train_model.py
FITUR_UTAMA = [
    "ram",
    "battery_power",
    "int_memory",
    "px_width",
    "px_height",
    "pc",
]

# Metadata label kelas dan kisaran harga ilustratif
INFO_KATEGORI = {
    0: {
        "nama": "Murah",
        "kisaran": "Sekitar Rp 1.000.000 - Rp 1.999.000",
        "deskripsi": "Segmen entry-level untuk kebutuhan komunikasi dan penggunaan ringan.",
        "badge_class": "badge-low",
    },
    1: {
        "nama": "Menengah",
        "kisaran": "Sekitar Rp 2.000.000 - Rp 3.999.000",
        "deskripsi": "Segmen mid-range dengan keseimbangan performa harian, baterai, dan layar.",
        "badge_class": "badge-mid",
    },
    2: {
        "nama": "Mahal",
        "kisaran": "Sekitar Rp 4.000.000 - Rp 6.999.000",
        "deskripsi": "Segmen menengah-atas dengan kapasitas memori dan resolusi layar lebih tinggi.",
        "badge_class": "badge-high",
    },
    3: {
        "nama": "Flagship",
        "kisaran": "Di atas Rp 7.000.000",
        "deskripsi": "Segmen atas dengan kombinasi spesifikasi hardware terbaik pada dataset.",
        "badge_class": "badge-flagship",
    },
}

# Batas bawaan apabila metrics_summary.json belum tersedia
BATAS_DEFAULT = {
    "ram": {"min": 256, "max": 3998, "median": 2146},
    "battery_power": {"min": 501, "max": 1998, "median": 1226},
    "int_memory": {"min": 2, "max": 64, "median": 32},
    "px_width": {"min": 500, "max": 1998, "median": 1247},
    "px_height": {"min": 0, "max": 1960, "median": 564},
    "pc": {"min": 0, "max": 20, "median": 10},
}


@lru_cache(maxsize=1)
def muat_model():
    """Memuat Pipeline model KNN dari file joblib (di-cache di memori)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "File model_knn_smartphone.joblib tidak ditemukan. Jalankan `python train_model.py`."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def muat_metrik() -> dict:
    """Memuat ringkasan metrik evaluasi dan rentang fitur dari file JSON."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "File metrics_summary.json tidak ditemukan. Jalankan `python train_model.py`."
        )
    with open(METRICS_PATH, "r", encoding="utf-8") as file_json:
        return json.load(file_json)


def ambil_rentang_fitur() -> dict:
    """Mengembalikan batas min, max, dan median seluruh 6 fitur."""
    try:
        metrik = muat_metrik()
        return metrik.get("feature_ranges", BATAS_DEFAULT)
    except FileNotFoundError:
        return BATAS_DEFAULT


def prediksi_kategori_harga(data_fitur: dict) -> dict:
    """Menjalankan prediksi kelas harga dan probabilitas 4 kelas dari input valid."""
    model = muat_model()
    baris = {kolom: int(data_fitur[kolom]) for kolom in FITUR_UTAMA}
    df_input = pd.DataFrame([baris], columns=FITUR_UTAMA)

    kelas = int(model.predict(df_input)[0])
    daftar_prob = model.predict_proba(df_input)[0]
    info = INFO_KATEGORI[kelas]

    probabilitas_kelas = []
    for idx in range(4):
        persen = round(float(daftar_prob[idx]) * 100, 1)
        probabilitas_kelas.append(
            {
                "kelas": idx,
                "nama": INFO_KATEGORI[idx]["nama"],
                "persen": persen,
                "terpilih": idx == kelas,
            }
        )

    return {
        "kelas": kelas,
        "nama": info["nama"],
        "kisaran": info["kisaran"],
        "deskripsi": info["deskripsi"],
        "badge_class": info["badge_class"],
        "keyakinan": round(float(daftar_prob[kelas]) * 100, 1),
        "probabilitas_kelas": probabilitas_kelas,
        "input_dipakai": baris,
    }


def siapkan_data_evaluasi() -> dict:
    """Memformat data metrik JSON agar siap ditampilkan pada tab Evaluasi Model."""
    metrik = muat_metrik()
    akurasi_latih = round(float(metrik["train_accuracy"]) * 100, 2)
    akurasi_uji = round(float(metrik["test_accuracy"]) * 100, 2)
    selisih = round(abs(akurasi_latih - akurasi_uji), 2)

    cm = metrik["confusion_matrix"]
    nilai_maks_cm = max(max(baris) for baris in cm) or 1
    label_kelas = [f"{i} ({INFO_KATEGORI[i]['nama']})" for i in range(4)]

    # Susun matriks sel dengan tingkat intensitas warna (0-4) untuk visualisasi heatmap
    baris_cm = []
    for idx_baris, baris in enumerate(cm):
        sel_baris = []
        for idx_kolom, nilai in enumerate(baris):
            rasio = nilai / nilai_maks_cm
            if nilai == 0:
                level = 0
            elif rasio < 0.25:
                level = 1
            elif rasio < 0.55:
                level = 2
            elif rasio < 0.85:
                level = 3
            else:
                level = 4
            sel_baris.append(
                {
                    "nilai": int(nilai),
                    "level": level,
                    "diagonal": idx_baris == idx_kolom,
                }
            )
        baris_cm.append(
            {
                "label": label_kelas[idx_baris],
                "sel": sel_baris,
            }
        )

    # Format eksperimen perbandingan nilai k ganjil
    eksperimen_k = []
    for item in metrik.get("k_experiment", []):
        latih_pct = round(float(item["train_accuracy"]) * 100, 2)
        uji_pct = round(float(item["test_accuracy"]) * 100, 2)
        gap_pct = round(float(item["gap"]) * 100, 2)
        eksperimen_k.append(
            {
                "k": int(item["k"]),
                "train_pct": latih_pct,
                "test_pct": uji_pct,
                "gap_pct": gap_pct,
                "terpilih": int(item["k"]) == int(metrik["best_k"]),
            }
        )

    # Format classification report per kelas
    laporan_raw = metrik.get("classification_report", {})
    baris_laporan = []
    for lbl in label_kelas:
        if lbl in laporan_raw:
            skor = laporan_raw[lbl]
            baris_laporan.append(
                {
                    "kelas": lbl,
                    "precision": f"{float(skor['precision']):.4f}",
                    "recall": f"{float(skor['recall']):.4f}",
                    "f1_score": f"{float(skor['f1-score']):.4f}",
                    "support": int(skor["support"]),
                }
            )

    return {
        "best_k": int(metrik["best_k"]),
        "train_accuracy": akurasi_latih,
        "test_accuracy": akurasi_uji,
        "gap": selisih,
        "dataset_rows": int(metrik.get("dataset_rows", 2000)),
        "train_rows": int(metrik.get("train_rows", 1500)),
        "test_rows": int(metrik.get("test_rows", 500)),
        "label_kelas": label_kelas,
        "confusion_matrix_rows": baris_cm,
        "k_experiment": eksperimen_k,
        "classification_report_rows": baris_laporan,
    }
