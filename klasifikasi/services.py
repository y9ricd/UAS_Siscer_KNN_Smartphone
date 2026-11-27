"""
Layanan Machine Learning untuk aplikasi Django.
Memuat model KNN terlatih dan ringkasan metrik sekali saja di memori
menggunakan path berbasis settings.BASE_DIR.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings

MODEL_PATH = Path(settings.BASE_DIR) / "model_knn_smartphone.joblib"
METRICS_PATH = Path(settings.BASE_DIR) / "metrics_summary.json"

# Urutan 6 fitur hardware utama (harus identik dengan train_model.py)
FITUR_UTAMA = [
    "ram",
    "battery_power",
    "int_memory",
    "px_width",
    "px_height",
    "pc",
]

LABEL_FITUR = {
    "ram": "RAM",
    "battery_power": "Baterai",
    "int_memory": "Memori Internal",
    "px_width": "Lebar Layar",
    "px_height": "Tinggi Layar",
    "pc": "Kamera Utama",
}

SATUAN_FITUR = {
    "ram": "MB",
    "battery_power": "mAh",
    "int_memory": "GB",
    "px_width": "px",
    "px_height": "px",
    "pc": "MP",
}

# Informasi kategori harga dan kisaran harga ilustratif untuk hasil prediksi
INFO_KATEGORI = {
    0: {
        "nama": "Murah",
        "kisaran": "Rp 1.000.000 – Rp 1.999.000",
        "deskripsi": "Segmen entry-level untuk komunikasi dasar, media sosial, dan aplikasi ringan.",
    },
    1: {
        "nama": "Menengah",
        "kisaran": "Rp 2.000.000 – Rp 3.999.000",
        "deskripsi": "Segmen mid-range dengan keseimbangan performa harian, baterai, dan layar.",
    },
    2: {
        "nama": "Mahal",
        "kisaran": "Rp 4.000.000 – Rp 6.999.000",
        "deskripsi": "Segmen menengah-atas dengan kapasitas RAM, layar, dan penyimpanan lebih lega.",
    },
    3: {
        "nama": "Flagship",
        "kisaran": "> Rp 7.000.000",
        "deskripsi": "Segmen teratas dengan kombinasi spesifikasi hardware paling tinggi di dataset.",
    },
}

# Nilai batas bawaan jika metrics_summary.json belum tersedia
BATAS_DEFAULT = {
    "ram": {"min": 256, "max": 3998, "median": 2146},
    "battery_power": {"min": 501, "max": 1998, "median": 1226},
    "int_memory": {"min": 2, "max": 64, "median": 32},
    "px_width": {"min": 500, "max": 1998, "median": 1247},
    "px_height": {"min": 0, "max": 1960, "median": 564},
    "pc": {"min": 0, "max": 20, "median": 10},
}

# Preset cepat untuk simulasi 3 segmen pasar
PRESET_SPESIFIKASI = {
    "Entry-Level": {
        "ram": 512,
        "battery_power": 600,
        "int_memory": 8,
        "px_width": 640,
        "px_height": 480,
        "pc": 2,
    },
    "Mid-Range": {
        "ram": 2048,
        "battery_power": 1200,
        "int_memory": 32,
        "px_width": 1200,
        "px_height": 720,
        "pc": 10,
    },
    "Flagship": {
        "ram": 3800,
        "battery_power": 1900,
        "int_memory": 64,
        "px_width": 1900,
        "px_height": 1800,
        "pc": 16,
    },
}


@lru_cache(maxsize=1)
def muat_model():
    """Memuat model KNN terlatih dari disk satu kali dan menyimpannya di cache memori."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "File model_knn_smartphone.joblib tidak ditemukan. Jalankan `python train_model.py`."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def muat_metrik() -> dict:
    """Memuat ringkasan metrik evaluasi dari metrics_summary.json satu kali di memori."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "File metrics_summary.json tidak ditemukan. Jalankan `python train_model.py`."
        )
    with open(METRICS_PATH, "r", encoding="utf-8") as file_json:
        return json.load(file_json)


def ambil_rentang_fitur() -> dict:
    """Mengembalikan batas min, max, dan median untuk ke-6 fitur hardware."""
    try:
        metrik = muat_metrik()
        return metrik.get("feature_ranges", BATAS_DEFAULT)
    except FileNotFoundError:
        return BATAS_DEFAULT


def validasi_input_spesifikasi(payload: dict) -> tuple[dict | None, dict]:
    """
    Memvalidasi input JSON dari klien agar berupa angka valid dalam rentang min-max dataset.
    Mengembalikan (data_bersih, dict_error).
    """
    if not isinstance(payload, dict):
        return None, {"_general": "Format payload harus berupa objek JSON."}

    rentang = ambil_rentang_fitur()
    data_bersih = {}
    errors = {}

    for fitur in FITUR_UTAMA:
        if fitur not in payload or payload[fitur] is None or payload[fitur] == "":
            errors[fitur] = f"Field '{fitur}' wajib diisi."
            continue

        nilai_mentah = payload[fitur]
        # Tolak tipe boolean karena bool di Python adalah turunan int
        if isinstance(nilai_mentah, bool):
            errors[fitur] = f"Field '{fitur}' harus berupa angka, bukan boolean."
            continue

        try:
            nilai_float = float(nilai_mentah)
        except (TypeError, ValueError):
            errors[fitur] = f"Field '{fitur}' harus berupa angka valid."
            continue

        if not math.isfinite(nilai_float):
            errors[fitur] = f"Field '{fitur}' tidak boleh bernilai NaN atau tak hingga."
            continue

        nilai_int = int(round(nilai_float))
        f_min = int(rentang[fitur]["min"])
        f_max = int(rentang[fitur]["max"])

        if nilai_int < f_min or nilai_int > f_max:
            errors[fitur] = (
                f"Nilai '{fitur}' ({nilai_int}) harus berada di antara {f_min} hingga {f_max}."
            )
            continue

        data_bersih[fitur] = nilai_int

    if errors:
        return None, errors
    return data_bersih, {}


def buat_ringkasan_analisis(
    kelas: int,
    daftar_prob: list[float],
    data_bersih: dict,
    rentang: dict,
) -> list[str]:
    """Menyusun poin analisis kontekstual dari hasil probabilitas dan posisi fitur input."""
    urutan = sorted(range(4), key=lambda idx: daftar_prob[idx], reverse=True)
    runner_up = urutan[1]
    prob_utama = round(float(daftar_prob[kelas]) * 100, 1)
    prob_runner_up = round(float(daftar_prob[runner_up]) * 100, 1)
    selisih = round(prob_utama - prob_runner_up, 1)

    ram_min, ram_max = int(rentang["ram"]["min"]), int(rentang["ram"]["max"])
    rasio_ram = round(((data_bersih["ram"] - ram_min) / max(1, ram_max - ram_min)) * 100)

    bat_min = int(rentang["battery_power"]["min"])
    bat_max = int(rentang["battery_power"]["max"])
    rasio_bat = round(
        ((data_bersih["battery_power"] - bat_min) / max(1, bat_max - bat_min)) * 100
    )

    poin = [
        (
            f"Model KNN menempatkan spesifikasi ini di kelas {INFO_KATEGORI[kelas]['nama']} "
            f"(Kelas {kelas}) dengan keyakinan {prob_utama}%."
        )
    ]

    if prob_runner_up > 0:
        poin.append(
            f"Kelas terdekat berikutnya adalah {INFO_KATEGORI[runner_up]['nama']} "
            f"({prob_runner_up}%), terpaut {selisih} poin persentase."
        )
    else:
        poin.append(
            "Seluruh tetangga terdekat sepakat pada satu kelas (probabilitas 100%)."
        )

    poin.append(
        f"Kapasitas RAM ({data_bersih['ram']} MB) berada di persentil rentang {rasio_ram}% "
        f"dan baterai ({data_bersih['battery_power']} mAh) di {rasio_bat}% dari rentang dataset."
    )
    return poin


def prediksi_kategori_harga(data_fitur: dict) -> dict:
    """Menjalankan prediksi kategori harga dan probabilitas 4 kelas dari input tervalidasi."""
    model = muat_model()
    rentang = ambil_rentang_fitur()
    baris = {kolom: int(data_fitur[kolom]) for kolom in FITUR_UTAMA}
    df_input = pd.DataFrame([baris], columns=FITUR_UTAMA)

    kelas = int(model.predict(df_input)[0])
    daftar_prob = [float(p) for p in model.predict_proba(df_input)[0]]
    info = INFO_KATEGORI[kelas]

    probabilitas_kelas = []
    for idx in range(4):
        probabilitas_kelas.append(
            {
                "kelas": idx,
                "nama": INFO_KATEGORI[idx]["nama"],
                "persen": round(daftar_prob[idx] * 100, 1),
                "terpilih": idx == kelas,
            }
        )

    return {
        "kelas": kelas,
        "nama": info["nama"],
        "kisaran": info["kisaran"],
        "deskripsi": info["deskripsi"],
        "keyakinan": round(daftar_prob[kelas] * 100, 1),
        "probabilitas_kelas": probabilitas_kelas,
        "ringkasan_analisis": buat_ringkasan_analisis(kelas, daftar_prob, baris, rentang),
        "input_dipakai": baris,
    }


def siapkan_data_evaluasi() -> dict:
    """Memformat metrik evaluasi dari JSON untuk ditampilkan pada tab Evaluasi Model."""
    metrik = muat_metrik()
    akurasi_latih = round(float(metrik["train_accuracy"]) * 100, 2)
    akurasi_uji = round(float(metrik["test_accuracy"]) * 100, 2)
    selisih = round(abs(akurasi_latih - akurasi_uji), 2)

    cm = metrik["confusion_matrix"]
    nilai_maks_cm = max(max(baris) for baris in cm) or 1
    label_kelas = [f"{i} ({INFO_KATEGORI[i]['nama']})" for i in range(4)]

    # Hitung tingkat intensitas warna (0-4) untuk sel Confusion Matrix monokrom
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
        baris_cm.append({"label": label_kelas[idx_baris], "sel": sel_baris})

    # Format eksperimen perbandingan nilai k ganjil
    eksperimen_k = []
    for item in metrik.get("k_experiment", []):
        eksperimen_k.append(
            {
                "k": int(item["k"]),
                "train_pct": round(float(item["train_accuracy"]) * 100, 2),
                "test_pct": round(float(item["test_accuracy"]) * 100, 2),
                "gap_pct": round(float(item["gap"]) * 100, 2),
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

    # Status kestabilan model berdasarkan selisih akurasi latih dan uji
    status_stabil = selisih <= 10.0

    return {
        "best_k": int(metrik["best_k"]),
        "train_accuracy": akurasi_latih,
        "test_accuracy": akurasi_uji,
        "gap": selisih,
        "status_stabil": status_stabil,
        "dataset_rows": int(metrik.get("dataset_rows", 2000)),
        "train_rows": int(metrik.get("train_rows", 1500)),
        "test_rows": int(metrik.get("test_rows", 500)),
        "label_kelas": label_kelas,
        "confusion_matrix_rows": baris_cm,
        "k_experiment": eksperimen_k,
        "classification_report_rows": baris_laporan,
    }
