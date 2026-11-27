"""
Skrip Pelatihan Model Klasifikasi Rentang Harga Smartphone Menggunakan KNN.
Mata Kuliah: Sistem Cerdas (TI24317) - Proyek Akhir UAS
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "mobile_price.csv"
MODEL_PATH = BASE_DIR / "model_knn_smartphone.joblib"
METRICS_PATH = BASE_DIR / "metrics_summary.json"

# 6 fitur hardware utama yang digunakan sebagai input model
FITUR_UTAMA = [
    "ram",
    "battery_power",
    "int_memory",
    "px_width",
    "px_height",
    "pc",
]
TARGET_KOLOM = "price_range"

# Pemetaan label kelas target ke nama kategori yang mudah dipahami pengguna
NAMA_KELAS = {
    0: "Murah",
    1: "Menengah",
    2: "Mahal",
    3: "Flagship",
}

# Daftar nilai k ganjil yang diuji untuk mencari performa terbaik tanpa overfitting
DAFTAR_K = [3, 5, 7, 9, 11]
K_DEFAULT = 7


def muat_dan_periksa_data(path_csv: Path) -> pd.DataFrame:
    """Memuat dataset CSV, memeriksa missing values, dan mencetak distribusi kelas."""
    if not path_csv.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {path_csv}")

    df = pd.read_csv(path_csv)
    print("=" * 60)
    print("1. PEMERIKSAAN DATASET")
    print("=" * 60)
    print(f"Jumlah baris & kolom : {df.shape[0]} baris, {df.shape[1]} kolom")

    # Cek apakah ada nilai kosong (missing values) pada 6 fitur utama dan target
    kolom_dicek = FITUR_UTAMA + [TARGET_KOLOM]
    jumlah_kosong = df[kolom_dicek].isnull().sum()
    print("\nPengecekan data kosong (missing values) pada fitur & target:")
    print(jumlah_kosong.to_string())
    print(f"Total missing values : {int(jumlah_kosong.sum())}")

    # Tampilkan distribusi kelas target untuk memastikan data seimbang
    print("\nDistribusi kelas target (price_range):")
    distribusi = df[TARGET_KOLOM].value_counts().sort_index()
    for kelas, jumlah in distribusi.items():
        print(f"  Kelas {kelas} ({NAMA_KELAS[int(kelas)]}): {jumlah} data")

    return df


def buat_pipeline_knn(nilai_k: int) -> Pipeline:
    """Membuat Pipeline StandardScaler + KNeighborsClassifier untuk mencegah data leakage."""
    # StandardScaler dipasang di dalam Pipeline agar rata-rata dan standar deviasi
    # hanya dihitung dari data latih saat fit(), lalu diterapkan konsisten saat prediksi
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=nilai_k)),
        ]
    )


def eksperimen_nilai_k(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    daftar_k: list[int],
) -> tuple[list[dict], int]:
    """Menguji beberapa nilai k ganjil untuk membandingkan akurasi latih vs uji."""
    print("\n" + "=" * 60)
    print("2. EKSPERIMEN PERBANDINGAN NILAI K (3, 5, 7, 9, 11)")
    print("=" * 60)
    print(f"{'k':>4} | {'Akurasi Latih':>14} | {'Akurasi Uji':>12} | {'Selisih':>9}")
    print("-" * 48)

    hasil_eksperimen = []
    for k in daftar_k:
        pipeline = buat_pipeline_knn(k)
        pipeline.fit(X_train, y_train)

        akurasi_latih = accuracy_score(y_train, pipeline.predict(X_train))
        akurasi_uji = accuracy_score(y_test, pipeline.predict(X_test))
        selisih = abs(akurasi_latih - akurasi_uji)

        print(
            f"{k:>4} | {akurasi_latih:>14.4f} | {akurasi_uji:>12.4f} | {selisih:>9.4f}"
        )

        hasil_eksperimen.append(
            {
                "k": int(k),
                "train_accuracy": round(float(akurasi_latih), 4),
                "test_accuracy": round(float(akurasi_uji), 4),
                "gap": round(float(selisih), 4),
            }
        )

    # Pilih k dengan akurasi uji tertinggi; jika seri, utamakan selisih latih-uji terkecil
    terbaik = max(
        hasil_eksperimen,
        key=lambda item: (item["test_accuracy"], -item["gap"], item["k"] == K_DEFAULT),
    )
    k_terbaik = int(terbaik["k"])
    print(
        f"\nNilai k terpilih: k={k_terbaik} "
        f"(Akurasi Uji={terbaik['test_accuracy']:.4f}, "
        f"Selisih Latih-Uji={terbaik['gap']:.4f})"
    )
    return hasil_eksperimen, k_terbaik


def latih_dan_evaluasi_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    nilai_k: int,
) -> tuple[Pipeline, dict]:
    """Melatih model KNN final dan menghitung akurasi, confusion matrix, dan report."""
    model = buat_pipeline_knn(nilai_k)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    akurasi_latih = accuracy_score(y_train, y_pred_train)
    akurasi_uji = accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)

    target_names = [f"{k} ({v})" for k, v in NAMA_KELAS.items()]
    laporan_teks = classification_report(
        y_test, y_pred_test, target_names=target_names, digits=4
    )
    laporan_dict = classification_report(
        y_test, y_pred_test, target_names=target_names, output_dict=True
    )

    print("\n" + "=" * 60)
    print(f"3. EVALUASI MODEL FINAL (KNN k={nilai_k} + StandardScaler)")
    print("=" * 60)
    print(f"Akurasi Data Latih : {akurasi_latih:.4f} ({akurasi_latih * 100:.2f}%)")
    print(f"Akurasi Data Uji   : {akurasi_uji:.4f} ({akurasi_uji * 100:.2f}%)")
    print("\nConfusion Matrix (Baris=Aktual, Kolom=Prediksi):")
    print(cm)
    print("\nClassification Report:")
    print(laporan_teks)

    metrik = {
        "best_k": int(nilai_k),
        "train_accuracy": round(float(akurasi_latih), 4),
        "test_accuracy": round(float(akurasi_uji), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": laporan_dict,
    }
    return model, metrik


def hitung_statistik_fitur(df: pd.DataFrame) -> dict:
    """Mengambil nilai min, max, dan median tiap fitur untuk batas validasi dan slider web."""
    rentang = {}
    for fitur in FITUR_UTAMA:
        rentang[fitur] = {
            "min": int(df[fitur].min()),
            "max": int(df[fitur].max()),
            "median": int(df[fitur].median()),
        }
    return rentang


def main() -> None:
    """Alur utama pelatihan model dari pemuatan data hingga penyimpanan artefak."""
    # 1. Muat dan periksa dataset
    df = muat_dan_periksa_data(DATASET_PATH)

    # 2. Pisahkan fitur (X) dan target (y), lalu bagi data latih 75% dan uji 25%
    X = df[FITUR_UTAMA]
    y = df[TARGET_KOLOM]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    print(
        f"\nPembagian data: {len(X_train)} data latih (75%), "
        f"{len(X_test)} data uji (25%)"
    )

    # 3. Jalankan eksperimen nilai k ganjil
    hasil_k, k_terbaik = eksperimen_nilai_k(
        X_train, X_test, y_train, y_test, DAFTAR_K
    )

    # 4. Latih dan evaluasi model terbaik berdasarkan hasil eksperimen
    model_final, metrik_final = latih_dan_evaluasi_model(
        X_train, X_test, y_train, y_test, k_terbaik
    )

    # 5. Lengkapi ringkasan metrik dan simpan artefak model + JSON ke disk
    ringkasan_json = {
        **metrik_final,
        "feature_names": FITUR_UTAMA,
        "feature_ranges": hitung_statistik_fitur(df),
        "class_labels": {str(k): v for k, v in NAMA_KELAS.items()},
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "k_experiment": hasil_k,
    }

    joblib.dump(model_final, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as file_json:
        json.dump(ringkasan_json, file_json, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("4. PENYIMPANAN ARTEFAK")
    print("=" * 60)
    print(f"Model tersimpan   : {MODEL_PATH}")
    print(f"Metrik tersimpan  : {METRICS_PATH}")


if __name__ == "__main__":
    main()
