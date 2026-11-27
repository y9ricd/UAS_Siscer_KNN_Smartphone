# Sistem Klasifikasi Rentang Harga Smartphone Berdasarkan Spesifikasi Hardware Berbasis Machine Learning (KNN)

## Masalah yang diselesaikan

Menentukan kategori harga smartphone secara objektif dari spesifikasi perangkat keras sering kali membingungkan bagi calon pembeli maupun penjual karena banyaknya kombinasi komponen (RAM, baterai, memori internal, resolusi layar, dan kamera). Proyek ini membangun aplikasi web berbasis Django yang mengklasifikasikan 6 spesifikasi hardware utama smartphone ke dalam 4 kategori harga (`0 = Murah`, `1 = Menengah`, `2 = Mahal`, `3 = Flagship`) menggunakan algoritma *K-Nearest Neighbors* (KNN) secara *real-time*.

## Dataset

- Sumber: Kaggle — Mobile Price Classification (oleh Abhishek Sharma)
- Tautan: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
- Lisensi: Unknown (tidak dicantumkan secara eksplisit oleh pengunggah di Kaggle; digunakan khusus untuk keperluan akademik/pembelajaran)
- Jumlah data: 2.000 baris (`train.csv` yang diubah namanya menjadi `mobile_price.csv`), dengan 21 kolom dan 4 kelas target seimbang (`price_range`: 0 = Murah sebanyak 500 data, 1 = Menengah sebanyak 500 data, 2 = Mahal sebanyak 500 data, 3 = Flagship sebanyak 500 data).
- Kualitas data: `0` *missing values* pada ke-6 fitur utama (`ram`, `battery_power`, `int_memory`, `px_width`, `px_height`, `pc`) dan target `price_range`.

## Metode

- **Pemilihan Fitur & Pembagian Data**: Menggunakan 6 fitur hardware utama (`ram`, `battery_power`, `int_memory`, `px_width`, `px_height`, `pc`) dengan pembagian *stratified* `train_test_split(test_size=0.25, random_state=42, stratify=y)` menghasilkan 1.500 data latih (75%) dan 500 data uji (25%).
- **Preprocessing & Pipeline ML**: Menggunakan `sklearn.pipeline.Pipeline` berisi `StandardScaler()` dan `KNeighborsClassifier`. Penyimpanan scaler di dalam `Pipeline` mencegah *data leakage* sekaligus memastikan fitur berangka ribuan (`ram`, `battery_power`) tidak mendominasi fitur puluhan (`int_memory`, `pc`) saat menghitung jarak Euclidean.
- **Arsitektur Web Django**: Model terlatih (`model_knn_smartphone.joblib`) dan metrik evaluasi (`metrics_summary.json`) dimuat sekali pada level modul dengan path berbasis `settings.BASE_DIR`. Antarmuka web menyajikan 2 tab utama (**Simulasi Harga** yang memanggil endpoint JSON `POST /api/predict/` via `fetch` secara *real-time* dan **Evaluasi Model**), dilengkapi toggle Light/Dark Mode serta konfigurasi *environment variable* dan `WhiteNoise` untuk kesiapan deploy.

## Hasil & evaluasi

Berikut hasil aktual dari eksekusi `python train_model.py`:

### 1. Eksperimen Pemilihan Nilai `k` Ganjil (`3, 5, 7, 9, 11`)

| Nilai `k` | Akurasi Data Latih | Akurasi Data Uji | Selisih (Latih - Uji) |
| --- | --- | --- | --- |
| 3 | 0.9080 (90.80%) | 0.7700 (77.00%) | 0.1380 (13.80%) |
| 5 | 0.8880 (88.80%) | 0.7820 (78.20%) | 0.1060 (10.60%) |
| 7 | 0.8840 (88.40%) | 0.7960 (79.60%) | 0.0880 (8.80%) |
| **9 (Terpilih)** | **0.8780 (87.80%)** | **0.8060 (80.60%)** | **0.0720 (7.20%)** |
| 11 | 0.8733 (87.33%) | 0.8020 (80.20%) | 0.0713 (7.13%) |

### 2. Performa Model Final (`KNN k=9` + `StandardScaler`)

- **Akurasi Data Latih**: `0.8780` (`87.80%`)
- **Akurasi Data Uji**: `0.8060` (`80.60%`)
- **Selisih Latih vs Uji**: `7.20%` (model stabil dan tidak mengalami *overfitting* ekstrem seperti `k=3`)

**Confusion Matrix (500 Data Uji; Baris = Aktual, Kolom = Prediksi):**

| Aktual \ Prediksi | 0 (Murah) | 1 (Menengah) | 2 (Mahal) | 3 (Flagship) |
| --- | --- | --- | --- | --- |
| **0 (Murah)** | 112 | 13 | 0 | 0 |
| **1 (Menengah)** | 7 | 99 | 19 | 0 |
| **2 (Mahal)** | 0 | 21 | 86 | 18 |
| **3 (Flagship)** | 0 | 0 | 19 | 106 |

**Classification Report (Data Uji):**

| Kelas | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| 0 (Murah) | 0.9412 | 0.8960 | 0.9180 | 125 |
| 1 (Menengah) | 0.7444 | 0.7920 | 0.7674 | 125 |
| 2 (Mahal) | 0.6935 | 0.6880 | 0.6908 | 125 |
| 3 (Flagship) | 0.8548 | 0.8480 | 0.8514 | 125 |
| **Macro Avg** | **0.8085** | **0.8060** | **0.8069** | **500** |
| **Weighted Avg** | **0.8085** | **0.8060** | **0.8069** | **500** |

## Cara menjalankan

1. Pastikan Python 3 sudah terpasang, lalu instal seluruh dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan skrip pelatihan untuk menghasilkan `model_knn_smartphone.joblib` dan `metrics_summary.json`:
   ```bash
   python train_model.py
   ```
3. Jalankan server web Django:
   ```bash
   python manage.py runserver
   ```
4. Buka `http://127.0.0.1:8000/` di browser untuk mencoba tab **Simulasi Harga** dan **Evaluasi Model**.

## Refleksi singkat

- **Yang berhasil**: Seluruh alur pelatihan `Pipeline(StandardScaler + KNN)` hingga penyajian melalui arsitektur Django (`POST /api/predict/` + frontend *real-time*) berjalan stabil. Pemilihan `k=9` menekan selisih latih-uji dari `13.80%` (pada `k=3`) menjadi `7.20%` dengan akurasi uji `80.60%`.
- **Keterbatasan**:
  1. Rentang nilai dataset merefleksikan spesifikasi smartphone generasi lama, misalnya kapasitas `ram` dibatasi hingga `3.998 MB` (~4 GB), `int_memory` maksimal `64 GB`, dan `battery_power` maksimal `1.998 mAh`, sehingga tidak langsung cocok untuk perangkat modern dengan RAM 8–16 GB.
  2. Dataset belum menyertakan faktor penting penentu harga pasar seperti merek/brand, jenis chipset/SoC, teknologi layar (AMOLED/*refresh rate*), maupun konektivitas 5G.
  3. Target `price_range` (0–3) merupakan label kelas relatif tanpa harga nominal resmi, sehingga kisaran harga Rupiah di antarmuka hanya bersifat ilustratif.
  4. Kelas yang berdekatan (`Menengah` dan `Mahal`) masih memiliki tumpang tindih fitur hardware sehingga F1-score kelas `2 (Mahal)` berada di `0.6908`.
- **Rencana perbaikan**: Mengumpulkan dataset smartphone keluaran terbaru yang mencakup variabel chipset, brand, dan harga riil pasar, melakukan eksperimen perbandingan dengan algoritma *ensemble* seperti *Random Forest* atau *XGBoost*, serta menambahkan *cross-validation* pada proses pemilihan *hyperparameter*.
