# Sistem Klasifikasi Rentang Harga Smartphone Berdasarkan Spesifikasi Hardware Berbasis Machine Learning (KNN)

## Masalah yang diselesaikan

Menentukan kategori harga smartphone secara objektif dari spesifikasi perangkat keras sering kali membingungkan bagi calon pembeli maupun penjual, karena banyak kombinasi spesifikasi (RAM, baterai, memori internal, resolusi layar, dan kamera) yang saling memengaruhi. Proyek ini menyelesaikan masalah klasifikasi multi-kelas untuk memetakan 6 spesifikasi hardware utama smartphone ke dalam 4 kategori harga (`0 = Murah`, `1 = Menengah`, `2 = Mahal`, `3 = Flagship`) menggunakan algoritma *K-Nearest Neighbors* (KNN) yang disajikan melalui antarmuka web interaktif berbasis Streamlit.

## Dataset

- Sumber: Kaggle — Mobile Price Classification (oleh Abhishek Sharma)
- Tautan: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
- Lisensi: Unknown (tidak dicantumkan secara eksplisit oleh pengunggah di Kaggle; digunakan khusus untuk keperluan akademik/pembelajaran)
- Jumlah data: 2.000 baris (`train.csv` yang diubah namanya menjadi `mobile_price.csv`), dengan 21 kolom dan 4 kelas target seimbang (`price_range`: 0 = Murah sebanyak 500 data, 1 = Menengah sebanyak 500 data, 2 = Mahal sebanyak 500 data, 3 = Flagship sebanyak 500 data).
- Pemeriksaan kualitas data: `0` *missing values* pada seluruh 6 fitur utama (`ram`, `battery_power`, `int_memory`, `px_width`, `px_height`, `pc`) dan target `price_range`.

## Metode

- **Pemilihan Fitur & Pembagian Data**: Menggunakan 6 fitur hardware utama (`ram`, `battery_power`, `int_memory`, `px_width`, `px_height`, `pc`) dengan pembagian data *stratified* `train_test_split(test_size=0.25, random_state=42, stratify=y)` menghasilkan 1.500 data latih (75%) dan 500 data uji (25%).
- **Preprocessing & Pipeline**: Menggunakan `sklearn.pipeline.Pipeline` yang menggabungkan `StandardScaler()` dan `KNeighborsClassifier`. `StandardScaler` diperlukan karena skala fitur sangat bervariasi (misalnya `ram` 256–3998 MB dibandingkan `pc` 0–20 MP), sehingga fitur berangka besar tidak mendominasi perhitungan jarak Euclidean pada KNN.
- **Pemilihan Nilai `k`**: Dilakukan eksperimen perbandingan nilai `k` ganjil (`k = 3, 5, 7, 9, 11`) untuk menghindari *overfitting* dan mencari akurasi uji terbaik. Dari eksperimen, `k = 9` dipilih sebagai model terbaik karena menghasilkan akurasi uji tertinggi dengan selisih latih-uji yang sehat.

## Hasil & evaluasi

Berikut hasil aktual dari eksekusi `python train_model.py`:

### 1. Perbandingan Nilai `k` (3, 5, 7, 9, 11)

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
- **Selisih Latih vs Uji**: `7.20%` (tidak terjadi *overfitting* ekstrem seperti pada `k=3`)

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

Terlihat bahwa kelas ekstrem (`0 = Murah` dengan F1-score `0.9180` dan `3 = Flagship` dengan F1-score `0.8514`) sangat mudah dipisahkan, sedangkan kesalahan prediksi paling banyak terjadi di batas kelas yang berdekatan (`1 = Menengah` dan `2 = Mahal`).

## Cara menjalankan

1. Pastikan Python 3 sudah terpasang, lalu instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. (Opsional jika ingin melatih ulang model dan memperbarui metrik):
   ```bash
   python train_model.py
   ```
3. Jalankan aplikasi web Streamlit (utama sesuai ketentuan UAS):
   ```bash
   streamlit run app.py
   ```
4. Alternatif antarmuka web berbasis Django (menggunakan model dan metrik yang sama):
   ```bash
   python manage.py runserver
   ```
   Lalu buka `http://127.0.0.1:8000/` di browser.

## Refleksi singkat

- **Yang berhasil**: Pipeline `StandardScaler` + `KNN` berjalan mulus dan eksperimen `k` ganjil berhasil menemukan `k=9` yang menurunkan *gap* latih-uji dari `13.80%` (pada `k=3`) menjadi `7.20%` dengan akurasi uji `80.60%`. Aplikasi web Streamlit dapat memuat model dan metrik secara otomatis untuk melakukan prediksi real-time beserta probabilitas tiap kelas.
- **Keterbatasan**:
  1. Rentang fitur pada dataset terbatas pada spesifikasi generasi lama — misalnya kapasitas `ram` hanya mencapai maksimal `3.998 MB` (~4 GB), `int_memory` maksimal `64 GB`, dan `battery_power` maksimal `1.998 mAh` — sehingga kurang akurat jika langsung diterapkan pada smartphone modern dengan RAM 8–16 GB atau baterai 5.000+ mAh.
  2. Dataset belum memasukkan faktor non-hardware atau spesifikasi penting masa kini seperti merek/brand, jenis chipset/SoC, teknologi layar (AMOLED/refresh rate), dan dukungan 5G.
  3. Target `price_range` (0–3) merupakan kelas relatif tanpa harga nominal resmi, sehingga estimasi harga Rupiah di aplikasi hanya bersifat ilustratif.
  4. Kelas berdekatan (`Menengah` dan `Mahal`) masih memiliki tumpang tindih fitur sehingga F1-score kelas `2 (Mahal)` berada di `0.6908`.
- **Rencana perbaikan**: Memperbarui dataset dengan data smartphone keluaran terbaru (termasuk chipset, RAM > 4 GB, dan harga aktual pasar), menambah fitur relevan, serta membandingkan performa KNN dengan algoritma lain seperti *Random Forest* atau *Gradient Boosting* menggunakan *cross-validation*.
