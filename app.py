"""
Aplikasi Web Streamlit: What-If Simulator Klasifikasi Harga Smartphone (KNN).
Mata Kuliah: Sistem Cerdas (TI24317) - Proyek Akhir UAS
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Konfigurasi halaman utama Streamlit (wajib dipanggil paling awal)
st.set_page_config(
    page_title="SmartPrice Lab · What-If Simulator Smartphone (KNN)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lokasi file dataset, model terlatih, dan ringkasan metrik (tidak diubah oleh app)
DATASET_PATH = Path("mobile_price.csv")
MODEL_PATH = Path("model_knn_smartphone.joblib")
METRICS_PATH = Path("metrics_summary.json")

# Urutan fitur wajib sama persis dengan train_model.py
FITUR_UTAMA = [
    "ram",
    "battery_power",
    "int_memory",
    "px_width",
    "px_height",
    "pc",
]

LABEL_FITUR = {
    "ram": "RAM (MB)",
    "battery_power": "Baterai (mAh)",
    "int_memory": "Memori Internal (GB)",
    "px_width": "Lebar Layar (px)",
    "px_height": "Tinggi Layar (px)",
    "pc": "Kamera Utama (MP)",
}

# Nilai preset cepat untuk memudahkan simulasi 3 segmen pasar saat demo UAS
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

# Informasi kategori harga dan kisaran harga ilustratif untuk tiap kelas target
INFO_KATEGORI = {
    0: {
        "nama": "Murah",
        "segmen": "Segmen Entry-Level",
        "kisaran_singkat": "Rp 1 - 2 Juta",
        "kisaran": "Sekitar Rp 1.000.000 - Rp 1.999.000",
        "deskripsi": "Cocok untuk kebutuhan dasar seperti komunikasi, media sosial, dan aplikasi ringan.",
    },
    1: {
        "nama": "Menengah",
        "segmen": "Segmen Mid-Range",
        "kisaran_singkat": "Rp 2 - 4 Juta",
        "kisaran": "Sekitar Rp 2.000.000 - Rp 3.999.000",
        "deskripsi": "Keseimbangan performa harian, kapasitas baterai, dan kualitas layar.",
    },
    2: {
        "nama": "Mahal",
        "segmen": "Segmen Upper Mid-Range",
        "kisaran_singkat": "Rp 4 - 7 Juta",
        "kisaran": "Sekitar Rp 4.000.000 - Rp 6.999.000",
        "deskripsi": "Performa multitasking tinggi dengan RAM, penyimpanan, dan layar lebih tajam.",
    },
    3: {
        "nama": "Flagship",
        "segmen": "Segmen Flagship",
        "kisaran_singkat": "> Rp 7 Juta",
        "kisaran": "Di atas Rp 7.000.000",
        "deskripsi": "Kombinasi spesifikasi hardware tertinggi di seluruh rentang dataset.",
    },
}

# Palet warna grafik matplotlib agar selaras dengan tema gelap What-If Simulator
WARNA_LATAR = "#1D1A17"
WARNA_TEKS = "#F5F2EE"
WARNA_GRID = "#3A342E"
WARNA_ORANYE = "#EA580C"
WARNA_TEAL = "#14B8A6"
WARNA_NETRAL = "#57534E"


@st.cache_resource
def muat_model(path_model: Path):
    """Memuat Pipeline model KNN dari file joblib dan menyimpannya di cache."""
    return joblib.load(path_model)


@st.cache_data
def muat_metrik(path_metrik: Path) -> dict:
    """Memuat ringkasan metrik evaluasi dari file JSON."""
    with open(path_metrik, "r", encoding="utf-8") as file_json:
        return json.load(file_json)


@st.cache_data
def muat_rata_rata_kelas(path_csv: Path) -> pd.DataFrame | None:
    """Menghitung rata-rata tiap fitur per kelas target untuk grafik pembanding."""
    if not path_csv.exists():
        return None
    df = pd.read_csv(path_csv)
    return df.groupby("price_range")[FITUR_UTAMA].mean()


def ambil_batas_fitur(
    rentang: dict, nama_fitur: str, fallback: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Mengambil nilai min, max, dan median fitur dari metadata dataset."""
    info = rentang.get(nama_fitur, {})
    return (
        int(info.get("min", fallback[0])),
        int(info.get("max", fallback[1])),
        int(info.get("median", fallback[2])),
    )


def daftar_batas_lengkap(rentang: dict) -> dict[str, tuple[int, int, int]]:
    """Menyusun batas min, max, dan median seluruh 6 fitur."""
    fallback_map = {
        "ram": (256, 3998, 2146),
        "battery_power": (501, 1998, 1226),
        "int_memory": (2, 64, 32),
        "px_width": (500, 1998, 1247),
        "px_height": (0, 1960, 564),
        "pc": (0, 20, 10),
    }
    return {
        fitur: ambil_batas_fitur(rentang, fitur, fb)
        for fitur, fb in fallback_map.items()
    }


def inisialisasi_state(batas_fitur: dict[str, tuple[int, int, int]]) -> None:
    """Menyiapkan state awal simulator dan daftar skenario tersimpan."""
    for fitur, (_, _, median_val) in batas_fitur.items():
        kunci = f"sim_{fitur}"
        if kunci not in st.session_state:
            st.session_state[kunci] = median_val

    if "preset_pill" not in st.session_state:
        st.session_state["preset_pill"] = "Custom"
    if "toggle_grafik" not in st.session_state:
        st.session_state["toggle_grafik"] = "Probabilitas"
    if "skenario_tersimpan" not in st.session_state:
        st.session_state["skenario_tersimpan"] = []


def terapkan_preset_pill() -> None:
    """Callback saat pengguna memilih preset dari segmented_control."""
    pilihan = st.session_state.get("preset_pill")
    if pilihan in PRESET_SPESIFIKASI:
        for fitur, nilai in PRESET_SPESIFIKASI[pilihan].items():
            st.session_state[f"sim_{fitur}"] = int(nilai)


def reset_parameter(batas_fitur: dict[str, tuple[int, int, int]]) -> None:
    """Mengembalikan seluruh slider ke nilai median dataset."""
    for fitur, (_, _, median_val) in batas_fitur.items():
        st.session_state[f"sim_{fitur}"] = int(median_val)
    st.session_state["preset_pill"] = "Custom"


def tandai_custom() -> None:
    """Menandai profil aktif sebagai Custom saat slider digeser manual."""
    st.session_state["preset_pill"] = "Custom"


def hitung_skor_ekspektasi_kelas(model, input_dict: dict) -> tuple[int, list[float], float]:
    """Menghitung kelas prediksi, probabilitas 4 kelas, dan skor ekspektasi (0.0 - 3.0)."""
    df_in = pd.DataFrame([input_dict], columns=FITUR_UTAMA)
    kelas = int(model.predict(df_in)[0])
    probs = [float(p) for p in model.predict_proba(df_in)[0]]
    ekspektasi = sum(i * probs[i] for i in range(4))
    return kelas, probs, ekspektasi


def analisis_sensitivitas_what_if(
    model,
    input_sekarang: dict,
    batas_fitur: dict[str, tuple[int, int, int]],
) -> pd.DataFrame:
    """Menguji dampak perubahan +/-25% tiap fitur terhadap skor kelas harga KNN."""
    _, _, skor_dasar = hitung_skor_ekspektasi_kelas(model, input_sekarang)
    baris_hasil = []

    for fitur in FITUR_UTAMA:
        f_min, f_max, _ = batas_fitur[fitur]
        delta_langkah = max(1, int(round((f_max - f_min) * 0.25)))

        nilai_turun = max(f_min, int(input_sekarang[fitur]) - delta_langkah)
        nilai_naik = min(f_max, int(input_sekarang[fitur]) + delta_langkah)

        uji_turun = {**input_sekarang, fitur: nilai_turun}
        uji_naik = {**input_sekarang, fitur: nilai_naik}

        _, _, skor_turun = hitung_skor_ekspektasi_kelas(model, uji_turun)
        _, _, skor_naik = hitung_skor_ekspektasi_kelas(model, uji_naik)

        dampak_turun = skor_turun - skor_dasar
        dampak_naik = skor_naik - skor_dasar
        rentang_pengaruh = abs(dampak_naik - dampak_turun) + abs(dampak_naik) + abs(dampak_turun)

        baris_hasil.append(
            {
                "fitur": fitur,
                "label": LABEL_FITUR[fitur],
                "nilai_turun": nilai_turun,
                "nilai_naik": nilai_naik,
                "dampak_turun": dampak_turun,
                "dampak_naik": dampak_naik,
                "pengaruh_total": rentang_pengaruh,
            }
        )

    df_sens = pd.DataFrame(baris_hasil).sort_values("pengaruh_total", ascending=True)
    return df_sens


def atur_gaya_axes_gelap(ax) -> None:
    """Menerapkan warna latar, sumbu, dan grid tema gelap pada figure Matplotlib."""
    ax.set_facecolor(WARNA_LATAR)
    ax.tick_params(colors=WARNA_TEKS, labelsize=9)
    ax.xaxis.label.set_color(WARNA_TEKS)
    ax.yaxis.label.set_color(WARNA_TEKS)
    ax.title.set_color(WARNA_TEKS)
    for sisi in ("top", "right"):
        ax.spines[sisi].set_visible(False)
    for sisi in ("bottom", "left"):
        ax.spines[sisi].set_color(WARNA_GRID)


def buat_grafik_probabilitas(probabilitas: list[float], kelas_terpilih: int):
    """Membuat grafik batang probabilitas 4 kelas dengan sorotan oranye pada kelas terpilih."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=WARNA_LATAR)
    atur_gaya_axes_gelap(ax)

    kategori = [f"{i} · {INFO_KATEGORI[i]['nama']}" for i in range(4)]
    persen = [p * 100 for p in probabilitas]
    warna = [WARNA_ORANYE if i == kelas_terpilih else WARNA_NETRAL for i in range(4)]

    bars = ax.bar(kategori, persen, color=warna, width=0.55, edgecolor="none")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Probabilitas (%)")
    ax.grid(axis="y", linestyle="--", color=WARNA_GRID, alpha=0.7)
    ax.set_axisbelow(True)

    for bar, nilai in zip(bars, persen):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            nilai + 2.5,
            f"{nilai:.1f}%",
            ha="center",
            va="bottom",
            color=WARNA_TEKS,
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


def buat_grafik_profil_fitur(
    input_sekarang: dict, batas_fitur: dict[str, tuple[int, int, int]]
):
    """Membuat grafik persentase posisi 6 fitur input terhadap rentang min-max dataset."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=WARNA_LATAR)
    atur_gaya_axes_gelap(ax)

    labels = [LABEL_FITUR[f] for f in FITUR_UTAMA]
    persen_fitur = []
    for f in FITUR_UTAMA:
        f_min, f_max, _ = batas_fitur[f]
        rasio = (input_sekarang[f] - f_min) / max(1, (f_max - f_min))
        persen_fitur.append(rasio * 100)

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, persen_fitur, color=WARNA_ORANYE, height=0.55)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel("Posisi Relatif pada Rentang Dataset (%)")
    ax.grid(axis="x", linestyle="--", color=WARNA_GRID, alpha=0.7)
    ax.set_axisbelow(True)

    for bar, nilai, f in zip(bars, persen_fitur, FITUR_UTAMA):
        ax.text(
            nilai + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{nilai:.0f}% ({input_sekarang[f]})",
            va="center",
            ha="left",
            color=WARNA_TEKS,
            fontsize=8.5,
        )

    fig.tight_layout()
    return fig


def buat_grafik_perbandingan_ram_kelas(
    df_rata: pd.DataFrame, input_sekarang: dict, kelas_terpilih: int
):
    """Membandingkan RAM input saat ini dengan rata-rata RAM tiap kelas di dataset."""
    fig, ax = plt.subplots(figsize=(6.0, 3.3), facecolor=WARNA_LATAR)
    atur_gaya_axes_gelap(ax)

    kategori = [f"{i} · {INFO_KATEGORI[i]['nama']}" for i in range(4)]
    ram_rata = [float(df_rata.loc[i, "ram"]) for i in range(4)]
    warna = [WARNA_ORANYE if i == kelas_terpilih else WARNA_NETRAL for i in range(4)]

    bars = ax.bar(kategori, ram_rata, color=warna, width=0.52)
    ax.axhline(
        input_sekarang["ram"],
        color=WARNA_TEAL,
        linestyle="--",
        linewidth=1.8,
        label=f"Input Saat Ini ({input_sekarang['ram']} MB)",
    )
    ax.set_ylim(0, 4400)
    ax.set_ylabel("Rata-rata RAM (MB)")
    ax.grid(axis="y", linestyle="--", color=WARNA_GRID, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(facecolor=WARNA_LATAR, edgecolor=WARNA_GRID, labelcolor=WARNA_TEKS)

    for bar, nilai in zip(bars, ram_rata):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            nilai + 80,
            f"{nilai:.0f}",
            ha="center",
            va="bottom",
            color=WARNA_TEKS,
            fontsize=8.5,
        )

    fig.tight_layout()
    return fig


def buat_grafik_tornado_what_if(df_sens: pd.DataFrame):
    """Membuat tornado chart pengaruh naik/turun fitur terhadap skor kelas (ala What moves profit)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.3), facecolor=WARNA_LATAR)
    atur_gaya_axes_gelap(ax)

    y_pos = range(len(df_sens))
    ax.barh(
        y_pos,
        df_sens["dampak_turun"],
        color=WARNA_ORANYE,
        height=0.52,
        label="Turun (-25% rentang)",
    )
    ax.barh(
        y_pos,
        df_sens["dampak_naik"],
        color=WARNA_TEAL,
        height=0.52,
        label="Naik (+25% rentang)",
    )
    ax.axvline(0, color=WARNA_TEKS, linewidth=1, alpha=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_sens["label"])
    ax.set_xlabel("Perubahan Skor Kelas (0.0 – 3.0)")
    ax.grid(axis="x", linestyle="--", color=WARNA_GRID, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(facecolor=WARNA_LATAR, edgecolor=WARNA_GRID, labelcolor=WARNA_TEKS, fontsize=8.5)

    batas_x = max(
        0.35,
        float(df_sens["dampak_turun"].abs().max()),
        float(df_sens["dampak_naik"].abs().max()),
    ) * 1.28
    ax.set_xlim(-batas_x, batas_x)

    fig.tight_layout()
    return fig


def buat_heatmap_confusion_matrix(cm: list[list[int]], label_kelas: list[str]):
    """Membuat visualisasi heatmap Confusion Matrix bertema gelap dengan aksen oranye."""
    fig, ax = plt.subplots(figsize=(5.5, 4.0), facecolor=WARNA_LATAR)
    atur_gaya_axes_gelap(ax)

    im = ax.imshow(cm, interpolation="nearest", cmap="Oranges")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=WARNA_TEKS)

    ax.set(
        xticks=range(len(label_kelas)),
        yticks=range(len(label_kelas)),
        xticklabels=label_kelas,
        yticklabels=label_kelas,
        xlabel="Prediksi Model",
        ylabel="Label Aktual",
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    ambang = max(max(baris) for baris in cm) / 2.0
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            warna_teks = "white" if cm[i][j] > ambang else "#141210"
            ax.text(
                j,
                i,
                str(cm[i][j]),
                ha="center",
                va="center",
                color=warna_teks,
                fontweight="bold",
            )

    fig.tight_layout()
    return fig


def buat_grafik_eksperimen_k(hasil_k: list[dict]):
    """Membuat grafik garis perbandingan Akurasi Latih vs Uji untuk tiap nilai k."""
    df_k = pd.DataFrame(hasil_k)
    fig, ax = plt.subplots(figsize=(5.8, 3.6), facecolor=WARNA_LATAR)
    atur_gaya_axes_gelap(ax)

    ax.plot(
        df_k["k"],
        df_k["train_accuracy"] * 100,
        marker="o",
        linewidth=2,
        color=WARNA_TEAL,
        label="Akurasi Latih (%)",
    )
    ax.plot(
        df_k["k"],
        df_k["test_accuracy"] * 100,
        marker="s",
        linewidth=2.2,
        color=WARNA_ORANYE,
        label="Akurasi Uji (%)",
    )

    ax.set_xticks(df_k["k"])
    ax.set_xlabel("Nilai k (Jumlah Tetangga Terdekat)")
    ax.set_ylabel("Akurasi (%)")
    ax.grid(True, linestyle="--", color=WARNA_GRID, alpha=0.7)
    ax.legend(facecolor=WARNA_LATAR, edgecolor=WARNA_GRID, labelcolor=WARNA_TEKS)
    fig.tight_layout()
    return fig


def render_sidebar(
    batas_fitur: dict[str, tuple[int, int, int]], metrik: dict
) -> dict[str, int]:
    """Merender panel kontrol slider di sidebar kiri seperti referensi What-If Simulator."""
    with st.sidebar:
        st.markdown("## SmartPrice Lab")
        st.caption("Panel Kontrol Spesifikasi Hardware")
        st.divider()

        ram = st.slider(
            "RAM (MB)",
            min_value=batas_fitur["ram"][0],
            max_value=batas_fitur["ram"][1],
            step=16,
            key="sim_ram",
            on_change=tandai_custom,
            help=f"Rentang dataset: {batas_fitur['ram'][0]} – {batas_fitur['ram'][1]} MB.",
        )
        battery_power = st.slider(
            "Kapasitas Baterai (mAh)",
            min_value=batas_fitur["battery_power"][0],
            max_value=batas_fitur["battery_power"][1],
            step=10,
            key="sim_battery_power",
            on_change=tandai_custom,
            help=f"Rentang dataset: {batas_fitur['battery_power'][0]} – {batas_fitur['battery_power'][1]} mAh.",
        )
        int_memory = st.slider(
            "Memori Internal (GB)",
            min_value=batas_fitur["int_memory"][0],
            max_value=batas_fitur["int_memory"][1],
            step=1,
            key="sim_int_memory",
            on_change=tandai_custom,
            help=f"Rentang dataset: {batas_fitur['int_memory'][0]} – {batas_fitur['int_memory'][1]} GB.",
        )
        px_width = st.slider(
            "Lebar Resolusi Layar (px)",
            min_value=batas_fitur["px_width"][0],
            max_value=batas_fitur["px_width"][1],
            step=10,
            key="sim_px_width",
            on_change=tandai_custom,
            help=f"Rentang dataset: {batas_fitur['px_width'][0]} – {batas_fitur['px_width'][1]} px.",
        )
        px_height = st.slider(
            "Tinggi Resolusi Layar (px)",
            min_value=batas_fitur["px_height"][0],
            max_value=batas_fitur["px_height"][1],
            step=10,
            key="sim_px_height",
            on_change=tandai_custom,
            help=f"Rentang dataset: {batas_fitur['px_height'][0]} – {batas_fitur['px_height'][1]} px.",
        )
        pc = st.slider(
            "Kamera Utama (MP)",
            min_value=batas_fitur["pc"][0],
            max_value=batas_fitur["pc"][1],
            step=1,
            key="sim_pc",
            on_change=tandai_custom,
            help=f"Rentang dataset: {batas_fitur['pc'][0]} – {batas_fitur['pc'][1]} MP.",
        )

        st.divider()
        with st.expander("Pengaturan & Info Model"):
            st.write(f"**Algoritma:** KNN (k = {metrik['best_k']})")
            st.write("**Scaler:** StandardScaler")
            st.write(f"**Akurasi Uji:** {float(metrik['test_accuracy']) * 100:.2f}%")
            st.write(f"**Total Dataset:** {metrik.get('dataset_rows', 2000)} baris")

        st.caption(
            "Setiap perubahan slider di panel ini langsung memperbarui hasil prediksi "
            "dan grafik analisis secara real-time."
        )

    return {
        "ram": int(ram),
        "battery_power": int(battery_power),
        "int_memory": int(int_memory),
        "px_width": int(px_width),
        "px_height": int(px_height),
        "pc": int(pc),
    }


def render_tab_simulasi(
    model,
    metrik: dict,
    input_sekarang: dict[str, int],
    batas_fitur: dict[str, tuple[int, int, int]],
    df_rata: pd.DataFrame | None,
) -> None:
    """Menampilkan Tab Simulasi Harga dengan KPI sparkline, grafik utama, brief, dan tornado What-If."""
    # Hitung inferensi real-time terlebih dahulu
    kelas_prediksi, probabilitas, skor_ekspektasi = hitung_skor_ekspektasi_kelas(
        model, input_sekarang
    )
    info_hasil = INFO_KATEGORI[kelas_prediksi]
    persen_4_kelas = [round(p * 100, 1) for p in probabilitas]
    keyakinan = persen_4_kelas[kelas_prediksi]

    # Hitung kelas runner-up dan skor kekuatan spek (0 - 100)
    urutan_kelas = sorted(range(4), key=lambda i: probabilitas[i], reverse=True)
    kelas_runner_up = urutan_kelas[1]
    selisih_peluang = keyakinan - persen_4_kelas[kelas_runner_up]

    persen_tiap_fitur = [
        round(
            ((input_sekarang[f] - batas_fitur[f][0]) / max(1, batas_fitur[f][1] - batas_fitur[f][0]))
            * 100,
            1,
        )
        for f in FITUR_UTAMA
    ]
    skor_spek = sum(persen_tiap_fitur) / len(persen_tiap_fitur)

    # Baris Header Halaman + Tombol Simpan Skenario & Reset
    col_head, col_act1, col_act2 = st.columns([2.6, 0.85, 0.85], gap="small")
    with col_head:
        st.header("Simulasi Harga")
        st.caption(
            f"RAM {input_sekarang['ram']} MB · Baterai {input_sekarang['battery_power']} mAh · "
            f"Memori {input_sekarang['int_memory']} GB · Layar {input_sekarang['px_width']}×{input_sekarang['px_height']} px · "
            f"Kamera {input_sekarang['pc']} MP"
        )
    with col_act1:
        if st.button("Simpan skenario", use_container_width=True):
            st.session_state["skenario_tersimpan"].append(
                {
                    "Profil": st.session_state["preset_pill"],
                    "Kategori": f"{kelas_prediksi} - {info_hasil['nama']}",
                    "Keyakinan (%)": keyakinan,
                    "RAM (MB)": input_sekarang["ram"],
                    "Baterai (mAh)": input_sekarang["battery_power"],
                    "Memori (GB)": input_sekarang["int_memory"],
                    "Layar (px)": f"{input_sekarang['px_width']}x{input_sekarang['px_height']}",
                    "Kamera (MP)": input_sekarang["pc"],
                }
            )
            st.toast(f"Skenario {info_hasil['nama']} disimpan ke tab Skenario Tersimpan.")
    with col_act2:
        st.button(
            "Reset parameter",
            on_click=reset_parameter,
            args=(batas_fitur,),
            use_container_width=True,
        )

    # Pill Preset Cepat (seperti Jackets | Boots | Tents | Daypacks di referensi)
    st.segmented_control(
        "Preset Cepat Skenario",
        options=["Entry-Level", "Mid-Range", "Flagship", "Custom"],
        key="preset_pill",
        on_change=terapkan_preset_pill,
        label_visibility="collapsed",
    )

    # BARIS 1: 4 Kartu KPI dengan mini sparkline bawaan st.metric
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.metric(
        "Kategori Harga",
        info_hasil["nama"],
        delta=f"Kelas {kelas_prediksi} dari 0–3",
        border=True,
        chart_data=persen_4_kelas,
        chart_type="bar",
    )
    k2.metric(
        "Tingkat Keyakinan",
        f"{keyakinan:.1f}%",
        delta=f"+{selisih_peluang:.1f}% vs {INFO_KATEGORI[kelas_runner_up]['nama']}",
        border=True,
        chart_data=persen_4_kelas,
        chart_type="area",
    )
    k3.metric(
        "Kisaran Harga",
        info_hasil["kisaran_singkat"],
        delta=info_hasil["segmen"],
        border=True,
        chart_data=[round(skor_ekspektasi, 2), round(skor_ekspektasi, 2)],
        chart_type="line",
    )
    k4.metric(
        "Skor Kekuatan Spek",
        f"{skor_spek:.0f} / 100",
        delta=f"Profil: {st.session_state['preset_pill']}",
        border=True,
        chart_data=persen_tiap_fitur,
        chart_type="bar",
    )

    # BARIS 2: Grafik Utama (Kiri) & Model Brief / Ringkasan Analitis (Kanan)
    df_sens = analisis_sensitivitas_what_if(model, input_sekarang, batas_fitur)
    fitur_pendorong_naik = df_sens.sort_values("dampak_naik", ascending=False).iloc[0]
    fitur_risiko_turun = df_sens.sort_values("dampak_turun", ascending=True).iloc[0]

    col_grafik, col_brief = st.columns([1.45, 1], gap="medium")
    with col_grafik:
        with st.container(border=True):
            cg_title, cg_toggle = st.columns([1.4, 1])
            with cg_title:
                st.subheader("Profil Probabilitas & Fitur")
                st.caption("Distribusi keyakinan model KNN pada 4 kelas harga")
            with cg_toggle:
                tampilan = st.segmented_control(
                    "Mode Grafik",
                    options=["Probabilitas", "Profil Spek"],
                    key="toggle_grafik",
                    label_visibility="collapsed",
                )

            if tampilan == "Profil Spek":
                fig_utama = buat_grafik_profil_fitur(input_sekarang, batas_fitur)
            else:
                fig_utama = buat_grafik_probabilitas(probabilitas, kelas_prediksi)
            st.pyplot(fig_utama, use_container_width=True)
            plt.close(fig_utama)

    with col_brief:
        with st.container(border=True):
            st.subheader("Ringkasan Simulasi")
            st.caption("Interpretasi otomatis dari model KNN berdasarkan input aktif")

            st.info(
                f"**Hasil Utama:** Smartphone diprediksi berada di kelas **{info_hasil['nama']} "
                f"(Kelas {kelas_prediksi})** dengan keyakinan **{keyakinan:.1f}%** "
                f"({info_hasil['kisaran']})."
            )
            st.markdown(
                f"- **Peluang Kelas Terdekat:** Kelas **{INFO_KATEGORI[kelas_runner_up]['nama']}** "
                f"memiliki probabilitas **{persen_4_kelas[kelas_runner_up]:.1f}%** "
                f"(selisih **{selisih_peluang:.1f}%**).\n"
                f"- **Pendorong Naik Kelas Terkuat:** Menaikkan **{fitur_pendorong_naik['label']}** "
                f"ke `{int(fitur_pendorong_naik['nilai_naik'])}` memberi perubahan skor "
                f"**{fitur_pendorong_naik['dampak_naik']:+.2f}**.\n"
                f"- **Risiko Turun Kelas Terbesar:** Menurunkan **{fitur_risiko_turun['label']}** "
                f"ke `{int(fitur_risiko_turun['nilai_turun'])}` menggeser skor sebesar "
                f"**{fitur_risiko_turun['dampak_turun']:+.2f}**.\n"
                f"- **Catatan Dataset:** Label kelas 0–3 merepresentasikan rentang harga relatif "
                f"dari `mobile_price.csv`, sehingga nominal Rupiah bersifat ilustratif."
            )

    # BARIS 3: Perbandingan Rata-Rata RAM per Kelas (Kiri) & Tornado What-If (Kanan)
    col_bawah_kiri, col_bawah_kanan = st.columns(2, gap="medium")
    with col_bawah_kiri:
        with st.container(border=True):
            st.subheader("Posisi RAM vs Rata-Rata Kelas")
            st.caption("Garis putus-putus menunjukkan nilai RAM aktif terhadap rata-rata tiap kelas di dataset")
            if df_rata is not None:
                fig_ram = buat_grafik_perbandingan_ram_kelas(
                    df_rata, input_sekarang, kelas_prediksi
                )
                st.pyplot(fig_ram, use_container_width=True)
                plt.close(fig_ram)
            else:
                df_prob_tbl = pd.DataFrame(
                    {
                        "Kategori": [f"{i} - {INFO_KATEGORI[i]['nama']}" for i in range(4)],
                        "Probabilitas (%)": persen_4_kelas,
                    }
                )
                st.dataframe(df_prob_tbl, use_container_width=True, hide_index=True)

    with col_bawah_kanan:
        with st.container(border=True):
            st.subheader("Pengaruh Tiap Fitur (What-If)")
            st.caption("Dampak menggeser tiap fitur ±25% rentang dataset terhadap skor kelas KNN")
            fig_tornado = buat_grafik_tornado_what_if(df_sens)
            st.pyplot(fig_tornado, use_container_width=True)
            plt.close(fig_tornado)


def render_tab_evaluasi(metrik: dict) -> None:
    """Menampilkan Tab 2: Skor akurasi latih vs uji, Confusion Matrix, eksperimen k, dan metode."""
    akurasi_latih = float(metrik["train_accuracy"]) * 100
    akurasi_uji = float(metrik["test_accuracy"]) * 100
    selisih = abs(akurasi_latih - akurasi_uji)
    k_terbaik = int(metrik["best_k"])

    st.header("Evaluasi Model")
    st.caption(
        f"Pembagian stratified: {metrik.get('train_rows', 1500)} data latih (75%) dan "
        f"{metrik.get('test_rows', 500)} data uji (25%) dari {metrik.get('dataset_rows', 2000)} baris dataset."
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Akurasi Data Latih", f"{akurasi_latih:.2f}%", border=True)
    c2.metric("Akurasi Data Uji", f"{akurasi_uji:.2f}%", border=True)
    c3.metric("Selisih Latih vs Uji", f"{selisih:.2f}%", border=True)
    c4.metric("Nilai k Terpilih", f"k = {k_terbaik}", border=True)

    cm = metrik["confusion_matrix"]
    label_kelas = [f"{i} ({INFO_KATEGORI[i]['nama']})" for i in range(4)]

    col_cm1, col_cm2 = st.columns(2, gap="medium")
    with col_cm1:
        with st.container(border=True):
            st.subheader("Tabel Confusion Matrix (Data Uji)")
            df_cm = pd.DataFrame(
                cm,
                index=[f"Aktual {lbl}" for lbl in label_kelas],
                columns=[f"Prediksi {lbl}" for lbl in label_kelas],
            )
            st.dataframe(df_cm, use_container_width=True)
            st.caption("Diagonal utama adalah jumlah prediksi benar pada 500 data uji.")

    with col_cm2:
        with st.container(border=True):
            st.subheader("Heatmap Confusion Matrix")
            fig_cm = buat_heatmap_confusion_matrix(cm, label_kelas)
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)

    hasil_k = metrik.get("k_experiment", [])
    laporan = metrik.get("classification_report", {})

    col_k, col_rep = st.columns(2, gap="medium")
    with col_k:
        with st.container(border=True):
            st.subheader("Eksperimen Nilai k (3, 5, 7, 9, 11)")
            if hasil_k:
                fig_k = buat_grafik_eksperimen_k(hasil_k)
                st.pyplot(fig_k, use_container_width=True)
                plt.close(fig_k)

                df_k = pd.DataFrame(hasil_k)
                df_k["train_accuracy"] = (df_k["train_accuracy"] * 100).round(2)
                df_k["test_accuracy"] = (df_k["test_accuracy"] * 100).round(2)
                df_k["gap"] = (df_k["gap"] * 100).round(2)
                df_k.columns = ["Nilai k", "Akurasi Latih (%)", "Akurasi Uji (%)", "Selisih (%)"]
                st.dataframe(df_k, use_container_width=True, hide_index=True)

    with col_rep:
        with st.container(border=True):
            st.subheader("Classification Report & Metode")
            baris_laporan = []
            for lbl in label_kelas:
                if lbl in laporan:
                    skor = laporan[lbl]
                    baris_laporan.append(
                        {
                            "Kelas": lbl,
                            "Precision": round(float(skor["precision"]), 4),
                            "Recall": round(float(skor["recall"]), 4),
                            "F1-Score": round(float(skor["f1-score"]), 4),
                            "Support": int(skor["support"]),
                        }
                    )
            if baris_laporan:
                st.dataframe(
                    pd.DataFrame(baris_laporan),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown(
                f"**Ringkasan Metode:**\n"
                f"- **KNN (k={k_terbaik}):** Menentukan kelas berdasarkan suara mayoritas dari {k_terbaik} "
                f"tetangga terdekat memakai jarak Euclidean (nilai `k` ganjil mencegah suara seri).\n"
                f"- **StandardScaler:** Menormalkan skala fitur (misal `ram` ribuan MB vs `pc` puluhan MP) "
                f"sebelum perhitungan jarak agar semua fitur berbobot seimbang."
            )


def render_tab_skenario() -> None:
    """Menampilkan Tab 3: Daftar skenario simulasi yang disimpan pengguna untuk perbandingan."""
    st.header("Skenario Tersimpan")
    st.caption(
        "Klik tombol **Simpan skenario** di Tab Simulasi Harga untuk membandingkan beberapa konfigurasi smartphone."
    )

    daftar = st.session_state.get("skenario_tersimpan", [])
    if not daftar:
        st.info("Belum ada skenario yang disimpan pada sesi ini.")
        return

    with st.container(border=True):
        st.dataframe(pd.DataFrame(daftar), use_container_width=True, hide_index=True)
        if st.button("Hapus semua skenario tersimpan"):
            st.session_state["skenario_tersimpan"] = []
            st.rerun()


def main() -> None:
    """Titik masuk utama aplikasi web Streamlit What-If Simulator."""
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        st.error(
            "File model (`model_knn_smartphone.joblib`) atau ringkasan metrik (`metrics_summary.json`) "
            "belum ditemukan. Jalankan `python train_model.py` terlebih dahulu di terminal."
        )
        st.stop()

    model = muat_model(MODEL_PATH)
    metrik = muat_metrik(METRICS_PATH)
    df_rata = muat_rata_rata_kelas(DATASET_PATH)

    batas_fitur = daftar_batas_lengkap(metrik.get("feature_ranges", {}))
    inisialisasi_state(batas_fitur)

    input_sekarang = render_sidebar(batas_fitur, metrik)

    tab_simulasi, tab_evaluasi, tab_skenario = st.tabs(
        ["Simulasi Harga", "Evaluasi Model", "Skenario Tersimpan"]
    )

    with tab_simulasi:
        render_tab_simulasi(model, metrik, input_sekarang, batas_fitur, df_rata)

    with tab_evaluasi:
        render_tab_evaluasi(metrik)

    with tab_skenario:
        render_tab_skenario()


if __name__ == "__main__":
    main()
