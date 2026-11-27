"""Pengujian otomatis untuk aplikasi Django Klasifikasi Harga Smartphone."""

from django.test import TestCase
from django.urls import reverse


class KlasifikasiViewTests(TestCase):
    """Memverifikasi rendering halaman, 3 skenario kontras prediksi, dan validasi input."""

    def test_beranda_dapat_diakses(self):
        response = self.client.get(reverse("beranda"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Prediksi Kategori Harga")
        self.assertContains(response, "Evaluasi Model")

    def test_health_endpoint_siap(self):
        response = self.client.get(reverse("health_check"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["best_k"], 9)

    def test_tiga_skenario_prediksi_kontras(self):
        # 1. Spek rendah -> Murah (0)
        res_low = self.client.post(
            reverse("beranda"),
            {
                "ram": 512,
                "battery_power": 600,
                "int_memory": 8,
                "px_width": 640,
                "px_height": 480,
                "pc": 2,
            },
        )
        self.assertEqual(res_low.status_code, 200)
        self.assertEqual(res_low.context["hasil_prediksi"]["kelas"], 0)
        self.assertEqual(res_low.context["hasil_prediksi"]["nama"], "Murah")

        # 2. Spek menengah -> Menengah (1) atau Mahal (2)
        res_mid = self.client.post(
            reverse("beranda"),
            {
                "ram": 2048,
                "battery_power": 1200,
                "int_memory": 32,
                "px_width": 1200,
                "px_height": 720,
                "pc": 10,
            },
        )
        self.assertEqual(res_mid.status_code, 200)
        self.assertIn(res_mid.context["hasil_prediksi"]["kelas"], (1, 2))

        # 3. Spek tinggi -> Flagship (3)
        res_high = self.client.post(
            reverse("beranda"),
            {
                "ram": 3800,
                "battery_power": 1900,
                "int_memory": 64,
                "px_width": 1900,
                "px_height": 1800,
                "pc": 16,
            },
        )
        self.assertEqual(res_high.status_code, 200)
        self.assertEqual(res_high.context["hasil_prediksi"]["kelas"], 3)
        self.assertEqual(res_high.context["hasil_prediksi"]["nama"], "Flagship")

    def test_validasi_input_di_luar_rentang_ditolak(self):
        response = self.client.post(
            reverse("beranda"),
            {
                "ram": 16000,  # Di luar batas maksimal dataset (3998 MB)
                "battery_power": 1200,
                "int_memory": 32,
                "px_width": 1200,
                "px_height": 720,
                "pc": 10,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context["hasil_prediksi"])
        self.assertTrue(response.context["form"].errors)
