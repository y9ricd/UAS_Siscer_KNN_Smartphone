"""Pengujian otomatis untuk halaman utama dan endpoint API prediksi Django."""

import json

from django.test import TestCase
from django.urls import reverse


class KlasifikasiAppTests(TestCase):
    """Memverifikasi rendering halaman utama, prediksi API valid, dan penolakan input tidak valid."""

    def test_halaman_utama_status_200(self):
        """Memastikan halaman utama dapat diakses dengan status HTTP 200 dan memuat kedua tab."""
        response = self.client.get(reverse("beranda"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Simulasi Harga")
        self.assertContains(response, "Evaluasi Model")

    def test_api_predict_mengembalikan_prediksi_valid(self):
        """Memastikan endpoint POST /api/predict/ mengembalikan kategori valid untuk 3 skenario kontras."""
        skenario = [
            (
                {
                    "ram": 512,
                    "battery_power": 600,
                    "int_memory": 8,
                    "px_width": 640,
                    "px_height": 480,
                    "pc": 2,
                },
                (0,),
            ),
            (
                {
                    "ram": 2048,
                    "battery_power": 1200,
                    "int_memory": 32,
                    "px_width": 1200,
                    "px_height": 720,
                    "pc": 10,
                },
                (1, 2),
            ),
            (
                {
                    "ram": 3800,
                    "battery_power": 1900,
                    "int_memory": 64,
                    "px_width": 1900,
                    "px_height": 1800,
                    "pc": 16,
                },
                (3,),
            ),
        ]

        for payload, kelas_diharapkan in skenario:
            response = self.client.post(
                reverse("api_predict"),
                data=json.dumps(payload),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertIn(body["kelas"], kelas_diharapkan)
            self.assertEqual(len(body["probabilitas_kelas"]), 4)
            self.assertIn("ringkasan_analisis", body)

    def test_api_predict_menolak_input_tidak_valid_dengan_400(self):
        """Memastikan input non-numerik atau di luar rentang mengembalikan HTTP 400 tanpa crash 500."""
        payload_salah = {
            "ram": 99999,
            "battery_power": "bukan-angka",
            "int_memory": 32,
            "px_width": 1200,
            "px_height": 720,
            "pc": 10,
        }
        response = self.client.post(
            reverse("api_predict"),
            data=json.dumps(payload_salah),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertIn("error", body)
        self.assertIn("ram", body["detail"])
        self.assertIn("battery_power", body["detail"])

    def test_health_dan_evaluasi_endpoint(self):
        """Memastikan endpoint health dan evaluasi JSON mengembalikan status 200."""
        res_health = self.client.get(reverse("health_check"))
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "ok")

        res_eval = self.client.get(reverse("api_evaluasi"))
        self.assertEqual(res_eval.status_code, 200)
        self.assertIn("confusion_matrix_rows", res_eval.json())
