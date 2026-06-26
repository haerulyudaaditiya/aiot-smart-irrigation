# Rencana Implementasi - AIoT Smart Irrigation System (MQTT & Web Dashboard)

## 1. Ringkasan Proyek

Sistem irigasi cerdas ini mengintegrasikan mikrokontroler ESP32, model Machine Learning (Random Forest Classifier), dan dashboard pemantauan berbasis web. Komunikasi antara perangkat keras dan server backend diubah dari protokol HTTP POST langsung menjadi arsitektur berbasis MQTT untuk memudahkan penempatan (deployment) dan pengujian tanpa ketergantungan pada alamat IP lokal.

### Arsitektur Sistem Baru (MQTT-Based)
* **ESP32**: Membaca sensor fisik dan mempublikasikan data telemetry ke topik sensor MQTT. ESP32 juga berlangganan ke topik kontrol MQTT untuk menerima perintah aktivasi pompa.
* **MQTT Broker**: Menggunakan broker publik `broker.emqx.io` pada port 1883 sebagai perantara transmisi data.
* **Flask Server & ML Inference**: Berlangganan ke topik sensor MQTT untuk memproses data menggunakan model ML, kemudian mempublikasikan keputusan ke topik kontrol MQTT. Server juga menyediakan REST API untuk menyajikan dashboard pemantauan.
* **Web Dashboard**: Menyajikan visualisasi data sensor real-time, log transaksi riwayat, dan panel kontrol konfigurasi jenis tanaman.

---

## 2. Struktur File Proyek

```text
latihan/
├── .gitignore
├── README.md
├── IMPLEMENTATION_PLAN.md          <- Dokumen rencana implementasi ini
├── latihan.ino                     <- Kode ESP32 (MQTT Client & Actuator)
└── model_training/
    ├── dataset/
    │   └── smart_agriculture.csv   <- Dataset publik
    ├── output/
    │   ├── model_pipeline.pkl      <- Serialisasi model & preprocessor
    │   └── training_summary.pkl    <- Metadata performa model
    ├── static/                     <- Aset untuk Web Dashboard
    │   ├── css/
    │   │   └── dashboard.css       <- Desain UI Premium (Dark Mode/Glassmorphism)
    │   └── js/
    │   │   └── dashboard.js        <- Logika real-time update & Chart.js
    │   └── assets/                 <- Gambar / icon pendukung dashboard
    ├── templates/
    │   └── index.html              <- Halaman utama Web Dashboard
    ├── train_model.py              <- Pelatihan model ML
    ├── server.py                   <- Flask Server & MQTT Daemon Thread
    └── requirements.txt            <- Dependencies Python (ditambah paho-mqtt & gunicorn)
```

---

## 3. Rencana Langkah Implementasi

### Fase 1: Setup MQTT di Flask Server & ESP32 secara Lokal [SELESAI]
**Target**: Memastikan pengiriman data telemetry dan pengiriman balik hasil instruksi kontrol ML berjalan lancar melalui MQTT Broker.

* **Task 1.1: Instalasi Library Python** (Selesai - `paho-mqtt` diinstal di venv)
* **Task 1.2: Pembaruan `server.py`** (Selesai - daemon thread berjalan dan terintegrasi dengan callbacks)
* **Task 1.3: Pembaruan `latihan.ino`** (Selesai - PubSubClient MQTT loop terpasang)
* **Task 1.4: Pengujian Telemetry Lokal** (Selesai - diverifikasi dengan test_mqtt.py)

---

### Fase 2: Pembuatan Dashboard Premium & Konfigurasi Tanaman [SELESAI]
**Target**: Menyajikan halaman antarmuka web interaktif yang modern untuk memantau status fisik tanaman dan mengubah parameter inferensi ML.

* **Task 2.1: Implementasi API Konfigurasi Dinamis** (Selesai - `ACTIVE_CONFIG` dict dan route `/api/config` GET/POST berfungsi)
* **Task 2.2: Pembuatan Halaman Dashboard Premium** (Selesai - templates/index.html, static/css/dashboard.css, dan static/js/dashboard.js dengan Chart.js, tema Light Mode profesional, dan status pompa/level air interaktif telah selesai dibuat dan diverifikasi)

---

### Fase 3: Deployment ke Cloud (Render)
**Target**: Memindahkan server backend ke server cloud agar dapat diakses secara publik dan terhubung dengan ESP32 dari jaringan mana pun tanpa mengubah firmware.

* **Task 3.1: Persiapan File Produksi**
  * Tambahkan `gunicorn` ke `requirements.txt` sebagai server WSGI produksi.
  * Pastikan binding port menggunakan variabel lingkungan dari Render (`PORT`).
* **Task 3.2: Push Perubahan ke GitHub**
  * Pastikan seluruh file baru terindeks oleh Git.
  * Lakukan commit dan push ke branch utama GitHub.
* **Task 3.3: Konfigurasi di Render**
  * Buat Web Service baru di Render yang terhubung ke repository GitHub.
  * Masukkan perintah start: `gunicorn --directory model_training server:app`.
* **Task 3.4: Pengujian Akhir**
  * Verifikasi web dashboard dapat diakses via URL publik Render.
  * Pastikan ESP32 tetap mengirim data dan menerima kontrol pompa secara otomatis tanpa perlu melakukan flash ulang (karena broker MQTT yang digunakan tetap sama).

---

## 4. Parameter Topik MQTT

Untuk menghindari interferensi data dengan sistem lain pada broker publik, berikut alokasi topik yang digunakan:

* **Sensor Telemetry**: `aiot/smart_irrigation/sensor`
  * Format Payload: `{"temperature": 32.5, "humidity": 60.0, "soil_moisture": 45.0, "water_level": 80.0}`
* **Control Actuator**: `aiot/smart_irrigation/control`
  * Format Payload: `{"prediction": 1, "label": "SIRAM", "action": 1, "confidence": 0.925}`
