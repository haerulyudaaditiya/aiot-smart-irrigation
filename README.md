# AIoT Smart Irrigation System

Sistem Irigasi Cerdas berbasis Artificial Intelligence of Things (AIoT) yang mengintegrasikan sensor perangkat keras (ESP32), Machine Learning (Random Forest Classifier), protokol komunikasi HTTP, dan web dashboard pemantauan.

Sistem ini dirancang untuk mendeteksi kebutuhan penyiraman tanaman secara otomatis berdasarkan kondisi lingkungan aktual, dengan fitur keselamatan pengaman level air.

## Arsitektur Sistem

Sistem terdiri dari tiga komponen utama:
1. **Perangkat Keras (ESP32 Edge Node)**: Berfungsi membaca data dari sensor DHT22 (Suhu dan Kelembapan Udara), Soil Moisture Sensor (Kelembapan Tanah), dan Water Level Sensor (Volume Air Cadangan). Mengontrol relay pompa air berdasarkan instruksi dari server.
2. **Protokol HTTP (Komunikasi)**: Menghubungkan ESP32 dan server Flask menggunakan request HTTP POST untuk mengirimkan data sensor dan menerima respons prediksi.
3. **Server Flask & Machine Learning**: Memproses data sensor masuk menggunakan model Random Forest Classifier yang sudah dilatih untuk menentukan status penyiraman (TIDAK, SIRAM, BANYAK).

## Skema Sambungan Kabel (Wiring)

Berikut adalah konfigurasi pin yang digunakan pada ESP32:
* **DHT22 (Sensor Suhu & Kelembapan)**: Data -> GPIO 14, VCC -> 3.3V, GND -> GND
* **Soil Moisture Sensor (Analog)**: AO -> GPIO 32, VCC -> 3.3V, GND -> GND
* **Water Level Sensor (Analog)**: Analog Out -> GPIO 34, VCC -> 3.3V, GND -> GND
* **Relay Pompa**: Control Pin -> GPIO 26, VCC -> 5V, GND -> GND
* **I2C LCD 16x2**: SDA -> GPIO 21, SCL -> GPIO 22, VCC -> 5V, GND -> GND

## Struktur Direktori Proyek

```text
aiot-smart-irrigation/
├── latihan.ino          # Kode program Arduino/C++ untuk ESP32
├── model_training/      # Direktori server Python dan pelatihan ML
│   ├── dataset/         # Dataset pelatihan (.csv)
│   ├── output/          # Model pipeline (.pkl) yang siap pakai
│   ├── server.py        # Server Flask API
│   ├── train_model.py   # Script pelatihan model machine learning
│   └── requirements.txt # Dependensi library Python
├── .gitignore           # File konfigurasi abaikan git
└── README.md            # Dokumentasi proyek ini
```

## Fitur Unggulan

* **Klasifikasi Machine Learning**: Menghitung keputusan penyiraman menggunakan Random Forest Classifier berdasarkan kombinasi suhu, kelembapan udara, kelembapan tanah, tipe tanah, jenis tanaman, dan fase pertumbuhan.
* **Protokol HTTP POST**: Pengiriman data berkala dari perangkat keras ke server untuk inferensi model secara real-time.
* **Fitur Keselamatan Fisik (Water Level Protection)**: ESP32 akan mengabaikan instruksi menyiram dari model AI jika sensor mendeteksi cadangan air habis (level air di bawah 15%) untuk mencegah kerusakan motor pompa (dry run).
* **Konfigurasi Default**: Server menggunakan konfigurasi bawaan untuk tipe tanaman (Wheat), tipe tanah (Black Soil), dan fase pertumbuhan (Germination) dalam pemrosesan inferensi.

## Panduan Instalasi dan Penggunaan

### 1. Menjalankan Server Flask
1. Masuk ke direktori server:
   ```bash
   cd model_training
   ```
2. Buat virtual environment dan aktifkan:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Untuk Linux/macOS
   # atau
   venv\Scripts\activate     # Untuk Windows
   ```
3. Instal semua dependensi:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan server Flask:
   ```bash
   python server.py
   ```

### 2. Mengunggah Program ke ESP32
1. Buka file `latihan.ino` menggunakan Arduino IDE.
2. Pastikan library berikut sudah terinstal di Arduino IDE:
   * `DHT sensor library` (oleh Adafruit)
   * `LiquidCrystal I2C`
   * `ArduinoJson` (oleh Benoit Blanchon)
3. Sesuaikan nama WiFi dan kata sandi sesuai dengan konfigurasi jaringan yang digunakan.
4. Upload program ke board ESP32.
