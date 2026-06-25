"""
================================================================================
 AIoT Smart Irrigation System — Flask API Server (V2.0)
================================================================================
 Menerima data sensor dari ESP32 via HTTP POST
 Menjalankan inferensi model ML menggunakan pipeline preprocessed pkl
 Mengembalikan prediksi ke ESP32
 
 Kelas:
   0 = Tidak Siram
   1 = Siram
   2 = Siram Banyak
================================================================================
"""

import os
import json
import datetime
import csv
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# ==============================================================================
# KONFIGURASI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
LOG_FILE = os.path.join(OUTPUT_DIR, 'prediction_log.csv')
HOST = '0.0.0.0'
PORT = 5000

# Label mapping
LABEL_MAP = {0: 'TIDAK', 1: 'SIRAM', 2: 'BANYAK'}
ESP32_ACTION = {0: 0, 1: 1, 2: 1}  # 0 = relay OFF, 1 atau 2 = relay ON

# Nilai default untuk data kategorikal jika tidak dikirim oleh ESP32
DEFAULT_CROP = 'Wheat'
DEFAULT_SOIL = 'Black Soil'
DEFAULT_STAGE = 'Germination'

# ==============================================================================
# INISIALISASI SERVER
# ==============================================================================
app = Flask(__name__)
CORS(app)

print("Memuat pipeline model...")
try:
    pipeline = joblib.load(os.path.join(OUTPUT_DIR, 'model_pipeline.pkl'))
    summary = joblib.load(os.path.join(OUTPUT_DIR, 'training_summary.pkl'))
    print(f"Pipeline model berhasil dimuat. Akurasi Pelatihan: {summary['accuracy']*100:.3f}%")
    MODEL_LOADED = True
except Exception as e:
    print(f"Gagal memuat pipeline model: {e}")
    print("Pastikan untuk menjalankan train_model.py terlebih dahulu.")
    MODEL_LOADED = False
    pipeline = None
    summary = None

prediction_history = []


def log_prediction(sensor_data, prediction, confidence, label):
    """Menyimpan data transaksi prediksi ke dalam file log CSV."""
    try:
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'temperature', 'humidity',
                    'soil_moisture', 'crop_id', 'soil_type',
                    'seedling_stage', 'prediction', 'label', 'confidence'
                ])
            writer.writerow([
                datetime.datetime.now().isoformat(),
                sensor_data.get('temperature'),
                sensor_data.get('humidity'),
                sensor_data.get('soil_moisture'),
                sensor_data.get('crop_id'),
                sensor_data.get('soil_type'),
                sensor_data.get('seedling_stage'),
                prediction,
                label,
                f'{confidence:.4f}'
            ])
    except Exception as e:
        print(f"Gagal menulis log: {e}")


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.route('/', methods=['GET'])
def home():
    """Endpoint informasi status server."""
    return jsonify({
        'name': 'AIoT Smart Irrigation API Server',
        'version': '2.0',
        'status': 'running',
        'model_loaded': MODEL_LOADED,
        'model_accuracy': f"{summary['accuracy']*100:.3f}%" if summary else None,
        'endpoints': {
            'POST /predict': 'Mengirimkan data sensor untuk diprediksi',
            'GET /health': 'Memeriksa kesehatan koneksi server',
            'GET /history': 'Mengambil riwayat prediksi terbaru',
            'GET /model-info': 'Mengambil detail performa model'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Endpoint pengecekan koneksi."""
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL_LOADED,
        'timestamp': datetime.datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint utama prediksi irigasi cerdas.
    Menerima format data JSON dari ESP32.
    """
    if not MODEL_LOADED:
        return jsonify({'error': 'Pipeline model belum dimuat.'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Payload data tidak valid atau kosong.'}), 400

        # Ekstraksi dan penyesuaian nama parameter sensor
        # ESP32: temperature, humidity, soil_moisture
        # Model: temp, humidity, MOI
        temp = data.get('temperature')
        humidity = data.get('humidity')
        moi = data.get('soil_moisture')

        # Cek fallback jika nama parameter berbeda
        if temp is None:
            temp = data.get('temp') or data.get('suhu')
        if humidity is None:
            humidity = data.get('hum') or data.get('kelembapan')
        if moi is None:
            moi = data.get('MOI') or data.get('moisture')

        # Validasi input numerik
        if temp is None or humidity is None or moi is None:
            return jsonify({
                'error': 'Fitur input numerik tidak lengkap.',
                'required': ['temperature', 'humidity', 'soil_moisture'],
                'received': data
            }), 400

        # Ekstraksi parameter kategorikal (menggunakan default jika tidak dikirim)
        crop_id = data.get('crop_id', DEFAULT_CROP)
        soil_type = data.get('soil_type', DEFAULT_SOIL)
        seedling_stage = data.get('seedling_stage', DEFAULT_STAGE)

        # Buat DataFrame agar sesuai dengan pipeline preprocessor sklearn
        input_data = pd.DataFrame([{
            'crop ID': crop_id,
            'soil_type': soil_type,
            'Seedling Stage': seedling_stage,
            'MOI': float(moi),
            'temp': float(temp),
            'humidity': float(humidity)
        }])

        # Lakukan inferensi prediksi menggunakan Pipeline
        prediction = int(pipeline.predict(input_data)[0])
        probabilities = pipeline.predict_proba(input_data)[0]
        confidence = float(max(probabilities))

        label = LABEL_MAP.get(prediction, 'UNKNOWN')
        action = ESP32_ACTION.get(prediction, 0)

        # Output payload
        result = {
            'prediction': prediction,
            'label': label,
            'action': action,
            'confidence': round(confidence, 5),
            'probabilities': {
                LABEL_MAP[i]: round(float(p), 5)
                for i, p in enumerate(probabilities)
            },
            'processed_features': {
                'temperature': float(temp),
                'humidity': float(humidity),
                'soil_moisture': float(moi),
                'crop_id': crop_id,
                'soil_type': soil_type,
                'seedling_stage': seedling_stage
            },
            'timestamp': datetime.datetime.now().isoformat()
        }

        # Logging ke file
        log_prediction({
            'temperature': temp,
            'humidity': humidity,
            'soil_moisture': moi,
            'crop_id': crop_id,
            'soil_type': soil_type,
            'seedling_stage': seedling_stage
        }, prediction, confidence, label)

        # Simpan ke memori riwayat
        prediction_history.append(result)
        if len(prediction_history) > 100:
            prediction_history.pop(0)

        # Print log server tanpa emotikon
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
              f"Prediksi: {label} ({confidence*100:.2f}%) | "
              f"Relay: {'ON' if action else 'OFF'} | "
              f"Input: temp={temp}, hum={humidity}, soil_moisture={moi}, "
              f"crop={crop_id}, soil={soil_type}, stage={seedling_stage}")

        return jsonify(result)

    except Exception as e:
        print(f"Error pada inferensi model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['GET'])
def history():
    """Mengambil riwayat prediksi server."""
    limit = request.args.get('limit', 50, type=int)
    return jsonify({
        'count': len(prediction_history),
        'predictions': prediction_history[-limit:]
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    """Informasi metadata model."""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model belum dimuat.'}), 503

    return jsonify({
        'model_type': summary.get('model_type'),
        'accuracy': summary.get('accuracy'),
        'precision_w': summary.get('precision_w'),
        'recall_w': summary.get('recall_w'),
        'f1_score_w': summary.get('f1_score_w'),
        'n_estimators': summary.get('n_estimators'),
        'training_samples': summary.get('dataset_shape'),
    })


# ==============================================================================
# RUN SERVER
# ==============================================================================
if __name__ == '__main__':
    print("="*80)
    print("  AIoT Smart Irrigation Flask API Server v2.0")
    print("="*80)
    print(f"  Server berjalan pada host: {HOST}:{PORT}")
    print("="*80 + "\n")
    app.run(host=HOST, port=PORT, debug=True)
