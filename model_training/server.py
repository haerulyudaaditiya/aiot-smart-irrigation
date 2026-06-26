"""
================================================================================
 AIoT Smart Irrigation System — Flask API Server
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
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import paho.mqtt.client as mqtt

# ==============================================================================
# KONFIGURASI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
LOG_FILE = os.path.join(OUTPUT_DIR, 'prediction_log.csv')
HOST = '0.0.0.0'
PORT = int(os.environ.get('PORT', 5000))

# MQTT CONFIGURATION
MQTT_BROKER = os.environ.get('MQTT_BROKER', "broker.emqx.io")
MQTT_PORT = int(os.environ.get('MQTT_PORT', 1883))
MQTT_TOPIC_SENSOR = os.environ.get('MQTT_TOPIC_SENSOR', "aiot/smart_irrigation/sensor")
MQTT_TOPIC_CONTROL = os.environ.get('MQTT_TOPIC_CONTROL', "aiot/smart_irrigation/control")

# Label mapping
LABEL_MAP = {0: 'TIDAK', 1: 'SIRAM', 2: 'BANYAK'}
ESP32_ACTION = {0: 0, 1: 1, 2: 1}  # 0 = relay OFF, 1 atau 2 = relay ON

# Nilai default untuk data kategorikal jika tidak dikirim oleh ESP32
DEFAULT_CROP = 'Wheat'
DEFAULT_SOIL = 'Black Soil'
DEFAULT_STAGE = 'Germination'

# Dynamic ML Configuration
ACTIVE_CONFIG = {
    'crop_id': DEFAULT_CROP,
    'soil_type': DEFAULT_SOIL,
    'seedling_stage': DEFAULT_STAGE
}

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


def load_prediction_history_from_csv():
    """Memuat data riwayat dari CSV ke memori saat startup."""
    global prediction_history
    if not os.path.exists(LOG_FILE):
        return
    try:
        temp_history = []
        with open(LOG_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pred = int(row.get('prediction', 0))
                    confidence = float(row.get('confidence', 0.0))
                    label = row.get('label', 'TIDAK')
                    action = ESP32_ACTION.get(pred, 0)
                    
                    item = {
                        'prediction': pred,
                        'label': label,
                        'action': action,
                        'confidence': round(confidence, 5),
                        'water_level': int(row.get('water_level', 100)),
                        'probabilities': {
                            'TIDAK': 0.0,
                            'SIRAM': 0.0,
                            'BANYAK': 0.0
                        },
                        'processed_features': {
                            'temperature': float(row.get('temperature', 0)),
                            'humidity': float(row.get('humidity', 0)),
                            'soil_moisture': float(row.get('soil_moisture', 0)),
                            'crop_id': row.get('crop_id', 'Wheat'),
                            'soil_type': row.get('soil_type', 'Black Soil'),
                            'seedling_stage': row.get('seedling_stage', 'Germination')
                        },
                        'timestamp': row.get('timestamp', datetime.datetime.now().isoformat())
                    }
                    temp_history.append(item)
                except Exception:
                    continue
        prediction_history = temp_history[-100:]
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [LOG] Berhasil memuat {len(prediction_history)} riwayat dari CSV.")
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [LOG] Gagal memuat riwayat dari CSV: {e}")


# Muat riwayat saat startup
load_prediction_history_from_csv()



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
    """Render Web Dashboard utama."""
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Mengambil konfigurasi parameter aktif."""
    return jsonify(ACTIVE_CONFIG)


@app.route('/api/config', methods=['POST'])
def update_config():
    """Memperbarui konfigurasi parameter aktif."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Payload tidak valid.'}), 400

        if 'crop_id' in data:
            ACTIVE_CONFIG['crop_id'] = str(data['crop_id'])
        if 'soil_type' in data:
            ACTIVE_CONFIG['soil_type'] = str(data['soil_type'])
        if 'seedling_stage' in data:
            ACTIVE_CONFIG['seedling_stage'] = str(data['seedling_stage'])

        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [CONFIG] Konfigurasi diubah: {ACTIVE_CONFIG}")
        return jsonify({'status': 'success', 'config': ACTIVE_CONFIG})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Endpoint pengecekan koneksi."""
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL_LOADED,
        'timestamp': datetime.datetime.now().isoformat()
    })


def perform_inference(data):
    """
    Menjalankan inferensi prediksi irigasi cerdas berdasarkan data input.
    Mengembalikan dictionary hasil prediksi atau raises Exception.
    """
    if not MODEL_LOADED:
        raise ValueError('Pipeline model belum dimuat.')

    # Ekstraksi dan penyesuaian nama parameter sensor
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
        raise ValueError('Fitur input numerik tidak lengkap.')

    # Ekstraksi parameter kategorikal (menggunakan ACTIVE_CONFIG jika tidak dikirim)
    crop_id = data.get('crop_id', ACTIVE_CONFIG['crop_id'])
    soil_type = data.get('soil_type', ACTIVE_CONFIG['soil_type'])
    seedling_stage = data.get('seedling_stage', ACTIVE_CONFIG['seedling_stage'])

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
        'water_level': int(data.get('water_level', 100)),
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

    return result


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint utama prediksi irigasi cerdas via HTTP POST.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Payload data tidak valid atau kosong.'}), 400

        result = perform_inference(data)

        # Print log server tanpa emotikon
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [HTTP] "
              f"Prediksi: {result['label']} ({result['confidence']*100:.2f}%) | "
              f"Relay: {'ON' if result['action'] else 'OFF'} | "
              f"Input: temp={result['processed_features']['temperature']}, hum={result['processed_features']['humidity']}, "
              f"soil_moisture={result['processed_features']['soil_moisture']}, crop={result['processed_features']['crop_id']}")

        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Error pada inferensi model: {e}")
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# MQTT CLIENT DAEMON
# ==============================================================================
mqtt_client = None
mqtt_error = None

def on_connect(client, userdata, flags, rc, properties=None):
    """Callback saat terhubung ke broker MQTT."""
    global mqtt_error
    if rc == 0:
        mqtt_error = "Connected successfully"
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MQTT] Terhubung ke Broker {MQTT_BROKER}")
        client.subscribe(MQTT_TOPIC_SENSOR)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MQTT] Subscribed ke topik: {MQTT_TOPIC_SENSOR}")
    else:
        mqtt_error = f"Gagal terhubung, return code {rc}"
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MQTT] Gagal terhubung, return code {rc}")

def on_message(client, userdata, msg):
    """Callback saat menerima telemetry MQTT."""
    try:
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)

        result = perform_inference(data)

        # Kirim balik keputusan ke topik kontrol
        control_payload = {
            'prediction': result['prediction'],
            'label': result['label'],
            'action': result['action'],
            'confidence': result['confidence']
        }
        client.publish(MQTT_TOPIC_CONTROL, json.dumps(control_payload))

        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MQTT] "
              f"Telemetry Diterima. Prediksi: {result['label']} ({result['confidence']*100:.2f}%) | "
              f"Relay: {'ON' if result['action'] else 'OFF'} | "
              f"Published ke kontrol: {MQTT_TOPIC_CONTROL}")

    except json.JSONDecodeError:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MQTT] Payload bukan JSON: {msg.payload}")
    except ValueError as ve:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MQTT] Validasi telemetry gagal: {ve}")
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MQTT] Error: {e}")

def start_mqtt_client():
    """Inisialisasi MQTT client."""
    global mqtt_client, mqtt_error
    mqtt_error = "Initializing..."
    try:
        from paho.mqtt.enums import CallbackAPIVersion
        mqtt_client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2)
    except (ImportError, AttributeError):
        mqtt_client = mqtt.Client()

    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    try:
        mqtt_error = "Connecting..."
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
        mqtt_error = "Loop started, waiting for connection callback..."
    except Exception as e:
        mqtt_error = f"Exception: {str(e)}"
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MQTT] Gagal memulai client: {e}")


@app.route('/history', methods=['GET'])
def history():
    """Mengambil riwayat prediksi server."""
    limit = request.args.get('limit', 50, type=int)
    return jsonify({
        'count': len(prediction_history),
        'predictions': prediction_history[-limit:],
        'server_time': datetime.datetime.now().isoformat()
    })


@app.route('/mqtt-status', methods=['GET'])
def mqtt_status():
    """Mengecek status koneksi MQTT client."""
    global mqtt_client, mqtt_error
    is_initialized = mqtt_client is not None
    is_connected = False
    if is_initialized:
        try:
            is_connected = mqtt_client.is_connected()
        except Exception:
            pass
    return jsonify({
        'initialized': is_initialized,
        'connected': is_connected,
        'error': mqtt_error,
        'broker': MQTT_BROKER,
        'port': MQTT_PORT,
        'topic_sensor': MQTT_TOPIC_SENSOR,
        'topic_control': MQTT_TOPIC_CONTROL
    })


@app.route('/test-connection', methods=['GET'])
def test_connection():
    """Menguji konektivitas soket TCP langsung ke broker."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect((MQTT_BROKER, MQTT_PORT))
        s.close()
        return jsonify({
            'status': 'success',
            'message': f'Berhasil terhubung ke {MQTT_BROKER}:{MQTT_PORT} via TCP socket.'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Gagal terhubung ke {MQTT_BROKER}:{MQTT_PORT}: {e}'
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
# Jalankan client MQTT untuk mode produksi (Gunicorn) atau mode pengembangan lokal
if __name__ != '__main__' or not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    start_mqtt_client()

if __name__ == '__main__':
    print("="*80)
    print("  AIoT Smart Irrigation Flask API Server")
    print("="*80)
    print(f"  Server berjalan pada host: {HOST}:{PORT}")
    print("="*80 + "\n")

    app.run(host=HOST, port=PORT, debug=True)
