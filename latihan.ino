/*
 * ============================================================
 *  AIoT Smart Irrigation System
 *  ESP32 + ML (Flask Server) + Multi-Sensor
 * ============================================================
 *  Sensor Input:
 *    - DHT22 (Suhu & Kelembapan Udara)
 *    - Capacitive Soil Moisture Sensor
 *    - Rain Sensor (Water Level)
 *
 *  Output:
 *    - Relay → Solenoid (simulasi valve/pompa)
 *    - LCD 16x2 I2C (status & prediksi ML)
 *    - LED Merah (perlu siram) & Hijau (aman)
 *
 *  Komunikasi:
 *    - WiFi HTTP POST ke Flask Server (ML Inference)
 *
 *  Library yang dibutuhkan (install via Arduino Library Manager):
 *    1. DHT sensor library (by Adafruit)
 *    2. Adafruit Unified Sensor
 *    3. ArduinoJson (by Benoit Blanchon)
 *    4. LiquidCrystal I2C (by Frank de Brabander)
 * ============================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ============================================================
// CONFIGURATION
// ============================================================

const char* WIFI_SSID     = "S21 FE";
const char* WIFI_PASSWORD  = "12345678";
const char* SERVER_URL     = "http://10.107.135.216:5000/predict";

// ============================================================
// PIN DEFINITIONS
// ============================================================

// Sensor Pins
#define DHTPIN            14    // DHT22 data pin
#define DHTTYPE           DHT22
#define SOIL_MOISTURE_PIN 32    // Capacitive Soil Moisture (Analog)
#define WATER_LEVEL_PIN   34    // Water Level Sensor (Analog)

// Output Pins
#define RELAY_PIN         26    // Relay module signal
#define LED_RED_PIN       4     // LED Merah (perlu siram)
#define LED_GREEN_PIN     5     // LED Hijau (aman)

// ============================================================
// SENSOR CALIBRATION VALUES
// ============================================================

const int SOIL_DRY   = 3400;
const int SOIL_WET   = 1500;
const int WATER_EMPTY = 0;
const int WATER_FULL  = 3000;

// ============================================================
// INISIALISASI OBJEK
// ============================================================

DHT dht(DHTPIN, DHTTYPE);
LiquidCrystal_I2C lcd(0x27, 16, 2);  // Alamat I2C: 0x27 (umum), coba 0x3F jika tidak tampil

// ============================================================
// VARIABEL GLOBAL
// ============================================================

float temperature    = 0;
float humidity       = 0;
int   soilMoisture   = 0;
int   waterLevel     = 0;

int   prediction     = -1;     // -1=belum ada, 0=tidak siram, 1=siram, 2=siram banyak
int   action         = 0;      // 0=relay OFF, 1=relay ON (dari server)
float confidence     = 0;
String predLabel     = "---";

unsigned long lastSendTime = 0;
const unsigned long SEND_INTERVAL = 5000;  // Kirim setiap 5 detik

bool serverConnected = false;

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("╔════════════════════════════════════════╗");
  Serial.println("║  AIoT Smart Irrigation System v1.0    ║");
  Serial.println("║  ESP32 + Machine Learning             ║");
  Serial.println("╚════════════════════════════════════════╝");

  // Inisialisasi pin output
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);

  // Matikan semua output awal
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(LED_RED_PIN, LOW);
  digitalWrite(LED_GREEN_PIN, LOW);

  // Inisialisasi DHT22
  dht.begin();

  // Inisialisasi LCD
  Wire.begin(21, 22);  // SDA=21, SCL=22
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("AIoT Irrigation");
  lcd.setCursor(0, 1);
  lcd.print("Connecting WiFi");

  // Koneksi WiFi
  connectWiFi();

  // Tampilkan IP di LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("IP:");
  lcd.print(WiFi.localIP());
  lcd.setCursor(0, 1);
  lcd.print("Sistem Siap!");
  delay(2000);
}

// ============================================================
// LOOP UTAMA
// ============================================================

void loop() {
  // Cek koneksi WiFi
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[!] WiFi terputus, reconnecting...");
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("WiFi Putus!");
    lcd.setCursor(0, 1);
    lcd.print("Reconnecting...");
    connectWiFi();
  }

  // Kirim data setiap SEND_INTERVAL
  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();

    // 1. Baca semua sensor
    readSensors();

    // 2. Tampilkan data sensor ke Serial
    printSensorData();

    // 3. Kirim ke Flask server & terima prediksi
    sendToServer();

    // 4. Kontrol output berdasarkan prediksi
    controlOutputs();

    // 5. Update LCD
    updateLCD();
  }

  // 6. Pembaruan LED (termasuk kedipan non-blocking jika air habis)
  updateLEDs();
}

// ============================================================
// FUNGSI: Koneksi WiFi
// ============================================================

void connectWiFi() {
  Serial.print("[WiFi] Connecting to ");
  Serial.print(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(" Connected!");
    Serial.print("[WiFi] IP Address: ");
    Serial.println(WiFi.localIP());

    // Indikator: kedua LED berkedip cepat 3x
    for (int i = 0; i < 3; i++) {
      digitalWrite(LED_GREEN_PIN, HIGH);
      digitalWrite(LED_RED_PIN, HIGH);
      delay(200);
      digitalWrite(LED_GREEN_PIN, LOW);
      digitalWrite(LED_RED_PIN, LOW);
      delay(200);
    }
  } else {
    Serial.println(" FAILED!");
    Serial.println("[WiFi] Cek SSID dan Password.");
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("WiFi GAGAL!");
    lcd.setCursor(0, 1);
    lcd.print("Cek SSID/Pass");
  }
}

// ============================================================
// FUNGSI: Baca Semua Sensor
// ============================================================

void readSensors() {
  // --- DHT22: Suhu & Kelembapan Udara ---
  float t = dht.readTemperature();
  float h = dht.readHumidity();

  if (!isnan(t) && !isnan(h)) {
    temperature = t;
    humidity = h;
  } else {
    Serial.println("[!] DHT22 error, pakai data terakhir.");
  }

  // --- Soil Moisture Sensor (Analog) ---
  int rawSoil = analogRead(SOIL_MOISTURE_PIN);
  soilMoisture = map(rawSoil, SOIL_DRY, SOIL_WET, 0, 100);
  soilMoisture = constrain(soilMoisture, 0, 100);

  // --- Water Level Sensor (Analog) ---
  int rawWater = analogRead(WATER_LEVEL_PIN);
  waterLevel = map(rawWater, WATER_EMPTY, WATER_FULL, 0, 100);
  waterLevel = constrain(waterLevel, 0, 100);
}

// ============================================================
// FUNGSI: Print Data Sensor ke Serial
// ============================================================

void printSensorData() {
  Serial.println("-------------------------------------");
  Serial.printf("Suhu        : %.1f °C\n", temperature);
  Serial.printf("Kelembapan  : %.1f %%\n", humidity);
  Serial.printf("Tanah       : %d %%\n", soilMoisture);
  Serial.printf("Level Air   : %d %%\n", waterLevel);
}

// ============================================================
// FUNGSI: Kirim Data ke Flask Server
// ============================================================

void sendToServer() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[!] WiFi tidak terhubung, skip pengiriman.");
    serverConnected = false;
    return;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(5000);  // Timeout 5 detik

  // Buat JSON payload
  JsonDocument doc;
  doc["temperature"]    = temperature;
  doc["humidity"]       = humidity;
  doc["soil_moisture"]  = soilMoisture;
  doc["water_level"]    = waterLevel;

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  Serial.print("[HTTP] POST → ");
  Serial.println(SERVER_URL);
  Serial.print("[HTTP] Payload: ");
  Serial.println(jsonPayload);

  // Kirim HTTP POST
  int httpCode = http.POST(jsonPayload);

  if (httpCode == 200) {
    String response = http.getString();
    Serial.print("[HTTP] Response: ");
    Serial.println(response);

    // Parse JSON response
    JsonDocument resDoc;
    DeserializationError error = deserializeJson(resDoc, response);

    if (!error) {
      prediction = resDoc["prediction"].as<int>();
      action     = resDoc["action"].as<int>();  // 0=OFF, 1=ON
      confidence = resDoc["confidence"].as<float>();
      predLabel  = resDoc["label"].as<String>();
      serverConnected = true;

      Serial.printf("ML Prediksi: %s (confidence: %.1f%%) -> Relay: %s\n",
                     predLabel.c_str(), confidence * 100,
                     action == 1 ? "ON" : "OFF");
    } else {
      Serial.println("[!] JSON parse error!");
      serverConnected = false;
    }
  } else {
    Serial.printf("[!] HTTP error code: %d\n", httpCode);
    Serial.println("[!] Pastikan Flask server jalan di laptop.");
    serverConnected = false;
    prediction = -1;
    predLabel = "ERR";
  }

  http.end();
}

// ============================================================
// FUNGSI: Kontrol Output (Relay, LED)
// ============================================================

void controlOutputs() {
  if (prediction < 0) {
    // BELUM ADA PREDIKSI / ERROR
    digitalWrite(RELAY_PIN, LOW);      // Safety: relay OFF
    Serial.println("[AKSI] Menunggu prediksi dari server...");
    return;
  }

  // LOGIKA OVERRIDE: Proteksi Tangki Air Kosong
  if (waterLevel < 15) {
    digitalWrite(RELAY_PIN, LOW);      // Matikan penyiraman
    Serial.println("[AKSI] Override: Sumber air habis! Penyiraman dinonaktifkan untuk proteksi.");
    return;
  }

  // Gunakan field 'action' dari server (0=OFF, 1=ON)
  if (action == 1) {
    // PERLU SIRAM (prediction 1 atau 2)
    digitalWrite(RELAY_PIN, HIGH);     // Relay ON -> Pompa menyala
    Serial.printf("[AKSI] Relay ON -> POMPA AKTIF (%s)\n", predLabel.c_str());
  } else {
    // TIDAK PERLU SIRAM (prediction 0)
    digitalWrite(RELAY_PIN, LOW);      // Relay OFF -> Pompa mati
    Serial.println("[AKSI] Relay OFF -> Pompa MATI (aman)");
  }
}

// ============================================================
// FUNGSI: Pembaruan LED Indikator (Non-Blocking)
// ============================================================

unsigned long lastBlinkTime = 0;
bool ledState = LOW;

void updateLEDs() {
  if (prediction < 0) {
    // BELUM ADA PREDIKSI / ERROR (Merah & Hijau menyala statis)
    digitalWrite(LED_RED_PIN, HIGH);
    digitalWrite(LED_GREEN_PIN, HIGH);
    return;
  }

  if (waterLevel < 15) {
    // AIR TANGKI HABIS -> LED Merah Kedap-kedip (500ms), LED Hijau OFF
    digitalWrite(LED_GREEN_PIN, LOW);
    if (millis() - lastBlinkTime >= 500) {
      lastBlinkTime = millis();
      ledState = !ledState;
      digitalWrite(LED_RED_PIN, ledState);
    }
  } else {
    // KONDISI AIR NORMAL (Mengikuti data prediksi)
    if (action == 1) {
      // SEDANG MENYIRAM (Merah ON, Hijau OFF)
      digitalWrite(LED_RED_PIN, HIGH);
      digitalWrite(LED_GREEN_PIN, LOW);
    } else {
      // KONDISI AMAN / TIDAK SIRAM (Merah OFF, Hijau ON)
      digitalWrite(LED_RED_PIN, LOW);
      digitalWrite(LED_GREEN_PIN, HIGH);
    }
  }
}

// ============================================================
// FUNGSI: Update LCD Display
// ============================================================

void updateLCD() {
  lcd.clear();

  if (!serverConnected && prediction == -1) {
    // Belum terhubung ke server
    lcd.setCursor(0, 0);
    lcd.print("T:");
    lcd.print(temperature, 0);
    lcd.print("C H:");
    lcd.print(humidity, 0);
    lcd.print("%");

    lcd.setCursor(0, 1);
    lcd.print("NO SERVER!");
    return;
  }

  // Baris 1: Data sensor (ringkas)
  lcd.setCursor(0, 0);
  // Format: T:32C H:65% S:45%
  lcd.print("T:");
  lcd.print(temperature, 0);
  lcd.print("C H:");
  lcd.print(humidity, 0);
  lcd.print("% S:");
  lcd.print(soilMoisture);
  lcd.print("%");

  // Baris 2: Prediksi ML & Status Air Tangki
  lcd.setCursor(0, 1);

  if (waterLevel < 15) {
    lcd.print("ML:KOSONG");
  } else if (prediction == 0) {
    lcd.print("ML:AMAN  ");
  } else if (prediction == 1) {
    lcd.print("ML:SIRAM ");
  } else if (prediction == 2) {
    lcd.print("ML:BNYK! ");
  } else {
    lcd.print("ML:WAIT  ");
  }

  // Tampilkan sisa ruang dengan kapasitas air tangki (W: XX%)
  lcd.print(" W:");
  lcd.print(waterLevel);
  lcd.print("%");
}
