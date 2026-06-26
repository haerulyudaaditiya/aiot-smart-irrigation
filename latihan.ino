/*
 * ============================================================
 *  AIoT Smart Irrigation System
 *  ESP32 + ML (Flask Server) + Multi-Sensor (MQTT Version)
 * ============================================================
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
// LCD removed (tidak dipakai)

// WiFi Configuration
const char* WIFI_SSID     = "S21 FE";
const char* WIFI_PASSWORD  = "12345678";

// MQTT Configuration
const char* MQTT_BROKER       = "broker.emqx.io";
const int   MQTT_PORT         = 1883;
const char* MQTT_TOPIC_SENSOR = "aiot/smart_irrigation/sensor";
const char* MQTT_TOPIC_CONTROL = "aiot/smart_irrigation/control";

// Pin Definitions
#define DHTPIN            14    // DHT22 data pin
#define DHTTYPE           DHT22
#define SOIL_MOISTURE_PIN 32    // Capacitive Soil Moisture (Analog)
#define WATER_LEVEL_PIN   34    // Water Level Sensor (Analog)
#define RELAY_PIN         26    // Relay module signal
#define LED_RED_PIN       4     // LED Merah (perlu siram)
#define LED_GREEN_PIN     5     // LED Hijau (aman)

// Relay Logic Configuration (Active-Low)
#define RELAY_ON          LOW
#define RELAY_OFF         HIGH

// Sensor Calibration Values
const int SOIL_DRY   = 3400;
const int SOIL_WET   = 1500;
const int WATER_EMPTY = 0;
const int WATER_FULL  = 3000;

// Object Initialization
DHT dht(DHTPIN, DHTTYPE);
// LCD removed
WiFiClient espClient;
PubSubClient mqttClient(espClient);

// Global Variables
float temperature    = 0;
float humidity       = 0;
int   soilMoisture   = 0;
int   waterLevel     = 0;

int   prediction     = -1;     // -1=belum ada, 0=tidak siram, 1=siram, 2=siram banyak
int   action         = 0;      // 0=relay OFF, 1=relay ON
float confidence     = 0;
String predLabel     = "---";

unsigned long lastSendTime = 0;
const unsigned long SEND_INTERVAL = 5000;  // Kirim setiap 5 detik

bool serverConnected = false;

unsigned long lastWiFiRetry = 0;
const unsigned long WIFI_RETRY_INTERVAL = 10000;  // Retry WiFi setiap 10 detik

// Function Prototypes
void connectWiFi();
void reconnectMQTT();
void readSensors();
void printSensorData();
void publishTelemetry();
void controlOutputs();
void updateLEDs();
// void updateLCD(); // LCD removed
void mqttCallback(char* topic, byte* payload, unsigned int length);

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("╔════════════════════════════════════════╗");
  Serial.println("║      AIoT Smart Irrigation System      ║");
  Serial.println("╚════════════════════════════════════════╝");

  // Initialize output pins
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);

  // Turn off all outputs initially
  pinMode(RELAY_PIN, INPUT_PULLUP); // Set as INPUT_PULLUP to turn OFF 5V active-low relay cleanly with pull-up
  digitalWrite(LED_RED_PIN, LOW);
  digitalWrite(LED_GREEN_PIN, LOW);

  // Initialize DHT22
  dht.begin();

  // LCD removed

  // Connect to WiFi
  connectWiFi();

  // LCD removed

  // Initialize MQTT
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
  reconnectMQTT();

  delay(500);
}

void loop() {
  // Feed watchdog
  yield();

  // Reconnect WiFi if disconnected (non-blocking with cooldown)
  if (WiFi.status() != WL_CONNECTED) {
    if (millis() - lastWiFiRetry >= WIFI_RETRY_INTERVAL) {
      lastWiFiRetry = millis();
      Serial.println("[!] WiFi terputus, reconnecting...");
      connectWiFi();
    }
  }

  // Reconnect MQTT if disconnected
  if (WiFi.status() == WL_CONNECTED && !mqttClient.connected()) {
    reconnectMQTT();
  }

  // Run MQTT loop
  mqttClient.loop();

  // Send telemetry at SEND_INTERVAL
  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();

    // 1. Read all sensors
    readSensors();

    // 2. Print sensor data to Serial
    printSensorData();

    // 3. Publish to MQTT Broker
    publishTelemetry();

    // 4. Control outputs based on prediction
    controlOutputs();

    // LCD removed
  }

  // 6. Update LEDs (including non-blocking blinking if tank is empty)
  updateLEDs();
}

void connectWiFi() {
  Serial.print("[WiFi] Connecting to ");
  Serial.print(WIFI_SSID);
  Serial.println("...");

  WiFi.disconnect(true);  // Disconnect dulu untuk clean state
  delay(100);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(250);
    yield();  // Feed watchdog timer
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(" Connected!");
    Serial.print("[WiFi] IP Address: ");
    Serial.println(WiFi.localIP());

    // Indicator: both LEDs blink fast 3 times
    for (int i = 0; i < 3; i++) {
      digitalWrite(LED_GREEN_PIN, HIGH);
      digitalWrite(LED_RED_PIN, HIGH);
      delay(150);
      yield();
      digitalWrite(LED_GREEN_PIN, LOW);
      digitalWrite(LED_RED_PIN, LOW);
      delay(150);
      yield();
    }
  } else {
    Serial.println(" FAILED!");
    Serial.println("[WiFi] Cek SSID dan Password.");
  }
}

void reconnectMQTT() {
  int attempts = 0;
  while (!mqttClient.connected() && attempts < 3) {
    yield();  // Feed watchdog timer
    Serial.print("[MQTT] Mencoba menghubungkan ke broker...");
    // Generate unique client ID
    String clientId = "ESP32Client-" + String(random(0xffff), HEX);
    
    if (mqttClient.connect(clientId.c_str())) {
      Serial.println(" Connected!");
      // Subscribe to control topic
      mqttClient.subscribe(MQTT_TOPIC_CONTROL);
      Serial.printf("[MQTT] Subscribed ke: %s\n", MQTT_TOPIC_CONTROL);
    } else {
      Serial.print(" Gagal, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" Coba lagi dalam 1 detik...");
      delay(500);
      yield();  // Feed watchdog timer
      delay(500);
      yield();  // Feed watchdog timer
      attempts++;
    }
  }
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  Serial.print("[MQTT] Pesan masuk di topik: ");
  Serial.println(topic);

  // Parse payload JSON
  JsonDocument doc;
  DeserializationError error = deserializeJson(doc, payload, length);

  if (!error) {
    prediction = doc["prediction"].as<int>();
    action     = doc["action"].as<int>();  // 0=OFF, 1=ON
    confidence = doc["confidence"].as<float>();
    predLabel  = doc["label"].as<String>();
    serverConnected = true;

    Serial.printf("[MQTT] ML Prediksi: %s (confidence: %.1f%%) -> Relay: %s\n",
                   predLabel.c_str(), confidence * 100,
                   action == 1 ? "ON" : "OFF");
  } else {
    Serial.print("[!] Gagal parse JSON MQTT callback: ");
    Serial.println(error.c_str());
  }
}

void readSensors() {
  // DHT22: Temperature & Humidity
  float t = dht.readTemperature();
  float h = dht.readHumidity();

  if (!isnan(t) && !isnan(h)) {
    temperature = t;
    humidity = h;
  } else {
    Serial.println("[!] DHT22 error, pakai data terakhir.");
  }

  // Soil Moisture Sensor (Analog)
  int rawSoil = analogRead(SOIL_MOISTURE_PIN);
  soilMoisture = map(rawSoil, SOIL_DRY, SOIL_WET, 0, 100);
  soilMoisture = constrain(soilMoisture, 0, 100);

  // Water Level Sensor (Analog)
  int rawWater = analogRead(WATER_LEVEL_PIN);
  waterLevel = map(rawWater, WATER_EMPTY, WATER_FULL, 0, 100);
  waterLevel = constrain(waterLevel, 0, 100);
}

void printSensorData() {
  Serial.println("-------------------------------------");
  Serial.printf("Suhu        : %.1f °C\n", temperature);
  Serial.printf("Kelembapan  : %.1f %%\n", humidity);
  Serial.printf("Kelembapan Tanah : %d %%\n", soilMoisture);
  Serial.printf("Level Tangki Air : %d %%\n", waterLevel);
}

void publishTelemetry() {
  if (!mqttClient.connected()) {
    Serial.println("[!] MQTT tidak terhubung, skip publish.");
    serverConnected = false;
    return;
  }

  // Create JSON payload
  JsonDocument doc;
  doc["temperature"]    = temperature;
  doc["humidity"]       = humidity;
  doc["soil_moisture"]  = soilMoisture;
  doc["water_level"]    = waterLevel;

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  Serial.print("[MQTT] Publish → ");
  Serial.println(MQTT_TOPIC_SENSOR);
  Serial.print("[MQTT] Payload: ");
  Serial.println(jsonPayload);

  if (mqttClient.publish(MQTT_TOPIC_SENSOR, jsonPayload.c_str())) {
    Serial.println("[MQTT] Telemetry berhasil dipublish!");
  } else {
    Serial.println("[!] Gagal publish telemetry!");
  }
}

void controlOutputs() {
  if (prediction < 0) {
    pinMode(RELAY_PIN, INPUT_PULLUP);        // Set as INPUT_PULLUP to turn OFF relay cleanly
    Serial.println("[AKSI] Menunggu prediksi dari server...");
    return;
  }

  // Override: Protection when water tank is empty
  if (waterLevel < 15) {
    pinMode(RELAY_PIN, INPUT_PULLUP);        // Set as INPUT_PULLUP to turn OFF relay cleanly
    Serial.println("[AKSI] Override: Sumber air habis! Penyiraman dinonaktifkan untuk proteksi.");
    return;
  }

  // Use action from server
  if (action == 1) {
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, RELAY_ON);      // Relay ON -> Valve/Pump ON
    Serial.printf("[AKSI] Relay ON -> POMPA AKTIF (%s)\n", predLabel.c_str());
  } else {
    pinMode(RELAY_PIN, INPUT_PULLUP);        // Set as INPUT_PULLUP to turn OFF relay cleanly
    Serial.println("[AKSI] Relay OFF -> Pompa MATI (aman)");
  }
}

unsigned long lastBlinkTime = 0;
bool ledState = LOW;

void updateLEDs() {
  if (prediction < 0) {
    // Belum terhubung / error: Merah & Hijau menyala statis
    digitalWrite(LED_RED_PIN, HIGH);
    digitalWrite(LED_GREEN_PIN, HIGH);
    return;
  }

  if (waterLevel < 15) {
    // Air tangki habis -> LED Merah Kedap-kedip (500ms), LED Hijau OFF
    digitalWrite(LED_GREEN_PIN, LOW);
    if (millis() - lastBlinkTime >= 500) {
      lastBlinkTime = millis();
      ledState = !ledState;
      digitalWrite(LED_RED_PIN, ledState);
    }
  } else {
    // Kondisi air normal
    if (action == 1) {
      // Sedang menyiram: Merah ON, Hijau OFF
      digitalWrite(LED_RED_PIN, HIGH);
      digitalWrite(LED_GREEN_PIN, LOW);
    } else {
      // Kondisi aman: Merah OFF, Hijau ON
      digitalWrite(LED_RED_PIN, LOW);
      digitalWrite(LED_GREEN_PIN, HIGH);
    }
  }
}

// LCD removed - semua fungsi updateLCD() dihapus
