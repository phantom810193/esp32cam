#include <Arduino.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include "esp_camera.h"

#include "secrets.h"

#ifndef WIFI_SSID
#error "Please define WIFI_SSID in include/secrets.h"
#endif
#ifndef WIFI_PASSWORD
#error "Please define WIFI_PASSWORD in include/secrets.h"
#endif
#ifndef SERVER_URL
#error "Please define SERVER_URL in include/secrets.h"
#endif

// AI-Thinker ESP32-CAM pinout configuration
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

constexpr uint32_t kCaptureIntervalMs = 5000;

static bool initCamera();
static void ensureWiFi();
static bool captureAndSend();

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("Booting ESP32-CAM MVP...");

  if (!initCamera()) {
    Serial.println("Camera init failed, halting.");
    while (true) {
      delay(1000);
    }
  }

  ensureWiFi();
}

void loop() {
  ensureWiFi();
  if (captureAndSend()) {
    Serial.println("Photo uploaded successfully.");
  } else {
    Serial.println("Upload failed.");
  }
  delay(kCaptureIntervalMs);
}

static bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }
  sensor_t * s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 1);
    s->set_saturation(s, 0);
  }
  return true;
}

static void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.printf("Connecting to Wi-Fi %s...\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  uint8_t retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 20) {
    delay(500);
    Serial.print(".");
    retries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("Connected! IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("Failed to connect to Wi-Fi.");
  }
}

static bool captureAndSend() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return false;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "image/jpeg");
#ifdef DEVICE_ID
  http.addHeader("X-Device-ID", DEVICE_ID);
#endif

  Serial.printf("Uploading %u bytes to %s\n", fb->len, SERVER_URL);
  int httpResponseCode = http.POST(fb->buf, fb->len);

  bool success = httpResponseCode > 0 && httpResponseCode < 400;
  if (success) {
    Serial.printf("Server response code: %d\n", httpResponseCode);
    String payload = http.getString();
    Serial.println(payload);
  } else {
    Serial.printf("HTTP POST failed: %d\n", httpResponseCode);
  }

  http.end();
  esp_camera_fb_return(fb);
  return success;
}

