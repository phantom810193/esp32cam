#include <Arduino.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <WiFiManager.h>
#include "esp_camera.h"

#include <cstring>

#if __has_include("dl_lib.h") && __has_include("fd_forward.h") && \
    __has_include("img_converters.h") && __has_include("fb_gfx.h")
#define HAS_FACE_DETECTION 1
extern "C" {
#include "dl_lib.h"
#include "fd_forward.h"
#include "img_converters.h"
#include "fb_gfx.h"
}
#else
#define HAS_FACE_DETECTION 0
#endif

#include "secrets.h"

#ifndef WIFI_SSID_1
#error "Please define WIFI_SSID_1 in include/secrets.h"
#endif
#ifndef WIFI_PASSWORD_1
#error "Please define WIFI_PASSWORD_1 in include/secrets.h"
#endif
#ifndef WIFI_SSID_2
#error "Please define WIFI_SSID_2 in include/secrets.h"
#endif
#ifndef WIFI_PASSWORD_2
#error "Please define WIFI_PASSWORD_2 in include/secrets.h"
#endif
#ifndef SERVER_URL
#error "Please define SERVER_URL in include/secrets.h"
#endif
#ifndef WIFI_MANAGER_AP_NAME
#define WIFI_MANAGER_AP_NAME "ESP32CamSetup"
#endif
#ifndef WIFI_MANAGER_AP_PASSWORD
#define WIFI_MANAGER_AP_PASSWORD ""
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
constexpr uint32_t kWiFiRetryDelayMs = 500;
constexpr uint8_t kMaxRetriesPerNetwork = 20;
constexpr uint16_t kConfigPortalTimeoutSeconds = 180;

struct WifiCredential {
  const char *ssid;
  const char *password;
};

static const WifiCredential kWifiCredentials[] = {
  {WIFI_SSID_1, WIFI_PASSWORD_1},
  {WIFI_SSID_2, WIFI_PASSWORD_2},
};

enum class CaptureResult {
  kUploaded,
  kNoFaceDetected,
  kError,
};

static WiFiManager wifiManager;

#if HAS_FACE_DETECTION
template <typename T>
inline auto assign_min_face(T &config, int value) -> decltype(config.min_face = value, void()) {
  config.min_face = value;
}
inline void assign_min_face(...) {}

template <typename T>
inline auto assign_pyramid(T &config, float value) -> decltype(config.pyramid = value, void()) {
  config.pyramid = value;
}
inline void assign_pyramid(...) {}

template <typename T>
inline auto assign_threshold(T &config, float value) -> decltype(config.threshold = value, void()) {
  config.threshold = value;
}
inline void assign_threshold(...) {}

template <typename T>
inline auto assign_threshold_score(T &config, float value) -> decltype(config.score_threshold = value, void()) {
  config.score_threshold = value;
}
inline void assign_threshold_score(...) {}

template <typename T>
inline auto assign_threshold_short(T &config, float value) -> decltype(config.score_thresh = value, void()) {
  config.score_thresh = value;
}
inline void assign_threshold_short(...) {}

template <typename T>
inline auto assign_nms(T &config, float value) -> decltype(config.nms_threshold = value, void()) {
  config.nms_threshold = value;
}
inline void assign_nms(...) {}

template <typename T>
inline auto assign_nms_short(T &config, float value) -> decltype(config.nms = value, void()) {
  config.nms = value;
}
inline void assign_nms_short(...) {}

template <typename T>
inline auto assign_nms_score(T &config, float value) -> decltype(config.nms_thresh = value, void()) {
  config.nms_thresh = value;
}
inline void assign_nms_score(...) {}
#endif

static bool initCamera();
static bool ensureWiFi();
static bool tryConnectToConfiguredNetworks();
static bool connectToNetwork(const WifiCredential &credential);
static CaptureResult captureAndSend();
static bool detectFace(const camera_fb_t *fb);
static bool loadStaticIpConfig(IPAddress *local_ip, IPAddress *gateway, IPAddress *subnet,
                               IPAddress *primary_dns, bool *has_primary_dns,
                               IPAddress *secondary_dns, bool *has_secondary_dns);
static bool applyStaticIpConfiguration();
static void configureWifiManagerStaticIp();

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("Booting ESP32-CAM MVP...");

  WiFi.mode(WIFI_STA);
  wifiManager.setConfigPortalTimeout(kConfigPortalTimeoutSeconds);
  configureWifiManagerStaticIp();

  if (!initCamera()) {
    Serial.println("Camera init failed, halting.");
    while (true) {
      delay(1000);
    }
  }

  if (!ensureWiFi()) {
    Serial.println("Wi-Fi not connected; waiting for configuration.");
  }
}

void loop() {
  if (!ensureWiFi()) {
    Serial.println("Wi-Fi connection unavailable; retrying after delay.");
    delay(kCaptureIntervalMs);
    return;
  }

  switch (captureAndSend()) {
    case CaptureResult::kUploaded:
      Serial.println("Photo uploaded successfully.");
      break;
    case CaptureResult::kNoFaceDetected:
      Serial.println("No face detected; skipping upload.");
      break;
    case CaptureResult::kError:
    default:
      Serial.println("Capture or upload failed.");
      break;
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

static bool ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return true;
  }

  if (tryConnectToConfiguredNetworks()) {
    return true;
  }

  Serial.println("Starting WiFiManager configuration portal...");
  bool connected = false;
  if (WIFI_MANAGER_AP_PASSWORD[0] == '\0') {
    connected = wifiManager.autoConnect(WIFI_MANAGER_AP_NAME);
  } else {
    connected = wifiManager.autoConnect(WIFI_MANAGER_AP_NAME, WIFI_MANAGER_AP_PASSWORD);
  }
  if (connected) {
    Serial.print("Connected via WiFiManager. IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFiManager portal timed out or failed to connect.");
  }
  return connected;
}

static bool tryConnectToConfiguredNetworks() {
  for (const WifiCredential &credential : kWifiCredentials) {
    if (credential.ssid == nullptr || credential.ssid[0] == '\0') {
      continue;
    }
    if (connectToNetwork(credential)) {
      return true;
    }
  }
  return false;
}

static bool connectToNetwork(const WifiCredential &credential) {
  Serial.printf("Connecting to Wi-Fi %s...\n", credential.ssid);
  WiFi.disconnect(true);
  delay(100);

#if defined(STATIC_IP_ADDRESS) && defined(STATIC_GATEWAY) && defined(STATIC_SUBNET)
  static bool logged_dhcp_fallback = false;
  if (!applyStaticIpConfiguration()) {
    if (!logged_dhcp_fallback) {
      Serial.println(
          "Static IP configuration missing or invalid; falling back to DHCP.");
      logged_dhcp_fallback = true;
    }
    WiFi.config(IPAddress(), IPAddress(), IPAddress());
  }
#else
  WiFi.config(IPAddress(), IPAddress(), IPAddress());
#endif

  WiFi.begin(credential.ssid, credential.password);

  uint8_t retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < kMaxRetriesPerNetwork) {
    delay(kWiFiRetryDelayMs);
    Serial.print(".");
    retries++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Connected! IP: ");
    Serial.println(WiFi.localIP());
    return true;
  }

  Serial.println("Failed to connect with stored credentials.");
  WiFi.disconnect(true);
  delay(100);
  return false;
}

static CaptureResult captureAndSend() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return CaptureResult::kError;
  }

  CaptureResult result = CaptureResult::kError;

  if (!detectFace(fb)) {
    result = CaptureResult::kNoFaceDetected;
    goto cleanup;
  }

  {
    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "image/jpeg");
#ifdef DEVICE_ID
    http.addHeader("X-Device-ID", DEVICE_ID);
#endif

    Serial.printf("Uploading %u bytes to %s\n", fb->len, SERVER_URL);
    int httpResponseCode = http.POST(fb->buf, fb->len);

    if (httpResponseCode > 0 && httpResponseCode < 400) {
      Serial.printf("Server response code: %d\n", httpResponseCode);
      String payload = http.getString();
      Serial.println(payload);
      result = CaptureResult::kUploaded;
    } else {
      Serial.printf("HTTP POST failed: %d\n", httpResponseCode);
      result = CaptureResult::kError;
    }

    http.end();
  }

cleanup:
  esp_camera_fb_return(fb);
  return result;
}

#if defined(STATIC_IP_ADDRESS) && defined(STATIC_GATEWAY) && defined(STATIC_SUBNET)
static bool loadStaticIpConfig(IPAddress *local_ip, IPAddress *gateway, IPAddress *subnet,
                               IPAddress *primary_dns, bool *has_primary_dns,
                               IPAddress *secondary_dns, bool *has_secondary_dns) {
  static bool logged_invalid_core = false;
  static bool logged_primary_warning = false;
  static bool logged_secondary_warning = false;

  if (has_primary_dns) {
    *has_primary_dns = false;
  }
  if (has_secondary_dns) {
    *has_secondary_dns = false;
  }

  if (STATIC_IP_ADDRESS[0] == '\0' || STATIC_GATEWAY[0] == '\0' ||
      STATIC_SUBNET[0] == '\0') {
    return false;
  }

  if (!local_ip->fromString(STATIC_IP_ADDRESS) ||
      !gateway->fromString(STATIC_GATEWAY) ||
      !subnet->fromString(STATIC_SUBNET)) {
    if (!logged_invalid_core) {
      Serial.println(
          "Static IP core settings invalid; check STATIC_IP_ADDRESS, STATIC_GATEWAY, and STATIC_SUBNET.");
      logged_invalid_core = true;
    }
    return false;
  }

#if defined(STATIC_PRIMARY_DNS)
  if (primary_dns && STATIC_PRIMARY_DNS[0] != '\0') {
    if (primary_dns->fromString(STATIC_PRIMARY_DNS)) {
      if (has_primary_dns) {
        *has_primary_dns = true;
      }
    } else if (!logged_primary_warning) {
      Serial.printf("Failed to parse STATIC_PRIMARY_DNS (%s); ignoring.\n",
                    STATIC_PRIMARY_DNS);
      logged_primary_warning = true;
    }
  }
#else
  (void)primary_dns;
#endif

#if defined(STATIC_SECONDARY_DNS)
  if (secondary_dns && STATIC_SECONDARY_DNS[0] != '\0') {
    if (secondary_dns->fromString(STATIC_SECONDARY_DNS)) {
      if (has_secondary_dns) {
        *has_secondary_dns = true;
      }
    } else if (!logged_secondary_warning) {
      Serial.printf(
          "Failed to parse STATIC_SECONDARY_DNS (%s); ignoring.\n",
          STATIC_SECONDARY_DNS);
      logged_secondary_warning = true;
    }
  }
#else
  (void)secondary_dns;
#endif

  return true;
}

static bool applyStaticIpConfiguration() {
  IPAddress local_ip;
  IPAddress gateway;
  IPAddress subnet;
  IPAddress primary_dns;
  IPAddress secondary_dns;
  bool has_primary_dns = false;
  bool has_secondary_dns = false;

  if (!loadStaticIpConfig(&local_ip, &gateway, &subnet, &primary_dns,
                          &has_primary_dns, &secondary_dns,
                          &has_secondary_dns)) {
    return false;
  }

  bool configured = false;
  if (has_primary_dns && has_secondary_dns) {
    configured = WiFi.config(local_ip, gateway, subnet, primary_dns,
                             secondary_dns);
  } else if (has_primary_dns) {
    configured = WiFi.config(local_ip, gateway, subnet, primary_dns);
  } else {
    configured = WiFi.config(local_ip, gateway, subnet);
  }

  if (!configured) {
    Serial.println("Failed to apply static IP configuration.");
    return false;
  }

  Serial.print("Using static IP: ");
  Serial.println(local_ip);
  return true;
}

static void configureWifiManagerStaticIp() {
  IPAddress local_ip;
  IPAddress gateway;
  IPAddress subnet;
  IPAddress primary_dns;
  IPAddress secondary_dns;
  bool has_primary_dns = false;
  bool has_secondary_dns = false;

  if (!loadStaticIpConfig(&local_ip, &gateway, &subnet, &primary_dns,
                          &has_primary_dns, &secondary_dns,
                          &has_secondary_dns)) {
    return;
  }

  if (has_primary_dns && has_secondary_dns) {
    wifiManager.setSTAStaticIPConfig(local_ip, gateway, subnet, primary_dns,
                                     secondary_dns);
  } else if (has_primary_dns) {
    wifiManager.setSTAStaticIPConfig(local_ip, gateway, subnet, primary_dns);
  } else {
    wifiManager.setSTAStaticIPConfig(local_ip, gateway, subnet);
  }
  Serial.print("Configured WiFiManager static IP: ");
  Serial.println(local_ip);
}
#else
static bool loadStaticIpConfig(IPAddress *local_ip, IPAddress *gateway, IPAddress *subnet,
                               IPAddress *primary_dns, bool *has_primary_dns,
                               IPAddress *secondary_dns, bool *has_secondary_dns) {
  (void)local_ip;
  (void)gateway;
  (void)subnet;
  (void)primary_dns;
  (void)has_primary_dns;
  (void)secondary_dns;
  (void)has_secondary_dns;
  return false;
}

static bool applyStaticIpConfiguration() {
  return false;
}

static void configureWifiManagerStaticIp() {}
#endif

#if HAS_FACE_DETECTION
static bool detectFace(const camera_fb_t *fb) {
  dl_matrix3du_t *image_matrix = dl_matrix3du_alloc(1, fb->width, fb->height, 3);
  if (!image_matrix) {
    Serial.println("Failed to allocate matrix for face detection.");
    return false;
  }

  bool detected = false;
  if (!fmt2rgb888(fb->buf, fb->len, fb->format, image_matrix->item)) {
    Serial.println("Failed to convert frame for detection.");
  } else {
    face_detect_config_t config = {};
    assign_min_face(config, 80);
    assign_pyramid(config, 0.707f);
    assign_threshold(config, 0.6f);
    assign_threshold_score(config, 0.6f);
    assign_threshold_short(config, 0.6f);
    assign_nms(config, 0.3f);
    assign_nms_short(config, 0.3f);
    assign_nms_score(config, 0.3f);
    box_array_t *faces = face_detect(image_matrix, &config);
    if (faces) {
      detected = faces->len > 0;
      dl_lib_free(faces->score);
      dl_lib_free(faces->box);
      dl_lib_free(faces);
    }
  }

  dl_matrix3du_free(image_matrix);
  return detected;
}
#else
static bool detectFace(const camera_fb_t *fb) {
  (void)fb;
  static bool warned = false;
  if (!warned) {
    Serial.println("Face detection library not available; defaulting to always send.");
    warned = true;
  }
  return true;
}
#endif

