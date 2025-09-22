#pragma once

// Copy this file to `secrets.h` and replace the placeholder values before compiling.
#define WIFI_SSID_1 "YOUR_PRIMARY_WIFI_SSID"
#define WIFI_PASSWORD_1 "YOUR_PRIMARY_WIFI_PASSWORD"
#define WIFI_SSID_2 "YOUR_BACKUP_WIFI_SSID"
#define WIFI_PASSWORD_2 "YOUR_BACKUP_WIFI_PASSWORD"

// Static IP configuration. Update to match your LAN; set to "" if you prefer DHCP.
#define STATIC_IP_ADDRESS "192.168.1.123"
#define STATIC_GATEWAY "192.168.1.1"
#define STATIC_SUBNET "255.255.255.0"
#define STATIC_PRIMARY_DNS "8.8.8.8"
#define STATIC_SECONDARY_DNS "8.8.4.4"

#define WIFI_MANAGER_AP_NAME "ESP32CamSetup"
#define WIFI_MANAGER_AP_PASSWORD ""  // Leave empty for an open configuration portal

#define SERVER_URL "http://34.80.216.88:5000/upload_face"

// Optional: provide a device identifier header
#define DEVICE_ID "cam-01"
