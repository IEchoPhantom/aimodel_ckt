#include <Arduino.h>
#include "esp_camera.h"

// Configure one sketch per lane. Change to 1..4 per ESP32-CAM.
#ifndef LANE_ID
#define LANE_ID 1
#endif

namespace {
constexpr int kFrameWidth = 160;
constexpr int kFrameHeight = 120;
constexpr size_t kFrameBytes = kFrameWidth * kFrameHeight;

constexpr unsigned long kSendIntervalMs = 300;
constexpr uint8_t kSampleStride = 2;

// Tune these thresholds in your scene.
constexpr float kDiffLow = 1.8f;
constexpr float kDiffHigh = 22.0f;

uint8_t g_prev_frame[kFrameBytes];
bool g_has_prev_frame = false;
float g_smoothed_score = 0.0f;
unsigned long g_last_send_ms = 0;

camera_config_t BuildCameraConfig() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sccb_sda = 26;
  config.pin_sccb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_QQVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;
  return config;
}

bool InitCamera() {
  camera_config_t config = BuildCameraConfig();
  const esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    return false;
  }

  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor != nullptr) {
    sensor->set_vflip(sensor, 1);
    sensor->set_hmirror(sensor, 0);
    sensor->set_brightness(sensor, 0);
    sensor->set_contrast(sensor, 0);
  }
  return true;
}

int ComputeCongestionScore(const uint8_t* current) {
  uint32_t sum_diff = 0;
  uint32_t sample_count = 0;

  for (int y = 0; y < kFrameHeight; y += kSampleStride) {
    const int row = y * kFrameWidth;
    for (int x = 0; x < kFrameWidth; x += kSampleStride) {
      const int idx = row + x;
      const int diff = static_cast<int>(current[idx]) - static_cast<int>(g_prev_frame[idx]);
      sum_diff += static_cast<uint32_t>(abs(diff));
      ++sample_count;
    }
  }

  if (sample_count == 0) {
    return 0;
  }

  const float mean_abs_diff = static_cast<float>(sum_diff) / static_cast<float>(sample_count);
  float normalized = (mean_abs_diff - kDiffLow) / (kDiffHigh - kDiffLow);
  if (normalized < 0.0f) {
    normalized = 0.0f;
  }
  if (normalized > 1.0f) {
    normalized = 1.0f;
  }

  const float raw_score = normalized * 100.0f;
  g_smoothed_score = (0.70f * g_smoothed_score) + (0.30f * raw_score);
  return static_cast<int>(g_smoothed_score + 0.5f);
}
}  // namespace

void setup() {
  Serial.begin(115200);
  delay(300);

  if (!InitCamera()) {
    Serial.println("TRAFFIC:camera_error=1");
    return;
  }

  Serial.println("TRAFFIC_CAM_READY");
}

void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (fb == nullptr) {
    delay(20);
    return;
  }

  if (fb->format != PIXFORMAT_GRAYSCALE || fb->width != kFrameWidth || fb->height != kFrameHeight) {
    esp_camera_fb_return(fb);
    delay(20);
    return;
  }

  int score = 0;
  if (g_has_prev_frame) {
    score = ComputeCongestionScore(fb->buf);
  }

  memcpy(g_prev_frame, fb->buf, kFrameBytes);
  g_has_prev_frame = true;
  esp_camera_fb_return(fb);

  const unsigned long now = millis();
  if ((now - g_last_send_ms) >= kSendIntervalMs) {
    g_last_send_ms = now;
    char line[32];
    snprintf(line, sizeof(line), "TRAFFIC:lane%d=%d", LANE_ID, score);
    Serial.println(line);
  }

  delay(5);
}
