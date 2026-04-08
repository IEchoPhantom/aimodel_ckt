/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ==============================================================================*/

#include "image_provider.h"

#include "model_settings.h"

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include <Arduino.h>

#include "esp_camera.h"

namespace {
constexpr int kCaptureWidth = 160;
constexpr int kCaptureHeight = 120;

bool g_camera_initialized = false;
uint8_t g_active_slot = 0;

const SlotRoi kSlotRois[kParkingSlotCount] = {
    {0, 0, 80, 60},
    {80, 0, 80, 60},
    {0, 60, 80, 60},
    {80, 60, 80, 60},
};

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

TfLiteStatus EnsureCameraInitialized(tflite::ErrorReporter* error_reporter) {
  if (g_camera_initialized) {
    return kTfLiteOk;
  }

  camera_config_t config = BuildCameraConfig();
  const esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    TF_LITE_REPORT_ERROR(error_reporter, "esp_camera_init failed: %d", err);
    return kTfLiteError;
  }

  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor != nullptr) {
    sensor->set_vflip(sensor, 1);
    sensor->set_hmirror(sensor, 0);
    sensor->set_brightness(sensor, 0);
    sensor->set_contrast(sensor, 0);
  }

  g_camera_initialized = true;
  return kTfLiteOk;
}

void ExtractAndQuantizeSlot(const camera_fb_t* fb, const SlotRoi& roi,
                            int image_width, int image_height,
                            int8_t* image_data) {
  const int src_width = fb->width;
  const int src_height = fb->height;
  const uint8_t* src = fb->buf;

  for (int y = 0; y < image_height; ++y) {
    const int src_y = roi.y + ((y * roi.height) / image_height);
    const int clamped_y = (src_y >= src_height) ? (src_height - 1) : src_y;
    for (int x = 0; x < image_width; ++x) {
      const int src_x = roi.x + ((x * roi.width) / image_width);
      const int clamped_x = (src_x >= src_width) ? (src_width - 1) : src_x;
      const uint8_t gray = src[clamped_y * src_width + clamped_x];
      image_data[(y * image_width) + x] = static_cast<int8_t>(gray - 128);
    }
  }
}
}  // namespace

TfLiteStatus InitCamera(tflite::ErrorReporter* error_reporter) {
  return EnsureCameraInitialized(error_reporter);
}

void SetActiveSlot(uint8_t slot_index) {
  if (slot_index >= kParkingSlotCount) {
    g_active_slot = 0;
    return;
  }
  g_active_slot = slot_index;
}

uint8_t GetActiveSlot() {
  return g_active_slot;
}

const SlotRoi* GetSlotRoiTable() {
  return kSlotRois;
}

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data) {
  if (channels != 1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Only grayscale input is supported");
    return kTfLiteError;
  }

  if (EnsureCameraInitialized(error_reporter) != kTfLiteOk) {
    return kTfLiteError;
  }

  camera_fb_t* fb = esp_camera_fb_get();
  if (fb == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter, "esp_camera_fb_get failed");
    return kTfLiteError;
  }

  if (fb->format != PIXFORMAT_GRAYSCALE || fb->width != kCaptureWidth ||
      fb->height != kCaptureHeight) {
    TF_LITE_REPORT_ERROR(error_reporter, "Unexpected frame format %d (%dx%d)",
                         fb->format, fb->width, fb->height);
    esp_camera_fb_return(fb);
    return kTfLiteError;
  }

  const SlotRoi& roi = kSlotRois[g_active_slot];
  ExtractAndQuantizeSlot(fb, roi, image_width, image_height, image_data);
  esp_camera_fb_return(fb);
  return kTfLiteOk;
}

#endif  // ARDUINO_EXCLUDE_CODE