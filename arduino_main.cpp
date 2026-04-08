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

#include "main_functions.h"

#include <Arduino.h>
#include <TensorFlowLite.h>

#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

namespace {
tflite::ErrorReporter* g_error_reporter = nullptr;
const tflite::Model* g_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;

constexpr int kTensorArenaSize = 120 * 1024;
static uint8_t g_tensor_arena[kTensorArenaSize];

constexpr unsigned long kInferenceIntervalMs = 250;
unsigned long g_last_inference_ms = 0;

int8_t ReadOutputScore(const TfLiteTensor* output, int index) {
    if (output->type == kTfLiteInt8) {
        return output->data.int8[index];
    }
    if (output->type == kTfLiteUInt8) {
        return static_cast<int8_t>(static_cast<int16_t>(output->data.uint8[index]) -
                                                             128);
    }
    return -128;
}
}  // namespace

void setup() {
    Serial.begin(115200);
    delay(300);

    static tflite::MicroErrorReporter micro_error_reporter;
    g_error_reporter = &micro_error_reporter;

    g_model = tflite::GetModel(g_person_detect_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(g_error_reporter,
                                                 "Schema mismatch: model=%d runtime=%d",
                                                 g_model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::MicroMutableOpResolver<8> op_resolver;
    op_resolver.AddConv2D();
    op_resolver.AddDepthwiseConv2D();
    op_resolver.AddFullyConnected();
    op_resolver.AddAveragePool2D();
    op_resolver.AddMaxPool2D();
    op_resolver.AddReshape();
    op_resolver.AddSoftmax();
    op_resolver.AddLogistic();

    static tflite::MicroInterpreter static_interpreter(
            g_model, op_resolver, g_tensor_arena, kTensorArenaSize, g_error_reporter);
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(g_error_reporter, "AllocateTensors failed");
        return;
    }

    g_input = g_interpreter->input(0);
    if (g_input->dims->size != 4 || g_input->dims->data[1] != kNumRows ||
            g_input->dims->data[2] != kNumCols || g_input->dims->data[3] != 1) {
        TF_LITE_REPORT_ERROR(g_error_reporter,
                                                 "Model input must be [1,96,96,1], got [%d,%d,%d,%d]",
                                                 g_input->dims->data[0], g_input->dims->data[1],
                                                 g_input->dims->data[2], g_input->dims->data[3]);
        return;
    }

    if (g_input->type != kTfLiteInt8) {
        TF_LITE_REPORT_ERROR(g_error_reporter,
                                                 "Expected int8 input model. Current type=%d",
                                                 g_input->type);
    }

    if (InitCamera(g_error_reporter) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(g_error_reporter, "Camera init failed");
        return;
    }

    Serial.println("PARKING_CAM_READY");
}

void loop() {
    if (g_interpreter == nullptr || g_input == nullptr) {
        delay(200);
        return;
    }

    const unsigned long now = millis();
    if ((now - g_last_inference_ms) < kInferenceIntervalMs) {
        delay(5);
        return;
    }
    g_last_inference_ms = now;

    bool occupied[kParkingSlotCount] = {false, false, false, false};

    for (uint8_t slot = 0; slot < kParkingSlotCount; ++slot) {
        SetActiveSlot(slot);

        if (GetImage(g_error_reporter, kNumCols, kNumRows, kNumChannels,
                                 g_input->data.int8) != kTfLiteOk) {
            TF_LITE_REPORT_ERROR(g_error_reporter, "GetImage failed on slot %d", slot);
            return;
        }

        if (g_interpreter->Invoke() != kTfLiteOk) {
            TF_LITE_REPORT_ERROR(g_error_reporter, "Invoke failed on slot %d", slot);
            return;
        }

        TfLiteTensor* output = g_interpreter->output(0);
        const int8_t empty_score = ReadOutputScore(output, kEmptyIndex);
        const int8_t occupied_score = ReadOutputScore(output, kOccupiedIndex);
        occupied[slot] = (occupied_score > empty_score);
    }

    char line[128];
    snprintf(line, sizeof(line),
                     "PARK:slot1=%s,slot2=%s,slot3=%s,slot4=%s",
                     occupied[0] ? "occupied" : "empty",
                     occupied[1] ? "occupied" : "empty",
                     occupied[2] ? "occupied" : "empty",
                     occupied[3] ? "occupied" : "empty");
    Serial.println(line);
}
