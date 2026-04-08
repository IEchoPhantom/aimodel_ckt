#include <Arduino.h>
#include <Servo.h>
#include <U8g2lib.h>
#include <Wire.h>

namespace {
constexpr uint8_t kLaneCount = 4;
constexpr uint8_t kTotalSlots = 4;

constexpr uint8_t kEntrySensorPin = 2;
constexpr uint8_t kExitSensorPin = 3;
constexpr uint8_t kServoPin = 11;

// Lane pins: [lane][0=R,1=Y,2=G]
const uint8_t kLaneLedPins[kLaneCount][3] = {
    {4, 5, 6},
    {7, 8, 9},
    {10, 12, 13},
    {A0, A1, A2},
};

constexpr unsigned long kYellowMs = 1200;
constexpr unsigned long kMinGreenMs = 3000;
constexpr unsigned long kMaxGreenMs = 9000;

constexpr uint8_t kBarrierClosedDeg = 5;
constexpr uint8_t kBarrierOpenDeg = 80;
constexpr unsigned long kBarrierOpenMs = 1800;

char g_line_buffer[96];
uint8_t g_line_len = 0;

int8_t g_lane_scores[kLaneCount] = {0, 0, 0, 0};
bool g_slot_occupied[kTotalSlots] = {false, false, false, false};
uint8_t g_camera_occupied_count = 0;
uint8_t g_sensor_occupied_count = 0;

uint8_t g_current_green_lane = 0;
bool g_is_yellow_phase = false;
unsigned long g_phase_deadline_ms = 0;

bool g_prev_entry_state = false;
bool g_prev_exit_state = false;

Servo g_barrier_servo;
bool g_barrier_open = false;
unsigned long g_barrier_close_ms = 0;

unsigned long g_last_display_ms = 0;
U8G2_SH1106_128X64_NONAME_1_HW_I2C g_display(U8G2_R0, U8X8_PIN_NONE);

int ClampInt(int value, int lo, int hi) {
  if (value < lo) return lo;
  if (value > hi) return hi;
  return value;
}

uint8_t FusedOccupiedCount() {
  uint8_t fused = g_camera_occupied_count;
  if (g_sensor_occupied_count > fused) {
    fused = g_sensor_occupied_count;
  }
  if (fused > kTotalSlots) {
    fused = kTotalSlots;
  }
  return fused;
}

void SetLaneLights(uint8_t lane, bool red, bool yellow, bool green) {
  digitalWrite(kLaneLedPins[lane][0], red ? HIGH : LOW);
  digitalWrite(kLaneLedPins[lane][1], yellow ? HIGH : LOW);
  digitalWrite(kLaneLedPins[lane][2], green ? HIGH : LOW);
}

void ApplyAllRed() {
  for (uint8_t lane = 0; lane < kLaneCount; ++lane) {
    SetLaneLights(lane, true, false, false);
  }
}

void ApplyGreenForLane(uint8_t active_lane) {
  for (uint8_t lane = 0; lane < kLaneCount; ++lane) {
    if (lane == active_lane) {
      SetLaneLights(lane, false, false, true);
    } else {
      SetLaneLights(lane, true, false, false);
    }
  }
}

void ApplyYellowForLane(uint8_t lane) {
  for (uint8_t i = 0; i < kLaneCount; ++i) {
    if (i == lane) {
      SetLaneLights(i, false, true, false);
    } else {
      SetLaneLights(i, true, false, false);
    }
  }
}

uint8_t PickNextLane() {
  int best_score = -1;
  uint8_t best_lane = (g_current_green_lane + 1) % kLaneCount;

  for (uint8_t lane = 0; lane < kLaneCount; ++lane) {
    int score = ClampInt(g_lane_scores[lane], 0, 100);
    if (score > best_score) {
      best_score = score;
      best_lane = lane;
    }
  }

  return best_lane;
}

unsigned long GreenDurationForLane(uint8_t lane) {
  const int score = ClampInt(g_lane_scores[lane], 0, 100);
  const unsigned long extra = static_cast<unsigned long>(score) * 60UL;
  unsigned long duration = kMinGreenMs + extra;
  if (duration > kMaxGreenMs) {
    duration = kMaxGreenMs;
  }
  return duration;
}

void UpdateTrafficScheduler() {
  const unsigned long now = millis();
  if (now < g_phase_deadline_ms) {
    return;
  }

  if (!g_is_yellow_phase) {
    g_is_yellow_phase = true;
    ApplyYellowForLane(g_current_green_lane);
    g_phase_deadline_ms = now + kYellowMs;
    return;
  }

  g_is_yellow_phase = false;
  g_current_green_lane = PickNextLane();
  ApplyGreenForLane(g_current_green_lane);
  g_phase_deadline_ms = now + GreenDurationForLane(g_current_green_lane);
}

void OpenBarrierTemporarily() {
  g_barrier_servo.write(kBarrierOpenDeg);
  g_barrier_open = true;
  g_barrier_close_ms = millis() + kBarrierOpenMs;
}

void UpdateBarrier() {
  if (!g_barrier_open) {
    return;
  }
  if (millis() >= g_barrier_close_ms) {
    g_barrier_servo.write(kBarrierClosedDeg);
    g_barrier_open = false;
  }
}

void UpdateSensors() {
  const bool entry_now = (digitalRead(kEntrySensorPin) == HIGH);
  const bool exit_now = (digitalRead(kExitSensorPin) == HIGH);

  if (entry_now && !g_prev_entry_state) {
    const uint8_t occupied = FusedOccupiedCount();
    if (occupied < kTotalSlots) {
      if (g_sensor_occupied_count < kTotalSlots) {
        ++g_sensor_occupied_count;
      }
      OpenBarrierTemporarily();
    }
  }

  if (exit_now && !g_prev_exit_state) {
    if (g_sensor_occupied_count > 0) {
      --g_sensor_occupied_count;
    }
  }

  g_prev_entry_state = entry_now;
  g_prev_exit_state = exit_now;
}

void ParseParkToken(char* token) {
  if (strncmp(token, "slot", 4) != 0) {
    return;
  }

  char* eq = strchr(token, '=');
  if (eq == nullptr) {
    return;
  }

  *eq = '\0';
  const uint8_t slot_index = static_cast<uint8_t>(atoi(token + 4));
  const char* value = eq + 1;

  if (slot_index == 0 || slot_index > kTotalSlots) {
    return;
  }

  g_slot_occupied[slot_index - 1] = (strcmp(value, "occupied") == 0);
}

void ParseParkLine(char* payload) {
  for (uint8_t i = 0; i < kTotalSlots; ++i) {
    g_slot_occupied[i] = false;
  }

  char* token = strtok(payload, ",");
  while (token != nullptr) {
    ParseParkToken(token);
    token = strtok(nullptr, ",");
  }

  uint8_t count = 0;
  for (uint8_t i = 0; i < kTotalSlots; ++i) {
    if (g_slot_occupied[i]) {
      ++count;
    }
  }
  g_camera_occupied_count = count;
}

void ParseTrafficToken(char* token) {
  if (strncmp(token, "lane", 4) != 0) {
    return;
  }

  char* eq = strchr(token, '=');
  if (eq == nullptr) {
    return;
  }

  *eq = '\0';
  const uint8_t lane_index = static_cast<uint8_t>(atoi(token + 4));
  const int score = ClampInt(atoi(eq + 1), 0, 100);

  if (lane_index == 0 || lane_index > kLaneCount) {
    return;
  }

  g_lane_scores[lane_index - 1] = static_cast<int8_t>(score);
}

void ParseTrafficLine(char* payload) {
  char* token = strtok(payload, ",");
  while (token != nullptr) {
    ParseTrafficToken(token);
    token = strtok(nullptr, ",");
  }
}

void ParseMessage(char* line) {
  if (strncmp(line, "PARK:", 5) == 0) {
    ParseParkLine(line + 5);
    return;
  }

  if (strncmp(line, "TRAFFIC:", 8) == 0) {
    ParseTrafficLine(line + 8);
    return;
  }
}

void PollSerialMessages() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\r') {
      continue;
    }

    if (c == '\n') {
      g_line_buffer[g_line_len] = '\0';
      if (g_line_len > 0) {
        ParseMessage(g_line_buffer);
      }
      g_line_len = 0;
      continue;
    }

    if (g_line_len < (sizeof(g_line_buffer) - 1)) {
      g_line_buffer[g_line_len++] = c;
    } else {
      g_line_len = 0;
    }
  }
}

void UpdateDisplay() {
  const unsigned long now = millis();
  if ((now - g_last_display_ms) < 500) {
    return;
  }
  g_last_display_ms = now;

  const uint8_t occupied = FusedOccupiedCount();
  const uint8_t available = (occupied <= kTotalSlots) ? (kTotalSlots - occupied) : 0;

  g_display.firstPage();
  do {
    g_display.setFont(u8g2_font_6x12_tf);
    g_display.setCursor(0, 12);
    g_display.print("Parking System");

    g_display.setCursor(0, 28);
    g_display.print("Total: ");
    g_display.print(kTotalSlots);

    g_display.setCursor(0, 40);
    g_display.print("Occupied: ");
    g_display.print(occupied);

    g_display.setCursor(0, 52);
    g_display.print("Available: ");
    g_display.print(available);

    g_display.setCursor(0, 64);
    g_display.print("L1:");
    g_display.print(static_cast<int>(g_lane_scores[0]));
    g_display.print(" L2:");
    g_display.print(static_cast<int>(g_lane_scores[1]));
  } while (g_display.nextPage());
}
}  // namespace

void setup() {
  Serial.begin(115200);

  pinMode(kEntrySensorPin, INPUT);
  pinMode(kExitSensorPin, INPUT);

  for (uint8_t lane = 0; lane < kLaneCount; ++lane) {
    for (uint8_t c = 0; c < 3; ++c) {
      pinMode(kLaneLedPins[lane][c], OUTPUT);
    }
  }

  ApplyAllRed();
  g_current_green_lane = 0;
  ApplyGreenForLane(g_current_green_lane);
  g_phase_deadline_ms = millis() + GreenDurationForLane(g_current_green_lane);

  g_barrier_servo.attach(kServoPin);
  g_barrier_servo.write(kBarrierClosedDeg);

  Wire.begin();
  g_display.begin();
  g_display.clearBuffer();
}

void loop() {
  PollSerialMessages();
  UpdateSensors();
  UpdateBarrier();
  UpdateTrafficScheduler();
  UpdateDisplay();
}
