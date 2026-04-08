import argparse
import time
from typing import List, Tuple

import cv2
import numpy as np

try:
    import serial
except ImportError:
    serial = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Laptop webcam bridge for Uno protocol testing")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--mode", choices=["parking", "traffic", "both"], default="both")
    parser.add_argument("--lane-id", type=int, default=1, help="Lane number for TRAFFIC messages")
    parser.add_argument("--send-ms", type=int, default=300, help="Send interval in milliseconds")
    parser.add_argument("--serial-port", type=str, default="", help="COM port, e.g. COM6")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--slots", type=int, default=4, choices=[1, 2, 4], help="Parking slots in frame")
    parser.add_argument("--show", action="store_true", help="Show debug window")
    parser.add_argument("--diff-low", type=float, default=2.0)
    parser.add_argument("--diff-high", type=float, default=22.0)
    parser.add_argument("--tflite", type=str, default="", help="Optional .tflite model for parking")
    return parser


def open_serial(port: str, baud: int):
    if not port:
        return None
    if serial is None:
        raise RuntimeError("pyserial is not installed. Install with: pip install pyserial")
    return serial.Serial(port, baudrate=baud, timeout=0)


def emit(line: str, ser) -> None:
    print(line)
    if ser is not None:
        ser.write((line + "\n").encode("ascii", errors="ignore"))


def get_slot_rois(width: int, height: int, slot_count: int) -> List[Tuple[int, int, int, int]]:
    if slot_count == 1:
        return [(0, 0, width, height)]
    if slot_count == 2:
        half = width // 2
        return [(0, 0, half, height), (half, 0, width - half, height)]

    half_w = width // 2
    half_h = height // 2
    return [
        (0, 0, half_w, half_h),
        (half_w, 0, width - half_w, half_h),
        (0, half_h, half_w, height - half_h),
        (half_w, half_h, width - half_w, height - half_h),
    ]


def traffic_score(curr_gray: np.ndarray, prev_gray: np.ndarray, low: float, high: float) -> int:
    abs_diff = cv2.absdiff(curr_gray, prev_gray)
    mean_diff = float(np.mean(abs_diff))
    norm = (mean_diff - low) / max(1e-6, (high - low))
    norm = min(1.0, max(0.0, norm))
    return int(round(norm * 100.0))


def parking_states_simple(gray: np.ndarray, rois: List[Tuple[int, int, int, int]]) -> List[str]:
    states: List[str] = []
    for x, y, w, h in rois:
        patch = gray[y : y + h, x : x + w]
        patch = cv2.resize(patch, (96, 96), interpolation=cv2.INTER_AREA)

        # Heuristic proxy: visible printed number usually creates stronger local edges.
        lap = cv2.Laplacian(patch, cv2.CV_16S)
        edge_energy = float(np.mean(np.abs(lap)))

        state = "empty" if edge_energy >= 12.0 else "occupied"
        states.append(state)
    return states


def try_load_tflite(path: str):
    if not path:
        return None, None

    interpreter = None
    try:
        import tflite_runtime.interpreter as tflite

        interpreter = tflite.Interpreter(model_path=path)
    except Exception:
        try:
            import tensorflow as tf

            interpreter = tf.lite.Interpreter(model_path=path)
        except Exception:
            return None, None

    interpreter.allocate_tensors()
    in_info = interpreter.get_input_details()[0]
    out_info = interpreter.get_output_details()[0]
    return interpreter, (in_info, out_info)


def parking_states_tflite(gray: np.ndarray, rois: List[Tuple[int, int, int, int]], interpreter, io_info) -> List[str]:
    in_info, out_info = io_info
    states: List[str] = []

    for x, y, w, h in rois:
        patch = gray[y : y + h, x : x + w]
        patch = cv2.resize(patch, (96, 96), interpolation=cv2.INTER_AREA)

        if in_info["dtype"] == np.int8:
            tensor = patch.astype(np.int16) - 128
            tensor = tensor.astype(np.int8)
        else:
            tensor = patch.astype(np.float32) / 255.0

        tensor = np.expand_dims(np.expand_dims(tensor, axis=0), axis=-1)
        interpreter.set_tensor(in_info["index"], tensor)
        interpreter.invoke()
        out = interpreter.get_tensor(out_info["index"])

        vec = out[0].reshape(-1)
        if vec.shape[0] < 2:
            states.append("empty")
            continue

        states.append("occupied" if float(vec[1]) > float(vec[0]) else "empty")

    return states


def main() -> int:
    args = build_arg_parser().parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return 1

    ser = open_serial(args.serial_port, args.baud)
    interpreter, io_info = try_load_tflite(args.tflite)

    prev_gray = None
    last_send = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        rois = get_slot_rois(w, h, args.slots)

        now = time.time() * 1000.0
        if (now - last_send) >= args.send_ms:
            last_send = now

            if args.mode in ("traffic", "both") and prev_gray is not None:
                score = traffic_score(gray, prev_gray, args.diff_low, args.diff_high)
                emit(f"TRAFFIC:lane{args.lane_id}={score}", ser)

            if args.mode in ("parking", "both"):
                if interpreter is not None and io_info is not None:
                    states = parking_states_tflite(gray, rois, interpreter, io_info)
                else:
                    states = parking_states_simple(gray, rois)

                parts = []
                for i, state in enumerate(states):
                    parts.append(f"slot{i + 1}={state}")
                emit("PARK:" + ",".join(parts), ser)

        prev_gray = gray

        if args.show:
            vis = frame.copy()
            for i, (x, y, rw, rh) in enumerate(rois):
                cv2.rectangle(vis, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
                cv2.putText(vis, f"slot{i+1}", (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("webcam-bridge", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    if ser is not None:
        ser.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
