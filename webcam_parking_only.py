import argparse
from typing import List, Tuple

import cv2
import numpy as np


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Webcam-only parking occupancy demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--slots", type=int, default=4, choices=[1, 2, 4], help="Number of parking slots")
    parser.add_argument("--model", type=str, default="", help="Optional quantized .tflite model")
    parser.add_argument(
        "--reference",
        type=str,
        default="",
        help="Optional reference image of the empty parking scene",
    )
    parser.add_argument("--show-rois", action="store_true", help="Draw slot regions on screen")
    parser.add_argument("--mirror", action="store_true", help="Mirror the preview window")
    return parser


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


def load_tflite(path: str):
    if not path:
        return None, None

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


def load_reference(path: str):
    if not path:
        return None

    ref = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        return None
    return ref


def score_from_reference(patch: np.ndarray, reference_patch: np.ndarray) -> float:
    patch = cv2.GaussianBlur(patch, (5, 5), 0)
    reference_patch = cv2.GaussianBlur(reference_patch, (5, 5), 0)

    diff = cv2.absdiff(patch, reference_patch)
    mean_diff = float(np.mean(diff))

    patch_std = float(np.std(patch))
    ref_std = float(np.std(reference_patch))

    # Higher score means more likely occupied.
    score = (mean_diff * 1.7) + max(0.0, ref_std - patch_std)
    return score


def classify_with_model(gray: np.ndarray, rois, interpreter, io_info) -> List[str]:
    in_info, out_info = io_info
    states: List[str] = []

    for x, y, w, h in rois:
        patch = gray[y : y + h, x : x + w]
        patch = cv2.resize(patch, (96, 96), interpolation=cv2.INTER_AREA)

        if in_info["dtype"] == np.int8:
            tensor = (patch.astype(np.int16) - 128).astype(np.int8)
        else:
            tensor = patch.astype(np.float32) / 255.0

        tensor = tensor.reshape(1, 96, 96, 1)
        interpreter.set_tensor(in_info["index"], tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(out_info["index"])[0].reshape(-1)

        if output.shape[0] < 2:
            states.append("empty")
        else:
            states.append("occupied" if float(output[1]) > float(output[0]) else "empty")

    return states


def classify_with_heuristic(gray: np.ndarray, rois, reference_gray: np.ndarray | None) -> List[str]:
    states: List[str] = []
    for x, y, w, h in rois:
        patch = gray[y : y + h, x : x + w]
        patch = cv2.resize(patch, (96, 96), interpolation=cv2.INTER_AREA)

        if reference_gray is not None:
            reference_patch = reference_gray[y : y + h, x : x + w]
            reference_patch = cv2.resize(reference_patch, (96, 96), interpolation=cv2.INTER_AREA)
            score = score_from_reference(patch, reference_patch)
            # Tune this threshold after taking one empty reference frame.
            state = "occupied" if score >= 18.0 else "empty"
        else:
            edges = cv2.Laplacian(patch, cv2.CV_16S)
            edge_energy = float(np.mean(np.abs(edges)))
            brightness = float(np.mean(patch))
            score = (edge_energy * 1.6) + max(0.0, 180.0 - brightness) * 0.04
            state = "occupied" if score >= 20.0 else "empty"
        states.append(state)
    return states


def draw_overlay(frame, rois, states):
    for i, ((x, y, w, h), state) in enumerate(zip(rois, states), start=1):
        color = (0, 200, 0) if state == "empty" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"slot{i}:{state}"
        cv2.putText(frame, label, (x + 6, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main() -> int:
    args = build_arg_parser().parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return 1

    interpreter, io_info = load_tflite(args.model)
    reference_gray = load_reference(args.reference)

    if reference_gray is not None:
        print("Loaded reference image for occupancy comparison")

    print("Keys: q=quit, r=reload reference from current frame, s=save current frame as reference")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        if args.mirror:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rois = get_slot_rois(frame.shape[1], frame.shape[0], args.slots)

        if interpreter is not None and io_info is not None:
            states = classify_with_model(gray, rois, interpreter, io_info)
        else:
            states = classify_with_heuristic(gray, rois, reference_gray)

        occupied = sum(1 for state in states if state == "occupied")
        available = len(states) - occupied

        preview = frame.copy()
        if args.show_rois:
            draw_overlay(preview, rois, states)

        header = f"Occupied: {occupied} | Available: {available}"
        cv2.putText(preview, header, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(preview, header, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 1)

        y = 70
        for i, state in enumerate(states, start=1):
            line = f"Slot {i}: {state}"
            cv2.putText(preview, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y += 32

        cv2.imshow("Parking Monitor", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            reference_gray = gray.copy()
            print("Saved current webcam frame as reference")
        elif key == ord("r") and reference_gray is not None:
            reference_gray = gray.copy()
            print("Reloaded reference from current frame")
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
