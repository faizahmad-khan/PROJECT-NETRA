import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import csv
import time
import argparse
from typing import cast
from datetime import datetime
from src.tracker import VehicleTracker
from src.model_runtime import (
    RuntimeConfig,
    build_predict_kwargs,
    format_runtime_summary,
    resolve_device,
    resolve_precision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NETRA main runtime with edge optimization options."
    )
    parser.add_argument(
        "--video",
        default="videos/traffic.mp4",
        help="Path to input video",
    )
    parser.add_argument(
        "--traffic-model",
        default="models/yolov8m.pt",
        help="Traffic detector model path (.pt/.onnx/.engine/.mlpackage)",
    )
    parser.add_argument(
        "--ambulance-model",
        default="models/best.pt",
        help="Ambulance detector model path",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device",
    )
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp16"],
        help="Inference precision (fp16 enabled only on CUDA)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Skip N frames between processed frames to increase throughput",
    )
    return parser.parse_args()


# --- 1. SETUP VIDEO & MODELS ---
args = parse_args()

cap = cv2.VideoCapture(args.video)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

device = resolve_device(args.device)
use_half = resolve_precision(args.precision, device)

runtime_config = RuntimeConfig(
    device=device,
    use_half=use_half,
    imgsz=args.imgsz,
)

predict_kwargs = build_predict_kwargs(runtime_config)

print("Loading Intelligent Models...")
model_traffic = YOLO(args.traffic_model)    # The Generalist (Cars)
model_ambulance = YOLO(args.ambulance_model)  # The Specialist (Ambulance)
print("Models Loaded!")
print(f"Runtime Optimized: {format_runtime_summary(runtime_config)}")

if args.precision == "fp16" and not use_half:
    print(
        "FP16 requested but unsupported on this device. "
        "Falling back to FP32."
    )

# --- 2. SETUP VEHICLE TRACKER (ByteTrack) ---
tracker = VehicleTracker(frame_rate=fps)
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike", "bicycle"]
vehicle_class_ids = [
    cid for cid, name in model_traffic.names.items()
    if name in VEHICLE_CLASSES
]
print(f"🔍 ByteTrack enabled @ {fps} FPS | Tracking: {VEHICLE_CLASSES}")

# --- 3. SETUP DATA LOGGING ---
file_name = (
    "data/traffic_logs/Traffic_Data_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

# Write the Header Row (Column Names — extended with tracking data)
with open(file_name, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Timestamp", "Lane1_Count", "Lane2_Count",
        "Lane1_Unique", "Lane2_Unique",
        "Avg_Speed_L1", "Avg_Speed_L2",
        "Ambulance_Detected", "Green_Time_L1", "Green_Time_L2"
    ])
print(f"✅ Logging data to: {file_name}")

# --- 4. CONFIGURATION (LANE BOXES) ---
# [x_min, y_min, x_max, y_max]
lane1_limits = [50, 100, 350, 500]   # Left Lane (Red Box)
lane2_limits = [400, 100, 700, 500]  # Right Lane (Blue Box)

frame_idx = 0
while True:
    success, img = cap.read()
    if not success:
        # Loop video forever for demo — reset tracker on loop
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        tracker.reset()
        frame_idx = 0
        continue

    frame_idx += 1
    if args.skip_frames > 0 and (frame_idx % (args.skip_frames + 1)) != 1:
        continue

    ambulance_detected = False

    # --- 5. TRAFFIC DETECTION + TRACKING (Brain 1 + ByteTrack) ---
    results = model_traffic(img, **predict_kwargs)
    detections = cast(
        sv.Detections,
        sv.Detections.from_ultralytics(results[0]),
    )

    if detections.class_id is None or detections.confidence is None:
        continue

    # Filter: keep only vehicle classes above confidence threshold
    mask = (np.isin(detections.class_id, vehicle_class_ids)
            & (detections.confidence > 0.15))
    detections = cast(sv.Detections, detections[mask])

    # Track across frames (assigns persistent IDs)
    tracked = tracker.update(detections)
    info = tracker.process(tracked, lane1_limits, lane2_limits)

    count_lane1 = info["lane1_count"]
    count_lane2 = info["lane2_count"]
    avg_speed_l1 = (
        np.mean(info["lane1_speeds"]) if info["lane1_speeds"] else 0.0
    )
    avg_speed_l2 = (
        np.mean(info["lane2_speeds"]) if info["lane2_speeds"] else 0.0
    )
    uniq_l1, uniq_l2 = tracker.get_unique_counts()

    # Draw tracked vehicles with IDs and movement trails
    for t in info["tracks"]:
        x1, y1, x2, y2 = t["bbox"]
        tid = t["id"]
        spd = t["speed"]
        trail = t["trail"]

        if t["lane"] == "lane1":
            color = (0, 0, 255)        # Red (BGR)
        elif t["lane"] == "lane2":
            color = (255, 0, 0)        # Blue (BGR)
        else:
            color = (180, 180, 180)    # Grey for vehicles outside lanes

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label: Track ID + speed
        label = f"ID:{tid} {spd:.0f}px/s"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Movement trail (fading line)
        for j in range(1, len(trail)):
            thickness = max(1, int(2 * j / len(trail)))
            cv2.line(img, trail[j - 1], trail[j], color, thickness)

    # --- 6. AMBULANCE DETECTION (Brain 2) ---
    results_amb = model_ambulance(img, **predict_kwargs)

    for box in results_amb[0].boxes:
        cls = int(box.cls[0])
        # Check if your model uses 'Ambulance' or 'ambulance'
        # Also check your confidence threshold (adjust 0.7 as needed)
        if model_ambulance.names[cls] == 'Ambulance' and box.conf[0] > 0.7:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Aspect Ratio & Size Filter to stop false positives
            w, h = x2 - x1, y2 - y1
            if (w * h) > 3000 and (w / h) < 2.0:
                ambulance_detected = True
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(img, "AMBULANCE", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- 7. CALCULATE TIMERS ---
    t1 = min(5 + (count_lane1 * 2), 60)
    t2 = min(5 + (count_lane2 * 2), 60)

    # --- 8. DISPLAY LOGIC ---
    # Draw Lane Boundaries
    cv2.rectangle(img, (lane1_limits[0], lane1_limits[1]),
                  (lane1_limits[2], lane1_limits[3]), (0, 0, 255), 2)
    cv2.rectangle(img, (lane2_limits[0], lane2_limits[1]),
                  (lane2_limits[2], lane2_limits[3]), (255, 0, 0), 2)

    # Dashboard Background (Height 130 to fit tracking info)
    h_img, w_img = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w_img, 130), (0, 0, 0), -1)

    if ambulance_detected:
        cv2.putText(img, '!!! EMERGENCY OVERRIDE !!!', (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    else:
        # LANE 1 STATS (Top Line - RED)
        cv2.putText(img,
                    f'LANE 1: {count_lane1} ({uniq_l1} unique) | Time: {t1}s | Spd: {avg_speed_l1:.0f}px/s',
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # LANE 2 STATS (Middle Line - BLUE)
        cv2.putText(img,
                    f'LANE 2: {count_lane2} ({uniq_l2} unique) | Time: {t2}s | Spd: {avg_speed_l2:.0f}px/s',
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        # TRACKER STATUS (Bottom Line - GREEN)
        cv2.putText(img,
                    f'TRACKER: {tracker.active_tracks()} active tracks | ByteTrack',
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # --- 9. DATA LOGGING (every 5 seconds) ---
    # We log only once every 5 seconds to avoid flooding the CSV file
    if int(time.time()) % 5 == 0:
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Open in APPEND mode ('a')
        with open(file_name, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp, count_lane1, count_lane2,
                uniq_l1, uniq_l2,
                round(avg_speed_l1, 1), round(avg_speed_l2, 1),
                ambulance_detected, t1, t2
            ])

        print(f"Logged: {timestamp} | L1: {count_lane1} (uniq:{uniq_l1}) | "
              f"L2: {count_lane2} (uniq:{uniq_l2}) | Tracks: {tracker.active_tracks()}")

    cv2.imshow("NETRA + ByteTrack", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Session Summary
uniq_l1, uniq_l2 = tracker.get_unique_counts()
print(f"\n{'='*50}")
print("📊 SESSION SUMMARY")
print(f"{'='*50}")
print(f"  Unique Vehicles — Lane 1: {uniq_l1}  |  Lane 2: {uniq_l2}")
print(f"  Total Unique: {uniq_l1 + uniq_l2}")
print(f"  Data saved to: {file_name}")
print(f"{'='*50}")
