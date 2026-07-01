"""
🚦 PROJECT NETRA - Main Application with Error Handling & Logging
Intelligent Traffic Management System using YOLOv8 + ByteTrack
"""

import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import csv
import time
import argparse
import logging
import sys
import yaml
from typing import Dict, List, Tuple, Optional, Any, cast
from datetime import datetime
from pathlib import Path

from src.tracker import VehicleTracker
from src.model_runtime import (
    RuntimeConfig,
    build_predict_kwargs,
    format_runtime_summary,
    resolve_device,
    resolve_precision,
)

# ==================== SETUP LOGGING ====================

def setup_logger(config: Dict[str, Any]) -> logging.Logger:
    """Initialize structured logging system.
    
    Args:
        config: Configuration dictionary with logger settings
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger_config: Dict[str, Any] = config.get("logger", {})
    log_level: int = getattr(logging, logger_config.get("level", "INFO"))
    log_format: str = logger_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logger: logging.Logger = logging.getLogger("NETRA")
    logger.setLevel(log_level)
    
    # Console handler
    if logger_config.get("console_enabled", True):
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
    
    # File handler
    if logger_config.get("file_enabled", True):
        log_file: str = logger_config.get("file", "logs/netra.log")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler: logging.FileHandler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dict: Configuration dictionary or empty dict if loading fails
    """
    try:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        print("   Using command-line arguments instead.")
        return {}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (can override config file).
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="NETRA main runtime with edge optimization options."
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--video",
        help="Path to input video (overrides config)",
    )
    parser.add_argument(
        "--traffic-model",
        help="Traffic detector model path (overrides config)",
    )
    parser.add_argument(
        "--ambulance-model",
        help="Ambulance detector model path (overrides config)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device (overrides config)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        help="Inference precision (overrides config)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        help="Inference image size (overrides config)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        help="Skip N frames between processed frames (overrides config)",
    )
    return parser.parse_args()


def merge_config_and_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge YAML config with command-line arguments (CLI takes precedence).
    
    Args:
        config: Configuration dictionary from YAML
        args: Parsed command-line arguments
    
    Returns:
        Dict: Merged configuration
    """
    if args.video:
        config.setdefault("video", {})["input_path"] = args.video
    
    if args.traffic_model:
        config.setdefault("models", {})["traffic_model"] = args.traffic_model
    
    if args.ambulance_model:
        config.setdefault("models", {})["ambulance_model"] = args.ambulance_model
    
    if args.device:
        config.setdefault("runtime", {})["device"] = args.device
    
    if args.precision:
        config.setdefault("runtime", {})["precision"] = args.precision
    
    if args.imgsz:
        config.setdefault("runtime", {})["imgsz"] = args.imgsz
    
    if args.skip_frames is not None:
        config.setdefault("runtime", {})["skip_frames"] = args.skip_frames
    
    return config


def main() -> None:
    """Main application entry point with comprehensive error handling."""
    
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    config = merge_config_and_args(config, args)
    
    # Setup logging
    logger = setup_logger(config)
    logger.info("🚦 PROJECT NETRA - Starting...")
    logger.info(f"Configuration file: {args.config}")
    
    try:
        # ==================== SETUP VIDEO & MODELS ====================
        
        video_path = config.get("video", {}).get("input_path", "videos/traffic.mp4")
        logger.info(f"Opening video: {video_path}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Failed to open video: {video_path}")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"✅ Video loaded: {frame_width}x{frame_height} @ {fps} FPS ({total_frames} frames)")
        except Exception as e:
            logger.error(f"❌ Video loading failed: {e}")
            raise
        
        # Resolve device and precision
        runtime_cfg = config.get("runtime", {})
        device = resolve_device(runtime_cfg.get("device", "auto"))
        precision = runtime_cfg.get("precision", "fp32")
        use_half = resolve_precision(precision, device)
        imgsz = runtime_cfg.get("imgsz", 640)
        
        runtime_config = RuntimeConfig(
            device=device,
            use_half=use_half,
            imgsz=imgsz,
        )
        
        predict_kwargs = build_predict_kwargs(runtime_config)
        logger.info(f"Runtime: {format_runtime_summary(runtime_config)}")
        
        # Load models
        model_paths = config.get("models", {})
        traffic_model_path = model_paths.get("traffic_model", "models/yolov8m.pt")
        ambulance_model_path = model_paths.get("ambulance_model", "models/best.pt")
        
        try:
            logger.info("Loading models...")
            model_traffic = YOLO(traffic_model_path)
            model_ambulance = YOLO(ambulance_model_path)
            logger.info("✅ Models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
        
        # ==================== SETUP VEHICLE TRACKER ====================
        
        tracking_cfg = config.get("tracking", {})
        frame_rate = tracking_cfg.get("frame_rate", fps)
        trail_length = tracking_cfg.get("trail_length", 30)
        
        tracker = VehicleTracker(frame_rate=frame_rate, trail_length=trail_length)
        
        detection_cfg = config.get("detection", {})
        vehicle_classes = detection_cfg.get("vehicle_classes", ["car", "truck", "bus", "motorbike", "bicycle"])
        vehicle_confidence = detection_cfg.get("vehicle_confidence", 0.15)
        ambulance_confidence = detection_cfg.get("ambulance_confidence", 0.7)
        
        vehicle_class_ids = [
            cid for cid, name in model_traffic.names.items()
            if name.lower() in [v.lower() for v in vehicle_classes]
        ]
        logger.info(f"🔍 ByteTrack enabled @ {frame_rate} FPS | Tracking: {vehicle_classes}")
        
        # ==================== SETUP DATA LOGGING ====================
        
        log_cfg = config.get("logging", {})
        if log_cfg.get("enabled", True):
            log_dir = log_cfg.get("output_dir", "data/traffic_logs")
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            
            file_name = (
                f"{log_dir}/Traffic_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            try:
                with open(file_name, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(log_cfg.get("columns", [
                        "Timestamp", "Lane1_Count", "Lane2_Count",
                        "Lane1_Unique", "Lane2_Unique",
                        "Avg_Speed_L1", "Avg_Speed_L2",
                        "Ambulance_Detected", "Green_Time_L1", "Green_Time_L2"
                    ]))
                logger.info(f"✅ Logging enabled: {file_name}")
            except Exception as e:
                logger.error(f"❌ Failed to create log file: {e}")
                log_cfg["enabled"] = False
        else:
            file_name = None
            logger.info("⏭️  Data logging disabled")
        
        # ==================== LANE CONFIGURATION ====================
        
        lanes_cfg = config.get("lanes", {})
        lane1_limits = lanes_cfg.get("lane1", {}).get("roi", [50, 100, 350, 500])
        lane2_limits = lanes_cfg.get("lane2", {}).get("roi", [400, 100, 700, 500])
        
        logger.info(f"Lane 1 ROI: {lane1_limits}")
        logger.info(f"Lane 2 ROI: {lane2_limits}")
        
        # ==================== SIGNAL TIMING CONFIGURATION ====================
        
        signal_cfg = config.get("signal_timing", {})
        base_time = signal_cfg.get("base_time", 5)
        multiplier = signal_cfg.get("multiplier", 2)
        min_time = signal_cfg.get("min_time", 5)
        max_time = signal_cfg.get("max_time", 60)
        
        # ==================== AMBULANCE CONFIGURATION ====================
        
        amb_cfg = config.get("ambulance", {})
        ambulance_enabled = amb_cfg.get("enabled", True)
        ambulance_min_area = amb_cfg.get("min_area", 3000)
        ambulance_max_aspect = amb_cfg.get("max_aspect_ratio", 2.0)
        
        # ==================== MAIN PROCESSING LOOP ====================
        
        frame_idx = 0
        log_interval = log_cfg.get("log_interval_seconds", 5)
        last_log_time = time.time()
        display_enabled = config.get("video", {}).get("display_enabled", True)
        skip_frames = runtime_cfg.get("skip_frames", 0)
        
        logger.info("🎬 Starting frame processing...")
        
        while True:
            try:
                success, img = cap.read()
                
                if not success:
                    # Loop video forever for demo
                    logger.info("🔄 Video looped")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    tracker.reset()
                    frame_idx = 0
                    continue
                
                frame_idx += 1
                
                if skip_frames > 0 and (frame_idx % (skip_frames + 1)) != 1:
                    continue
                
                ambulance_detected = False
                
                # ===== TRAFFIC DETECTION + TRACKING =====
                try:
                    results = model_traffic(img, **predict_kwargs)
                    detections = cast(sv.Detections, sv.Detections.from_ultralytics(results[0]))
                    
                    if detections.class_id is None or detections.confidence is None:
                        tracked = sv.Detections()
                    else:
                        # Filter vehicles
                        mask = (np.isin(detections.class_id, vehicle_class_ids)
                                & (detections.confidence > vehicle_confidence))
                        detections = cast(sv.Detections, detections[mask])
                        
                        # Track across frames
                        tracked = tracker.update(detections)
                except Exception as e:
                    logger.warning(f"⚠️ Traffic detection error: {e}")
                    tracked = sv.Detections()
                
                try:
                    info = tracker.process(tracked, lane1_limits, lane2_limits)
                except Exception as e:
                    logger.warning(f"⚠️ Tracking error: {e}")
                    info = {
                        "lane1_count": 0, "lane2_count": 0,
                        "lane1_speeds": [], "lane2_speeds": [], "tracks": []
                    }
                
                count_lane1 = info["lane1_count"]
                count_lane2 = info["lane2_count"]
                avg_speed_l1 = np.mean(info["lane1_speeds"]) if info["lane1_speeds"] else 0.0
                avg_speed_l2 = np.mean(info["lane2_speeds"]) if info["lane2_speeds"] else 0.0
                uniq_l1, uniq_l2 = tracker.get_unique_counts()
                
                # ===== DRAW TRACKED VEHICLES =====
                try:
                    for t in info["tracks"]:
                        x1, y1, x2, y2 = t["bbox"]
                        tid = t["id"]
                        spd = t["speed"]
                        trail = t["trail"]
                        
                        if t["lane"] == "lane1":
                            color = (0, 0, 255)
                        elif t["lane"] == "lane2":
                            color = (255, 0, 0)
                        else:
                            color = (180, 180, 180)
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"ID:{tid} {spd:.0f}px/s"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(img, label, (x1 + 2, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                        
                        for j in range(1, len(trail)):
                            thickness = max(1, int(2 * j / len(trail)))
                            cv2.line(img, trail[j - 1], trail[j], color, thickness)
                except Exception as e:
                    logger.warning(f"⚠️ Drawing error: {e}")
                
                # ===== AMBULANCE DETECTION =====
                if ambulance_enabled:
                    try:
                        results_amb = model_ambulance(img, **predict_kwargs)
                        
                        for box in results_amb[0].boxes:
                            cls = int(box.cls[0])
                            model_name = model_ambulance.names.get(cls, "").lower()
                            conf = float(box.conf[0])
                            
                            if 'ambulance' in model_name and conf > ambulance_confidence:
                                x1, y1, x2, y2 = box.xyxy[0]
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                w, h = x2 - x1, y2 - y1
                                if (w * h) > ambulance_min_area and (w / h) < ambulance_max_aspect:
                                    ambulance_detected = True
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                                    cv2.putText(img, "AMBULANCE", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    except Exception as e:
                        logger.warning(f"⚠️ Ambulance detection error: {e}")
                
                # ===== CALCULATE TIMERS =====
                t1 = min(base_time + (count_lane1 * multiplier), max_time)
                t2 = min(base_time + (count_lane2 * multiplier), max_time)
                
                # ===== DISPLAY LOGIC =====
                try:
                    h_img, w_img = img.shape[:2]
                    
                    cv2.rectangle(img, (lane1_limits[0], lane1_limits[1]),
                                  (lane1_limits[2], lane1_limits[3]), (0, 0, 255), 2)
                    cv2.rectangle(img, (lane2_limits[0], lane2_limits[1]),
                                  (lane2_limits[2], lane2_limits[3]), (255, 0, 0), 2)
                    
                    cv2.rectangle(img, (0, 0), (w_img, 130), (0, 0, 0), -1)
                    
                    if ambulance_detected:
                        cv2.putText(img, '!!! EMERGENCY OVERRIDE !!!', (50, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    else:
                        cv2.putText(img,
                                    f'LANE 1: {count_lane1} ({uniq_l1} unique) | Time: {t1}s | Spd: {avg_speed_l1:.0f}px/s',
                                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                        
                        cv2.putText(img,
                                    f'LANE 2: {count_lane2} ({uniq_l2} unique) | Time: {t2}s | Spd: {avg_speed_l2:.0f}px/s',
                                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
                        
                        cv2.putText(img,
                                    f'TRACKER: {tracker.active_tracks()} active tracks | ByteTrack',
                                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                except Exception as e:
                    logger.warning(f"⚠️ Display error: {e}")
                
                # ===== DATA LOGGING =====
                if log_cfg.get("enabled", False) and file_name:
                    try:
                        current_time = time.time()
                        if current_time - last_log_time >= log_interval:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            
                            with open(file_name, mode='a', newline='', encoding='utf-8') as file:
                                writer = csv.writer(file)
                                writer.writerow([
                                    timestamp, count_lane1, count_lane2,
                                    uniq_l1, uniq_l2,
                                    round(avg_speed_l1, 1), round(avg_speed_l2, 1),
                                    int(ambulance_detected), int(t1), int(t2)
                                ])
                            
                            logger.info(f"Logged: {timestamp} | L1: {count_lane1} (uniq:{uniq_l1}) | "
                                      f"L2: {count_lane2} (uniq:{uniq_l2}) | Tracks: {tracker.active_tracks()}")
                            last_log_time = current_time
                    except Exception as e:
                        logger.error(f"❌ Logging error: {e}")
                
                # ===== DISPLAY VIDEO =====
                if display_enabled:
                    try:
                        cv2.imshow("NETRA + ByteTrack", img)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("🛑 User requested exit (q key)")
                            break
                    except Exception as e:
                        logger.warning(f"⚠️ Display error: {e}")
                        display_enabled = False
            
            except KeyboardInterrupt:
                logger.info("🛑 Interrupted by user (Ctrl+C)")
                break
            except Exception as e:
                logger.error(f"❌ Frame processing error: {e}", exc_info=True)
                continue
        
        # ==================== CLEANUP ====================
        
        logger.info("Shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        
        uniq_l1, uniq_l2 = tracker.get_unique_counts()
        logger.info(f"\n{'='*50}")
        logger.info("📊 SESSION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Unique Vehicles — Lane 1: {uniq_l1}  |  Lane 2: {uniq_l2}")
        logger.info(f"Total Unique: {uniq_l1 + uniq_l2}")
        if log_cfg.get("enabled", False):
            logger.info(f"Data saved to: {file_name}")
        logger.info(f"{'='*50}")
        logger.info("✅ NETRA shutdown complete")
    
    except Exception as e:
        logger.critical(f"❌ Critical error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
