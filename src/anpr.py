"""
🚦 PROJECT NETRA - ANPR (Automatic Number Plate Recognition) Module
Standalone license plate detection and OCR for traffic management

Example usage:
    python src/anpr.py
    python src/anpr.py --video videos/traffic.mp4 --device cpu --imgsz 480
    python src/anpr.py --skip-frames 2 --imgsz 416
"""

import cv2
import csv
import time
import argparse
import logging
import sys
import yaml
import re
import easyocr
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.model_runtime import resolve_device, resolve_precision

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ==================== SETUP LOGGING ====================

def setup_logger(config: Dict[str, Any]) -> logging.Logger:
    """Initialize structured logging system for ANPR module.
    
    Args:
        config: Configuration dictionary with logger settings
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger_config: Dict[str, Any] = config.get("logger", {})
    log_level: int = getattr(logging, logger_config.get("level", "INFO"))
    log_format: str = logger_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logger: logging.Logger = logging.getLogger("NETRA.ANPR")
    
    # Avoid adding duplicate handlers
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(log_level)
    
    # Console handler
    if logger_config.get("console_enabled", True):
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
    
    # File handler
    if logger_config.get("file_enabled", True):
        log_file: str = "logs/netra_anpr.log"
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
        
        with open(config_path, 'r') as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        
        return config if config else {}
    except Exception as e:
        print(f"⚠️  Error loading config: {e}")
        return {}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (can override config file).
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="NETRA ANPR module: standalone license plate recognition"
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
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device (overrides config)",
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
    
    if args.device:
        config.setdefault("runtime", {})["device"] = args.device
    
    if args.imgsz:
        config.setdefault("runtime", {})["imgsz"] = args.imgsz
    
    if args.skip_frames is not None:
        config.setdefault("runtime", {})["skip_frames"] = args.skip_frames
    
    return config


# ==================== PLATE DETECTION & OCR ====================

@dataclass
class PlateDetection:
    """Data class for plate detection results"""
    plate_text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    frame_id: int
    timestamp: str


class ANPREngine:
    """ANPR detection and recognition engine"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str,
        imgsz: int,
        logger: logging.Logger
    ) -> None:
        """Initialize ANPR engine.
        
        Args:
            config: ANPR configuration section
            device: Target device (cpu, mps, cuda)
            imgsz: Inference image size
            logger: Logger instance
        """
        self.config = config
        self.device = device
        self.imgsz = imgsz
        self.logger = logger
        
        # Initialize OCR reader (languages: English + common ANPR plates)
        try:
            self.reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))
            self.logger.info("✅ easyocr initialized for OCR")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize easyocr: {e}")
            raise
        
        # Try to load plate detector model
        self.plate_detector: Optional[Any] = None
        self.use_plate_detector = False
        
        if YOLO_AVAILABLE:
            plate_model_path = "models/plate_detector.pt"
            if Path(plate_model_path).exists():
                try:
                    self.plate_detector = YOLO(plate_model_path)
                    self.use_plate_detector = True
                    self.logger.info(f"✅ Plate detector loaded from {plate_model_path}")
                    self.logger.info("   → Using PRIMARY path: YOLOv8 plate detector + easyocr OCR")
                except Exception as e:
                    self.logger.warning(f"⚠️  Failed to load plate detector: {e}")
                    self.logger.info("   → Using FALLBACK path: easyocr OCR on frame regions")
            else:
                self.logger.info(f"⚠️  Plate detector not found at {plate_model_path}")
                self.logger.info("   → Using FALLBACK path: easyocr OCR on frame regions")
        else:
            self.logger.info("   → Using FALLBACK path: easyocr OCR on frame regions")
        
        # OCR thresholds
        self.min_plate_length = config.get("min_plate_length", 4)
        self.min_ocr_confidence = config.get("confidence", 0.5)
    
    def detect_and_recognize(
        self,
        frame: Any,
        frame_id: int,
        timestamp: str
    ) -> List[PlateDetection]:
        """Detect and recognize license plates in frame.
        
        Args:
            frame: Video frame (numpy array)
            frame_id: Frame index
            timestamp: Frame timestamp string
        
        Returns:
            List of PlateDetection objects
        """
        detections: List[PlateDetection] = []
        
        try:
            if self.use_plate_detector:
                # PRIMARY path: detect plate regions with YOLOv8, then OCR
                detections = self._detect_with_yolo_ocr(frame, frame_id, timestamp)
            else:
                # FALLBACK path: scan frame regions with easyocr
                detections = self._detect_with_ocr_fallback(frame, frame_id, timestamp)
        except Exception as e:
            self.logger.warning(f"⚠️  ANPR detection error on frame {frame_id}: {e}")
        
        return detections
    
    def _detect_with_yolo_ocr(
        self,
        frame: Any,
        frame_id: int,
        timestamp: str
    ) -> List[PlateDetection]:
        """PRIMARY path: YOLOv8 plate detector + easyocr.
        
        Args:
            frame: Video frame
            frame_id: Frame index
            timestamp: Timestamp
        
        Returns:
            List of PlateDetection objects
        """
        detections: List[PlateDetection] = []
        
        try:
            # Run plate detector
            results = self.plate_detector(frame, imgsz=self.imgsz, verbose=False, device=self.device)
            
            if not results or len(results) == 0:
                return detections
            
            # Extract bounding boxes
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Crop plate region
                plate_region = frame[y1:y2, x1:x2]
                if plate_region.size == 0:
                    continue
                
                # OCR on plate region
                try:
                    ocr_results = self.reader.readtext(plate_region, detail=1)
                    if not ocr_results:
                        continue
                    
                    # Combine OCR results
                    plate_text = "".join([text for (_, text, conf) in ocr_results])
                    avg_confidence = sum([conf for (_, text, conf) in ocr_results]) / len(ocr_results)
                    
                    # Filter and clean
                    if self._is_valid_plate(plate_text, avg_confidence):
                        plate_text = self._clean_plate_text(plate_text)
                        detection = PlateDetection(
                            plate_text=plate_text,
                            confidence=avg_confidence,
                            bbox=(x1, y1, x2, y2),
                            frame_id=frame_id,
                            timestamp=timestamp
                        )
                        detections.append(detection)
                except Exception as e:
                    self.logger.debug(f"   OCR error on plate region: {e}")
                    continue
        
        except Exception as e:
            self.logger.debug(f"   Plate detector error: {e}")
        
        return detections
    
    def _detect_with_ocr_fallback(
        self,
        frame: Any,
        frame_id: int,
        timestamp: str
    ) -> List[PlateDetection]:
        """FALLBACK path: scan frame with easyocr directly.
        
        Scans the entire frame for text regions that look like license plates.
        
        Args:
            frame: Video frame
            frame_id: Frame index
            timestamp: Timestamp
        
        Returns:
            List of PlateDetection objects
        """
        detections: List[PlateDetection] = []
        
        try:
            # Run easyocr on full frame
            ocr_results = self.reader.readtext(frame, detail=1)
            
            if not ocr_results:
                return detections
            
            # Filter results for plate-like text
            for (bbox, text, conf) in ocr_results:
                # Get bounding box from OCR
                bbox_array = bbox  # List of 4 corner points
                if len(bbox_array) < 4:
                    continue
                
                # Extract min/max coordinates
                x_coords = [pt[0] for pt in bbox_array]
                y_coords = [pt[1] for pt in bbox_array]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                
                # Filter and clean
                if self._is_valid_plate(text, conf):
                    plate_text = self._clean_plate_text(text)
                    detection = PlateDetection(
                        plate_text=plate_text,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        frame_id=frame_id,
                        timestamp=timestamp
                    )
                    detections.append(detection)
        
        except Exception as e:
            self.logger.debug(f"   OCR fallback error: {e}")
        
        return detections
    
    def _is_valid_plate(self, text: str, confidence: float) -> bool:
        """Check if OCR result looks like a valid plate.
        
        Args:
            text: OCR text
            confidence: OCR confidence score
        
        Returns:
            bool: True if valid plate
        """
        if not text:
            return False
        
        # Check confidence
        if confidence < self.min_ocr_confidence:
            return False
        
        # Minimum length
        if len(text.strip()) < self.min_plate_length:
            return False
        
        return True
    
    def _clean_plate_text(self, text: str) -> str:
        """Clean and standardize plate text.
        
        Args:
            text: Raw OCR text
        
        Returns:
            Cleaned plate text (uppercase alphanumeric only)
        """
        # Strip whitespace
        text = text.strip()
        
        # Remove all non-alphanumeric characters
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        return text


# ==================== LANE ASSIGNMENT ====================

def get_lane_for_plate(
    bbox: Tuple[int, int, int, int],
    lanes_config: Dict[str, Any]
) -> str:
    """Determine which lane a plate detection belongs to.
    
    Uses ROI boxes from config to assign lanes based on plate bbox center.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        lanes_config: Lane configuration from config
    
    Returns:
        str: Lane name ('lane1', 'lane2', or 'unknown')
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2  # Center X
    cy = (y1 + y2) // 2  # Center Y
    
    # Check lane1 ROI
    lane1_config = lanes_config.get("lane1", {})
    lane1_roi = lane1_config.get("roi", [50, 100, 350, 500])
    if (lane1_roi[0] <= cx <= lane1_roi[2]) and (lane1_roi[1] <= cy <= lane1_roi[3]):
        return "lane1"
    
    # Check lane2 ROI
    lane2_config = lanes_config.get("lane2", {})
    lane2_roi = lane2_config.get("roi", [400, 100, 700, 500])
    if (lane2_roi[0] <= cx <= lane2_roi[2]) and (lane2_roi[1] <= cy <= lane2_roi[3]):
        return "lane2"
    
    return "unknown"


# ==================== VISUALIZATION ====================

def draw_plate_detection(
    frame: Any,
    detection: PlateDetection,
    lane: str,
    lanes_config: Dict[str, Any]
) -> None:
    """Draw plate detection on frame.
    
    Args:
        frame: Video frame (modified in place)
        detection: PlateDetection object
        lane: Lane assignment
        lanes_config: Lane configuration
    """
    try:
        x1, y1, x2, y2 = detection.bbox
        
        # Green box around plate
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Plate text label with green background
        label = f"{detection.plate_text} ({detection.confidence:.2f})"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Green background rectangle
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - 8),
            (x1 + label_w + 8, y1),
            (0, 255, 0),
            -1
        )
        
        # White text
        cv2.putText(
            frame,
            label,
            (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    except Exception as e:
        pass  # Silently skip drawing errors


def draw_hud(
    frame: Any,
    total_plates: int,
    last_plate_text: Optional[str]
) -> None:
    """Draw HUD bar at top of frame.
    
    Args:
        frame: Video frame (modified in place)
        total_plates: Total plates detected in session
        last_plate_text: Last detected plate text or None
    """
    try:
        h, w = frame.shape[:2]
        
        # Black HUD bar (80px height)
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        
        # Left: Total plates
        left_text = f"Total Plates: {total_plates}"
        cv2.putText(
            frame,
            left_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Right: Last plate
        if last_plate_text:
            right_text = f"Last: {last_plate_text}"
        else:
            right_text = "Last: -"
        
        (text_w, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(
            frame,
            right_text,
            (w - text_w - 20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
    except Exception as e:
        pass  # Silently skip HUD errors


# ==================== CSV LOGGING ====================

def initialize_csv_log(output_dir: str, logger: logging.Logger) -> Tuple[str, bool]:
    """Create and initialize CSV log file.
    
    Args:
        output_dir: Output directory for CSV
        logger: Logger instance
    
    Returns:
        Tuple of (file_path, success_bool)
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        file_name = (
            f"{output_dir}/ANPR_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        with open(file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp", "Frame_ID", "Track_ID", "Plate_Text",
                "Confidence", "Lane", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"
            ])
        
        logger.info(f"✅ CSV log initialized: {file_name}")
        return file_name, True
    except Exception as e:
        logger.error(f"❌ Failed to create CSV log: {e}")
        return "", False


def log_detections_to_csv(
    file_name: str,
    detections: List[Tuple[PlateDetection, str]],
    logger: logging.Logger
) -> None:
    """Append detections to CSV log.
    
    Args:
        file_name: CSV file path
        detections: List of (PlateDetection, lane) tuples
        logger: Logger instance
    """
    if not file_name or not detections:
        return
    
    try:
        with open(file_name, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for detection, lane in detections:
                x1, y1, x2, y2 = detection.bbox
                writer.writerow([
                    detection.timestamp,
                    detection.frame_id,
                    -1,  # Track_ID is -1 (no tracking)
                    detection.plate_text,
                    round(detection.confidence, 4),
                    lane,
                    x1, y1, x2, y2
                ])
    except Exception as e:
        logger.error(f"❌ CSV write error: {e}")


def save_plate_image(
    frame: Any,
    detection: PlateDetection,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """Save cropped plate image to disk.
    
    Args:
        frame: Video frame
        detection: PlateDetection object
        output_dir: Output directory for plate images
        logger: Logger instance
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        x1, y1, x2, y2 = detection.bbox
        plate_crop = frame[y1:y2, x1:x2]
        
        if plate_crop.size == 0:
            return
        
        # Generate filename
        filename = (
            f"{output_dir}/plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"f{detection.frame_id:06d}.jpg"
        )
        
        cv2.imwrite(filename, plate_crop)
    except Exception as e:
        logger.debug(f"⚠️  Failed to save plate image: {e}")


# ==================== MAIN ENTRY POINT ====================

def main() -> None:
    """Main ANPR application entry point."""
    
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    config = merge_config_and_args(config, args)
    
    # Setup logging
    logger = setup_logger(config)
    logger.info("🚦 PROJECT NETRA - ANPR Module Starting...")
    logger.info(f"Configuration file: {args.config}")
    
    try:
        # ==================== SETUP VIDEO ====================
        
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
        
        # ==================== SETUP DEVICE & PRECISION ====================
        
        runtime_cfg = config.get("runtime", {})
        device = resolve_device(runtime_cfg.get("device", "auto"))
        precision = runtime_cfg.get("precision", "fp32")
        use_half = resolve_precision(precision, device)  # Respects CPU/MPS limitations
        imgsz = runtime_cfg.get("imgsz", 640)
        
        logger.info(f"Runtime: device={device}, precision={precision}, imgsz={imgsz}")
        
        # ==================== SETUP ANPR ENGINE ====================
        
        anpr_cfg = config.get("anpr", {})
        if not anpr_cfg.get("enabled", True):
            logger.warning("⚠️  ANPR is disabled in config. Enable 'anpr.enabled: true'")
        
        try:
            anpr = ANPREngine(anpr_cfg, device, imgsz, logger)
        except Exception as e:
            logger.error(f"❌ Failed to initialize ANPR engine: {e}")
            raise
        
        # ==================== SETUP DATA LOGGING ====================
        
        output_dir = anpr_cfg.get("output_dir", "data/anpr_logs")
        csv_file, csv_enabled = initialize_csv_log(output_dir, logger)
        
        log_interval = anpr_cfg.get("log_interval_seconds", 5)
        last_log_time = time.time()
        
        save_plates = anpr_cfg.get("save_plate_images", False)
        plates_dir = anpr_cfg.get("plate_images_dir", "data/anpr_logs/plates")
        
        # ==================== LANE CONFIGURATION ====================
        
        lanes_cfg = config.get("lanes", {})
        logger.info(f"Lane 1 ROI: {lanes_cfg.get('lane1', {}).get('roi', [50, 100, 350, 500])}")
        logger.info(f"Lane 2 ROI: {lanes_cfg.get('lane2', {}).get('roi', [400, 100, 700, 500])}")
        
        # ==================== MAIN PROCESSING LOOP ====================
        
        frame_idx = 0
        display_enabled = config.get("video", {}).get("display_enabled", True)
        skip_frames = runtime_cfg.get("skip_frames", 0)
        
        total_plates_session = 0
        unique_plates = set()
        last_plate_text: Optional[str] = None
        frame_detections_buffer: List[Tuple[PlateDetection, str]] = []
        
        logger.info("🎬 Starting frame processing...")
        
        while True:
            try:
                success, img = cap.read()
                
                if not success:
                    # Loop video forever for demo
                    logger.info("🔄 Video looped")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    continue
                
                frame_idx += 1
                
                # Skip frames if configured
                if skip_frames > 0 and (frame_idx % (skip_frames + 1)) != 1:
                    continue
                
                timestamp = datetime.now().strftime('%H:%M:%S')
                
                # ===== ANPR DETECTION & RECOGNITION =====
                try:
                    detections = anpr.detect_and_recognize(img, frame_idx, timestamp)
                except Exception as e:
                    logger.warning(f"⚠️  ANPR error on frame {frame_idx}: {e}")
                    detections = []
                
                # ===== ASSIGN LANES AND PREPARE FOR LOGGING =====
                frame_with_lanes: List[Tuple[PlateDetection, str]] = []
                for detection in detections:
                    lane = get_lane_for_plate(detection.bbox, lanes_cfg)
                    frame_with_lanes.append((detection, lane))
                    frame_detections_buffer.append((detection, lane))
                    
                    # Track totals
                    total_plates_session += 1
                    unique_plates.add(detection.plate_text)
                    last_plate_text = detection.plate_text
                    
                    # Save plate image if enabled
                    if save_plates:
                        try:
                            save_plate_image(img, detection, plates_dir, logger)
                        except Exception as e:
                            logger.debug(f"⚠️  Plate save error: {e}")
                
                # ===== DRAW DETECTIONS =====
                try:
                    for detection, lane in frame_with_lanes:
                        draw_plate_detection(img, detection, lane, lanes_cfg)
                except Exception as e:
                    logger.warning(f"⚠️  Drawing error: {e}")
                
                # ===== DRAW HUD =====
                try:
                    draw_hud(img, total_plates_session, last_plate_text)
                except Exception as e:
                    logger.debug(f"⚠️  HUD error: {e}")
                
                # ===== LOG TO CSV =====
                if csv_enabled:
                    try:
                        current_time = time.time()
                        if current_time - last_log_time >= log_interval:
                            if frame_detections_buffer:
                                log_detections_to_csv(csv_file, frame_detections_buffer, logger)
                                logger.info(
                                    f"Logged: {len(frame_detections_buffer)} detections "
                                    f"| Total session: {total_plates_session} "
                                    f"| Unique: {len(unique_plates)}"
                                )
                                frame_detections_buffer = []
                            last_log_time = current_time
                    except Exception as e:
                        logger.error(f"❌ Logging error: {e}")
                
                # ===== DISPLAY VIDEO =====
                if display_enabled:
                    try:
                        cv2.imshow("NETRA - ANPR", img)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("🛑 User requested exit (q key)")
                            break
                    except Exception as e:
                        logger.warning(f"⚠️  Display error: {e}")
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
        
        # Flush remaining detections to CSV
        if csv_enabled and frame_detections_buffer:
            try:
                log_detections_to_csv(csv_file, frame_detections_buffer, logger)
            except Exception as e:
                logger.error(f"❌ Final log flush error: {e}")
        
        # ==================== SESSION SUMMARY ====================
        
        logger.info("")
        logger.info("=" * 50)
        logger.info("📊 ANPR SESSION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Plates Detected: {total_plates_session}")
        logger.info(f"Unique Plate Texts: {len(unique_plates)}")
        if csv_enabled:
            logger.info(f"Data saved to: {csv_file}")
        logger.info("=" * 50)
        logger.info("✅ NETRA ANPR shutdown complete")
    
    except Exception as e:
        logger.critical(f"❌ Critical error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
