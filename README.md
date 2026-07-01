# 🚦 Project NETRA (Network Enabled Traffic Regulation & Analysis)

**An Intelligent Traffic Management System using Computer Vision & Deep Learning**

Project NETRA addresses the critical issue of urban traffic congestion and delayed emergency services. Unlike traditional fixed-timer traffic lights, NETRA uses real-time camera feeds to calculate traffic density and adjust signal timings dynamically. It features **ByteTrack Multi-Object Tracking** for persistent vehicle identification, speed estimation, and unique vehicle counting — plus an **Automatic Ambulance Detection System** that overrides signals to provide a "Green Corridor" for emergency vehicles.

---

## 🌐 Live Demo

**🚀 Web Dashboard:** [https://project-netra-dr457xmtu7dhxvakejabwz.streamlit.app/](https://project-netra-dr457xmtu7dhxvakejabwz.streamlit.app/)

Experience the interactive analytics dashboard with real-time traffic visualizations, data explorer, and comprehensive system insights!

---

## 📸 Demo & Screenshots

![NETRA System Demo](assets/demo.png)

*Above: The NETRA system in action — Real-time detection and tracking of vehicles across two lanes. Each vehicle receives a persistent Track ID and speed label. Lane 1 (Red) and Lane 2 (Blue) show live counts, unique vehicle totals, signal times, and average speeds. Movement trails visualize each vehicle's trajectory.*

---

## ✨ Key Features

🧠 **Hybrid AI Architecture**: Utilizes a dual-model strategy:
- **Generalist Model (yolov8m)**: Detects standard vehicles (Cars, Trucks, Buses, Rickshaws) with high accuracy.
- **Specialist Model (Custom Trained)**: A dedicated model trained via Transfer Learning to specifically detect Ambulances.

🔍 **ByteTrack Vehicle Tracking**: Persistent multi-object tracking powered by the `supervision` library:
- **Unique Vehicle Counting**: Each vehicle gets a stable Track ID — eliminates per-frame double-counting.
- **Speed Estimation**: Real-time speed (px/s) calculated from position history over recent frames.
- **Movement Trails**: Fading trajectory lines visualize each vehicle's path across the frame.
- **Session Statistics**: Cumulative unique vehicle totals per lane across the entire session.

⏱️ **Dynamic Signal Timer**: Replaces static timers with an adaptive algorithm ($T = 5 + 2n$) that allocates green light duration based on real-time lane density.

🚑 **Emergency Override Module**: Instantly detects approaching ambulances (with geometric & confidence filtering) to clear the lane immediately.

🛣️ **Multi-Lane Logic**: Supports distinct ROI (Region of Interest) definitions to manage multiple lanes independently.

📊 **Traffic Analytics**: Automatically logs extended traffic data (vehicle counts, unique counts, speeds, timestamps, signal times) to a CSV database for urban planning analysis.

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Computer Vision**: OpenCV (cv2)
- **Deep Learning**: YOLOv8 (Ultralytics)
- **Object Tracking**: ByteTrack via Supervision library
- **Data Processing**: NumPy, Pandas (for analytics)
- **Web Dashboard**: Streamlit
- **Training Environment**: Google Colab (Tesla T4 GPU)

---

## 📁 Project Structure

```
PROJECT-NETRA/
├── main.py                          # Main app (detection + ByteTrack tracking)
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── LICENSE                          # MIT License
│
├── src/                            # Source code modules
│   ├── tracker.py                  # ByteTrack vehicle tracking module
│   ├── model_runtime.py            # Device + precision runtime tuning helpers
│   ├── export_model.py             # Export utility (ONNX/CoreML/TensorRT)
│   ├── analytics.py                # Interactive analytics dashboard
│   ├── analytics_report.py         # Headless analytics (no GUI)
│   ├── web_dashboard.py            # Streamlit web dashboard
│   └── utils/                      # Utility scripts
│       ├── check_brain.py          # Model verification tool
│       └── mouse_finder.py         # Coordinate selection helper
│
├── models/                         # AI model weights
│   ├── best.pt                     # Custom ambulance detection model
│   └── yolov8m.pt                  # YOLOv8 traffic detection model
│
├── data/                           # Data storage
│   └── traffic_logs/               # CSV traffic logs (auto-generated)
│
├── reports/                        # Generated analytics
│   └── analytics_output/           # Visualizations & summaries
│
├── docs/                           # Documentation
│   ├── ANALYTICS_GUIDE.md          # Analytics usage guide
│   └── CONTRIBUTING.md             # Contribution guidelines
│
├── assets/                         # Project assets
│   └── demo.png                    # Demo screenshot
│
└── videos/                         # Video files
    └── traffic.mp4                 # Test video input
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Project-NETRA.git
cd Project-NETRA
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install ultralytics opencv-python pandas matplotlib seaborn streamlit pillow supervision
```

### Setup Models

Place the model files in the `models/` directory:
- `yolov8m.pt` - General traffic detection (will auto-download on first run)
- `best.pt` - Your custom trained ambulance detection model

### Add Video Source

Place your test video in the `videos/` folder and rename it to `traffic.mp4` (or update the path in [main.py](main.py)).

---

## 🚀 Usage

Run the main application:

```bash
python3 main.py
```

### Performance-Friendly Runtime (Recommended)

```bash
python3 main.py --traffic-model models/yolov8n.pt --device auto --precision fp32 --imgsz 640 --skip-frames 1
```

Useful runtime flags:
- `--traffic-model`: choose a lighter model (`models/yolov8n.pt`) for edge devices.
- `--device`: `auto`, `cpu`, `mps`, `cuda`.
- `--precision`: `fp16` (CUDA only), otherwise auto-falls back to FP32.
- `--imgsz`: reduce from 640 to 512/416 for more FPS.
- `--skip-frames`: process every `(N+1)`th frame to improve throughput.

### Model Export for Edge Deployment

Export helper:

```bash
python3 src/export_model.py --model models/yolov8n.pt --format onnx --imgsz 640
```

MacBook M2 (Apple Silicon):

```bash
python3 src/export_model.py --model models/yolov8n.pt --format coreml --imgsz 640
```

Jetson / NVIDIA edge (TensorRT FP16):

```bash
python3 src/export_model.py --model models/yolov8n.pt --format engine --half --device cuda:0 --imgsz 640
```

Jetson / NVIDIA edge (TensorRT INT8):

```bash
python3 src/export_model.py --model models/yolov8n.pt --format engine --int8 --device cuda:0 --imgsz 640 --data coco8.yaml
```

Notes:
- INT8 requires representative calibration data (`--data`) for stable accuracy.
- If your platform does not support requested precision, the runtime keeps working with FP32.

### 🎮 Controls

- **q**: Quit the application.

The system will automatically generate a CSV file (e.g., `Traffic_Data_20260131.csv`) in the project folder.

### 📊 Analytics Dashboard

Run the comprehensive traffic analytics dashboard to visualize and analyze your collected data:

```bash
python3 src/analytics_report.py
```

**The analytics module provides:**

✅ **Traffic Pattern Graphs**: Visualize hourly and daily traffic trends  
✅ **Peak Hour Identification**: Automatically detect high-traffic periods  
✅ **Lane Utilization Comparison**: Analyze traffic distribution across lanes  
✅ **Ambulance Frequency Analytics**: Track emergency vehicle patterns  
✅ **Average Wait Time Calculations**: Calculate signal timing efficiency  
✅ **Correlation Heatmaps**: Understand relationships between traffic variables  
✅ **Automated Reports**: Generate PDF-ready summary reports  

**Output Files Generated:**
- `Traffic_Analysis_YYYYMMDD_HHMMSS.png` - 4-panel visualization dashboard
- `Correlation_Heatmap_YYYYMMDD_HHMMSS.png` - Data correlation matrix
- `Traffic_Summary_YYYYMMDD_HHMMSS.txt` - Text-based statistics report

### 🌐 Web Dashboard (NEW!)

**Launch the interactive web interface for real-time monitoring and analysis:**

```bash
streamlit run src/web_dashboard.py
```

Or use the quick launcher:
```bash
bash start_dashboard.sh
```

**Web Dashboard Features:**

🏠 **Home Page**
- Real-time KPIs and metrics
- Lane utilization charts
- Recent activity overview

📊 **Analytics Page**
- Interactive traffic trends
- Correlation analysis
- Detailed statistics
- Historical reports

🔍 **Data Explorer**
- Filter and search traffic data
- Export custom datasets
- Interactive data tables

⚙️ **System Info**
- Project structure
- Model status
- Quick reference commands

**Access:** Dashboard opens automatically at `http://localhost:8501`

**Perfect for:**
- Live demonstrations
- Project presentations
- Real-time monitoring
- Interactive data exploration

---

## 🏗️ System Architecture

1. **Input Acquisition**: Video frames are captured from CCTV/Video feed.
2. **Preprocessing**: Frames are resized for the neural network.
3. **Object Detection**:
   - **Parallel Execution**: Frame is passed to both the Traffic Model and Ambulance Model.
4. **Multi-Object Tracking (ByteTrack)**:
   - Detection results are fed into ByteTrack via the `supervision` library.
   - Each vehicle receives a **persistent Track ID** that survives across frames.
   - Position history is stored per track for speed estimation and trail rendering.
5. **Heuristic Filtering**:
   - Confidence Threshold > 0.15 (for background vehicles).
   - Ambulance Aspect Ratio Check (< 2.0) to filter out buses.
6. **Decision Logic**:
   - **Case A (Ambulance)**: Trigger Override → Set Signal to GREEN.
   - **Case B (Normal)**: Count Vehicles → Calculate Time → Update Display.
7. **Output**: Render Bounding Boxes with Track IDs, Speed Labels, Movement Trails, Timer Overlay, and write extended CSV.

---

## 📊 Data Analytics

The system logs traffic patterns every 5 seconds. This data can be used to generate reports on **"Peak Traffic Hours."**

### Sample CSV Output:

| Timestamp | Lane1_Count | Lane2_Count | Lane1_Unique | Lane2_Unique | Avg_Speed_L1 | Avg_Speed_L2 | Ambulance_Detected | Green_Time_L1 | Green_Time_L2 |
|:----------|:------------|:------------|:-------------|:-------------|:-------------|:-------------|:-------------------|:--------------|:--------------|
| 10:45:05  | 12          | 4           | 28           | 15           | 45.2         | 38.7         | False              | 29            | 13            |
| 10:45:10  | 14          | 3           | 34           | 16           | 52.1         | 41.3         | False              | 33            | 11            |
| 10:45:15  | 8           | 0           | 38           | 16           | 30.5         | 0.0          | True               | 21            | 5             |

---

## � Configuration Management

NETRA now supports **YAML-based configuration** for easy deployment across different environments.

### Quick Start with Config

```bash
# Use default configuration
python3 main.py

# Use custom configuration file
python3 main.py --config config/my-deployment.yaml

# Override specific settings via CLI
python3 main.py --config config/default.yaml --device cuda --precision fp16 --imgsz 480
```

### Configuration File Structure

Edit `config/default.yaml` to customize:

```yaml
# Video input source
video:
  input_path: "videos/traffic.mp4"
  display_enabled: true

# Detection thresholds
detection:
  vehicle_confidence: 0.15        # Lower = more detections
  ambulance_confidence: 0.7       # Higher = fewer false positives
  vehicle_classes:
    - car
    - truck
    - bus

# Lane definitions (ROI boxes)
lanes:
  lane1:
    roi: [50, 100, 350, 500]      # [x_min, y_min, x_max, y_max]

# Dynamic signal timing
signal_timing:
  base_time: 5                    # Base green duration (seconds)
  multiplier: 2                   # Additional seconds per vehicle
  max_time: 60                    # Maximum green duration

# Data logging
logging:
  enabled: true
  output_dir: "data/traffic_logs"
  log_interval_seconds: 5
```

---

## 🚨 Troubleshooting

### Issue: "Model failed to load"
**Solution:** 
- Ensure model files exist: `ls models/yolov8m.pt models/best.pt`
- Download if missing: `python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"`
- Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`

### Issue: Low FPS / Slow Processing
**Solution:**
- Use lighter model: `--traffic-model models/yolov8n.pt`
- Reduce image size: `--imgsz 416` (from 640)
- Skip frames: `--skip-frames 2` (process every 3rd frame)
- Check device: `--device cuda` (if GPU available)
- Monitor CPU/GPU: Use `nvidia-smi` (GPU) or `top` (CPU)

### Issue: Video Won't Open
**Solution:**
- Check file path: `ls -l videos/traffic.mp4`
- Verify video format: `ffprobe videos/traffic.mp4`
- Convert if needed: `ffmpeg -i video.mkv -c:v libx264 -preset medium traffic.mp4`
- Use absolute path: `--video /absolute/path/to/video.mp4`

### Issue: "No module named 'supervision'"
**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or specific package
pip install supervision>=0.18.0
```

### Issue: CSV File Not Created
**Solution:**
- Check logging is enabled: `logging: enabled: true` in config
- Verify write permissions: `ls -ld data/traffic_logs/`
- Create directory if missing: `mkdir -p data/traffic_logs`
- Check disk space: `df -h`

### Issue: Incorrect Lane Counts
**Solution:**
- Verify ROI boxes in config: Draw boxes on sample frame to confirm
- Adjust confidence thresholds: Lower `vehicle_confidence` for more detections
- Use `src/utils/mouse_finder.py` to find correct ROI coordinates

### Issue: MacBook/Apple Silicon Issues
**Solution:**
```bash
# Use MPS device explicitly
python3 main.py --device mps

# Or fallback to CPU if unstable
python3 main.py --device cpu
```

### Issue: "CUDA out of memory"
**Solution:**
- Reduce batch size (if applicable)
- Use lighter model: `yolov8n.pt` instead of `yolov8m.pt`
- Reduce image size: `--imgsz 416`
- Skip frames: `--skip-frames 1`

### Issue: Ambulance Not Detected
**Solution:**
- Lower confidence threshold: `ambulance_confidence: 0.5` (in config)
- Verify model: `python src/utils/check_brain.py --model models/best.pt`
- Check video has ambulances in it
- Retrain if needed with proper labeled data

---

## ⚡ Performance Optimization Tips

### For Real-Time Performance (30+ FPS)

1. **Choose Right Model**
   ```bash
   # YOLOv8n (Nano) - Fastest, ~10M parameters
   python3 main.py --traffic-model models/yolov8n.pt --imgsz 416
   
   # YOLOv8m (Medium) - Balanced, ~25M parameters  
   python3 main.py --traffic-model models/yolov8m.pt --imgsz 640
   ```

2. **Use GPU Acceleration**
   ```bash
   # NVIDIA GPU (CUDA)
   python3 main.py --device cuda --precision fp16 --imgsz 640
   
   # Apple Silicon (MPS)
   python3 main.py --device mps --precision fp32 --imgsz 480
   ```

3. **Frame Skipping** (Process every Nth frame)
   ```bash
   # Process every 3rd frame (3x speedup, slight accuracy loss)
   python3 main.py --skip-frames 2
   ```

4. **Reduce Input Resolution**
   ```bash
   # Smaller input = faster processing
   python3 main.py --imgsz 480   # Default is 640
   python3 main.py --imgsz 416   # For edge devices
   ```

### For Best Accuracy

```bash
# Use larger model + higher resolution
python3 main.py --traffic-model models/yolov8m.pt --imgsz 640 --device cuda --precision fp16
```

### For Edge Devices (Jetson Nano, RPi, etc.)

```bash
# Export to optimized format
python3 src/export_model.py --model models/yolov8n.pt --format tflite --imgsz 416

# Or TensorRT for NVIDIA edge
python3 src/export_model.py --model models/yolov8n.pt --format engine --half --device cuda:0 --imgsz 480
```

### Benchmark Your Setup

Monitor performance with logging:
```bash
python3 main.py 2>&1 | grep -E "FPS|Time|Frames"
```

Check resource usage:
```bash
# Linux/Mac
watch -n1 'nvidia-smi'           # GPU (NVIDIA)
top -p $(pgrep -f "python main") # CPU

# macOS specific
vm_stat                           # Memory
power_log                         # Power usage
```

---

## 📋 Running Tests

Unit tests verify tracker accuracy:

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test
python3 -m pytest tests/test_tracker.py::TestTrackerSpeedEstimation -v

# Generate coverage report
python3 -m pytest tests/ --cov=src --cov-report=html
```

Tests cover:
- ✅ Speed estimation calculations
- ✅ Lane assignment logic
- ✅ Unique vehicle counting
- ✅ Movement trail history
- ✅ Error handling

---

## �🔮 Future Scope

- **Traffic Flow Prediction**: LSTM / Prophet time-series model to predict congestion 15–30 minutes ahead using collected CSV data.
- **License Plate Recognition (ANPR)**: Add OCR to detect number plates for red-light violation logging.
- **Night Vision**: Integrating CLAHE preprocessing or thermal imaging for low-light traffic detection.
- **IoT Integration**: Connecting the Python logic to Arduino/Raspberry Pi to control physical traffic lights.
- **Multi-Intersection Coordination**: Scale from 2-lane to full 4-way intersections with graph-based signal optimization.

---

## 📝 License

This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide to get started.

## 👨‍💻 Author

**Faiz Ahmad Khan**

---

**⭐ If you find this project useful, please consider giving it a star!**
