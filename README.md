# 🚦 Project NETRA (Network Enabled Traffic Regulation & Analysis)

**An Intelligent Traffic Management System using Computer Vision & Deep Learning**

Project NETRA addresses the critical issue of urban traffic congestion and delayed emergency services. Unlike traditional fixed-timer traffic lights, NETRA uses real-time camera feeds to calculate traffic density and adjust signal timings dynamically. Crucially, it features an **Automatic Ambulance Detection System** that overrides signals to provide a "Green Corridor" for emergency vehicles.

---

## 📸 Demo & Screenshots

![NETRA System Demo](screenshots/demo.png)

*Above: The NETRA system in action - Real-time detection of multiple vehicles across two lanes. Lane 1 (Red bounding boxes) shows 11 vehicles with 27s green time, while Lane 2 (Blue bounding boxes) shows 14 vehicles with 33s green time. The system dynamically calculates signal timings based on traffic density.*

---

## ✨ Key Features

🧠 **Hybrid AI Architecture**: Utilizes a dual-model strategy:
- **Generalist Model (yolov8m)**: Detects standard vehicles (Cars, Trucks, Buses, Rickshaws) with high accuracy.
- **Specialist Model (Custom Trained)**: A dedicated model trained via Transfer Learning to specifically detect Ambulances.

⏱️ **Dynamic Signal Timer**: Replaces static timers with an adaptive algorithm ($T = 5 + 2n$) that allocates green light duration based on real-time lane density.

🚑 **Emergency Override Module**: Instantly detects approaching ambulances (with geometric & confidence filtering) to clear the lane immediately.

🛣️ **Multi-Lane Logic**: Supports distinct ROI (Region of Interest) definitions to manage multiple lanes independently.

📊 **Traffic Analytics**: Automatically logs traffic data (vehicle counts, timestamps, signal times) to a CSV database for urban planning analysis.

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Computer Vision**: OpenCV (cv2)
- **Deep Learning**: YOLOv8 (Ultralytics)
- **Data Processing**: NumPy, Pandas (for analytics)
- **Training Environment**: Google Colab (Tesla T4 GPU)

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Project-NETRA.git
cd Project-NETRA
```

### Install Dependencies

```bash
pip install ultralytics opencv-python
```

### Setup Models

The system will automatically download `yolov8m.pt`.

**Important**: Ensure your custom trained model `best.pt` is placed in the root directory.

### Add Video Source

Place your test video in a folder named `videos/` and rename it to `traffic.mp4` (or update the path in [main.py](main.py)).

---

## 🚀 Usage

Run the main application:

```bash
python main.py
```

### 🎮 Controls

- **q**: Quit the application.

The system will automatically generate a CSV file (e.g., `Traffic_Data_20260131.csv`) in the project folder.

---

## 🏗️ System Architecture

1. **Input Acquisition**: Video frames are captured from CCTV/Video feed.
2. **Preprocessing**: Frames are resized for the neural network.
3. **Object Detection**:
   - **Parallel Execution**: Frame is passed to both the Traffic Model and Ambulance Model.
4. **Heuristic Filtering**:
   - Confidence Threshold > 0.15 (for background vehicles).
   - Ambulance Aspect Ratio Check (< 2.0) to filter out buses.
5. **Decision Logic**:
   - **Case A (Ambulance)**: Trigger Override → Set Signal to GREEN.
   - **Case B (Normal)**: Count Vehicles → Calculate Time → Update Display.
6. **Output**: Render Bounding Boxes, Timer Overlay, and write to CSV.

---

## 📊 Data Analytics

The system logs traffic patterns every 5 seconds. This data can be used to generate reports on **"Peak Traffic Hours."**

### Sample CSV Output:

| Timestamp | Lane1_Count | Lane2_Count | Ambulance_Detected | Green_Time_L1 |
|:----------|:------------|:------------|:-------------------|:--------------|
| 10:45:05  | 12          | 4           | False              | 29s           |
| 10:45:10  | 14          | 3           | False              | 33s           |
| 10:45:15  | 8           | 0           | True               | OVERRIDE      |

---

## 🔮 Future Scope

- **Night Vision**: Integrating thermal imaging for low-light traffic detection.
- **IoT Integration**: Connecting the Python logic to Arduino/Raspberry Pi to control physical traffic lights.
- **Cloud Dashboard**: Sending CSV data to a web dashboard for city-wide monitoring.

---

## 📝 License

This project is open-source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👨‍💻 Author

**Faiz Ahmad Khan**

---

**⭐ If you find this project useful, please consider giving it a star!**
