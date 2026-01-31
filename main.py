import cv2
from ultralytics import YOLO
import math
import csv
import time
from datetime import datetime

# --- 1. SETUP VIDEO & MODELS ---
cap = cv2.VideoCapture("videos/traffic.mp4") # Check your filename!

print("Loading Intelligent Models...")
model_traffic = YOLO('yolov8m.pt')  # The Generalist (Cars)
model_ambulance = YOLO('best.pt')   # The Specialist (Ambulance)
print("Models Loaded!")

# --- 2. SETUP DATA LOGGING ---
file_name = f"Traffic_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Write the Header Row (Column Names)
with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Lane1_Count", "Lane2_Count", "Ambulance_Detected", "Green_Time_L1", "Green_Time_L2"])
print(f"✅ Logging data to: {file_name}")

# --- 3. CONFIGURATION (LANE BOXES) ---
# [x_min, y_min, x_max, y_max]
lane1_limits = [50, 100, 350, 500]   # Left Lane (Red Box)
lane2_limits = [400, 100, 700, 500]  # Right Lane (Blue Box)

while True:
    success, img = cap.read()
    if not success: 
        # Loop video forever for demo
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count_lane1 = 0
    count_lane2 = 0
    ambulance_detected = False
    
    # --- 4. TRAFFIC DETECTION (Brain 1) ---
    results_traffic = model_traffic(img, stream=True)
    
    for r in results_traffic:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            currentClass = model_traffic.names[cls]
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Detect Car, Truck, Bus, Motorbike, Bicycle
            if currentClass in ["car", "truck", "bus", "motorbike", "bicycle"] and conf > 0.15:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # CHECK LANE 1 (Left)
                if lane1_limits[0] < cx < lane1_limits[2] and lane1_limits[1] < cy < lane1_limits[3]:
                    count_lane1 += 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red
                
                # CHECK LANE 2 (Right)
                elif lane2_limits[0] < cx < lane2_limits[2] and lane2_limits[1] < cy < lane2_limits[3]:
                    count_lane2 += 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue

    # --- 5. AMBULANCE DETECTION (Brain 2) ---
    results_amb = model_ambulance(img, stream=True)
    
    for r in results_amb:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            # Check if your model uses 'Ambulance' or 'ambulance'
            # Also check your confidence threshold (adjust 0.7 as needed)
            if model_ambulance.names[cls] == 'Ambulance' and box.conf[0] > 0.7:
                 x1, y1, x2, y2 = box.xyxy[0]
                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                 
                 # Aspect Ratio & Size Filter to stop false positives
                 w, h = x2 - x1, y2 - y1
                 if (w*h) > 3000 and (w/h) < 2.0:
                    ambulance_detected = True
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.putText(img, "AMBULANCE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- 6. CALCULATE TIMERS ---
    t1 = 5 + (count_lane1 * 2)
    t2 = 5 + (count_lane2 * 2)
    if t1 > 60: t1 = 60
    if t2 > 60: t2 = 60

    # --- 7. DISPLAY LOGIC ---
    # Draw Lane Boundaries
    cv2.rectangle(img, (lane1_limits[0], lane1_limits[1]), (lane1_limits[2], lane1_limits[3]), (0, 0, 255), 2)
    cv2.rectangle(img, (lane2_limits[0], lane2_limits[1]), (lane2_limits[2], lane2_limits[3]), (255, 0, 0), 2)

    # Dashboard Background (Height 110 to fit both lines comfortably)
    h, w, c = img.shape
    cv2.rectangle(img, (0, 0), (w, 110), (0, 0, 0), -1)
    
    if ambulance_detected:
        cv2.putText(img, f'!!! EMERGENCY OVERRIDE !!!', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    else:
        # LANE 1 STATS (Top Line - RED)
        cv2.putText(img, f'LANE 1: {count_lane1} | Time: {t1}s', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # LANE 2 STATS (Bottom Line - BLUE)
        cv2.putText(img, f'LANE 2: {count_lane2} | Time: {t2}s', (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # --- 8. DATA LOGGING (New Feature) ---
    # We log only once every 5 seconds to avoid flooding the CSV file
    if int(time.time()) % 5 == 0: 
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Open in APPEND mode ('a')
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, count_lane1, count_lane2, ambulance_detected, t1, t2])
        
        print(f"Logged: {timestamp} | L1: {count_lane1} | L2: {count_lane2}")

    cv2.imshow("NETRA Final System", img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()