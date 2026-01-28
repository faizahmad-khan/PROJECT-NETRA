import cv2
from ultralytics import YOLO
import math

cap = cv2.VideoCapture("videos/traffic.mp4")

# Load Models
model_traffic = YOLO('yolov8m.pt')
model_ambulance = YOLO('best.pt')

# --- CONFIGURATION (CRITICAL PART) ---
# You need to find the correct X coordinates for your video!

# LANE 1 (LEFT LANE) -> Defines the RED Box
# [x_min, y_min, x_max, y_max]
lane1_limits = [50, 100, 350, 500]   # <--- TWEAK THESE NUMBERS

# LANE 2 (RIGHT LANE) -> Defines the BLUE Box
# Notice I started x_min at 400 (leaving a 50px gap from Lane 1's 350)
lane2_limits = [400, 100, 700, 500]  # <--- TWEAK THESE NUMBERS

while True:
    success, img = cap.read()
    if not success: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count_lane1 = 0
    count_lane2 = 0
    ambulance_detected = False
    
    # --- TRAFFIC DETECTION ---
    results = model_traffic(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            currentClass = model_traffic.names[cls]
            conf = math.ceil((box.conf[0] * 100)) / 100

            if currentClass in ["car", "truck", "bus", "motorbike", "bicycle"] and conf > 0.15:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # CHECK LANE 1 (Left)
                if lane1_limits[0] < cx < lane1_limits[2] and lane1_limits[1] < cy < lane1_limits[3]:
                    count_lane1 += 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2) # Draw RED box
                
                # CHECK LANE 2 (Right)
                elif lane2_limits[0] < cx < lane2_limits[2] and lane2_limits[1] < cy < lane2_limits[3]:
                    count_lane2 += 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Draw BLUE box

    # --- AMBULANCE DETECTION ---
    results_amb = model_ambulance(img, stream=True)
    for r in results_amb:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if model_ambulance.names[cls] == 'Ambulance' and box.conf[0] > 0.7:
                 # Check Aspect Ratio and Area filters here (from previous step)
                 x1, y1, x2, y2 = box.xyxy[0]
                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                 w, h = x2 - x1, y2 - y1
                 if (w*h) > 3000 and (w/h) < 2.0:
                    ambulance_detected = True
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

    # --- DISPLAY BOXES (So you can fix the overlap!) ---
    # Lane 1 (Red Border)
    cv2.rectangle(img, (lane1_limits[0], lane1_limits[1]), (lane1_limits[2], lane1_limits[3]), (0, 0, 255), 2)
    # Lane 2 (Blue Border)
    cv2.rectangle(img, (lane2_limits[0], lane2_limits[1]), (lane2_limits[2], lane2_limits[3]), (255, 0, 0), 2)

    # --- DASHBOARD ---
    cv2.rectangle(img, (0, 0), (1280, 100), (0, 0, 0), -1)
    
    if ambulance_detected:
        cv2.putText(img, f'!!! EMERGENCY OVERRIDE !!!', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    else:
        # Display Separate Counts
        cv2.putText(img, f'LANE 1: {count_lane1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f'LANE 2: {count_lane2}', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Calculate Separate Times
        t1 = 5 + (count_lane1 * 2)
        t2 = 5 + (count_lane2 * 2)
        cv2.putText(img, f'Time: {t1}s', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f'Time: {t2}s', (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("NETRA Multi-Lane", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break