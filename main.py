import cv2
from ultralytics import YOLO
import math

# --- 1. SETUP VIDEO ---
cap = cv2.VideoCapture("videos/traffic.mp4")

# --- 2. LOAD BOTH BRAINS ---
print("Loading Intelligent Models...")
# CHANGED: Switched to 'yolov8m.pt' (Medium) for better detection
model_traffic = YOLO('yolov8m.pt')  
model_ambulance = YOLO('best.pt')   
print("Models Loaded!")

# --- CONFIGURATION ---
limits = [100, 100, 500, 500] 

while True:
    success, img = cap.read()
    if not success: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    car_count = 0
    ambulance_detected = False
    
    # --- 3. RUN TRAFFIC BRAIN (High Accuracy) ---
    results_traffic = model_traffic(img, stream=True)
    
    for r in results_traffic:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            currentClass = model_traffic.names[cls]
            conf = math.ceil((box.conf[0] * 100)) / 100

            # CHANGED: 
            # 1. Added "bicycle" (Auto-rickshaws often get detected as bikes or cars)
            # 2. Lowered confidence from 0.3 to 0.15 (Catches background cars)
            if currentClass in ["car", "truck", "bus", "motorbike", "bicycle"] and conf > 0.15:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if limits[0] < cx < limits[2] and limits[1] < cy < limits[3]:
                    car_count += 1
                    # Draw thinner boxes so we can see better
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- 4. RUN AMBULANCE BRAIN ---
    results_ambulance = model_ambulance(img, stream=True)
    for r in results_ambulance:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            currentClass = model_ambulance.names[cls]
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            area = w * h

            # Smart Filter (Size + Confidence)
            if currentClass == 'Ambulance' and conf > 0.6 and area > 3000:
                ambulance_detected = True
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(img, f"AMBULANCE {int(conf*100)}%", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- 5. LOGIC & DISPLAY ---
    cv2.rectangle(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 2)
    cv2.rectangle(img, (0, 0), (1280, 100), (0, 0, 0), -1)

    if ambulance_detected:
        cv2.rectangle(img, (0, 0), (1280, 100), (0, 0, 255), -1) 
        cv2.putText(img, f'!!! EMERGENCY OVERRIDE !!!', (50, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    else:
        # LOGIC EXPLANATION:
        # More Cars = More Time needed to clear them.
        # 1 Car = 2 seconds of green light.
        # 10 Cars = 20 seconds of green light.
        green_time = 5 + (car_count * 2) 
        if green_time > 60: green_time = 60 # Cap at 60s
        
        cv2.putText(img, f'Traffic Density: {car_count} Vehicles', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Dynamic Color for the Timer
        # Green = Short wait, Red = Long wait
        color = (0, 255, 0)
        if green_time > 20: color = (0, 255, 255) # Yellow
        if green_time > 40: color = (0, 0, 255)   # Red
        
        cv2.putText(img, f'Green Signal Time: {green_time}s', (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Project NETRA: Final System", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()