import cv2
from ultralytics import YOLO
import math

# --- 1. SETUP VIDEO ---
cap = cv2.VideoCapture("videos/traffic.mp4") # Check your filename!

# --- 2. LOAD BOTH BRAINS ---
print("Loading Models...")
model_traffic = YOLO('yolov8n.pt')  # The Generalist (Cars)
model_ambulance = YOLO('best.pt')   # The Specialist (Ambulance)
print("Models Loaded!")

# --- CONFIGURATION ---
# Coordinates for your Traffic Lane (Blue Box)
# Update these numbers to match your video!
limits = [100, 100, 500, 500] 

while True:
    success, img = cap.read()
    if not success: 
        # Optional: Loop video forever for demo
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    car_count = 0
    ambulance_detected = False
    
    # --- 3. RUN BRAIN 1: TRAFFIC (Cars/Buses/Trucks) ---
    results_traffic = model_traffic(img, stream=True)
    
    for r in results_traffic:
        boxes = r.boxes
        for box in boxes:
            # Check only for Car, Truck, Bus, Motorbike
            cls = int(box.cls[0])
            currentClass = model_traffic.names[cls]
            conf = math.ceil((box.conf[0] * 100)) / 100

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Check if inside the Lane Zone
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if limits[0] < cx < limits[2] and limits[1] < cy < limits[3]:
                    car_count += 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # (Optional: Don't draw red boxes to keep screen clean)

    # --- 4. RUN BRAIN 2: AMBULANCE OVERRIDE ---
    results_ambulance = model_ambulance(img, stream=True)
    
    for r in results_ambulance:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            currentClass = model_ambulance.names[cls]
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Calculate the SIZE of the object
            # (Width * Height)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            area = w * h

            # --- SMART FILTER ---
            # Rule 1: Confidence must be high (> 0.6 or 60%)
            # Rule 2: Object must be big enough (Area > 5000 pixels) to avoid background noise
            if currentClass == 'Ambulance' and conf > 0.6 and area > 5000:
                ambulance_detected = True
                
                # Draw the Emergency Box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(img, f"AMBULANCE {int(conf*100)}%", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- 5. LOGIC & DISPLAY ---
    # Draw the Lane Zone (Blue Box)
    cv2.rectangle(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 2)

    # Dashboard Background
    cv2.rectangle(img, (0, 0), (1280, 100), (0, 0, 0), -1)

    if ambulance_detected:
        # EMERGENCY UI
        cv2.rectangle(img, (0, 0), (1280, 100), (0, 0, 255), -1) # Red Top Bar
        cv2.putText(img, f'!!! EMERGENCY OVERRIDE !!!', (50, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(img, f'Status: CLEARING TRAFFIC', (50, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        # NORMAL TRAFFIC UI
        green_time = 5 + (car_count * 2)
        if green_time > 60: green_time = 60
        
        cv2.putText(img, f'Traffic Density: {car_count} Vehicles', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Color code the time (Green if low, Red if high wait)
        color = (0, 255, 0)
        if green_time > 30: color = (0, 165, 255) # Orange
        
        cv2.putText(img, f'Green Light Time: {green_time} seconds', (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Project NETRA: Final System", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()