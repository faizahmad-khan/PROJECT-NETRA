import cv2
from ultralytics import YOLO
import math

cap = cv2.VideoCapture("videos/traffic.mp4")
model = YOLO('yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# --- CONFIGURATION ---
# The Zone: [x1, y1, x2, y2]
# CHANGE THESE NUMBERS to match your specific video lane!
limits = [100, 100, 500, 500] 

while True:
    success, img = cap.read()
    if not success: break
    
    # 1. Reset count every frame (we are calculating DENSITY, not total passed)
    car_count = 0
    
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            conf = math.ceil((box.conf[0] * 100)) / 100

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                
                # Calculate Center Point of the Car
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # CHECK: Is the center inside our "Lane" box?
                if limits[0] < cx < limits[2] and limits[1] < cy < limits[3]:
                    car_count += 1
                    # Draw Green Box for counted cars
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    # Draw Red Box for ignored cars
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # --- LOGIC BLOCK ---
    # Draw the Lane Zone
    cv2.rectangle(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 2)

    # Calculate Green Light Time
    # Formula: Minimum 5 sec + (2 sec per car)
    green_time = 5 + (car_count * 2)
    # Cap the max time at 60 seconds
    if green_time > 60: green_time = 60

    # --- DISPLAY BLOCK ---
    # Create a nice dashboard on the top left
    cv2.rectangle(img, (0, 0), (400, 100), (0, 0, 0), -1) # Background
    
    cv2.putText(img, f'Vehicles: {car_count}', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
    cv2.putText(img, f'Signal Time: {green_time}s', (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Project NETRA - Phase 2", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()