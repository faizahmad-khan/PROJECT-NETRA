import cv2

def mouse_points(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinate: {x}, {y}")

cap = cv2.VideoCapture("videos/traffic.mp4") # Your video

while True:
    success, img = cap.read()
    if not success: break
    
    cv2.imshow("Find Coordinates", img)
    cv2.setMouseCallback("Find Coordinates", mouse_points)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break