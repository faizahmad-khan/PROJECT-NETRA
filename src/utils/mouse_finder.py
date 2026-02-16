import cv2
import os
import sys

# Get project root and video path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
video_path = os.path.join(project_root, 'videos', 'traffic.mp4')

print("="*50)
print("🖱️  MOUSE COORDINATE FINDER")
print("="*50)
print("Click on the video to get coordinates")
print("Press 'q' to quit")
print("="*50 + "\n")

def mouse_points(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinate: ({x}, {y})")

cap = cv2.VideoCapture(video_path)

while True:
    success, img = cap.read()
    if not success: break
    
    cv2.imshow("Find Coordinates", img)
    cv2.setMouseCallback("Find Coordinates", mouse_points)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break