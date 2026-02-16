from ultralytics import YOLO
import os
import sys

# Add project root to path and get model path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(project_root, 'models', 'best.pt')

print("\n" + "="*50)
print("🧠 MODEL CLASS NAMES (best.pt)")
print("="*50)

# Load your custom ambulance detection model
model = YOLO(model_path)

# Print the list of classes it can detect
print(model.names)
print("="*50 + "\n")