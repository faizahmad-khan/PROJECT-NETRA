from ultralytics import YOLO

# Load your new custom model
model = YOLO('best.pt')

# Print the list of names it knows
print(model.names)