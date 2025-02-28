"""
Real-Time Clothing Detection with YOLOv8
----------------------------------------
This script:
1. Loads a trained YOLOv8 model.
2. Captures frames from the webcam.
3. Detects clothing items in real-time.
4. Displays bounding boxes and labels on detected objects.
5. Allows the user to exit by pressing 'q'.
"""

import cv2
from ultralytics import YOLO

model_path = "runs/detect/train/weights/best.pt"  # Path to the trained model
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("🎥 Running real-time clothing detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    results = model.predict(frame, show=True, conf=0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🔴 Stopped clothing detection.")
