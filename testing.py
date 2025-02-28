"""
Real-Time Object Detection with YOLOv8
---------------------------------------
This script demonstrates how to perform real-time object detection using the Ultralytics YOLOv8 model and OpenCV.
It carries out the following tasks:
  - Loads a pre-trained YOLOv8 model from "yolov8n.pt".
  - Opens a video stream from a camera (default index is 1; change to 0 if needed).
  - Captures frames in a loop and runs the model's prediction on each frame.
  - Displays the frame with detection results overlaid.
  - Exits the loop when the user presses the 'q' key.
  - Releases the camera and closes all OpenCV windows upon exit.
"""

from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov8n.pt")   # Replace with the path to trained YOLO model

# Open the camera
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    results = model.predict(frame, conf=0.5, show=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()