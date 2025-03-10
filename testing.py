"""
Real-Time Object Detection with YOLOv8
---------------------------------------
- Loads a trained YOLOv8 model (`best.pt` from training).
- Opens a video stream from a camera.
- Captures frames and runs the model's prediction.
- Displays the frame with detection results.
- Exits when the user presses the 'q' key.
"""

from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    results = model.predict(frame, conf=0.5, show=True)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
