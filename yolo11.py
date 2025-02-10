from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolo11n.pt")  # Replace with the path to your YOLO model if needed

# Open the camera
cap = cv2.VideoCapture(0)  # 0 is the default camera index

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLO model on the frame
    results = model.predict(frame, show=True)  # show=True displays the frame with predictions

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
