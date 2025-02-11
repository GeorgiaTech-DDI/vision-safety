from ultralytics import YOLO
import cv2

# Load your trained model instead of the default one
model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Add confidence threshold and specific classes
    results = model.predict(
        frame,
        conf=0.5,  # Confidence threshold
        show=True
    )
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
