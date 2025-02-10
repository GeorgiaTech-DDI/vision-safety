from ultralytics import YOLO

# Load a detection model (using YOLOv8 nano for Raspberry Pi compatibility)
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="sh17.yaml",
    epochs=10,
    imgsz=640,
    patience=2
)
