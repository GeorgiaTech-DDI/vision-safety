from ultralytics import YOLO

# Load your trained detection model
model = YOLO('runs/detect/train/weights/best.pt')  # Note: 'detect' not 'classify'

# Validate the model
metrics = model.val()

# Test on a single image
results = model.predict("path_to_test_image.jpg", show=True)