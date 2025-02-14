from ultralytics import YOLO
import os

def train_model():
    # Get absolute path to current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load a model
    model = YOLO('yolov8n.pt')
    
    # Train the model with specified project directory
    results = model.train(
        data=os.path.join(current_dir, 'datasets/sh17/data.yaml'),
        epochs=10,
        imgsz=640,
        batch=16,
        project=current_dir,  # Set project directory to current directory
        name='runs/detect/train',  # Subfolder structure
        patience=2,
        save=True
    )

if __name__ == "__main__":
    train_model()
