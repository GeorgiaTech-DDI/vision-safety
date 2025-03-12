from ultralytics import YOLO
import os
import torch

def train_model():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
    # Get absolute path to current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    batch_size = 64

    model = YOLO("yolo11s.pt")  # from ultralytics
    # model = YOLO("runs/detect/train5/weights/last.pt")
    model.to("cuda")

    # Train the model with specified project directory
    results = model.train(
        data=os.path.join(current_dir, "old-datasets/goggles.yolov8/data.yaml"),
        epochs=150,
        imgsz=640,
        batch=batch_size,
        project=current_dir,
        name="runs/detect/train",
        patience=5,
        # resume=True,
        save=True,
        workers=1,
    )

if __name__ == "__main__":
    train_model()
