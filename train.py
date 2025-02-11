from ultralytics import YOLO
from torch.nn.modules.conv import Conv2d
import torch.serialization

import os

# Add Conv2d to safe globals before loading the model
torch.serialization.add_safe_globals([Conv2d])

# Set environment variable to disable weights_only
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Then load your model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="sh17.yaml",  # Point to the YAML file
    epochs=10,
    imgsz=1280,
    patience=2,
    save=True
)
