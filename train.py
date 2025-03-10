from ultralytics import YOLO

'''
Validate YOLOv8 Model
----------------------
This script performs validation of a pre-trained YOLOv8 model on a specified validation dataset.

The process includes:
1. Loading a pre-trained YOLOv8 model.
2. Running validation on the validation dataset defined in the `dataset.yaml` file.
3. Printing out the evaluation metrics, including mean Average Precision (mAP), loss, etc.

Parameters:
- `dataset_path`: Path to the dataset configuration file (e.g., `dataset.yaml`), which contains the paths to the images and labels.
- `imgsz`: The image size for validation (set to 640 pixels).
- `batch`: The batch size for validation (set to 16).

The script prints out the validation results after running the validation process on the model.
'''

dataset_path = "datasets/dataset.yaml"

model = YOLO("best.pt")

results = model.val(
    data=dataset_path, 
    imgsz=640, 
    batch=16,
)

print(f"Validation Results: {results}")
