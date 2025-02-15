from ultralytics import YOLO
import os

def verify_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, 'datasets', 'sh17')
    
    paths = {
        'train_images': os.path.join(base_dir, 'train', 'images'),
        'train_labels': os.path.join(base_dir, 'train', 'labels'),
        'val_images': os.path.join(base_dir, 'val', 'images'),
        'val_labels': os.path.join(base_dir, 'val', 'labels'),
        'data_yaml': os.path.join(base_dir, 'data.yaml')
    }
    
    for name, path in paths.items():
        exists = os.path.exists(path)
        print(f"{name}: {path} {'passed.' if exists else 'failed.'}")
        if os.path.isdir(path):
            print(f"  Files: {len(os.listdir(path))}")

if __name__ == "__main__":
    verify_paths()
    train_model()
