import glob
import os

def check_labels(labels_path):
    for label_file in glob.glob(os.path.join(labels_path, '*.txt')):
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                if class_id >= 12:
                    print(f"Invalid class {class_id} in {label_file}")

# Check both training and validation labels
train_path = "datasets/sh17/train/labels"
val_path = "datasets/sh17/val/labels"

print("Checking training labels...")
check_labels(train_path)
print("\nChecking validation labels...")
check_labels(val_path)
