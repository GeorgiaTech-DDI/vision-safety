import os
import shutil
import random

# Define paths
base_dir = "datasets/sh17"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
train_files_txt = os.path.join(base_dir, "train_files.txt")
val_files_txt = os.path.join(base_dir, "val_files.txt")

# Create directories if they don't exist
os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)

# Get all image files
all_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle and split files into train and validation sets (80/20)
random.shuffle(all_files)
split_idx = int(len(all_files) * 0.8)
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

# Write train and validation file lists
with open(train_files_txt, 'w') as f:
    for file in train_files:
        f.write(f"{file}\n")

with open(val_files_txt, 'w') as f:
    for file in val_files:
        f.write(f"{file}\n")

# Function to move files
def move_files(file_list, dest_images_dir, dest_labels_dir):
    for file_name in file_list:
        image_path = os.path.join(images_dir, file_name)
        label_path = os.path.join(labels_dir, file_name.replace('.jpg', '.txt'))

        if os.path.exists(image_path):
            shutil.copy2(image_path, dest_images_dir)
        if os.path.exists(label_path):
            shutil.copy2(label_path, dest_labels_dir)

# Move files
move_files(train_files, os.path.join(train_dir, "images"), os.path.join(train_dir, "labels"))
move_files(val_files, os.path.join(val_dir, "images"), os.path.join(val_dir, "labels"))
