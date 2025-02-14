import os
import shutil
from sklearn.model_selection import train_test_split
import glob

def create_dataset_structure():
    # Create directories with correct path
    base_dir = 'datasets/sh17'
    dirs = [
        f'{base_dir}/train/images', 
        f'{base_dir}/train/labels',
        f'{base_dir}/val/images', 
        f'{base_dir}/val/labels'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def split_data(images_path, labels_path):
    # Get all image files
    image_files = glob.glob(os.path.join(images_path, '*.*'))
    
    # Get corresponding label files
    label_files = [os.path.join(labels_path, os.path.splitext(os.path.basename(img))[0] + '.txt')
                  for img in image_files]
    
    # Split with 80/20 ratio
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        image_files, label_files, test_size=0.2, random_state=42
    )
    
    # Copy files to respective directories
    for img, label in zip(train_imgs, train_labels):
        shutil.copy(img, 'datasets/sh17/train/images/')
        shutil.copy(label, 'datasets/sh17/train/labels/')
    
    for img, label in zip(val_imgs, val_labels):
        shutil.copy(img, 'datasets/sh17/val/images/')
        shutil.copy(label, 'datasets/sh17/val/labels/')

if __name__ == "__main__":
    create_dataset_structure()
    split_data('datasets/sh17/images', 'datasets/sh17/labels')
