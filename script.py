"""
Dataset Preparation Script for YOLOv8 (Ignoring Invalid Categories)
------------------------------------------------------------------
1. Uses only the first 701 rows from `styles.csv`.
2. Maps `articleType` to a unique class ID.
3. **Ignores any rows where `articleType` contains a URL or NaN.**
4. Splits dataset into `train/` (80%) and `val/` (20%).
5. Creates `class_mapping.txt` with correct category mappings.

Output:
- Moves images and labels to `train/` and `val/`, excluding invalid categories.
- Generates `.txt` label files with correct class IDs.
- Saves `class_mapping.txt` without invalid entries.
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_path = "/storage/ice-shared/vip-vx4/spring2025/vision-safety/fashion-dataset/fashion-dataset"
image_dir = os.path.join(dataset_path, "images")
label_dir = os.path.join(dataset_path, "labels")
csv_file = os.path.join(dataset_path, "styles.csv")

for split in ["train", "val"]:
    os.makedirs(os.path.join(image_dir, split), exist_ok=True)
    os.makedirs(os.path.join(label_dir, split), exist_ok=True)

# Load only first 701 rows from `styles.csv`
df = pd.read_csv(csv_file, nrows=701)

if "id" not in df.columns or "articleType" not in df.columns:
    raise ValueError("CSV file must contain 'id' and 'articleType' columns.")
# Convert articleType to string and remove NaN values
df["articleType"] = df["articleType"].astype(str).str.strip()
df = df[df["articleType"].notna()]  # Drop rows where articleType is NaN

# **Remove entries where articleType is missing or incorrectly assigned as "0"**
# df = df[df["articleType"] != "0"]

# Create a mapping of articleType to numeric class ID (starting at 1)
unique_classes = sorted(df["articleType"].dropna().unique())
class_mapping = {cls: idx + 1 for idx, cls in enumerate(unique_classes)}  # Start at 1
df["class_id"] = df["articleType"].map(class_mapping)

# **Print filenames where class_id ≠ 0**
df["filename"] = df["id"].astype(str) + ".jpg"
valid_images = df[df["class_id"] != 0][["filename", "class_id"]]
print("\n✅ Files with class_id ≠ 0:")
print(valid_images.to_string(index=False))  # Print without index

# Save corrected class mapping
class_mapping_path = os.path.join(dataset_path, "class_mapping.txt")
with open(class_mapping_path, "w") as f:
    for cls, idx in class_mapping.items():
        f.write(f"{idx}: {cls}\n")

print("Class mapping saved in class_mapping.txt (Invalid categories removed)")

for _, row in df.iterrows():
    label_file = os.path.join(label_dir, f"{row['id']}.txt")
    with open(label_file, "w") as f:
        f.write(f"{row['class_id']}\n")

all_images = df["filename"].tolist()
all_labels = [f"{img.replace('.jpg', '.txt')}" for img in all_images]

train_images, val_images, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42
)

def move_files(file_list, source_folder, destination_folder):
    for file in file_list:
        src = os.path.join(source_folder, file)
        dst = os.path.join(destination_folder, file)
        if os.path.exists(src):
            shutil.move(src, dst)

move_files(train_images, image_dir, os.path.join(image_dir, "train"))
move_files(val_images, image_dir, os.path.join(image_dir, "val"))
move_files(train_labels, label_dir, os.path.join(label_dir, "train"))
move_files(val_labels, label_dir, os.path.join(label_dir, "val"))

print("Dataset split into train/ and val/ successfully (Invalid categories removed)!")
