"""
Training Data Extraction Script for YOLOv8
------------------------------------------
This script:
1. Loads `styles.csv` and ensures the dataset is not empty.
2. Extracts `articleType` (clothing category) and assigns class IDs.
3. Saves processed data to `typeName_data.csv`.
4. Creates YOLO `.txt` label files in `fashion-dataset/labels/`.
"""

import os
import pandas as pd

dataset_path = "fashion-dataset"
csv_file = os.path.join(dataset_path, "styles.csv")
labels_dir = os.path.join(dataset_path, "labels")

os.makedirs(labels_dir, exist_ok=True)

df = pd.read_csv(csv_file)

if df.empty:
    raise ValueError("🚨 ERROR: Dataset is empty! Check styles.csv filtering.")

required_columns = {"filename", "articleType", "id", "image_path", "class_id"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"🚨 ERROR: Missing required columns! Found: {df.columns}")

df = df.dropna(subset=["filename", "articleType", "id", "class_id"])

print("✅ Sample Data Before Processing:")
print(df.head())

output_csv = "typeName_data.csv"
df.to_csv(output_csv, index=False)
print(f"✅ Processed dataset saved to: {output_csv}")

for _, row in df.iterrows():
    label_filename = f"{row['id']}.txt"
    label_path = os.path.join(labels_dir, label_filename)

    with open(label_path, "w") as f:
        f.write(f"{int(row['class_id'])}\n")

print(f"✅ YOLO labels saved in: {labels_dir}/")
