import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Define dataset paths
dataset_path = "datasets"
label_dir = os.path.join(dataset_path, "labels")
train_label_dir = os.path.join(label_dir, "train")
val_label_dir = os.path.join(label_dir, "val")
csv_file = os.path.join(dataset_path, "updated_styles.csv")

# Ensure label directories exist
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# **🚨 Step 1: Delete old label files**
for folder in [train_label_dir, val_label_dir]:
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# print("🗑️ Deleted old label files in train/ and val/.")

df = pd.read_csv(csv_file, nrows=701)  # Load first 701 rows

# Ensure required columns exist
required_columns = {"filename", "articleType", "id", "class_id", "x_center", "y_center", "width", "height"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"🚨 ERROR: Missing required columns! Found: {df.columns}")

# Remove NaN values
df = df.dropna(subset=["filename", "articleType", "id", "class_id", "x_center", "y_center", "width", "height"])

valid_class_ids = set(range(15))
df = df[df["class_id"].isin(valid_class_ids)]

class_mapping_path = os.path.join(dataset_path, "class_mapping.txt")
class_mapping = df[["class_id", "articleType"]].drop_duplicates().set_index("class_id").to_dict()["articleType"]

with open(class_mapping_path, "w") as f:
    for cls_id, cls_name in class_mapping.items():
        f.write(f"{cls_id}: {cls_name}\n")

# print("✅ Class mapping saved in class_mapping.txt")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

def save_labels(df_split, label_folder):
    for _, row in df_split.iterrows():
        label_filename = f"{row['id']}.txt"
        label_path = os.path.join(label_folder, label_filename)

        label_content = f"{int(row['class_id'])} {row['x_center']} {row['y_center']} {row['width']} {row['height']}\n"

        with open(label_path, "w") as f:
            f.write(label_content)

save_labels(train_df, train_label_dir)
save_labels(val_df, val_label_dir)

# print("✅ YOLO labels generated successfully in train/ and val/")
