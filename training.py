import pandas as pd
import os
from sklearn.model_selection import train_test_split
from PIL import Image

# Path to dataset
dataset_path = "fashion-dataset"
images_folder = os.path.join(dataset_path, "images")
csv_file = os.path.join(dataset_path, "styles.csv")

# Load styles.csv
styles_df = pd.read_csv(csv_file, on_bad_lines='skip')

# Extract only the columns we care about: 'id' and 'articleType'
styles_df = styles_df[['id', 'articleType']]

# Assuming the 'id' column in styles.csv corresponds with image filenames (without extensions)
styles_df['image_path'] = styles_df['id'].apply(lambda x: f'fashion-dataset/images/{x}.jpg')  # Adjust extension if needed

# Check for missing images (optional)
# missing_images = styles_df[~styles_df['image_path'].apply(os.path.exists)]

# If any images are missing, you can choose to exclude them or log them
styles_df = styles_df[styles_df['image_path'].apply(os.path.exists)]

# Create a mapping of articleType to numeric class ID
article_types = styles_df["articleType"].unique()
article_type_to_id = {atype: idx for idx, atype in enumerate(article_types)}
styles_df["class_id"] = styles_df["articleType"].map(article_type_to_id)

# Split dataset into training and validation sets
train_df, val_df = train_test_split(styles_df, test_size=0.2, random_state=42)

# Create YOLOv8-style labels
labels_dir = os.path.join(dataset_path, "labels")
os.makedirs(labels_dir, exist_ok=True)

# Preview the dataframe
print(train_df.columns)
print(train_df.head())

# Check this logic 
for dataset, df in [("train", train_df), ("val", val_df)]:
    dataset_dir = os.path.join(dataset_path, "images", dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    for _, row in df.iterrows():
        image_filename = os.path.basename(row["image_path"])
        label_filename = image_filename.replace(".jpg", ".txt")
        label_path = os.path.join(labels_dir, dataset, label_filename)

        # YOLO format: class_id x_center y_center width height (classification: only class_id)
        with open(label_path, "w") as f:
            f.write(f"{row['class_id']}\n")

        # Move images to correct train/val directory
        os.rename(row["image_path"], os.path.join(dataset_dir, image_filename))

# Save class mappings for reference
with open(os.path.join(dataset_path, "class_mapping.txt"), "w") as f:
    for k, v in article_type_to_id.items():
        f.write(f"{v}: {k}\n")

print("Dataset processing complete.")