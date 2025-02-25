import os
import json
import pandas as pd

# Path to the folder containing JSON files
styles_path = "fashion-dataset/styles/"

# List to store extracted data
data = []

# Loop through all JSON files in the styles folder
for filename in os.listdir(styles_path):
    if filename.endswith(".json"):
        file_path = os.path.join(styles_path, filename)

        # Read JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                json_data = json.load(f)

                # Extract nested fields from articleType
                data_info = json_data["data"]
                image_id = data_info.get('id', None)  # Extract image id
                article_info = data_info.get('articleType', {})
                article_id = article_info.get('id', None)  # Extract article id
                type_name = article_info.get('typeName', None)  # Extract typeName

                if image_id and type_name and article_id:
                    data.append({"id": image_id, "typeName": type_name, "articleId": article_id})
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {filename}")

# Convert to DataFrame for easier handling
df = pd.DataFrame(data)

# Display sample extracted data
print(df.head())

# Save to CSV for training if needed
df.to_csv("typeName_data.csv", index=False)