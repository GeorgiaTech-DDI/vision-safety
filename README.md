
# Vision-Based Clothing Detection

This project utilizes **YOLOv8** for **real-time clothing detection** in images and videos. The model is trained on a **fashion dataset** to identify **16 different clothing categories**.

---

## **🚀 Initial Setup**
### **Download the Dataset**
Download the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) and place it inside the `fashion-dataset/` directory.

### **Create and Activate a Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # For Unix/MacOS
                           # OR
.venv\Scripts\activate     # For Windows
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **📂 Project Structure**
```text
vision-safety/
├── .venv/                  # Virtual environment
├── fashion-dataset/        # Dataset directory
├── script.py               # Dataset preprocessing script
├── train.py                # Model training script
├── testing.py              # Model testing script
└── README.md               # Project documentation
```

---

## **📌 Preparing the Dataset**
Run the dataset preparation script to **organize images and labels** into YOLO format:

```bash
python script.py
```

After execution, the `fashion-dataset/` directory should be structured as follows:

```text
fashion-dataset/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
├── labels/
│   ├── train/          # YOLO labels for training images
│   ├── val/            # YOLO labels for validation images
├── class_mapping.txt   # Maps class IDs to clothing categories
├── styles.csv          # Raw dataset metadata
└── data.yaml           # YOLO dataset configuration
```

---

## **📜 data.yaml Structure**
The `data.yaml` file defines the dataset structure for YOLOv8 training:

```yaml
ath: path/to/datasets
train: path/to/datasets/images/train
val: path/to/datastes/images/val

nc: 15
names: ["Short sleeve", "Long sleeve", "Open toe", "Closed toe", "Bag", "Hat", "Socks", "Jewelry", "Watch", "Jacket", "Dress", "Shorts", "Pants", "Scarf", "Glasses"]
```

---

## **🛠 Training the Model**
Train the YOLOv8 model using the dataset:

```bash
yolo task=detect mode=train model=yolov8n.pt data=datasets/data.yaml epochs=50 imgsz=640
```

This will:
- Train YOLOv8 on the **fashion dataset**.
- Save the **trained model** inside the `runs/detect/train/weights/` directory.

---

## **✅ Validating the Model**
To evaluate the model’s performance after training:

```bash
python train.py
```

This runs **model validation** on the test set to check accuracy.

---

## **🎥 Running Real-Time Clothing Detection**
To detect clothing **in real-time using your webcam**, run:

```bash
python yolo.py
```
- **Press 'q' to quit the application.**
- The model will **identify and label clothing** in live camera feed.

---

## **👕 Classes Detected (16 Total)**
The model is trained to recognize the following **clothing categories**:
- Short sleeve
- Long sleeve
- Open toe
- Closed toe
- Bag
- Hat
- Socks
- Jewelry
- Watch
- Jacket
- Hoodie
- Dress
- Shorts
- Pants
- Scarf
- Glasses

---

## **👥 Contribution**
- Esther Park
- Preyas Joshi