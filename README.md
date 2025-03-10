
# Vision-Based Clothing Detection

This project utilizes **YOLOv8** for **real-time clothing detection** in images and videos. The model is trained on a **fashion dataset** to identify **16 different clothing categories**.

---

## **🚀 Initial Setup**
### **Download the Dataset**
Download the [Datasets](https://drive.google.com/file/d/1tUPgF2-K9HhkK0eiXupadHky3VEm0pgL/view?usp=sharing) and place it inside the `vision-safety` directory.

### **Create and Activate a Virtual Environment**
### only necessary if not already using a conda environment
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

### **Update dataset.yaml**
Update 'path', 'train', and 'val' with the correct local absolute path to those directories

---

## **📂 Project Structure**
```text
vision-safety/
├── .venv/                  # Virtual environment
├── datasets/               # Dataset directory
├── testing.py              # Model testing script
└── README.md               # Project documentation
```

---

## **🛠 Training the Model**
Train the YOLOv8 model using the dataset:

```bash
yolo task=detect mode=train model=yolov8n.pt data=datasets/dataset.yaml epochs=50 imgsz=640
```

This will:
- Train YOLOv8 on the **dataset**.
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

## **👕 Classes Detected (4 Total)**
The model is trained to recognize the following **clothing categories**:
- Short sleeve
- Long sleeve
- Jewelry
- Watch

---

## **👥 Contribution**
- Esther Park
- Preyas Joshi