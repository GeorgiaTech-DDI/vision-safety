# Vision-Based Safety

## Initial Setup
Download the [dataset](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection/data). 

## Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # For Unix/MacOS
                           # OR
.venv\Scripts\activate     # For Windows
```

## Install dependencies:

```bash
pip install "ultralytics<=8.3.40" opencv-python scikit-learn
```

## Project structure:

```text
vision-safety/
├── .venv/
├── datasets/
│   └── sh17/
├── script.py       # Dataset preparation script
├── train.py        # Model training script
├── verify_paths.py # Verify the paths within the datasets directory
└── yolo.py         # Real-time detection script
```

Structure the datasets directory as follows:

```text
datasets/
├── images      # from dataset
├── labels      # from dataset
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Data.yaml structure:

```yaml
path: path/to/datasets/sh17  # absolute path to dataset
train: train/images
val: val/images

nc: 17
names: ['Person', 
        'Head', 
        'Face', 
        'Glasses', 
        'Face-mask-medical', 
        'Face-guard', 
        'Ear', 
        'Earmuffs', 
        'Hands', 
        'Gloves', 
        'Foot', 
        'Shoes', 
        'Safety-vest', 
        'Tools', 
        'Helmet', 
        'Medical-suit', 
        'Safety-suit']
```

## Split the data into an 80:20 training and validation ratio:

```bash
python script.py
```

## Train the model:

```bash
python train.py
```

## Validation
To validate the model's performance:

```bash
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/train/weights/best.pt'); model.val()"
```

## Running the program:
To run real-time PPE detection using your webcam:

```bash
python yolo.py
```
Press 'q' to quit the application.

## Requirements
- Python 3.x
- ultralytics<=8.3.40
- opencv-python
- scikit-learn
- torch

## Classes Detected (17 Total)
- Person
- Head
- Face
- Glasses
- Face-mask-medical
- Face-guard
- Ear
- Earmuffs
- Hands
- Gloves
- Foot
- Shoes
- Safety-vest
- Tools
- Helmet
- Medical-suit
- Safety-suit