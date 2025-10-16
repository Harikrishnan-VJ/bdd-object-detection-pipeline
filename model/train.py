import os
from ultralytics import YOLO


# Load a pretrained YOLOv11 model 
model = YOLO("/home/user/hari/test/bdd-object-detection-pipeline/runs/detect/bdd100k_yolo11/weights/best.pt")

# Train on your dataset
results = model.train(
    data="dataset/data.yaml",  # Path to data.yaml
    epochs=100,                # Number of epochs
    imgsz=640,                 # Image size
    batch=32,                  # Batch size
    name="bdd100k_yolo11",      # Custom name for the training run folder
    project='./runs',
    val=True,
    save_period=5
)