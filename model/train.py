from ultralytics import YOLO

# Load a pretrained YOLOv11 model 
model = YOLO("yolo11n.pt")

# Train on your dataset
results = model.train(
    data="dataset/data.yaml",  # Path to data.yaml
    epochs=100,                # Number of epochs
    imgsz=640,                 # Image size
    batch=32,                  # Batch size
    name="bdd100k_yolo11",      # Custom name for the training run folder
    val=True,
    save_period=5
)

# After training, the best model is saved in runs/train/bdd100k_yolo11/weights/best.pt
# You can then use it for inference: model = YOLO("runs/train/bdd100k_yolo11/weights/best.pt")