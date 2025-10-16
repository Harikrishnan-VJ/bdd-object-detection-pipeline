import os
from ultralytics import YOLO

# Load the trained YOLOv11 model
model = YOLO("/home/user/hari/test/bdd-object-detection-pipeline/runs/detect/bdd100k_yolo11/weights/best.pt")  # Path to your trained weights

# Evaluate the model on the validation dataset
results = model.val(
    data="dataset/data.yaml",         # Path to data.yaml 
    imgsz=640,                        # Image size (must match training configuration)
    batch=32,                         # Batch size for evaluation
    conf=0.001,                       # Confidence threshold for detections
    iou=0.6,                          # IoU threshold for NMS
    project='./runs',               # Save runs folder in the script's directory
    name="bdd100k_yolo11_val",        # Custom name for the validation run folder
    split="val",                      # Specify the validation split
    save_json=True,                   # Save results to JSON for further analysis
    save_txt=True,                    # Save prediction results as text files
    plots=True                        # Generate plots (e.g., PR curves, confusion matrix)
)

# Print key evaluation metrics
print("Validation Results:")
print(f"mAP@0.5: {results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {results.box.map:.4f}")
print(f"Precision: {results.box.p[0]:.4f}")  # Average precision across classes
print(f"Recall: {results.box.r[0]:.4f}")     # Average recall across classes

# Save metrics to a text file
output_dir = os.path.join("runs", "val", "bdd100k_yolo11_val")
os.makedirs(output_dir, exist_ok=True)
metrics_file = os.path.join(output_dir, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write("Validation Metrics:\n")
    f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")
    f.write(f"Precision: {results.box.p[0]:.4f}\n")
    f.write(f"Recall: {results.box.r[0]:.4f}\n")
    if hasattr(results.box, "maps"):
        f.write("Per-class mAP@0.5:\n")
        for i, map_value in enumerate(results.box.maps):
            class_name = results.names.get(i, f"Class_{i}")
            f.write(f"  {class_name}: {map_value:.4f}\n")

print(f"Metrics saved to: {metrics_file}")
# Results are also saved in runs/val/bdd100k_yolo11_val/