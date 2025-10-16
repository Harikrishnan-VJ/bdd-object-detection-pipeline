import json
import os
import sys
from PIL import Image

# Define the class mapping based on BDD100K object detection classes
CLASSES = ['traffic light', 'traffic sign', 'car', 'person', 'bus', 'truck', 'rider', 'bike', 'motor', 'train']
CLASS_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}


def convert_bdd100k_to_yolo(labels_json_path, images_folder, output_folder):
    """
    Convert BDD100K JSON annotations to YOLO format.

    Args:
    - labels_json_path (str): Path to the bdd100k labels.
    - images_folder (str): Path to the images folder.
    - output_folder (str): Path to the output folder for YOLO .txt files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(labels_json_path, 'r') as f:
        annotations = json.load(f)

    # Assuming annotations is a list of dicts, each for an image
    for anno in annotations:
        image_name = anno["name"]
        image_path = os.path.join(images_folder, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Get image dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        labels = anno.get("labels", [])
        yolo_lines = []
        
        for label in labels:
            if "box2d" not in label:
                continue  # Skip non-bbox annotations like poly2d
            
            category = label["category"]
            if category not in CLASS_MAP:
                print(f"Warning: Unknown category '{category}' in {image_name}")
                sys.exit(1)
                continue
            
            class_id = CLASS_MAP[category]
            box = label["box2d"]
            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]
            
            # Calculate YOLO format
            center_x = (x1 + x2) / 2 / img_width
            center_y = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # Skip invalid boxes
            if width <= 0 or height <= 0:
                continue
            
            yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        if yolo_lines:
            # Create .txt file with same name as image but .txt extension
            txt_name = os.path.splitext(image_name)[0] + ".txt"
            txt_path = os.path.join(output_folder, txt_name)
            with open(txt_path, 'w') as f:
                f.write("\n".join(yolo_lines) + "\n")
            print(f"Converted: {image_name} -> {txt_name}")
        else:
            print(f"No valid boxes for: {image_name}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python bdd100k_to_yolo_format.py <labels_json_path> <images_folder> <output_folder>")
        sys.exit(1)
        
    json_labels_path = sys.argv[1] # Labels json file path
    images_folder = sys.argv[2] # Images folder path
    output_folder = sys.argv[3] # Output folder path

    convert_bdd100k_to_yolo(json_labels_path, images_folder, output_folder)