import os
import sys
import shutil


def filter_unlabeled_images(images_folder, labels_folder, dest_folder):
    """Remove images with no .txt label files

    Args:
        images_folder (str): Path to the image folder
        labels_folder (str): Path to the labels folder (YOLO format)
        dest_folder (str): Path to move the unlabeled images 
    """
    for files in os.listdir(images_folder):
        if not files.endswith('.jpg'):
            continue
        
        label_file = files.replace('.jpg', '.txt')
        
        if not os.path.exists(os.path.join(labels_folder, label_file)):
            print(f"Removing image without label: {files}")
            image_path = os.path.join(images_folder, files)
            
            shutil.move(image_path, dest_folder)
        
  
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python filter_unlabeled_images.py <images_folder> <labels_folder> <dest_folder>")
        sys.exit(1)

    images_folder = sys.argv[1]
    labels_folder = sys.argv[2]
    dest_folder = sys.argv[3]

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filter_unlabeled_images(images_folder, labels_folder, dest_folder)   