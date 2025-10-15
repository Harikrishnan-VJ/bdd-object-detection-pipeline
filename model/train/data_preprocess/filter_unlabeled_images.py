import os
import sys
import shutil


images_folder = sys.argv[1]
labels_folder = sys.argv[2]
dest_folder = sys.argv[3]

os.makedirs(dest_folder, exist_ok=True)


for files in os.listdir(images_folder):
    if not files.endswith('.jpg'):
        continue
    
    label_file = files.replace('.jpg', '.txt')
    
    if not os.path.exists(os.path.join(labels_folder, label_file)):
        print(f"Removing image without label: {files}")
        image_path = os.path.join(images_folder, files)
        
        shutil.move(image_path, dest_folder)
    