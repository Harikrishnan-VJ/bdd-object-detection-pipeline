import os
import gdown
import zipfile
import shutil

# Define URLs and paths
url = "https://drive.google.com/uc?id=1feQiQetF8whnreYooFCQ1BygjZnlndvB"
output_zip = "bdd100k.zip"
extracted_folder = "bdd100k"
data_folder = "../data_analysis/data/"

# Create data folder if it doesn't exist
os.makedirs(data_folder, exist_ok=True)

# Download the dataset
print("Downloading dataset...")
gdown.download(url, output_zip, quiet=False)

# Unzip the downloaded file
print("Unzipping dataset...")
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall()

# Copy JSON files to data folder
print("Copying data to data folder...")
shutil.copytree(os.path.join(extracted_folder, 'images'), '../model/dataset/images', dirs_exist_ok=True)
shutil.copytree(os.path.join(extracted_folder, 'labels'), '../model/dataset/labels', dirs_exist_ok=True)
shutil.move(extracted_folder, data_folder)

# Clean up remove zip file and extracted folder
print("Cleaning up...")
os.remove(output_zip)
print("Done!")