import os
import gdown
import zipfile
import shutil

# Define URLs and paths
url = "https://drive.google.com/uc?id=1s8ruVWqWUYz4K_1sj-OoaAr2tlTHyHmF"
output_zip = "model.zip"
extracted_folder = "model"
data_folder = "../model/"

# Create data folder if it doesn't exist
os.makedirs(data_folder, exist_ok=True)

# Download the dataset
print("Downloading inference model...")
gdown.download(url, output_zip, quiet=False)

# Unzip the downloaded file
print("Unzipping dataset...")
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall()

# Copy JSON files to data folder
print("Copying inference model to model folder...")
shutil.move(os.path.join(extracted_folder, 'best.pt'), data_folder)

# Clean up remove zip file and extracted folder
print("Cleaning up...")
os.remove(output_zip)
print("Done!")