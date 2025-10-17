# Instructions for Downloading Dataset and Inference Model

Instructions for using the scripts to download the dataset and inference model.

## Prerequisites
Prerequisites are already installed in setup

## Scripts Overview
The scripts/ folder contains two Python scripts:

- 1. download_dataset.py: Downloads the dataset from Google Drive and saves it to data_analysis/data/.
- 2. download_model.py: Downloads the inference model from Google Drive and saves it to model/.

## How to Use the Scripts
1. Navigate to the Scripts Folder

Open a terminal and navigate to the scripts/ folder in your repository:
```
cd scripts/
```


2. Run the Dataset Download Script

The download_dataset.py script downloads the dataset from a specified Google Drive link and saves it to the data_analysis/data/ folder.
Run the script:
```
python3 download_dataset.py
```

3. Run the Model Download Script

The download_inference_model.py script downloads the inference model from a specified Google Drive link and saves it to the model/ folder.
Run the script:
```
python3 download_model.py
```

4. Verify Downloads

After running the scripts, check the following:
Dataset files should be in data_analysis/data/.
Model files should be in model/.


If files are missing, check the terminal output for errors.
