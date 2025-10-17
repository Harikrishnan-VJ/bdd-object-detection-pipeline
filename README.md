# BDD Object Detection Pipeline
This project implements a pipeline for object detection using the BDD100K dataset. It includes data preprocessing, data analysis, model training, evaluation, and visualization of results.


## Table of Contents

- Project Overview
- Setup
- Downloading Dataset and Inference Model
- Step 1: Data Preprocessing
- Step 2: Model Training
- Step 3: Model Evaluation and Visualization

## Project Overview
The pipeline provides tools for:

Data Preprocessing: Preparing the BDD100K dataset for training (e.g., converting to YOLO format, removing images without labels).
Data Analysis: Exploring the dataset with visualizations and reports.
Model Training and Evaluation: Training an object detection model and evaluating its performance.
Visualization: Generating plots for metrics like accuracy, mAP, and confusion matrices and validation results etc..

## Setup
To set up the Python virtual environment and install dependencies, use the provided install.sh script. For detailed instructions, refer to [Setup Instructions](docs/setup.md).

Activate the virtual environment as described in setup.md.

## Downloading Dataset and Inference Model
To download the BDD100K dataset and the pre-trained inference model, use the scripts in the scripts/ folder. For detailed instructions, refer to Dataset and [Model Download Instructions](docs/dataset_and_model_download_instructions.md).


## Step 1: Data Preprocessing
Before training the model, preprocess the BDD100K dataset using scripts in the model/data_preprocess/ folder. These scripts handle tasks such as:

Converting BDD100K data to YOLO format.
Removing images without labels.
Other necessary preprocessing steps.

### Instructions

Ensure the dataset is downloaded to data_analysis/data/ (see Downloading Dataset and Inference Model).
Navigate to the model/data_preprocess/ folder:
```
cd model/data_preprocess/
```

Run the preprocessing scripts (refer to individual script documentation or comments for specific usage).


Verify the output (e.g., YOLO-formatted files) in the appropriate directory, typically within model/data_preprocess/ or a specified output folder.

## Step 2: Data Analysis
This step generates a data analysis report and visualizations for the BDD100K dataset using a Docker container.

Navigate to the data_analysis/ folder:
```
cd data_analysis/
```

Make the script executable and run it:
```
chmod +x install.sh
./install.sh
```

The script builds and runs a Docker container with default variables specified in install.sh.

### Outputs:
- Plots: Saved in output/plots/, including:
    - background_images: Distribution of background images.
    - class_distribution: Per-class distribution chart.
    - empty_images: Distribution of empty images.
    - occluded_objects_per_class: Number of occluded images per class.
    - scene_distribution: Distribution of scenes (e.g., highway, street, residential).
    - time_of_day_distribution: Distribution by time of day (e.g., daytime, nighttime).
    - weather_distribution: Distribution by weather conditions.


- Report: A detailed analysis report saved as output/analysis_report.md, highlighting anomalies and insights.

For further details, refer to analysis_report.md.


## Step 3: Model Training
After preprocessing, train the object detection model using the train.py script.

Ensure the virtual environment is activated (see setup.md).
Ensure the preprocessed data is available (from Step 1: Data Preprocessing).
Navigate to the model/ folder:
```
cd model/
```

Run the training script:
```
python3 train.py
```

Outputs: Trained model weights and logs are saved in the model/ folder or a subdirectory (e.g., model/runs/), depending on the script configuration.

## Step 4: Model Evaluation and Visualization
This step evaluates the trained model and generates visualizations of performance metrics.

Ensure the virtual environment is activated (see setup.md).
Navigate to the model/ folder:
```
cd model/
```

Run the evaluation script:
```
python3 eval.py
```

### Outputs:
Evaluation results and visualizations (e.g., accuracy, mAP, confusion matrix) are saved in the model/runs/ folder.

For model evaluation and visualization details, see [Evaluation and Visualization](docs/evaluation_and_visualization.md).




### Additional Notes
If you encounter issues, check the respective .md files or script comments for troubleshooting tips.
