# Evaluation and Visualization

This document details the evaluation of the object detection model on the **BDD100K validation dataset**, including quantitative and qualitative analyses, performance insights, and suggested improvements.  
It connects evaluation results to visualizations and data analysis to identify patterns in model performance.

---

## Table of Contents
- [Quantitative Evaluation](#quantitative-evaluation)
- [Qualitative Evaluation](#qualitative-evaluation)
- [Performance Analysis](#performance-analysis)
- [Suggested Improvements](#suggested-improvements)
- [Conclusion](#conclusion)

---

## Quantitative Evaluation

The model was evaluated on the **BDD100K validation dataset** using standard object detection metrics to assess performance.

### Metrics Used for Evaluation
- **Mean Average Precision (mAP):** Measures precision across all classes at IoU thresholds of 0.5 (*mAP@0.5*) and 0.5 to 0.95 (*mAP@0.5:0.95*).  
- **Precision, Recall, F1-Score:** Evaluate the balance between true positive detections and false positives/negatives.  
- **Average Precision (AP) per Class:** Identifies strengths and weaknesses across object categories.  
- **Loss Metrics:** Validation bounding box and classification losses to assess model convergence.

### Results

The evaluation was performed using the `eval.py` script in the `model/` folder, with results saved in `model/runs/bdd100k_yolo11_val/`.  
Below are the key metrics:

| Metric             | Value   |
|--------------------|---------|
| **mAP@0.5**        | 0.4623  |
| **mAP@0.5:0.95**   | 0.2563  |
| **Precision**      | 0.6914  |
| **Recall**         | 0.4750  |
| **F1-Score**       | 0.5642  |

#### Class-Specific AP (mAP@0.5)

| Class          | AP     |
|----------------|--------|
| traffic light  | 0.1926 |
| traffic sign   | 0.2943 |
| car            | 0.4628 |
| person         | 0.2539 |
| bus            | 0.4275 |
| truck          | 0.4155 |
| rider          | 0.1643 |
| bike           | 0.1714 |
| motor          | 0.1791 |
| train          | 0.0018 |

### Visualizations

Quantitative results are visualized in the following plots, saved in `model/runs/bdd100k_yolo11_val/`:

- **Precision-Recall Curve:** Shows the trade-off between precision and recall for each class (`pr_curve.png`).
- **mAP Plot:** Displays *mAP@0.5* across classes (`map_per_class.png`).
- **Confusion Matrix:** Highlights true positives, false positives, and false negatives per class (`confusion_matrix.png`).
- **BoxF1 Curve:** Visualizes F1-score trends across IoU thresholds (`box_f1_curve.png`).
- **BoxP Curve:** Displays precision trends (`box_p_curve.png`).
- **BoxR Curve:** Displays recall trends (`box_r_curve.png`).
- **Validation Batch Samples:** Example validation predictions (`val_batch.png`).

These visualizations provide clear insights into model performance, highlighting lower-performing classes (e.g., *bike*, *motor*, *rider*) and potential issues such as false positives.

---

## Qualitative Evaluation

Qualitative analysis was conducted to visualize **ground truth** and **predicted bounding boxes** on validation images, identifying where the model succeeds or fails.

### Observations

**Success Cases:**
- Accurate detection of large objects (*cars, trucks*) in clear daytime conditions with minimal occlusion.

**Failure Cases:**
- **Occluded Objects:** Missed or incorrect predictions for occluded pedestrians and bicycles.  
- **Small Objects:** Distant motorcycles often undetected due to low resolution.  
- **Nighttime Images:** Lower recall in low-light conditions.  
- **Complex Scenes:** False positives in crowded urban areas.

---

## Performance Analysis

### Model Training Hyperparameters

The model was trained using the following **default YOLO hyperparameters** in `train.py`:

| Parameter | Value |
|------------|--------|
| Learning Rate | 0.01 (with scheduler) |
| Batch Size | 32 |
| Epochs | 100 |
| Image Size | 640×640 |
| Optimizer | SGD (momentum=0.937) |
| Weight Decay | 0.0005 |
| IoU Threshold (NMS) | 0.5 |
| Loss Components | CIoU, BCE (classification), BCE (objectness) |

These defaults were used with minimal tuning for faster experimentation.

### What Works

- Strong performance on **large objects** (cars, trucks) — e.g., AP 0.46 for cars.  
- High precision and recall for **daytime, low-occlusion images**, showing good feature learning for frequently occurring classes.

### What Doesn’t Work

- **Low AP for Small Objects:** Poor performance for small classes such as bicycles, motorcycles, and riders due to size and class imbalance.  
- **Occlusion Issues:** Pedestrian detection drops significantly in crowded scenes.  
- **Nighttime Performance:** Low recall linked to underrepresented low-light data.  
- **False Positives:** Common in urban environments with complex backgrounds.  
- **Data Verification Limitations:** Incomplete label and image validation affects generalization.

---

## Connection to Data Analysis

The **data analysis report** (`data_analysis/output/analysis_report.md`) highlights several factors correlated with model weaknesses:

- **Class Imbalance:** Underrepresented classes (*bicycles, motorcycles, riders*) → lower AP.  
- **Time of Day Distribution:** Only ~20% nighttime images → poor low-light performance.  
- **Occlusion Patterns:** Pedestrian-heavy scenes with high occlusion → missed detections.

These insights align with the evaluation, confirming that both data and model aspects need refinement.

---

## Suggested Improvements

### Data-Related Improvements
- **Augment Underrepresented Classes:** Apply rotation, scaling, and color jitter to balance class representation.  
- **Balance Time of Day:** Collect or augment more **nighttime data** to improve low-light detection.  
- **Handle Occlusion:** Use synthetic occlusion augmentation or filtering for improved robustness.  
- **Increase Scene Diversity:** Include more **urban residential** and **crowded scenes** to reduce false positives.  
- **Improve Data Verification:** Add scripts in `model/data_preprocess/` to validate labels and remove faulty or unannotated samples.

### Model-Related Improvements
- **Architecture:** Try deeper or more advanced models (e.g., **YOLOv8**, **Faster R-CNN**) for better small object detection.  
- **Hyperparameter Tuning:** Experiment with learning rate, batch size, and epochs for dataset-specific optimization.  
- **Loss Function:** Introduce **Focal Loss** to focus on hard examples like occluded pedestrians and small objects.

### Preprocessing Enhancements
- Modify `convert_to_yolo.py` to better scale small bounding boxes.  
- Add filtering for **blurry, low-quality, or heavily occluded** images to improve training data quality.

---

## Conclusion

The evaluation reveals strong performance for **large, clear objects** but challenges with **small, occluded, and nighttime** instances.  
By connecting **quantitative metrics** (e.g., low AP for bikes and riders) with **qualitative visualizations** (e.g., missed detections) and **data analysis** (e.g., class imbalance, occlusion), clear improvement paths emerge.

Implementing the suggested **data verification, augmentation, and model upgrades** should significantly enhance future performance.

