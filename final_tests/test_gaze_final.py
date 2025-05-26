import sys
sys.path.append('.')
from tqdm import tqdm
from gaze_estimation import gaze_estimator
import cv2
import numpy as np
import os
import shutil
import config
from metrics import compute_metrics


selected_directory = "/mnt/d/datasets/TONO/icao"
wrong_directory_1 = "/mnt/d/datasets/TONO/la_1"
wrong_directory_2 = "/mnt/d/datasets/TONO/la_2"
false_positive_directory = "/mnt/d/datasets/TONO/la_2_fp"
false_negative_directory = "/mnt/d/datasets/TONO/la_2_fn"

gaze_model = gaze_estimator.GazeEstimator()

# Initialize y_true and y_pred as lists
files = []
y_true = []
y_pred = []

# Process the selected_directory
for filename in tqdm(os.listdir(selected_directory)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(selected_directory, filename)
        gaze_compliant = gaze_model.calculate_gaze(img_path)

        files.append(img_path)
        y_pred.append(gaze_compliant)
        y_true.append(1)  # All images in selected_directory are considered true positives

# Process the wrong_directory
for filename in tqdm(os.listdir(wrong_directory_1)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(wrong_directory_1, filename)
        gaze_compliant = gaze_model.calculate_gaze(img_path)

        files.append(img_path)
        y_pred.append(gaze_compliant)
        y_true.append(0)  # All images in wrong_directory are considered true negatives

# Convert y_true and y_pred to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute the Equal Error Rate (EER)
eer, threshold, _ = compute_metrics(y_pred, y_true, 0.1)

print('EER: ', eer)
print('Threshold: ', threshold)

os.mkdir(false_positive_directory)
os.mkdir(false_negative_directory)
for file, pred, true in zip(files, y_pred, y_true):
    if pred == 0 and true == 1:
        shutil.copy(file, false_negative_directory)
    if pred == 1 and true == 0:
        shutil.copy(file, false_positive_directory)