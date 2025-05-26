import sys
sys.path.append('.')
from tqdm import tqdm
import cv2
from image_quality import qualitychecker
import numpy as np
import os
import config
import shutil
from metrics import compute_metrics


selected_directory = "/mnt/d/datasets/TONO/icao"
wrong_directory = "/mnt/d/datasets/TONO/pixel"
false_positive_directory = "/mnt/d/datasets/TONO/pixel_fp"
false_negative_directory = "/mnt/d/datasets/TONO/pixel_fn"

quality_checker = qualitychecker.QualityChecker()

# Initialize y_true and y_pred as lists
files = []
y_true = []
y_pred = []

# Process the selected_directory
for filename in tqdm(os.listdir(selected_directory)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(selected_directory, filename)
        is_pixelated = quality_checker.is_pixelated(img_path)

        files.append(img_path)
        y_pred.append(is_pixelated)
        y_true.append(1)  # All images in selected_directory are considered true positives

# Process the wrong_directory
for filename in tqdm(os.listdir(wrong_directory)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(wrong_directory, filename)
        is_pixelated = quality_checker.is_pixelated(img_path)

        files.append(img_path)
        y_pred.append(is_pixelated)
        y_true.append(0)  # All images in wrong_directory are considered true negatives

# Convert y_true and y_pred to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute the Equal Error Rate (EER)
eer, threshold, _ = compute_metrics(y_pred, y_true, 0.1)

print('EER: ', eer)
print('Threshold: ', threshold)

import matplotlib.pyplot as plt

plt.hist(y_pred[y_true == 1], bins=100, alpha=0.5, label='True Positives')
plt.hist(y_pred[y_true == 0], bins=100, alpha=0.5, label='True Negatives')
plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()

os.mkdir(false_positive_directory)
os.mkdir(false_negative_directory)
for file, pred, true in zip(files, y_pred, y_true):
    if pred < threshold and true == 1:
        shutil.copy(file, false_negative_directory)
    if pred >= threshold and true == 0:
        shutil.copy(file, false_positive_directory)

