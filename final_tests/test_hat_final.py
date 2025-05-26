import sys
sys.path.append('.')
from tqdm import tqdm
import cv2
import numpy as np
import shutil
import os
from face_parser import parserModel
from metrics import compute_metrics

selected_directory = "/mnt/d/datasets/TONO/icao"
wrong_directory = "/mnt/d/datasets/TONO/cap"
false_positive_directory = "/mnt/d/datasets/TONO/cap_fp"
false_negative_directory = "/mnt/d/datasets/TONO/cap_fn"

# Initialize the FaceParser model
parserAgent = parserModel.FaceParser()

# Initialize y_true and y_pred as lists
files = []
y_true = []
y_pred = []

# Process the selected_directory
for filename in tqdm(os.listdir(selected_directory)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(selected_directory, filename)
        has_hat = parserAgent.detect_hat(img_path)

        files.append(img_path)
        y_pred.append(has_hat)
        y_true.append(1)  # All images in selected_directory are considered true positives

# Process the wrong_directory
for filename in tqdm(os.listdir(wrong_directory)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(wrong_directory, filename)
        has_hat = parserAgent.detect_hat(img_path)

        files.append(img_path)
        y_pred.append(has_hat)
        y_true.append(0)  # All images in wrong_directory are considered true negatives

# Convert y_true and y_pred to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute the Equal Error Rate (EER)
eer, threshold, _ = compute_metrics(y_pred, y_true, 0.1)

print('EER: ', eer)
print('Threshold: ', threshold)

import matplotlib.pyplot as plt

plt.hist(y_pred[y_true == 0], bins=100, alpha=0.5, color='r', label='Negative')
plt.hist(y_pred[y_true == 1], bins=100, alpha=0.5, color='g', label='Positive')
plt.axvline(x=threshold, color='k', linestyle='--')
plt.legend()
plt.show()

os.mkdir(false_positive_directory)
os.mkdir(false_negative_directory)
for file, pred, true in zip(files, y_pred, y_true):
    if pred == 0 and true == 1:
        shutil.copy(file, false_negative_directory)
    if pred == 1 and true == 0:
        shutil.copy(file, false_positive_directory)