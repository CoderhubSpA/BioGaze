import sys
sys.path.append('.')
from tqdm import tqdm
import cv2
import numpy as np
import os
import config
import shutil
from landmarks import landmark
from metrics import compute_metrics

selected_directory = "/mnt/d/datasets/TONO/icao"
wrong_directory = "/mnt/d/datasets/TONO/ceg"
false_positive_directory = "/mnt/d/datasets/TONO/ceg_fp"
false_negative_directory = "/mnt/d/datasets/TONO/ceg_fn"

# Initialize the FaceParser model
land = landmark.LandmarkRecognizer()

# Initialize y_true and y_pred as lists
files = []
y_true = []
y_pred = []

# Process the selected_directory
for filename in tqdm(os.listdir(selected_directory)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(selected_directory, filename)
        eyes_open = land.eyes_open_check(img_path)

        files.append(img_path)
        y_pred.append(eyes_open)
        y_true.append(1)  # All images in selected_directory are considered true positives

# Process the wrong_directory
for filename in tqdm(os.listdir(wrong_directory)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(wrong_directory, filename)
        eyes_open = land.eyes_open_check(img_path)

        files.append(img_path)
        y_pred.append(eyes_open)
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