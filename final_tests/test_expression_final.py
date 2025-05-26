import sys
sys.path.append('.')
from tqdm import tqdm
from landmarks import landmark
import cv2
import numpy as np
import os
import config
import shutil
from metrics import compute_metrics
from emotion_recognizer import emotion_detector

land = landmark.LandmarkRecognizer()

emotion_recognizer = emotion_detector.EmotionDetector()

selected_directory = "/mnt/d/datasets/TONO/icao"
wrong_directory = "/mnt/d/datasets/TONO/sm"
false_positive_directory = "/mnt/d/datasets/TONO/sm_fp"
false_negative_directory = "/mnt/d/datasets/TONO/sm_fn"


# Initialize y_true and y_pred as lists
files = []
y_true = []
y_pred = []

def combine(emo, mouth, perc):
    combined = emo * 0.2 + mouth * 0.3 + perc * 0.5
    return combined

# Process the selected_directory
for filename in tqdm(os.listdir(selected_directory)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(selected_directory, filename)
        mouth_open, percentage = land.mouth_check(img_path)
        emo_compliant = emotion_recognizer.check_neutral_expression(img_path)

        files.append(img_path)
        y_pred.append(combine(emo_compliant, mouth_open, percentage))
        y_true.append(1)  # All images in selected_directory are considered true positives

# Process the wrong_directory
for filename in tqdm(os.listdir(wrong_directory)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(wrong_directory, filename)
        mouth_open, percentage = land.mouth_check(img_path)
        emo_compliant = emotion_recognizer.check_neutral_expression(img_path)

        files.append(img_path)
        y_pred.append(combine(emo_compliant, mouth_open, percentage))
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
    if pred < threshold and true == 1:
        shutil.copy(file, false_negative_directory)
    if pred >= threshold and true == 0:
        shutil.copy(file, false_positive_directory)