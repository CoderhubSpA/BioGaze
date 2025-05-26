import sys
sys.path.append('.')
from tqdm import tqdm
import cv2
import numpy as np
import os
import config
import shutil
import time
from detectors import detect
from metrics import compute_metrics

selected_directory = "/mnt/d/datasets/TONO/icao"
wrong_directory = "/mnt/d/datasets/TONO/expos"
false_positive_directory = "/mnt/d/datasets/TONO/expos_fp"
false_negative_directory = "/mnt/d/datasets/TONO/expos_fn"

os.environ["YOLO_VERBOSE"] = "False"

face_detector = detect.FaceDetector()

# Initialize y_true and y_pred as lists
files = []
y_true = []
y_pred = []
times = []

# Process the selected_directory
for filename in tqdm(os.listdir(selected_directory)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(selected_directory, filename)
        start = time.perf_counter()
        _, correct_exposure = face_detector.detector_analysis(img_path)
        end = time.perf_counter()
        times.append(end - start)

        files.append(img_path)
        y_pred.append(correct_exposure)
        y_true.append(1)  # All images in selected_directory are considered true positives

# Process the wrong_directory
for filename in tqdm(os.listdir(wrong_directory)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(wrong_directory, filename)
        start = time.perf_counter()
        _, correct_exposure = face_detector.detector_analysis(img_path)
        end = time.perf_counter()
        times.append(end - start)

        files.append(img_path)
        y_pred.append(correct_exposure)
        y_true.append(0)  # All images in wrong_directory are considered true negatives


# Convert y_true and y_pred to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# np.save("scores.npy", np.concatenate([y_pred, y_true], axis=1))

# sys.exit(0)

# Compute the Equal Error Rate (EER)
eer, threshold, _ = compute_metrics(y_pred, y_true, 0.1)

print('EER: ', eer)
print('Threshold: ', threshold)
print('Average time: ', np.mean(times), '+-', np.std(times))

# import matplotlib.pyplot as plt

# plt.hist(y_pred[y_true == 1], bins=100, alpha=0.5, label='True Positives')
# plt.hist(y_pred[y_true == 0], bins=100, alpha=0.5, label='True Negatives')
# plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
# plt.legend()
# plt.show()

os.mkdir(false_positive_directory)
os.mkdir(false_negative_directory)
for file, pred, true in zip(files, y_pred, y_true):
    if pred < threshold and true == 1:
        shutil.copy(file, false_negative_directory)
    if pred >= threshold and true == 0:
        shutil.copy(file, false_positive_directory)



'''
std 50 bright 80:
0.32

std 45 bright 80:
0.255


'''