import sys
sys.path.append('.')
from tqdm import tqdm
import cv2
import numpy as np
import os
import config
from face_parser import parserModel
from metrics import compute_metrics
import torch
import matplotlib.pyplot as plt


def compute_tpr_fpr_accuracies(distances: torch.Tensor, thresholds: torch.Tensor, actual: torch.Tensor, chunk_size: int = 1000) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # distances is a (n_samples,) tensor
    # thresholds is a (n_thresholds,) tensor
    # actual is a (n_samples,) tensor
    # Returns a tuple of (tpr, fpr, accuracies), each of which is a (n_thresholds,) tensor
    if thresholds.ndim == 0:
        thresholds = thresholds.unsqueeze(0)
    n_thresholds = thresholds.shape[0]
    actual_expanded = actual.unsqueeze(1)  # (n_samples, 1)
    tpr = torch.zeros(n_thresholds, device=distances.device)
    fpr = torch.zeros(n_thresholds, device=distances.device)
    acc = torch.zeros(n_thresholds, device=distances.device)
    for start in range(0, n_thresholds, chunk_size):
        end = min(start + chunk_size, n_thresholds)
        chunk_thresholds = thresholds[start:end]
        predictions = distances.unsqueeze(1) < chunk_thresholds.unsqueeze(0)  # (n_samples, chunk_size)
        tp = (predictions & actual_expanded).sum(dim=0, dtype=torch.int32)
        fp = (predictions & ~actual_expanded).sum(dim=0, dtype=torch.int32)
        tn = (~predictions & ~actual_expanded).sum(dim=0, dtype=torch.int32)
        fn = (~predictions & actual_expanded).sum(dim=0, dtype=torch.int32)
        tpr[start:end] = torch.nan_to_num(tp / (tp + fn), nan=0.0, posinf=0.0, neginf=0.0)
        fpr[start:end] = torch.nan_to_num(fp / (fp + tn), nan=0.0, posinf=0.0, neginf=0.0)
        acc[start:end] = (tp + tn) / distances.shape[0]
    return tpr, fpr, acc


root = "/mnt/d/datasets/BioLab-ICAO"
images_directory = os.path.join(root, "Img")
markers_directory = os.path.join(root, "Mrk")

# Initialize the FaceParser model
parser = parserModel.FaceParser()

# Initialize y_true and y_pred as lists
y_true = []
y_pred = []

# Process the selected_directory
for filename in tqdm(os.listdir(images_directory)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(images_directory, filename)
        mrk_path = os.path.join(markers_directory, filename[:-4] + ".mrk")
        with open(mrk_path, 'r') as f:
            gt = int(f.readlines()[11])
        if gt == -1:
            continue
        bg_mean = parser.detect_background(img_path)
        y_pred.append(bg_mean)
        y_true.append(gt)

# Convert y_true and y_pred to numpy arrays
y_true = torch.tensor(y_true)
y_pred = torch.tensor(y_pred)
thresholds = torch.linspace(0, y_pred.max(), 1000)

tpr, fpr, acc = compute_tpr_fpr_accuracies(y_pred, thresholds, y_true)
eer_idx = torch.argmin(torch.abs(fpr - (1 - tpr)))
eer = (fpr[eer_idx] + (1 - tpr[eer_idx])) / 2

print('EER: ', eer.item())
print('Threshold: ', thresholds[eer_idx].item())