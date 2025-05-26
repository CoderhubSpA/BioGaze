import os
os.environ["YOLO_VERBOSE"] = "False"

import numpy as np
from tqdm import tqdm

from detectors import detect
from face_parser import parserModel
from emotion_recognizer import emotion_detector
from gaze_estimation import gaze_estimator
from landmarks import landmark
from image_quality import qualitychecker
from head_pose import headpose

face_detector = detect.FaceDetector()
parser = parserModel.FaceParser()
land = landmark.LandmarkRecognizer()
emotion_recognizer = emotion_detector.EmotionDetector()
gaze_model = gaze_estimator.GazeEstimator()
quality_checker = qualitychecker.QualityChecker()
pose_estimator = headpose.HeadposeEstimator()

BACKGROUND_THRESHOLD = 0.9945776977045377
EXPOSURE_THRESHOLD = 0.0
EXPRESSION_THRESHOLD = 0.8109785629194833
EYE_THRESHOLD = 0.5021922918942692
GAZE_THRESHOLD = 0.05316425730352814
HAT_THRESHOLD = 1.0
LIGHT_THRESHOLD = 0.7067474680437039
MAKEUP_THRESHOLD = 0.8937117116889786
OOF_THRESHOLD = 0.5044080862890954
PIXEL_THRESHOLD = 0.38621218900191945
POSTER_THRESHOLD = 0.0
SATURATION_THRESHOLD = 0.8905963190266972
SUN_THRESHOLD = 0.8842969164252281
TQ_THRESHOLD = 0.5269291307526158


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_background(img_path):
    raw = parser.detect_background(img_path)
    score = sigmoid(raw - BACKGROUND_THRESHOLD)
    return score, raw >= BACKGROUND_THRESHOLD


def test_exposure(img_path):
    _, raw = face_detector.detector_analysis(img_path)
    score = sigmoid(raw - EXPOSURE_THRESHOLD)
    return score, raw >= EXPOSURE_THRESHOLD


def test_expression(img_path):
    mouth_open, percentage = land.mouth_check(img_path)
    emo_compliant = emotion_recognizer.check_neutral_expression(img_path)
    raw = emo_compliant * 0.2 + mouth_open * 0.3 + percentage * 0.5
    score = sigmoid(raw - EXPRESSION_THRESHOLD)
    return score, raw >= EXPRESSION_THRESHOLD


def test_eye(img_path):
    raw = land.eyes_open_check(img_path)
    score = sigmoid(raw - EYE_THRESHOLD)
    return score, raw >= EYE_THRESHOLD


def test_gaze(img_path):
    raw = gaze_model.calculate_gaze(img_path)
    score = sigmoid(raw - GAZE_THRESHOLD)
    return score, raw >= GAZE_THRESHOLD


def test_hat(img_path):
    raw = parser.detect_hat(img_path)
    score = sigmoid(raw - HAT_THRESHOLD)
    return score, raw >= HAT_THRESHOLD


def test_light(img_path):
    s1 = parser.detect_face_illumination(img_path)
    s2 = land.light_analysis(img_path)
    raw = 0.5 * s1 + 0.5 * s2
    score = sigmoid(raw - LIGHT_THRESHOLD)
    return score, raw >= LIGHT_THRESHOLD


def test_makeup(img_path):
    raw = land.makeup_check(img_path)
    score = sigmoid(raw - MAKEUP_THRESHOLD)
    return score, raw >= MAKEUP_THRESHOLD


def test_oof(img_path):
    raw = quality_checker.is_out_of_focus(img_path)
    score = sigmoid(raw - OOF_THRESHOLD)
    return score, raw >= OOF_THRESHOLD


def test_pixel(img_path):
    raw = quality_checker.is_pixelated(img_path)
    score = sigmoid(raw - PIXEL_THRESHOLD)
    return score, raw >= PIXEL_THRESHOLD


def test_poster(img_path):
    raw = quality_checker.is_posterized(img_path)
    score = sigmoid(raw - POSTER_THRESHOLD)
    return score, raw >= POSTER_THRESHOLD


def test_saturation(img_path):
    raw = parser.alternative_color_saturation(img_path)
    score = sigmoid(raw - SATURATION_THRESHOLD)
    return score, raw >= SATURATION_THRESHOLD


def test_sun(img_path):
    raw = parser.has_sunglasses(img_path)
    score = sigmoid(raw - SUN_THRESHOLD)
    return score, raw >= SUN_THRESHOLD


def test_tq(img_path):
    s1 = parser.shoulder_check(img_path)
    s2 = pose_estimator.headpose_compliant(img_path)
    raw = 0.5 * s1 + 0.5 * s2
    score = sigmoid(raw - TQ_THRESHOLD)
    return score, raw >= TQ_THRESHOLD


def test_file(img_path):
    try:
        bg_score, bg_compliant = test_background(img_path)
        exp_score, exp_compliant = test_exposure(img_path)
        expre_score, expre_compliant = test_expression(img_path)
        eye_score, eye_compliant = test_eye(img_path)
        gaze_score, gaze_compliant = test_gaze(img_path)
        hat_score, hat_compliant = test_hat(img_path)
        light_score, light_compliant = test_light(img_path)
        makeup_score, makeup_compliant = test_makeup(img_path)
        oof_score, oof_compliant = test_oof(img_path)
        pixel_score, pixel_compliant = test_pixel(img_path)
        poster_score, poster_compliant = test_poster(img_path)
        sat_score, sat_compliant = test_saturation(img_path)
        sun_score, sun_compliant = test_sun(img_path)
        tq_score, tq_compliant = test_tq(img_path)
        scores = [bg_score, exp_score, expre_score, eye_score, gaze_score, hat_score, light_score, makeup_score,
                    oof_score, pixel_score, poster_score, sat_score, sun_score, tq_score]
        score = sum(scores) / len(scores)
        compliants = [bg_compliant, exp_compliant, expre_compliant, eye_compliant, gaze_compliant, hat_compliant,
                            light_compliant, makeup_compliant, oof_compliant, pixel_compliant, poster_compliant, sat_compliant,
                            sun_compliant, tq_compliant]
        compliant = all(compliants)
        return scores, compliants, score, compliant
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return [0] * 14, [False] * 14, 0, False


# selected_directory = "/mnt/d/datasets/TONO/mix"
all_selected_directory = "D:/datasets/TONO/bkg"
fail_selected_directory = "D:/datasets/TONO/mix"
pass_selected_directory = "D:/datasets/TONO/icao"

all_files = []
all_scores = []
all_compliants = []
all_merged_scores = []
all_merged_compliants = []
gt = []

from pathlib import Path
found_files = list(x for x in Path(all_selected_directory).rglob("*") if x.is_file())
for filename in tqdm(found_files):
    if filename.name.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = str(filename)
        scores, compliants, score, compliant = test_file(img_path)
        all_files.append(img_path)
        all_scores.append(scores)
        all_compliants.append(compliants)
        all_merged_scores.append(score)
        all_merged_compliants.append(compliant)
        gt.append(1 if "icao" in img_path else 0)

import pandas as pd

df_dict = {
    "file": all_files,
    "bg_score": [s[0] for s in all_scores],
    "exp_score": [s[1] for s in all_scores],
    "expre_score": [s[2] for s in all_scores],
    "eye_score": [s[3] for s in all_scores],
    "gaze_score": [s[4] for s in all_scores],
    "hat_score": [s[5] for s in all_scores],
    "light_score": [s[6] for s in all_scores],
    "makeup_score": [s[7] for s in all_scores],
    "oof_score": [s[8] for s in all_scores],
    "pixel_score": [s[9] for s in all_scores],
    "poster_score": [s[10] for s in all_scores],
    "sat_score": [s[11] for s in all_scores],
    "sun_score": [s[12] for s in all_scores],
    "tq_score": [s[13] for s in all_scores],
    "bg_compliant": [c[0] for c in all_compliants],
    "exp_compliant": [c[1] for c in all_compliants],
    "expre_compliant": [c[2] for c in all_compliants],
    "eye_compliant": [c[3] for c in all_compliants],
    "gaze_compliant": [c[4] for c in all_compliants],
    "hat_compliant": [c[5] for c in all_compliants],
    "light_compliant": [c[6] for c in all_compliants],
    "makeup_compliant": [c[7] for c in all_compliants],
    "oof_compliant": [c[8] for c in all_compliants],
    "pixel_compliant": [c[9] for c in all_compliants],
    "poster_compliant": [c[10] for c in all_compliants],
    "sat_compliant": [c[11] for c in all_compliants],
    "sun_compliant": [c[12] for c in all_compliants],
    "tq_compliant": [c[13] for c in all_compliants],
    "score": all_merged_scores,
    "compliant": all_merged_compliants,
    "gt": gt,
}

df = pd.DataFrame(df_dict)
df.to_csv("results_bg.csv", index=False)

# for filename in tqdm(os.listdir(fail_selected_directory)):
#     if filename.lower().endswith((".png", ".jpg", ".jpeg")):
#         img_path = os.path.join(fail_selected_directory, filename)
#         score, compliant = test_file(img_path)
#         files.append(img_path)
#         scores.append(score)
#         compliants.append(compliant)
#         gt.append(0)

# for filename in tqdm(os.listdir(pass_selected_directory)):
#     if filename.lower().endswith((".png", ".jpg", ".jpeg")):
#         img_path = os.path.join(pass_selected_directory, filename)
#         score, compliant = test_file(img_path)
#         files.append(img_path)
#         scores.append(score)
#         compliants.append(compliant)
#         gt.append(1)


# tp = sum([1 for c, g in zip(compliants, gt) if c == g and g == 1])
# tn = sum([1 for c, g in zip(compliants, gt) if c == g and g == 0])
# fp = sum([1 for c, g in zip(compliants, gt) if c != g and g == 0])
# fn = sum([1 for c, g in zip(compliants, gt) if c != g and g == 1])

# accuracy = (tp + tn) / len(gt)
# print(f"Accuracy: {accuracy}")
# print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")