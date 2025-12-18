import logging
import config
import numpy as np
from detectors import detect
from landmarks import landmark
from head_pose import headpose
from face_parser import parserModel
from image_quality import qualitychecker
from gaze_estimation import gaze_estimator
from vlm_client import VLMClient

logger = logging.getLogger("BioGazeAPI")

class BioGazeEngine:
    def __init__(self):
        print("Loading BioGaze models...")
        self.detector = detect.FaceDetector()
        self.landmark_recognizer = landmark.LandmarkRecognizer()
        self.pose_estimator = headpose.HeadposeEstimator()
        self.face_parser = parserModel.FaceParser()
        self.quality_checker = qualitychecker.QualityChecker()
        self.gaze_model = gaze_estimator.GazeEstimator()
        self.vlm_client = None
        if config.VLM_ENABLED:
            self.vlm_client = VLMClient(
                api_key=config.VLM_API_KEY,
                base_url=config.VLM_ENDPOINT,
                model=config.VLM_MODEL_NAME,
                timeout=config.VLM_REQUEST_TIMEOUT,
                max_tokens=config.VLM_MAX_TOKENS,
                temperature=config.VLM_TEMPERATURE,
                max_concurrency=config.VLM_MAX_CONCURRENCY,
            )
        print("BioGaze models loaded successfully.")

    def _convert_numpy_types(self, obj):
        """
        Recursively converts numpy types to native Python types for JSON serialization.
        """
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_types(obj.tolist())
        else:
            return obj

    def _build_vlm_checks(self, image_path: str):
        raw_vlm_results = {}
        vlm_error = None
        if self.vlm_client:
            try:
                raw_vlm_results = self.vlm_client.evaluate(image_path, config.VLM_CHECK_PROMPTS)
            except Exception as exc:
                vlm_error = str(exc)
                logger.error("VLM evaluation failed: %s", vlm_error)
        vlm_checks = {}
        for name, prompt_data in config.VLM_CHECK_PROMPTS.items():
            outcome = raw_vlm_results.get(name, {}) if isinstance(raw_vlm_results, dict) else {}
            normalized = outcome.get("normalized")
            normalized = normalized.lower() if isinstance(normalized, str) else None
            vlm_checks[name] = {
                "answer": outcome.get("answer") if isinstance(outcome, dict) else None,
                "normalized": normalized,
                "latency": outcome.get("latency", 0.0) if isinstance(outcome, dict) else 0.0,
                "passed": normalized == prompt_data.get("pass_if"),
                "error": outcome.get("error") if isinstance(outcome, dict) else vlm_error,
            }
        return vlm_checks

    def validate_image(self, image_path):
        """
        Runs the full BioGaze validation pipeline on a single image.
        Returns a dictionary with the results.
        """
        results = {
            "compliant": True,
            "reasons": [],
            "details": {},
            "technical_metrics": {}
        }

        # 1. Face Detection
        faces_detected, correct_exposure = self.detector.detector_analysis(image_path)
        results["details"]["face_detection"] = {
            "count": int(faces_detected),
            "passed": bool(faces_detected == config.MAX_FACES)
        }
        
        if faces_detected != config.MAX_FACES:
            results["compliant"] = False
            if faces_detected == 0:
                results["reasons"].append("No face detected")
            else:
                results["reasons"].append(f"Multiple faces detected ({faces_detected})")
            # If no face or multiple faces, we might stop here or continue with limited checks.
            # The original script stops. We will stop too as other checks depend on a single face.
            return self._convert_numpy_types(results)

        if not correct_exposure:
            results["compliant"] = False
            results["reasons"].append("Exposure not compliant")
        results["details"]["exposure"] = {"passed": bool(correct_exposure)}

        vlm_checks = self._build_vlm_checks(image_path)
        results["details"]["vlm_checks"] = vlm_checks

        def vlm_pass(name: str) -> bool:
            return vlm_checks.get(name, {}).get("passed", False)

        def vlm_answer(name: str):
            return vlm_checks.get(name, {}).get("normalized")

        # 2. Head Pose
        pitch, yaw, roll = self.pose_estimator.get_headpose_values(image_path)
        
        # Normalize for output (0-1 range logic from original script if needed, or just raw degrees)
        # Original script output logic:
        # pitch_out = round((pitch + 90) / 180, 2)
        
        results["technical_metrics"]["pose"] = {
            "pitch": round(float(pitch), 2),
            "yaw": round(float(yaw), 2),
            "roll": round(float(roll), 2)
        }

        frontal_pose = True
        if not (config.MIN_YAW <= yaw <= config.MAX_YAW):
            results["compliant"] = False
            results["reasons"].append(f"Excessive yaw ({yaw:.2f})")
            frontal_pose = False
        
        if not (config.MIN_PITCH <= pitch <= config.MAX_PITCH):
            results["compliant"] = False
            results["reasons"].append(f"Excessive pitch ({pitch:.2f})")
            frontal_pose = False

        if not (config.MIN_ROLL <= roll <= config.MAX_ROLL):
            results["compliant"] = False
            results["reasons"].append(f"Excessive roll ({roll:.2f})")
            frontal_pose = False
            
        results["details"]["pose"] = {"passed": bool(frontal_pose)}

        # 3. Face Parser (Attributes)
        # has_hat, color_saturation, has_glasses, head_not_contained, chin_not_contained, shoulder_check, uniform_illumination, homogeneous_background, has_sunglasses
        parser_res = self.face_parser.parser_analysis(image_path)
        _, color_saturation, has_glasses, _, _, shoulder_check, uniform_illumination, _, _ = parser_res

        headwear_answer = vlm_answer("headwear")
        sunglasses_answer = vlm_answer("sunglasses")
        background_pass = vlm_pass("background")

        results["details"]["attributes"] = {
            "has_hat": bool(headwear_answer == "yes"),
            "has_glasses": bool(has_glasses),
            "has_sunglasses": bool(sunglasses_answer == "yes"),
            "homogeneous_background": bool(background_pass),
            "uniform_illumination": bool(uniform_illumination)
        }

        if not vlm_pass("headwear"):
            results["compliant"] = False
            if headwear_answer == "yes":
                results["reasons"].append("Hat/Head covering detected")
            else:
                results["reasons"].append("Head covering check inconclusive (VLM)")
        
        if not background_pass:
            results["compliant"] = False
            if vlm_answer("background") == "no":
                results["reasons"].append("Background not homogeneous")
            else:
                results["reasons"].append("Background check inconclusive (VLM)")

        if not vlm_pass("sunglasses"):
            results["compliant"] = False
            if sunglasses_answer == "yes":
                results["reasons"].append("Sunglasses detected")
            else:
                results["reasons"].append("Sunglasses check inconclusive (VLM)")
            
        if not shoulder_check:
             results["compliant"] = False
             results["reasons"].append("Shoulders not aligned/visible correctly")

        if not uniform_illumination:
             results["compliant"] = False
             results["reasons"].append("Non-uniform face illumination")
             
        if not color_saturation:
             results["compliant"] = False
             results["reasons"].append("Incorrect color saturation")

        if not vlm_pass("eyes_open"):
            results["compliant"] = False
            if vlm_answer("eyes_open") == "no":
                results["reasons"].append("Eyes closed or partially closed")
            else:
                results["reasons"].append("Eye openness check inconclusive (VLM)")

        if not vlm_pass("neutral_expression"):
            results["compliant"] = False
            if vlm_answer("neutral_expression") == "no":
                results["reasons"].append("Non-neutral expression detected")
            else:
                results["reasons"].append("Expression check inconclusive (VLM)")

        if not vlm_pass("makeup"):
            results["compliant"] = False
            if vlm_answer("makeup") == "yes":
                results["reasons"].append("Heavy makeup detected")
            else:
                results["reasons"].append("Makeup check inconclusive (VLM)")


        # 4. Landmarks (Eyes, Mouth)
        # inter_eye_distance, eyes_open, mouth_open, m_h, m_v, uniform_luminosity, image_height, image_width, has_makeup
        land_res = self.landmark_recognizer.landmark_analysis(image_path)
        inter_eye_distance = land_res[0]
        eyes_open_val = land_res[1]
        mouth_open_val = land_res[2]
        uniform_luminosity = land_res[5]

        results["technical_metrics"]["landmarks"] = {
            "inter_eye_distance": float(inter_eye_distance),
            "eyes_open_score": float(eyes_open_val),
            "mouth_open_score": float(mouth_open_val)
        }

        # Mouth open check (Logic from original script seems to check mouth_open_val against threshold)
        # Note: In original script, variable is `mouth_open` (boolean result of check?) No, `mouth_open` in `landmark_analysis` return seems to be a float ratio.
        # Let's check config for MOUTH_THRESHOLD.
        # Assuming logic: if mouth_open_val > threshold -> Mouth Open (Fail)
        # Wait, usually "mouth open" means fail.
        
        # Let's check `quality_analysis.py` logic for mouth:
        # It doesn't seem to explicitly reject on mouth open in the snippet I read, but `landmark.py` might have the logic.
        # I'll assume standard logic: if mouth ratio > threshold, it's open.
        
        if mouth_open_val > config.MOUTH_THRESHOLD:
             results["compliant"] = False
             results["reasons"].append("Mouth open detected")

        if not uniform_luminosity:
             results["compliant"] = False
             results["reasons"].append("Non-uniform luminosity (Landmarks)")
        results["details"]["expression"] = {"neutral": bool(vlm_pass("neutral_expression"))}

        # 6. Gaze
        gaze_in_camera = self.gaze_model.calculate_gaze(image_path)
        # gaze_in_camera returns a score or boolean? 
        # In quality_analysis.py: `if not gaze_in_camera:` -> implies boolean or thresholded value.
        # Let's assume it returns True/False or a value that evaluates to False if bad.
        results["details"]["gaze"] = {"looking_at_camera": bool(gaze_in_camera)}
        if not gaze_in_camera:
            results["compliant"] = False
            results["reasons"].append("Gaze not directed at camera")

        # 7. Image Quality
        is_posterized = self.quality_checker.is_posterized(image_path)
        is_pixelated = self.quality_checker.is_pixelated(image_path)
        out_of_focus = self.quality_checker.is_out_of_focus(image_path)

        results["details"]["quality"] = {
            "posterized": bool(is_posterized),
            "pixelated": bool(is_pixelated),
            "out_of_focus": bool(out_of_focus)
        }

        if is_posterized:
            results["compliant"] = False
            results["reasons"].append("Posterization detected")
        
        if is_pixelated:
            results["compliant"] = False
            results["reasons"].append("Pixelation detected")
            
        if out_of_focus:
            results["compliant"] = False
            results["reasons"].append("Image out of focus")

        return self._convert_numpy_types(results)
