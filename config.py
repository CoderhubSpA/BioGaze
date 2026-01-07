#DETECTION

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# INFRASTRUCTURE SETTINGS (Environment Variables)
# =============================================================================

MAX_FACES = 1
EARLY_STOP_ENABLED = os.getenv("EARLY_STOP_ENABLED", "false").lower() not in {"false", "0", "no"}
MIN_WIDTH = int(os.getenv("MIN_WIDTH", "413"))
MIN_HEIGHT = int(os.getenv("MIN_HEIGHT", "531"))
ASPECT_RATIO_RAW = os.getenv("ASPECT_RATIO", "7:9")

def _parse_aspect_ratio(raw: str) -> float:
    try:
        if ":" in raw:
            num, denom = raw.split(":", 1)
            return float(num) / float(denom)
        return float(raw)
    except Exception:
        return 7.0 / 9.0

ASPECT_RATIO = _parse_aspect_ratio(ASPECT_RATIO_RAW)
ASPECT_RATIO_THRESHOLD = float(os.getenv("ASPECT_RATIO_THRESHOLD", "0.03"))
SKIP_RESOLUTION_CHECK = os.getenv("SKIP_RESOLUTION_CHECK", "false").lower() in {"true", "1", "yes"}
SKIP_RATIO_CHECK = os.getenv("SKIP_RATIO_CHECK", "false").lower() in {"true", "1", "yes"}

# =============================================================================
# LOAD CHECKS CONFIGURATION FROM YAML
# =============================================================================

def _load_checks_config():
    """Load checks configuration from YAML file"""
    config_path = Path(__file__).parent / "checks_config.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. Using default values.")
        return {}
    except Exception as e:
        print(f"Error loading checks config: {e}. Using default values.")
        return {}

_CHECKS_CONFIG = _load_checks_config()

# =============================================================================
# HEADPOSE THRESHOLDS
# =============================================================================

MIN_YAW = _CHECKS_CONFIG.get('headpose', {}).get('min_yaw', -7)
MAX_YAW = _CHECKS_CONFIG.get('headpose', {}).get('max_yaw', 7)
MIN_PITCH = _CHECKS_CONFIG.get('headpose', {}).get('min_pitch', -7)
MAX_PITCH = _CHECKS_CONFIG.get('headpose', {}).get('max_pitch', 7)
MIN_ROLL = _CHECKS_CONFIG.get('headpose', {}).get('min_roll', -10)
MAX_ROLL = _CHECKS_CONFIG.get('headpose', {}).get('max_roll', 10)

# =============================================================================
# EYES OPEN, MOUTH OPEN THRESHOLDS
# =============================================================================

MOUTH_THRESHOLD = _CHECKS_CONFIG.get('facial_features', {}).get('mouth_threshold', 0.5)
EYES_THRESHOLD = _CHECKS_CONFIG.get('facial_features', {}).get('eyes_threshold', 0.09)

# =============================================================================
# IMAGE AND FACE DIMENSION THRESHOLDS
# =============================================================================

MIN_FACE_SIZE_RATIO = _CHECKS_CONFIG.get('face_dimensions', {}).get('min_face_size_ratio', 0.60)
MAX_FACE_SIZE_RATIO = _CHECKS_CONFIG.get('face_dimensions', {}).get('max_face_size_ratio', 0.85)
MIN_WIDTH_HEIGHT_RATIO = _CHECKS_CONFIG.get('face_dimensions', {}).get('min_width_height_ratio', 0.74)
MAX_WIDTH_HEIGHT_RATIO = _CHECKS_CONFIG.get('face_dimensions', {}).get('max_width_height_ratio', 0.80)
MIN_LEFT_DISTANCE = _CHECKS_CONFIG.get('face_dimensions', {}).get('min_left_distance', 0.45)
MAX_LEFT_DISTANCE = _CHECKS_CONFIG.get('face_dimensions', {}).get('max_left_distance', 0.55)
MIN_TOP_DISTANCE = _CHECKS_CONFIG.get('face_dimensions', {}).get('min_top_distance', 0.30)
MAX_TOP_DISTANCE = _CHECKS_CONFIG.get('face_dimensions', {}).get('max_top_distance', 0.50)
MIN_WIDTH_HEAD = _CHECKS_CONFIG.get('face_dimensions', {}).get('min_width_head', 0.50)
MAX_WIDTH_HEAD = _CHECKS_CONFIG.get('face_dimensions', {}).get('max_width_head', 0.75)
MIN_HEIGHT_HEAD = _CHECKS_CONFIG.get('face_dimensions', {}).get('min_height_head', 0.60)
MAX_HEIGHT_HEAD = _CHECKS_CONFIG.get('face_dimensions', {}).get('max_height_head', 0.90)
MINIMUM_IED = _CHECKS_CONFIG.get('face_dimensions', {}).get('minimum_ied', 0.50)

# =============================================================================
# EXPOSURE CONSTANTS
# =============================================================================

AVG_DARK_THRESHOLD = _CHECKS_CONFIG.get('exposure', {}).get('avg_dark_threshold', 0.0055)
MAX_DARK_THRESHOLD = _CHECKS_CONFIG.get('exposure', {}).get('max_dark_threshold', 0.015)
AVG_LIGHT_THRESHOLD = _CHECKS_CONFIG.get('exposure', {}).get('avg_light_threshold', 0.0035)
MAX_LIGHT_THRESHOLD = _CHECKS_CONFIG.get('exposure', {}).get('max_light_threshold', 0.015)

# =============================================================================
# PIXELATION THRESHOLDS
# =============================================================================

PIXELATED_MIN_THRESHOLD = _CHECKS_CONFIG.get('pixelation', {}).get('pixelated_min_threshold', 19)
PIXELATED_MAX_THRESHOLD = _CHECKS_CONFIG.get('pixelation', {}).get('pixelated_max_threshold', 106)
PIXEL_SCORE_THRESHOLD = _CHECKS_CONFIG.get('pixelation', {}).get('pixel_score_threshold', 0.5)
MIN_PIXEL_SCORE = _CHECKS_CONFIG.get('pixelation', {}).get('min_pixel_score', 10)
MAX_PIXEL_SCORE = _CHECKS_CONFIG.get('pixelation', {}).get('max_pixel_score', 110)

# =============================================================================
# EMOTION DETECTION THRESHOLD
# =============================================================================

MAX_RATIO_EMOTION = _CHECKS_CONFIG.get('emotion', {}).get('max_ratio_emotion', 0.35)

# =============================================================================
# PARSER THRESHOLDS
# =============================================================================

MAX_SHOULDER_PIXEL_RATIO = _CHECKS_CONFIG.get('parser', {}).get('max_shoulder_pixel_ratio', 0.55)
MAX_SHOULDER_Y_DISTANCE = _CHECKS_CONFIG.get('parser', {}).get('max_shoulder_y_distance', 5)
OVERSATURATION_THRESHOLD = _CHECKS_CONFIG.get('parser', {}).get('oversaturation_threshold', 0.002)
UNDERSATURATION_THRESHOLD = _CHECKS_CONFIG.get('parser', {}).get('undersaturation_threshold', 0.06)
MAX_BRIGHT_LIGHT = _CHECKS_CONFIG.get('parser', {}).get('max_bright_light', 5)
MAX_DARK_LIGHT = _CHECKS_CONFIG.get('parser', {}).get('max_dark_light', 70)
MAX_EDGES_THRESHOLD = _CHECKS_CONFIG.get('parser', {}).get('max_edges_threshold', 900000000)
AVG_VARIANCE_THRESHOLD = _CHECKS_CONFIG.get('parser', {}).get('avg_variance_threshold', 500)
HOMOGENEOUS_PROPORTION_THRESHOLD = _CHECKS_CONFIG.get('parser', {}).get('homogeneous_proportion_threshold', 0.90)
SUPERPIXEL_VARIANCE_THRESHOLD = _CHECKS_CONFIG.get('parser', {}).get('superpixel_variance_threshold', 45)
MAX_LIGHT_DARK_SUN = _CHECKS_CONFIG.get('parser', {}).get('max_light_dark_sun', 0.004)

# =============================================================================
# COMPUTER VISION THRESHOLDS
# =============================================================================

MINIMUM_FOCUS_THRESHOLD = _CHECKS_CONFIG.get('computer_vision', {}).get('minimum_focus_threshold', 110)
MINIMUM_FOCUS_SCORE = _CHECKS_CONFIG.get('computer_vision', {}).get('minimum_focus_score', 0.3)
MAX_GAPS_THRESHOLD = _CHECKS_CONFIG.get('computer_vision', {}).get('max_gaps_threshold', 417)
GAP_HISTOGRAM_THRESHOLD = _CHECKS_CONFIG.get('computer_vision', {}).get('gap_histogram_threshold', 0.001)

# =============================================================================
# LANDMARK THRESHOLDS
# =============================================================================

MIN_COLOR_RATIO_THRESOLD = _CHECKS_CONFIG.get('landmarks', {}).get('min_color_ratio_threshold', 0.5)
MAKEUP_HIGH_HUE_THRESHOLD = _CHECKS_CONFIG.get('landmarks', {}).get('makeup_high_hue_threshold', 0.15)
MAKEUP_DISTANCE_THRESHOLD = _CHECKS_CONFIG.get('landmarks', {}).get('makeup_distance_threshold', 14)

# =============================================================================
# GAZE ESTIMATION THRESHOLDS
# =============================================================================

MAXIMUM_RIGHT_THRESHOLD = _CHECKS_CONFIG.get('gaze_estimation', {}).get('maximum_right_threshold', -0.15)
MAXIMUM_LEFT_THRESHOLD = _CHECKS_CONFIG.get('gaze_estimation', {}).get('maximum_left_threshold', 0.08)

# =============================================================================
# HEADERS FOR TABLE OUTPUT
# =============================================================================

HEADERS = ["Filename", "Compliant", "IED", "Background", "Lighting", "Color/Saturation", "Roll", "Pitch", "Yaw",
           "Shoulders", "Mouth", "Eyes", "Glasses", "Hat", "Image ratio", "Head location", "Head dimension"]

HEADERS_SECONDARY = ["Filename", "Compliant", "Head without covering", "Eyes open", "No sunglasses", "No posterization", 
                     "Gaze in camera", "Neutral expression", "In focus", "Correct exposure", "No/light makeup", "No pixelation", 
                     "Frontal pose", "Correct saturation", "Uniform background", "Uniform face lighting"]

HEADERS_TABLE = ["Filename", "Compliant", "Head without covering", "Eyes open", "No sunglasses", "No posterization", 
                     "Gaze in camera", "Neutral expression", "In focus", "Correct exposure", "No/light makeup", "No pixelation", 
                     "Frontal pose", "Correct saturation", "Uniform background", "Uniform face lighting", "IED", "Roll", "Pitch", "Yaw", "Mouth", "Eyes", "Glasses"]

# =============================================================================
# CONSTANTS FOR INDIVIDUAL CHECKS
# =============================================================================

# Constants for Checks
HEAD_WITHOUT_COVERING = 0
EYES_OPEN = 1
NO_SUNGLASSES = 2
NO_POSTERIZATION = 3
GAZE_IN_CAMERA = 4
NEUTRAL_EXPRESSION = 5
IN_FOCUS_PHOTO = 6
CORRECT_EXPOSURE = 7
NO_LIGHT_MAKEUP = 8
NO_PIXELATION = 9
FRONTAL_POSE = 10
CORRECT_SATURATION = 11
UNIFORM_BACKGROUND = 12
UNIFORM_FACE_LIGHTING = 13

# Maps of checks to functions for landmark and parser
landmark_checks_map = {
    EYES_OPEN: 'eyes_open',
    NEUTRAL_EXPRESSION: 'mouth_open',
    NO_LIGHT_MAKEUP: 'has_makeup',
    UNIFORM_FACE_LIGHTING: 'uniform_luminosity'
}

parser_checks_map = {
    HEAD_WITHOUT_COVERING: 'has_hat',
    NO_SUNGLASSES: 'has_sunglasses',
    FRONTAL_POSE: 'shoulder_check',
    CORRECT_SATURATION: 'color_saturation',
    UNIFORM_BACKGROUND: 'homogeneous_background',
    UNIFORM_FACE_LIGHTING: 'uniform_illumination'
}

# =============================================================================
# VISUAL LANGUAGE MODEL (VLM) SETTINGS - Infrastructure (Environment Variables)
# =============================================================================
VLM_ENABLED = os.getenv("VLM_ENABLED", "true").lower() not in {"false", "0", "no"}
VLM_ENDPOINT = os.getenv("VLM_ENDPOINT", "http://localhost:8000/")
VLM_API_KEY = os.getenv("VLM_API_KEY", "")
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "Qwen3-VL")
VLM_REQUEST_TIMEOUT = int(os.getenv("VLM_REQUEST_TIMEOUT", "120"))
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "4"))
VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0"))
VLM_MAX_CONCURRENCY = int(os.getenv("VLM_MAX_CONCURRENCY", "4"))

VLM_CHECK_PROMPTS = {
    "background": {
        "prompt": "Does the photograph have a plain, featureless, light and uniform background? Answer only Yes or No.",
        "pass_if": "yes",
    },
    "eyes_open": {
        "prompt": "Are both of the person's eyes open and clearly visible in the photograph? Answer only Yes or No.",
        "pass_if": "yes",
    },
    "neutral_expression": {
        "prompt": "Does the person have a neutral facial expression and is not smiling? Answer only Yes or No.",
        "pass_if": "yes",
    },
    "headwear": {
        "prompt": "Is the person wearing any head covering (such as a hat, headband, cap or hood)? Answer only Yes or No.",
        "pass_if": "no",
    },
    "makeup": {
        "prompt": "Is the person wearing heavy or conspicuous makeup that significantly alters their natural appearance? Answer only Yes or No.",
        "pass_if": "no",
    },
    "sunglasses": {
        "prompt": "Is the person wearing sunglasses or dark tinted lenses that obscure the eyes? Answer only Yes or No.",
        "pass_if": "no",
    },
}

