import os
import config
import numpy as np
from PIL import Image
from interfaces import IPhotoValidator, ValidationResult
from detectors import detect
from landmarks import landmark
from head_pose import headpose
from face_parser import parserModel
from emotion_recognizer import emotion_detector
from image_quality import qualitychecker
from gaze_estimation import gaze_estimator

class BioGazeAdapter(IPhotoValidator):
    """
    Adaptador concreto que conecta la interfaz genérica con la librería BioGaze específica.
    Si mañana cambiamos BioGaze, solo borramos este archivo y creamos uno nuevo.
    """
    
    def __init__(self):
        self.detector = None
        self.landmark_recognizer = None
        self.pose_estimator = None
        self.face_parser = None
        self.emotion_recognizer = None
        self.quality_checker = None
        self.gaze_model = None

    def load_models(self):
        print("[BioGazeAdapter] Loading models...")
        self.detector = detect.FaceDetector()
        self.landmark_recognizer = landmark.LandmarkRecognizer()
        self.pose_estimator = headpose.HeadposeEstimator()
        self.face_parser = parserModel.FaceParser()
        self.emotion_recognizer = emotion_detector.EmotionDetector()
        self.quality_checker = qualitychecker.QualityChecker()
        self.gaze_model = gaze_estimator.GazeEstimator()
        print("[BioGazeAdapter] Models loaded.")

    def _convert_numpy_types(self, obj):
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

    def validate(self, image_path: str) -> ValidationResult:
        # --- Lógica original de BioGaze (Adaptada y Traducida) ---
        results = {
            "compliant": True,
            "reasons": [],
            "details": {},
            "technical_metrics": {}
        }

        # 0. Validación Previa: Dimensiones y Resolución (MINREL)
        # Requerimiento: 35x45 mm a 300 DPI.
        # Pixeles aprox: 413 x 531 px.
        MIN_WIDTH = 413
        MIN_HEIGHT = 531
        MIN_DPI = 300

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                dpi_info = img.info.get('dpi')
                
                # Chequeo de Dimensiones (Pixeles)
                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    results["compliant"] = False
                    results["reasons"].append(f"Dimensiones incorrectas (Debe ser de 35 x 45 mm)")
                
                # Chequeo de DPI
                if dpi_info:
                    x_dpi, y_dpi = dpi_info
                    if x_dpi < MIN_DPI or y_dpi < MIN_DPI:
                        results["compliant"] = False
                        results["reasons"].append(f"Baja resolución (Mínimo {MIN_DPI} DPI)")
                else:
                    # Por ahora, si cumple pixeles, asumimos que puede ser impreso a 300dpi.
                    pass

        except Exception as e:
            results["compliant"] = False
            results["reasons"].append(f"No se pudo leer la información de la imagen: {str(e)}")
            return ValidationResult(**self._convert_numpy_types(results))

        # Si falló la validación previa, retornamos inmediatamente
        if not results["compliant"]:
            return ValidationResult(**self._convert_numpy_types(results))

        # 1. Detección de Rostro
        faces_detected, correct_exposure = self.detector.detector_analysis(image_path)
        results["details"]["face_detection"] = {
            "count": int(faces_detected),
            "passed": bool(faces_detected == config.MAX_FACES)
        }
        
        if faces_detected != config.MAX_FACES:
            results["compliant"] = False
            if faces_detected == 0:
                results["reasons"].append("No se detectó ningún rostro")
            else:
                results["reasons"].append(f"Se detectaron múltiples rostros ({faces_detected})")
            return ValidationResult(**self._convert_numpy_types(results))

        # 1.1 Validación de Proporción y Centrado (OACI)
        face_box = self.detector.get_face_bbox(image_path)
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            face_height = y2 - y1
            face_center_x = (x1 + x2) / 2
            
            with Image.open(image_path) as img:
                img_width, img_height = img.size
                
                # Proporción del rostro (70-80% de la altura)
                face_ratio = face_height / img_height
                # Relajamos un poco el rango para evitar falsos positivos estrictos (60-85%)
                if not (0.60 <= face_ratio <= 0.85):
                    results["compliant"] = False
                    results["reasons"].append(f"El rostro no ocupa el tamaño adecuado (se requiere que ocupe el 70-80%)")
                
                # Centrado Horizontal
                img_center_x = img_width / 2
                offset_x = abs(face_center_x - img_center_x)
                # Permitir un desvío del 5-10% del ancho
                max_offset = img_width * 0.10
                if offset_x > max_offset:
                    results["compliant"] = False
                    results["reasons"].append("El rostro no está centrado horizontalmente")

        if not correct_exposure:
            results["compliant"] = False
            results["reasons"].append("Exposición incorrecta (muy clara o muy oscura)")
        results["details"]["exposure"] = {"passed": bool(correct_exposure)}

        # 2. Postura de la Cabeza (Head Pose)
        pitch, yaw, roll = self.pose_estimator.get_headpose_values(image_path)
        
        results["technical_metrics"]["pose"] = {
            "pitch": round(float(pitch), 2),
            "yaw": round(float(yaw), 2),
            "roll": round(float(roll), 2)
        }

        frontal_pose = True
        if not (config.MIN_YAW <= yaw <= config.MAX_YAW):
            results["compliant"] = False
            results["reasons"].append(f"Rostro rotado horizontalmente (mire de frente)")
            frontal_pose = False
        
        if not (config.MIN_PITCH <= pitch <= config.MAX_PITCH):
            results["compliant"] = False
            results["reasons"].append(f"Rostro inclinado verticalmente (levante o baje la cabeza)")
            frontal_pose = False

        if not (config.MIN_ROLL <= roll <= config.MAX_ROLL):
            results["compliant"] = False
            results["reasons"].append(f"Rostro inclinado lateralmente (enderece la cabeza)")
            frontal_pose = False
            
        results["details"]["pose"] = {"passed": bool(frontal_pose)}

        # 3. Análisis Facial (Face Parser)
        parser_res = self.face_parser.parser_analysis(image_path)
        has_hat = parser_res[0]
        color_saturation = parser_res[1]
        has_glasses = parser_res[2]
        shoulder_check = parser_res[5]
        uniform_illumination = parser_res[6]
        homogeneous_background = parser_res[7]
        has_sunglasses = parser_res[8]

        results["details"]["attributes"] = {
            "has_hat": bool(has_hat),
            "has_glasses": bool(has_glasses),
            "has_sunglasses": bool(has_sunglasses),
            "homogeneous_background": bool(homogeneous_background),
            "uniform_illumination": bool(uniform_illumination)
        }

        if has_hat:
            results["compliant"] = False
            results["reasons"].append("Se detectó sombrero o cubierta en la cabeza")
        
        # CORRECCIÓN FONDO: Usar umbral más estricto
        # homogeneous_background es un score (0.0 = Malo/Complejo, 1.0 = Liso/Bueno)
        MIN_BACKGROUND_SCORE = 0.85
        
        if homogeneous_background < MIN_BACKGROUND_SCORE:
            results["compliant"] = False
            results["reasons"].append("El fondo no es homogéneo (debe ser uniforme, claro y sin objetos)")

        if has_sunglasses:
            results["compliant"] = False
            results["reasons"].append("Se detectaron lentes oscuros/lentes de sol")
            
        if not shoulder_check:
             results["compliant"] = False
             results["reasons"].append("Hombros no alineados o no visibles correctamente")

        if not uniform_illumination:
             results["compliant"] = False
             results["reasons"].append("Iluminación del rostro no uniforme (sombras)")
             
        if not color_saturation:
             results["compliant"] = False
             results["reasons"].append("Saturación de color incorrecta")

        # 4. Puntos de Referencia (Landmarks)
        land_res = self.landmark_recognizer.landmark_analysis(image_path)
        inter_eye_distance = land_res[0]
        eyes_open_val = land_res[1]
        mouth_open_val = land_res[2]
        uniform_luminosity = land_res[5]
        has_makeup = land_res[8]

        results["technical_metrics"]["landmarks"] = {
            "inter_eye_distance": float(inter_eye_distance),
            "eyes_open_score": float(eyes_open_val),
            "mouth_open_score": float(mouth_open_val)
        }

        eyes_open_compliant = eyes_open_val >= config.EYES_THRESHOLD
        if not eyes_open_compliant:
            results["compliant"] = False
            results["reasons"].append("Ojos cerrados o parcialmente cerrados")
        
        if mouth_open_val < config.MOUTH_THRESHOLD:
             results["compliant"] = False
             results["reasons"].append("Boca abierta detectada")

        # CORRECCIÓN MAQUILLAJE: Usar umbral en vez de booleano
        # Basado en el código, parece ser score de naturalidad (1-mean).
        # Se rechaza si la naturalidad es baja.
        MAKEUP_THRESHOLD = 0.5 
        if has_makeup < MAKEUP_THRESHOLD:
             results["compliant"] = False
             results["reasons"].append("Maquillaje excesivo detectado")
             
        if not uniform_luminosity:
             results["compliant"] = False
             results["reasons"].append("Luminosidad no uniforme en el rostro")

        # 5. Emoción
        neutral_expression = self.emotion_recognizer.check_neutral_expression(image_path)
        results["details"]["expression"] = {"neutral": bool(neutral_expression)}
        if not neutral_expression:
            results["compliant"] = False
            results["reasons"].append("Expresión facial no neutral (sonrisa, gesto, etc.)")

        # 6. Mirada (Gaze)
        gaze_in_camera = self.gaze_model.calculate_gaze(image_path)
        results["details"]["gaze"] = {"looking_at_camera": bool(gaze_in_camera)}
        if not gaze_in_camera:
            results["compliant"] = False
            results["reasons"].append("Mirada no dirigida a la cámara")

        # 7. Calidad de Imagen
        # NOTA: Las funciones devuelven un SCORE de calidad (0.0 = Malo, 1.0 = Bueno)
        # is_posterized -> 1.0 = No posterizada (Bien)
        # out_of_focus -> 1.0 = Enfocada (Bien)
        
        quality_posterization_score = self.quality_checker.is_posterized(image_path)

        # is_pixelated = self.quality_checker.is_pixelated(image_path)
        
        quality_focus_score = self.quality_checker.is_out_of_focus(image_path)

        # Umbrales de aceptación (Hardcodeados por seguridad o mover a config)
        MIN_QUALITY_SCORE = 0.5

        results["details"]["quality"] = {
            "posterization_score": float(quality_posterization_score),
            "focus_score": float(quality_focus_score)
        }

        
        if quality_posterization_score < MIN_QUALITY_SCORE:
            results["compliant"] = False
            results["reasons"].append("Efecto de posterización detectado (baja calidad de color)")
        
        if quality_focus_score < MIN_QUALITY_SCORE:
            results["compliant"] = False
            results["reasons"].append("Imagen desenfocada")

        # Retornamos el objeto estandarizado
        return ValidationResult(**self._convert_numpy_types(results))
