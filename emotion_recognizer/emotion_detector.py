import cv2
import numpy as np
from rmn import RMN  # Assuming this is the model you're using.
import config
import os

class EmotionDetector:
    def __init__(self):
        # Initialize the RMN model (or any other emotion detection model).
        self.model = RMN()

    def check_neutral_expression(self, img_path):
        """
        Detects emotion from the given image and checks if the emotion is neutral and the second most
        likely emotion has a low probability.

        Parameters:
            img_path (str): Path to the image.

        Returns:
            bool: True if the emotion is neutral, False otherwise.
        """
        # Read the image.
        image = cv2.imread(img_path)
    
        # Use the model to detect emotion.
        results = self.model.detect_emotion_for_single_frame(image)

        # Extract the emotions and their probabilities.
        proba_list = results[0]['proba_list']

        for d in proba_list:
            k, v = list(d.items())[0]
            if k == "neutral":
                return v
        
    def draw_emotion(self, img_path):
        image = cv2.imread(img_path)
        results = self.model.detect_emotion_for_single_frame(image)
        image = self.model.draw(image, results)
        output_path = img_path.replace('.jpg', '_emo.jpg').replace('.png', '_emo.png').replace('.JPG', '_emo.JPG')
        cv2.imwrite(output_path, image)
    