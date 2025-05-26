import config
import numpy as np

def is_pixelated_difference_directional(image):
  """
  Checks if an image is pixelated by detecting repetitive patterns in a direction.

  Args:
    image: A NumPy array representing the image in BGR format.
    direction: Direction to calculate difference ("horizontal" or "vertical").

  Returns:
    True if the image is likely pixelated, False otherwise.
  """
  difference = np.abs(image[:, 1:] - image[:, :-1])

  # Calculate average difference
  avg_diff = np.mean(difference)

  return score_function(avg_diff)

def score_function(value):
    min_threshold = config.PIXELATED_MIN_THRESHOLD
    max_threshold = config.PIXELATED_MAX_THRESHOLD
    normalized_value = np.clip((value - min_threshold) / (max_threshold - min_threshold), 0, 1)
    return normalized_value
