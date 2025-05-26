import sys
sys.path.append('.')
import numpy as np
import cv2
import torch
from PIL import Image
from face_parser import parserModel
from skimage.color import rgb2lab
from skimage.segmentation import slic, mark_boundaries
from skimage.color import rgb2lab
import matplotlib.pyplot as plt


def show(title, img):
    cv2.imshow(title, img)
    while True:
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break        
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:        
            break        
    cv2.destroyAllWindows()


image_path = "/mnt/d/datasets/TONO/bkg/S464252649-GM-A60-EEA-T15-S1-LH-C1-I0045-F00.png"

parser = parserModel.FaceParser()

with torch.no_grad():
    img = Image.open(image_path)
    image = img.resize((512, 512), Image.BILINEAR)
    img = parser.to_tensor(image)
    img = torch.unsqueeze(img, 0)
    out = parser.net(img)[0]
    parsing = out.squeeze(0).cpu().numpy().argmax(0)

# Convert the PIL image to a numpy array
image_np = np.array(image)

# Extract the background pixels (where parsing == 0)
background_mask = (parsing == 0).astype(np.uint8)
image_lab = rgb2lab(image_np)

# Perform SLIC superpixel segmentation
segments = slic(image_lab, n_segments=300, compactness=10, sigma=0, start_label=1)

viz = mark_boundaries(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), segments, color=(0, 0, 1))
viz *= 255
viz = viz.astype(np.uint8)
cv2.imwrite("slic.png", viz)

# Find the unique segments
all_segments = np.unique(segments)

# Only keep segments where ALL pixels belong to the background
gradient_mask = np.zeros_like(background_mask)
for segment in all_segments:
    segment_mask = (segments == segment)
    # Check if all pixels of the segment are background pixels
    if np.all(background_mask[segment_mask] == 1):
        gradient_mask[segment_mask] = 1


viz = mark_boundaries(gradient_mask * 255, segments, color=(0, 0, 1))
viz *= 255
viz = viz.astype(np.uint8)
cv2.imwrite("gradient_mask.png", viz)


# Convert the image to grayscale
gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)
magnitude = cv2.magnitude(dx, dy)

viz = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
viz[gradient_mask != 1] = 0
viz = np.clip(viz, 0, 255)
cv2.imwrite("magnitude.png", viz)

# Mask out the non-background areas in the magnitude image
background_magnitude = magnitude[gradient_mask == 1]
mean = background_magnitude.mean()
# Compute 1 - sigmoid(mean - threshold)
sig = 1 / (1 + np.exp(10.793404579162598 - mean))
print(1 - sig)