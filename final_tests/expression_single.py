import sys
sys.path.append('.')
from landmarks import landmark
import cv2
import numpy as np
import matplotlib.pyplot as plt


land = landmark.LandmarkRecognizer()

# image = cv2.imread("/mnt/d/datasets/TONO/sm/S120660598-GM-A33-EEA-T00-S1-LH-C1-I0034-F00.png")
image = cv2.imread("/mnt/d/datasets/TONO/icao/S62389544-GF-A82-EEA-T14-S1-LH-C1-I0050-F00.png")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

detected_boxes = land.detector(image_rgb)
box = detected_boxes[0]
shape = land.predictor(image_rgb, box)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hue = hsv[:, :, 0]
lips_mask = np.zeros_like(hue)
lips_landmarks = np.array([(p.x, p.y) for p in shape.parts()[48:60]], dtype=np.int32)
teeth_landmarks = np.array([(p.x, p.y) for p in shape.parts()[60:68]], dtype=np.int32)
lips_mask = cv2.fillPoly(lips_mask, [lips_landmarks], 255)
lips_without_teeth_mask = lips_mask.copy()
lips_without_teeth_mask = cv2.fillPoly(lips_without_teeth_mask, [teeth_landmarks], 0)
teeth_mask = np.zeros_like(hue)
teeth_mask = cv2.fillPoly(teeth_mask, [teeth_landmarks], 255)

teeth_percentage = np.count_nonzero(lips_without_teeth_mask) / np.count_nonzero(lips_mask)
print(teeth_percentage)

# cv2.imshow("lips", lips_mask)
# cv2.waitKey(0)
# cv2.imshow("teeth", teeth_mask)
# cv2.waitKey(0)

# lips_hist = cv2.calcHist([hue], [0], lips_mask, [180], [0, 180])
# lips_hist /= lips_hist.sum()
# lips_without_teeth_hist = cv2.calcHist([hue], [0], lips_without_teeth_mask, [180], [0, 180])
# lips_without_teeth_hist /= lips_without_teeth_hist.sum()
# teeth_hist = cv2.calcHist([hue], [0], teeth_mask, [180], [0, 180])
# teeth_hist /= teeth_hist.sum()

# plt.bar(np.arange(180), lips_hist.flatten(), label="lips", alpha=0.5)
# plt.bar(np.arange(180), lips_without_teeth_hist.flatten(), label="lips no teeth", alpha=0.5)
# plt.bar(np.arange(180), teeth_hist.flatten(), label="teeth", alpha=0.5)
# plt.legend()
# plt.show()
# plt.close()

lips_hist = cv2.calcHist([image], [0, 1, 2], lips_mask, [256, 256, 256], [0, 256, 0, 256, 0, 256])
lips_hist /= lips_hist.sum()
teeth_hist = cv2.calcHist([image], [0, 1, 2], teeth_mask, [256, 256, 256], [0, 256, 0, 256, 0, 256])
teeth_hist /= teeth_hist.sum()


hist_comparison = cv2.compareHist(lips_hist, teeth_hist, cv2.HISTCMP_CORREL)
print(hist_comparison)

SCALE = 2
image = cv2.resize(image, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
for i, part in enumerate(shape.parts()):
    image = cv2.circle(image, (part.x * SCALE, part.y * SCALE), 1, (0, 0, 255), -1)
    image = cv2.putText(image, str(i), (part.x * SCALE, part.y * SCALE), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

cv2.imshow("img", image)
while True:
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break        
    if cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) < 1:
        break        
cv2.destroyAllWindows()