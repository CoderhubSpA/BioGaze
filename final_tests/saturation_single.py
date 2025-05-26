import sys
sys.path.append('.')
import numpy as np
import cv2
import torch
from PIL import Image
from face_parser import parserModel
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


image_path = "/mnt/d/datasets/TONO/sat_fn/S6470524-GF-A29-EAS-T14-S1-LH-C1-I0059-F00.png"

parser = parserModel.FaceParser()

with torch.no_grad():
    img = Image.open(image_path)
    image = img.resize((512, 512), Image.BILINEAR)
    img = parser.to_tensor(image)
    img = torch.unsqueeze(img, 0)
    out = parser.net(img)[0]
    parsing = out.squeeze(0).cpu().numpy().argmax(0)
image_rgb = np.array(image)
image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

mask = parsing == 1
sat_channel = image_hsv[..., 1][mask]
histogram = np.histogram(sat_channel, bins=256, range=(0, 256))[0]
histogram = histogram / np.sum(histogram)
total = np.count_nonzero(mask)
over_mask = sat_channel > 200
under_mask = sat_channel < 40
oversaturated = np.count_nonzero(over_mask) / total
undersaturated = np.count_nonzero(under_mask) / total

print("Mean saturation: ", np.mean(sat_channel))
print("Mean saturation (oversaturated): ", np.mean(sat_channel[over_mask]))
print("Mean saturation (undersaturated): ", np.mean(sat_channel[under_mask]))
print("Oversaturated: ", oversaturated)
print("Undersaturated: ", undersaturated)

plt.plot(histogram)
plt.show()
plt.close()

show("mask", mask.astype(np.uint8) * 255)
show("saturation", image_hsv[..., 1])
viz = image_bgr.copy()
over_mask_viz = image_hsv[..., 1] > 200
over_mask_viz = over_mask_viz & mask
under_mask_viz = image_hsv[..., 1] < 40
under_mask_viz = under_mask_viz & mask
viz[over_mask_viz] = (0, 0, 255)
viz[under_mask_viz] = (255, 0, 0)
show("viz", viz)
