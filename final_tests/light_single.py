import sys
sys.path.append('.')
import numpy as np
import cv2
import torch
from PIL import Image
from face_parser import parserModel
from landmarks import landmark


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


image_path = "/mnt/d/datasets/TONO/light_fn/S334613033-GM-A58-EAF-T14-S1-LH-C1-I0047-F00.png"

parser = parserModel.FaceParser()
land = landmark.LandmarkRecognizer()

with torch.no_grad():
    img = Image.open(image_path)
    image = img.resize((512, 512), Image.BILINEAR)
    img = parser.to_tensor(image)
    img = torch.unsqueeze(img, 0)
    out = parser.net(img)[0]
    parsing = out.squeeze(0).cpu().numpy().argmax(0)
image_rgb = np.array(image)
image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
detected_boxes = land.detector(image_rgb)
box = detected_boxes[0]
shape = land.predictor(image_rgb, box)
mask = (parsing == 1).astype(np.uint8)
masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


left_eye_start = 36
left_eye_end = 39
right_eye_start = 42
right_eye_end = 45

# Calculate middle points of each eye
left_eye_middle_x = (shape.part(left_eye_start).x + shape.part(left_eye_end).x) / 2
left_eye_middle_y = (shape.part(left_eye_start).y + shape.part(left_eye_end).y) / 2

right_eye_middle_x = (shape.part(right_eye_start).x + shape.part(right_eye_end).x) / 2
right_eye_middle_y = (shape.part(right_eye_start).y + shape.part(right_eye_end).y) / 2

middle_x = (left_eye_middle_x + right_eye_middle_x) / 2
middle_y = (left_eye_middle_y + right_eye_middle_y) / 2


# Calculate inter-eye distance (IED) between middle points
inter_eye_distance = landmark.math_distance(left_eye_middle_x, left_eye_middle_y, right_eye_middle_x, right_eye_middle_y)

mouth_up = shape.part(62)
mouth_down = shape.part(66)

# Calculate the midpoint
mouth_mid_x = (mouth_up.x + mouth_down.x) / 2
mouth_mid_y = (mouth_up.y + mouth_down.y) / 2

emd = landmark.math_distance(middle_x, middle_y, mouth_mid_x, mouth_mid_y)

# Calculate the point P (emd / 2 up from M)
P_x = middle_x
P_y = middle_y - (emd / 2)

# Calculate the side length of the square
side_length = 0.3 * inter_eye_distance

# Calculate the bottom-left corner of the square
bottom_left_x = P_x - (0.15 * inter_eye_distance)
bottom_left_y = P_y

# Draw the square
top_left_f = (int(bottom_left_x), int(bottom_left_y - side_length))
bottom_right_f = (int(bottom_left_x + side_length), int(bottom_left_y))

# Calculate the point P' (emd / 2 down from mouth_mid)
chin_x = mouth_mid_x
chin_y = mouth_mid_y + (emd / 2)

# Calculate the top-left corner of the square
top_left_x = chin_x - (0.15 * inter_eye_distance)
top_left_y = chin_y - (0.15 * inter_eye_distance)

# Calculate the coordinates of the square
top_left_c = (int(top_left_x), int(top_left_y))
bottom_right_c = (int(top_left_x + side_length), int(top_left_y + side_length))

P_x = middle_x
P_y = middle_y + (emd / 2)

# Calculate the top-right corner for the right cheek square (0.5 * IED to the left from P)
right_cheek_top_right_x = P_x - (0.5 * inter_eye_distance)
right_cheek_top_right_y = P_y

# Calculate the coordinates of the right cheek square
top_left_r = (int(right_cheek_top_right_x - side_length), int(right_cheek_top_right_y))
bottom_right_r = (int(right_cheek_top_right_x), int(right_cheek_top_right_y + side_length))

# Calculate the top-left corner for the left cheek square (0.5 * IED to the right from P)
left_cheek_top_left_x = P_x + (0.5 * inter_eye_distance)
left_cheek_top_left_y = P_y

# Calculate the coordinates of the left cheek square
top_left_l = (int(left_cheek_top_left_x), int(left_cheek_top_left_y))
bottom_right_l = (int(left_cheek_top_left_x + side_length), int(left_cheek_top_left_y + side_length))

mi_forehead = landmark.calculate_mean_intensity(image_rgb, top_left_f, bottom_right_f)
mi_chin = landmark.calculate_mean_intensity(image_rgb, top_left_c, bottom_right_c)
mi_right_cheek = landmark.calculate_mean_intensity(image_rgb, top_left_r, bottom_right_r)
mi_left_cheek = landmark.calculate_mean_intensity(image_rgb, top_left_l, bottom_right_l)

# Collect mean intensities in a dictionary for each channel
mi_squares = {
    "forehead": mi_forehead,
    "chin": mi_chin,
    "right_cheek": mi_right_cheek,
    "left_cheek": mi_left_cheek
}

means = []
for channel in range(3):
    channel_intensities = [mi_squares[patch][channel] for patch in mi_squares]
    low = min(channel_intensities)
    high = max(channel_intensities)
    means.append(low / high)
gray_value = 0.2126 * means[0] + 0.7152 * means[1] + 0.0722 * means[2] - 0.5
print(gray_value, 1 / (1 + np.exp(-gray_value)))


viz = image_bgr.copy()
viz = cv2.rectangle(viz, top_left_f, bottom_right_f, (0, 255, 0), 1)
viz = cv2.rectangle(viz, top_left_c, bottom_right_c, (0, 255, 0), 1)
viz = cv2.rectangle(viz, top_left_r, bottom_right_r, (0, 255, 0), 1)
viz = cv2.rectangle(viz, top_left_l, bottom_right_l, (0, 255, 0), 1)
viz = cv2.line(viz, (int(left_eye_middle_x), int(left_eye_middle_y)), (int(right_eye_middle_x), int(right_eye_middle_y),), (0, 255, 0), 1)
viz = cv2.circle(viz, (int(left_eye_middle_x), int(left_eye_middle_y)), 1, (0, 0, 255), 2)
viz = cv2.circle(viz, (int(right_eye_middle_x), int(right_eye_middle_y)), 1, (0, 0, 255), 2)
viz = cv2.circle(viz, (int(middle_x), int(middle_y)), 1, (0, 0, 255), 2)
viz = cv2.circle(viz, (int(mouth_mid_x), int(mouth_mid_y)), 1, (0, 0, 255), 2)
viz = cv2.circle(viz, (int(P_x), int(P_y)), 1, (0, 0, 255), 2)
viz = cv2.circle(viz, (int(chin_x), int(chin_y)), 1, (0, 0, 255), 2)
viz = cv2.circle(viz, (int(right_cheek_top_right_x), int(right_cheek_top_right_y)), 1, (0, 0, 255), 2)
viz = cv2.circle(viz, (int(left_cheek_top_left_x), int(left_cheek_top_left_y)), 1, (0, 0, 255), 2)

show("viz", viz)

print(mi_squares)

show("mask", masked)

gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
show("gray", gray)

# Highlight pixels that are too dark or bright
dark = cv2.inRange(gray, 0, 70)
# Ignore the mask in the pixels that are too dark
dark = cv2.bitwise_and(dark, mask)
bright = cv2.inRange(gray, 230, 255)

dark_percentage = np.count_nonzero(dark) / np.count_nonzero(mask)
bright_percentage = np.count_nonzero(bright) / np.count_nonzero(mask)

print("Global dark percentage: ", dark_percentage)
print("Global bright percentage: ", bright_percentage)
print("Score: ", 1 - dark_percentage - bright_percentage)

f_dark_percentage = np.count_nonzero(dark[top_left_f[1]:bottom_right_f[1], top_left_f[0]:bottom_right_f[0]]) / np.count_nonzero(mask[top_left_f[1]:bottom_right_f[1], top_left_f[0]:bottom_right_f[0]])
f_bright_percentage = np.count_nonzero(bright[top_left_f[1]:bottom_right_f[1], top_left_f[0]:bottom_right_f[0]]) / np.count_nonzero(mask[top_left_f[1]:bottom_right_f[1], top_left_f[0]:bottom_right_f[0]])

c_dark_percentage = np.count_nonzero(dark[top_left_c[1]:bottom_right_c[1], top_left_c[0]:bottom_right_c[0]]) / np.count_nonzero(mask[top_left_c[1]:bottom_right_c[1], top_left_c[0]:bottom_right_c[0]])
c_bright_percentage = np.count_nonzero(bright[top_left_c[1]:bottom_right_c[1], top_left_c[0]:bottom_right_c[0]]) / np.count_nonzero(mask[top_left_c[1]:bottom_right_c[1], top_left_c[0]:bottom_right_c[0]])

r_dark_percentage = np.count_nonzero(dark[top_left_r[1]:bottom_right_r[1], top_left_r[0]:bottom_right_r[0]]) / np.count_nonzero(mask[top_left_r[1]:bottom_right_r[1], top_left_r[0]:bottom_right_r[0]])
r_bright_percentage = np.count_nonzero(bright[top_left_r[1]:bottom_right_r[1], top_left_r[0]:bottom_right_r[0]]) / np.count_nonzero(mask[top_left_r[1]:bottom_right_r[1], top_left_r[0]:bottom_right_r[0]])

l_dark_percentage = np.count_nonzero(dark[top_left_l[1]:bottom_right_l[1], top_left_l[0]:bottom_right_l[0]]) / np.count_nonzero(mask[top_left_l[1]:bottom_right_l[1], top_left_l[0]:bottom_right_l[0]])
l_bright_percentage = np.count_nonzero(bright[top_left_l[1]:bottom_right_l[1], top_left_l[0]:bottom_right_l[0]]) / np.count_nonzero(mask[top_left_l[1]:bottom_right_l[1], top_left_l[0]:bottom_right_l[0]])

print("Forehead dark percentage: ", f_dark_percentage)
print("Forehead bright percentage: ", f_bright_percentage)
print("Chin dark percentage: ", c_dark_percentage)
print("Chin bright percentage: ", c_bright_percentage)
print("Right cheek dark percentage: ", r_dark_percentage)
print("Right cheek bright percentage: ", r_bright_percentage)
print("Left cheek dark percentage: ", l_dark_percentage)
print("Left cheek bright percentage: ", l_bright_percentage)

viz = masked.copy()
viz[dark > 0] = [0, 0, 255]
viz[bright > 0] = [255, 0, 0]
show("viz", viz)
