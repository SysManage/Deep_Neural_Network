import os

import cv2
import numpy as np

save_path = os.path.join(os.path.dirname(__file__), 'seg_images')
open_path = os.path.join(os.path.dirname(__file__), 'Input_large_images')


def openImage(dir, file_name):
    inImg = cv2.imread(dir + os.sep + file_name)
    img = cv2.cvtColor(inImg, cv2.COLOR_BGR2GRAY)
    return img


def binarizeImage(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def removeLines(bin_image):  # must be a binary image

    minLineLength = 100
    maxLineGap = 70
    lines = cv2.HoughLinesP(bin_image, 5, np.pi / 180, 1000, minLineLength, maxLineGap)
    if lines is not None:
        a, b, c = lines.shape
        for i in range(a):
            cv2.line(bin_image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), 0, 3, cv2.LINE_AA)


file_name = "word2.jpg"
img = openImage(open_path, file_name)
th3 = binarizeImage(img)
removeLines(th3)
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel, iterations=1)

erosion_clone = erosion.copy()
print(erosion.shape)
# close = cv2.morphologyEx(th3, cv2.MORPH_HITMISS, kernel)
# open = cv2.morphologyEx(close, cv2.MORPH_HITMISS, kernel)
_, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

count = 0
for contour in contours:
    # get rectangle bounding contour
    [x, y, w, h] = boundingRect = cv2.boundingRect(contour)

    # discard areas that are too large
    if h > 300 and w > 300:
        continue

    # discard areas that are too small
    elif h < 35 or w < 35:
        continue

    else:
        count += 1
        character_mask = np.zeros((h, w), dtype="uint8")
        width_fixed, height_fixed = 200, 200
        character_mask = cv2.drawContours(character_mask, [contour], -1, 1, cv2.FILLED, offset=(-x, -y))
        empty_mask = np.zeros((h, w), np.uint8)
        result = np.where(character_mask == 0, empty_mask,
                          erosion_clone[y:y + h, x:x + w])  # Take Region of interest from erosion_clone
        roi_height = result.shape[0]
        roi_width = result.shape[1]
        aspectRatio = roi_width / roi_height  # width/h
        height = width = 0
        if roi_height > roi_width:
            height = height_fixed  # threshold
            width = height * aspectRatio
        else:
            width = width_fixed
            height = width / aspectRatio

        res = cv2.resize(result, (int(width), int(height)),
                         interpolation=cv2.INTER_CUBIC )
        top = bottom = int((height_fixed - height) / 2)
        left = right = int((width_fixed - width) / 2)
        res = cv2.copyMakeBorder(res, top, bottom, left, right, cv2.BORDER_ISOLATED)
        cv2.imwrite(save_path + os.sep + str(count) + " of " + file_name, res)

print("Chars found: " + str(count))
