import cv2
from matplotlib import pyplot as plt
import numpy as np
from theanoOCR import openimage as op


def removeLines(bin_image):  # must be a binary image

    minLineLength = 100
    maxLineGap = 70
    lines = cv2.HoughLinesP(bin_image, 5, np.pi / 180, 1000, minLineLength, maxLineGap)
    if (lines != None):
        a, b, c = lines.shape
        for i in range(a):
            cv2.line(bin_image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), 0, 3, cv2.LINE_AA)


inImg = op.getImage()
img = cv2.cvtColor(inImg, cv2.COLOR_BGR2GRAY)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)

ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Blurred", th3)
cv2.waitKey()
removeLines(th3)
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel, iterations=1)
#####
#
# sure_bg = cv2.dilate(erosion, kernel, iterations=3)
#
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(erosion, cv2.DIST_L2, 5)
# ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
#
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg, sure_fg)
#
# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers + 1
# # Now, mark the region of unknown with zero
# markers[unknown == 255] = 0
#
# markers = cv2.watershed(inImg, markers)
# inImg[markers == -1] = [255, 0, 0]
#
# print(len(markers))
# cv2.imshow("dsa", inImg)
# cv2.imwrite("watershed.jpg", inImg)
# cv2.waitKey()
###########
erosion_clone = np.copy(erosion)  # np.array(erosion,copy=True)
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
        character_mask = cv2.drawContours(character_mask, [contour], -1, 1, cv2.FILLED, offset=(-x, -y))
        empty_mask = np.zeros((h, w), np.uint8)
        result = np.where(character_mask == 0, empty_mask,
                          erosion_clone[y:y + h, x:x + w])  # Take Region of interest from erosion_clone
        cv2.imwrite(str(count) + "contoured.jpg", result)

print("Chars found: "+count)

