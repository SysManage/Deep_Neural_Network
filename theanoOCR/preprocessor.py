import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def pre_processing():

    path = os.path.join(os.path.dirname(__file__), 'Intermidiate_Step')
    path2 = os.path.join(os.path.dirname(__file__), 'Original_input_letters')

    listing = os.listdir(path)

    listing.sort()
    for file in listing:

        inputImage = cv2.imread(path+"/"+file)

        img = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(img, (5, 5), 0)
        ret3,th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

        kernel = np.ones((6,6),np.uint8)
        morphed = cv2.morphologyEx(th3, cv2.MORPH_HITMISS, element)
        # open = cv2.morphologyEx(open1, cv2.MORPH_HITMISS, element)
        temp=np.copy(morphed)

        im2, contours, hierarchy = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask = np.ones(morphed.shape[:2], dtype="uint8") * 255

        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # discard areas that are too large
            if h > 300 and w > 300:
                cv2.drawContours(mask, [contour], -1, 0, cv2.FILLED)


            # discard areas that are too small
            if h < 35 or w < 35:
                cv2.drawContours(mask, [contour], -1, 0, cv2.FILLED)

        new = cv2.bitwise_and(temp,mask)
        eroded = cv2.erode(new, element)

        cv2.imwrite(path2+"/"+file, eroded)

pre_processing()
