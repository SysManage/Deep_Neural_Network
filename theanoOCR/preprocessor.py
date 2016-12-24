import cv2
import numpy as np
import os


def pre_processing():

    path = os.path.join(os.path.dirname(__file__), 'Intermidiate_Step')
    path2 = os.path.join(os.path.dirname(__file__), 'Original_input_letters')

    listing = os.listdir(path)

    listing.sort()
    for file in listing:

        img = cv2.imread(path+"/"+file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)

        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        done = False

        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=4)

        while not done:
            eroded = cv2.erode(img, kernel, element)
            temp = cv2.dilate(eroded, kernel, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        skel = cv2.dilate(skel, kernel)
        skel = cv2.morphologyEx(skel, cv2.MORPH_OPEN, kernel)
        # skel = cv2.GaussianBlur(skel, (3, 3), 0)
        skel = cv2.medianBlur(skel, 5)
        # skel = cv2.bilateralFilter(skel, 9, 75, 75)
        cv2.imwrite(path2+"/"+file, skel)

# pre_processing()
