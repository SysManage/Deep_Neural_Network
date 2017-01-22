import cv2

import os


def getImage():
    path = os.path.join(os.path.dirname(__file__), 'Original_letters')
    inImg = cv2.imread(path + os.sep + "ha-056.jpg")

    img = cv2.cvtColor(inImg, cv2.COLOR_BGR2GRAY)

    return img
